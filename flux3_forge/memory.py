"""Bound convolution workspace without tiling normalization or image attention."""
import torch
from torch import nn
from torch.nn import functional as F


def release_pressure():
    if torch.cuda.is_available():
        reserved,allocated=torch.cuda.memory_reserved(),torch.cuda.memory_allocated()
        total=torch.cuda.get_device_properties(0).total_memory
        if reserved>total*.72 and reserved-allocated>total*.12:
            torch.cuda.empty_cache()


def upsample_conv(conv,x,threshold=1024,chunk_rows=128):
    """Fuse nearest-neighbor expansion with row-wise convolution, preserving halos."""
    height,width=x.shape[-2]*2,x.shape[-1]*2
    if height<=threshold or conv.padding_mode!="zeros" or isinstance(conv.padding,str):
        return conv(F.interpolate(x,scale_factor=2.0,mode="nearest"))
    sh,sw=conv.stride
    ph,pw=conv.padding
    dh,dw=conv.dilation
    kh,kw=conv.kernel_size
    oh=(height+2*ph-dh*(kh-1)-1)//sh+1
    ow=(width+2*pw-dw*(kw-1)-1)//sw+1
    output=None
    for start in range(0,oh,chunk_rows):
        end=min(start+chunk_rows,oh)
        lo=start*sh-ph
        hi=(end-1)*sh-ph+dh*(kh-1)+1
        clipped_lo,clipped_hi=max(0,lo),min(height,hi)
        source_lo,source_hi=clipped_lo//2,(clipped_hi+1)//2
        part=F.interpolate(x[...,source_lo:source_hi,:],scale_factor=2.0,mode="nearest")
        offset=clipped_lo-source_lo*2
        part=part[...,offset:offset+clipped_hi-clipped_lo,:]
        if lo<0 or hi>height:
            part=F.pad(part,(0,0,max(0,-lo),max(0,hi-height)))
        result=F.conv2d(part,conv.weight,conv.bias,conv.stride,(0,pw),conv.dilation,conv.groups)
        if output is None:
            output=result.new_empty((x.shape[0],conv.out_channels,oh,ow))
        output[...,start:end,:]=result
    if output.is_cuda:
        release_pressure()
    return output


def norm_swish_conv(norm,conv,x,threshold=1024,chunk_rows=128,inplace=False):
    """Preserve full-image GroupNorm statistics while streaming its convolution."""
    if (x.shape[-2]<=threshold or x.dtype!=torch.float32 or not x.is_contiguous()
            or not isinstance(norm,nn.GroupNorm) or conv.padding_mode!="zeros" or isinstance(conv.padding,str)):
        value=norm(x)
        return conv(torch.sigmoid(value).mul_(value))
    batch,channels,height,width=x.shape
    groups=norm.num_groups
    variance,mean=torch.var_mean(x.reshape(batch,groups,-1),dim=-1,unbiased=False)
    mean=mean.reshape(batch,groups,1,1,1)
    inverse=torch.rsqrt(variance+norm.eps).reshape(batch,groups,1,1,1)
    sh,sw=conv.stride
    ph,pw=conv.padding
    dh,dw=conv.dilation
    kh,kw=conv.kernel_size
    oh=(height+2*ph-dh*(kh-1)-1)//sh+1
    ow=(width+2*pw-dw*(kw-1)-1)//sw+1
    inplace = bool(inplace and conv.in_channels==conv.out_channels and conv.stride==(1,1)
                   and conv.padding==(1,1) and conv.kernel_size==(3,3) and conv.dilation==(1,1))
    output=x if inplace else x.new_empty((batch,conv.out_channels,oh,ow))
    prefix=None
    for start in range(0,oh,chunk_rows):
        end=min(start+chunk_rows,oh)
        lo=start*sh-ph
        hi=(end-1)*sh-ph+dh*(kh-1)+1
        if inplace and start:
            part=torch.cat((prefix,x[...,start:min(hi,height),:]),dim=-2)
        else:
            part=x[...,max(lo,0):min(hi,height),:]
        tail=x[...,end-1:end,:].clone() if inplace else None
        value=(part.reshape(batch,groups,channels//groups,part.shape[-2],width)-mean)*inverse
        if norm.affine:
            value.mul_(norm.weight.reshape(1,groups,channels//groups,1,1))
            value.add_(norm.bias.reshape(1,groups,channels//groups,1,1))
        value=torch.sigmoid(value).mul_(value)
        value=value.reshape(batch,channels,part.shape[-2],width)
        if lo<0 or hi>height:
            value=F.pad(value,(0,0,max(0,-lo),max(0,hi-height)))
        result=F.conv2d(value,conv.weight,conv.bias,conv.stride,(0,pw),conv.dilation,conv.groups)
        output[...,start:end,:]=result
        prefix=tail
    if output.is_cuda:
        release_pressure()
    return output


class RowChunkedConv2d(nn.Conv2d):
    chunk_rows = 256
    threshold = 1024

    def forward(self, x):
        if x.shape[-2] <= self.threshold or self.padding_mode != "zeros" or isinstance(self.padding,str):
            return super().forward(x)
        height,width=x.shape[-2:]
        sh,sw=self.stride
        ph,pw=self.padding
        dh,dw=self.dilation
        kh,kw=self.kernel_size
        oh=(height+2*ph-dh*(kh-1)-1)//sh+1
        ow=(width+2*pw-dw*(kw-1)-1)//sw+1
        output=None
        for start in range(0,oh,self.chunk_rows):
            end=min(start+self.chunk_rows,oh)
            lo=start*sh-ph
            hi=(end-1)*sh-ph+dh*(kh-1)+1
            part=x[...,max(lo,0):min(hi,height),:]
            if lo<0 or hi>height:
                part=F.pad(part,(0,0,max(0,-lo),max(0,hi-height)))
            value=F.conv2d(part,self.weight,self.bias,self.stride,(0,pw),self.dilation,self.groups)
            if output is None:
                output=value.new_empty((x.shape[0],self.out_channels,oh,ow))
            output[...,start:end,:]=value
        if output.is_cuda:
            release_pressure()
        return output


def bound_vae_convolutions(module):
    """Keep original parameter names/shapes; constrain only inference workspaces."""
    for name,child in list(module.named_children()):
        if type(child) is nn.Conv2d:
            with torch.device("meta"):
                replacement=RowChunkedConv2d(child.in_channels,child.out_channels,child.kernel_size,
                    child.stride,child.padding,child.dilation,child.groups,child.bias is not None,
                    child.padding_mode,dtype=child.weight.dtype)
            replacement.weight=child.weight
            replacement.bias=child.bias
            replacement.train(child.training)
            setattr(module,name,replacement)
        else:
            bound_vae_convolutions(child)
