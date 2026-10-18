"""Strict Comfy INT8/ConvRot loading; original integer storage is retained.

Uses the installed Comfy-Kitchen layout implementation rather than inventing a
rotation/dequantization rule. CPU fixtures are not a real-model GPU verdict.
"""
from __future__ import annotations
import json
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

FORMAT = 'int8_tensorwise'

def _layout():
    try:
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    except ImportError as error:
        raise RuntimeError('INT8 ConvRot requires the existing Comfy-Kitchen TensorWiseINT8Layout; no packages are installed automatically') from error
    return QuantizedTensor, TensorWiseINT8Layout

def configuration(value):
    if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()>65536:
        raise ValueError('comfy_quant must be a bounded uint8 JSON tensor')
    try: conf=json.loads(bytes(value.cpu().tolist()).decode('utf-8'))
    except (ValueError,UnicodeError) as error: raise ValueError('Invalid comfy_quant JSON') from error
    if not isinstance(conf,dict) or conf.get('format')!=FORMAT:
        raise ValueError('Only int8_tensorwise (optionally ConvRot) is supported in this loader; no format substitution')
    unknown=set(conf)-{'format','convrot','convrot_groupsize','params','full_precision_matrix_mult'}
    if unknown: raise ValueError(f'Unsupported quantization settings: {sorted(unknown)}')
    params=conf.get('params',{})
    if not isinstance(params,dict) or set(params)-{'convrot','convrot_groupsize'}:
        raise ValueError('Unsupported INT8 layout params')
    for name in ('convrot','convrot_groupsize'):
        if name in conf and name in params and conf[name] != params[name]:
            raise ValueError('Conflicting top-level and nested ConvRot settings')
    convrot=conf.get('convrot',params.get('convrot',False))
    full=conf.get('full_precision_matrix_mult',False)
    if not isinstance(convrot,bool) or not isinstance(full,bool):raise ValueError('ConvRot/full-precision flags must be booleans')
    group=conf.get('convrot_groupsize',params.get('convrot_groupsize',256))
    if isinstance(group,bool) or not isinstance(group,int):raise ValueError('ConvRot group size must be an integer')
    if convrot and (group<4 or group&(group-1) or (group.bit_length()-1)%2):
        raise ValueError('Regular ConvRot Hadamard group size must be a power of four')
    return {'convrot':convrot,'group_size':group,'full_precision_matrix_mult':full}

class _Int8Weights(nn.Module):
    def __init__(self,qdata,scale,conf,compute_dtype):
        super().__init__()
        self.register_buffer('qdata',qdata,persistent=False)
        self.register_buffer('scale',scale,persistent=False)
        self.conf,self.compute_dtype=conf,compute_dtype

    @property
    def weight(self):
        QuantizedTensor,Layout=_layout()
        params=Layout.Params(scale=self.scale,orig_dtype=self.compute_dtype,
            orig_shape=tuple(self.qdata.shape),convrot=self.conf['convrot'],
            convrot_groupsize=self.conf['group_size'])
        return QuantizedTensor(self.qdata,'TensorWiseINT8Layout',params)

    def _apply(self,fn,recurse=True):
        # nn.Module.to(dtype=...) must not silently lower scale precision.
        scale=self._buffers.pop('scale')
        try:
            result=super()._apply(fn,recurse=recurse)
            if self.qdata.dtype!=torch.int8:raise RuntimeError('An operation attempted to change stored INT8 precision')
        finally:
            self._buffers['scale']=scale.to(device=self.qdata.device,dtype=torch.float32)
        return result

class Int8Linear(_Int8Weights):
    def __init__(self,qdata,scale,conf,compute_dtype,bias):
        super().__init__(qdata,scale,conf,compute_dtype)
        self.out_features,self.in_features=qdata.shape
        if bias:self.bias=nn.Parameter(torch.empty(self.out_features,device=qdata.device,dtype=compute_dtype),requires_grad=False)
        else:self.register_parameter('bias',None)

    def forward(self,x):
        weight=self.weight
        if self.conf['full_precision_matrix_mult']:weight=weight.dequantize()
        return F.linear(x,weight,self.bias)

class Int8Embedding(_Int8Weights):
    def __init__(self,qdata,scale,conf,compute_dtype):
        super().__init__(qdata,scale,conf,compute_dtype)
        self.num_embeddings,self.embedding_dim=qdata.shape

    def forward(self,indices):
        _,Layout=_layout()
        return Layout.dequantize_embedding(self.qdata,self.weight._params,indices)

def load_state(module,state,*,device='cpu',compute_dtype=torch.bfloat16,require_convrot=True,full_precision_matrix_mult=None):
    """Validate all packed leaves before replacement; load plain leaves strictly.

    The caller supplies a fresh/meta model. A failed load must discard it. This
    function never writes weights or converts the stored checkpoint format.
    """
    if compute_dtype not in (torch.float32,torch.float16,torch.bfloat16):raise ValueError('Unsupported compute dtype')
    if full_precision_matrix_mult is not None and not isinstance(full_precision_matrix_mult,bool):
        raise ValueError('Full precision override must be a boolean')
    from .comfy_checkpoint_layout import expand_state,attention_metadata
    state=attention_metadata(expand_state(dict(state),module));plans=[];consumed=set();has_convrot=False
    for key,value in state.items():
        if not key.endswith('.comfy_quant'):continue
        name=key[:-len('.comfy_quant')]
        try: layer=module.get_submodule(name)
        except AttributeError as error:raise ValueError(f'Quantized layer is not in the target architecture: {name}') from error
        if not isinstance(layer,(nn.Linear,nn.Embedding)):raise ValueError(f'Unsupported quantized layer type: {name}')
        conf=configuration(value);weight=state.get(name+'.weight');scale=state.get(name+'.weight_scale')
        if full_precision_matrix_mult is not None:conf['full_precision_matrix_mult']=full_precision_matrix_mult
        expected=tuple(layer.weight.shape)
        if not isinstance(weight,torch.Tensor) or weight.dtype!=torch.int8 or tuple(weight.shape)!=expected:
            raise ValueError(f'INT8 weight dtype/shape mismatch: {name}')
        if not isinstance(scale,torch.Tensor) or not scale.is_floating_point() or not (scale.numel()==1 or tuple(scale.shape)==(expected[0],1)):
            raise ValueError(f'Missing or invalid INT8 weight scale: {name}')
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):raise ValueError(f'INT8 scale must be finite and positive: {name}')
        if conf['convrot'] and (expected[1]%conf['group_size'] or tuple(scale.shape)!=(expected[0],1)):
            raise ValueError(f'ConvRot requires row scales and a divisible feature dimension: {name}')
        consumed.update([key,name+'.weight',name+'.weight_scale']);has_convrot|=conf['convrot']
        plans.append((name,layer,weight,scale,conf))
    if not plans or (require_convrot and not has_convrot):raise ValueError('Profile has no validated ConvRot layers')
    for key,value in state.items():
        if key not in consumed and key.endswith('.weight') and value.dtype not in (torch.bfloat16,torch.float16,torch.float32):
            raise ValueError(f'Unannotated integer or foreign weight cannot be interpreted as BF16: {key}')
    for name,old,weight,scale,conf in plans:
        packed=(Int8Linear(weight.to(device),scale.to(device=device,dtype=torch.float32),conf,compute_dtype,old.bias is not None)
                if isinstance(old,nn.Linear) else Int8Embedding(weight.to(device),scale.to(device=device,dtype=torch.float32),conf,compute_dtype))
        parent,_,leaf=name.rpartition('.');setattr(module.get_submodule(parent) if parent else module,leaf,packed)
    plain={key:(value.to(device=device,dtype=compute_dtype) if value.is_floating_point() else value.to(device))
           for key,value in state.items() if key not in consumed}
    module.load_state_dict(plain,strict=True,assign=True)
    return module.to(device).eval()

def load_checkpoint(module,path,**kwargs):
    return load_state(module,load_file(str(path),device='cpu'),**kwargs)
