"""Staged Base TXT2IMG execution. Matching real-weight acceptance is still absent."""
import math
import torch
from .runtime import Cancelled,_cancel
from .exact_base_sampling import latent_noise,simple_schedule,run_base_steps,run_flow_map_steps
from .workflow_support import controls
from .vendor.flux3.autoencoder import latent_ids,latent_shape
from .vendor.flux3.pipeline import pack,scaled_ids,to_pil

@torch.inference_mode()
def generate_exact(owner,request,progress=None,cancelled=None):
 # No allocations/loading before validating this exact graph's supported subset.
 family=owner.manifest.get('model_family')
 if family not in ('base','distilled') or owner.dit.flow_map!=(family=='distilled'):raise ValueError('Validated explicit Base/distilled profile required')
 if request.get('images'):raise ValueError('This staged exact-graph route is TXT2IMG; comparison images are not conditioning')
 if request.get('negative_prompt','')!='':raise ValueError('The supplied graph uses an empty unconditional caption')
 if request.get('batch_size',1)!=1:raise ValueError('Only exact graph batch_size1 is implemented')
 if request.get('sampler','euler')!='euler' or request.get('scheduler','simple')!='simple':raise ValueError('Only the supplied Euler/simple contract is implemented')
 prompt=request.get('prompt','')
 if not isinstance(prompt,str) or not prompt.strip():raise ValueError('A complete literal caption is required')
 width=request.get('width',1776);height=request.get('height',1776)
 if not all(isinstance(v,int) and not isinstance(v,bool) and 64<=v<=4096 and v%16==0 for v in [width,height]):raise ValueError('Dimensions must be64..4096 and divisible by16')
 cfg=request.get('cfg',3.5 if family=='base' else 1)
 if isinstance(cfg,bool) or not isinstance(cfg,(int,float)) or not math.isfinite(cfg) or not 0<=cfg<=20:raise ValueError('Experimental CFG must be finite0..20')
 if family=='distilled' and cfg!=1:raise ValueError('Distilled exact workflow requires CFG1')
 workflow=controls(dict(request,seed_bits=request.get('seed_bits',64)));steps=request.get('steps',25 if family=='base' else 16)
 schedule=simple_schedule(steps,workflow['alpha']);_cancel(cancelled)
 if progress:progress(0,steps)
 unconditional=owner._encode_prompt('',cancelled) if family=='base' else None
 conditional=owner._encode_prompt(prompt,cancelled)
 _cancel(cancelled);h,w=latent_shape(height,width)
 state=latent_noise(workflow['seed'],h,w).to('cuda');ids=latent_ids(h,w,'cuda')
 texts=[tuple(v.to('cuda') for v in pair) for pair in ([unconditional,conditional] if family=='base' else [conditional])]
 packed=pack([[text,None,(state,scaled_ids(ids.clone()))] for text in texts])
 dit=owner.dit
 # Keep unvalidated adapters/Spectrum/native patches outside the staged Base path.
 dit._forge_attention=None;dit._forge_spectrum=None
 try:
  owner._stage(dit,'cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   try:tokens=run_base_steps(dit,packed,schedule,cfg,progress,cancelled) if family=='base' else run_flow_map_steps(dit,packed,schedule,progress,cancelled)
   except InterruptedError as error:raise Cancelled(str(error)) from error
 finally:owner._offload(dit)
 _cancel(cancelled)
 ae=owner.pipeline.autoencoder
 try:
  owner._stage(ae,'cuda');image=to_pil(ae.decode(tokens,ids))
 finally:owner._offload(ae)
 return image,dict(model='flux-3-image-klein-base' if family=='base' else 'flux-3-image-klein',model_family=family,flow_map=family=='distilled',sampler='euler',scheduler='comfy-simple10000',noise_layout='Comfy BCHW CPU FP32; token-major permutation',prompt=prompt,negative_prompt='',steps=steps,cfg=cfg,seed=workflow['seed'],seed_bits=workflow['seed_bits'],alpha=workflow['alpha'],width=width,height=height,experimental=True,real_weight_acceptance=False)

def generate_base(owner,request,progress=None,cancelled=None):
 if owner.manifest.get('model_family')!='base':raise ValueError('Validated Base profile required')
 return generate_exact(owner,request,progress,cancelled)
