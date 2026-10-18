"""Experimental Comfy-shaped Base sampler primitives. Never used by the live bridge."""
import math
import torch

def simple_schedule(steps,alpha):
 if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=100:raise ValueError('Steps must be1..100')
 if isinstance(alpha,bool) or not math.isfinite(alpha) or not 0<alpha<=1000:raise ValueError('Alpha must be finite positive')
 table_t=torch.arange(1,10001,dtype=torch.float32)/10000
 # Same reviewed Comfy v0.41 ModelSamplingFlux expression, not a uniform-grid alias.
 exp_shift=math.exp(math.log(alpha));table=exp_shift/(exp_shift+(1/table_t-1))
 return torch.tensor([float(table[-(1+int(i*(len(table)/steps)))]) for i in range(steps)]+[0.0],dtype=torch.float32)

def latent_noise(seed,h,w):
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('Exact64-bit integer seed required')
 if not all(isinstance(n,int) and not isinstance(n,bool) and 1<=n<=256 for n in [h,w]):raise ValueError('Latent dimensions out of supported bounds')
 # Comfy draws BCHW CPU FP32; BFL native draws token-major BF16-rounded noise.
 # These are deliberately distinct and are not seed-equivalent layouts.
 noise=torch.randn((1,128,h,w),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
 return noise.permute(0,2,3,1).reshape(h*w,128)

def run_base_steps(dit,packed,schedule,guidance,progress=None,cancelled=None):
 if getattr(dit,'flow_map',False):raise ValueError('Base CFG sampler rejects a flow-map checkpoint')
 if isinstance(guidance,bool) or not isinstance(guidance,(int,float)) or not math.isfinite(guidance) or not 0<=guidance<=20:raise ValueError('Experimental Base CFG must be finite0..20')
 if packed.data['image'].shape[0]%2:raise ValueError('Base pack must contain unconditional and conditional samples')
 for i in range(len(schedule)-1):
  if cancelled and cancelled():raise InterruptedError('Base generation cancelled')
  packed.timesteps['image'].fill_(schedule[i].item())
  uncond,cond=dit(packed).chunk(2)
  velocity=uncond+guidance*(cond-uncond)
  state=packed.data['image'].chunk(2)[0]+(schedule[i+1]-schedule[i]).item()*velocity
  packed.data['image']=torch.cat([state,state])
  if progress:progress(i+1,len(schedule)-1)
 return packed.data['image'].chunk(2)[0]

def run_flow_map_steps(dit,packed,schedule,progress=None,cancelled=None):
 if not getattr(dit,'flow_map',False):raise ValueError('Distilled sampler requires a target-time flow-map checkpoint')
 for i in range(len(schedule)-1):
  if cancelled and cancelled():raise InterruptedError('Distilled generation cancelled')
  packed.timesteps['image'].fill_(schedule[i].item())
  target=schedule[i+1].reshape(1).to(packed.data['image'].device)
  packed.data['image']=packed.data['image']+(schedule[i+1]-schedule[i]).item()*dit(packed,timesteps_ref=target)
  if progress:progress(i+1,len(schedule)-1)
 return packed.data['image']
