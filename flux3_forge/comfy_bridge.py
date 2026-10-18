"""Opt-in native Forge routing for strictly validated Comfy component profiles."""
import threading
from pathlib import Path
_LOCK=threading.RLock()
_MODEL=None
_KEY=None

def pinned_file(folder,pin):
 """Resolve one canonical local asset by size/SHA, independent of AutoLink rename."""
 import hashlib
 folder=Path(folder).resolve();matches=[]
 for candidate in sorted(folder.glob('*.safetensors')):
  path=candidate.resolve()
  if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size!=pin[0]:continue
  digest=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
  if digest.hexdigest()==pin[1]:matches.append(path)
 if len(matches)!=1:raise ValueError(f'Expected exactly one pinned local component in {folder}; found {len(matches)}. No download or format fallback.')
 return matches[0]

def configured_asset(setting,models):
 from .comfy_profiles import validate_profile,require_profile,PINNED
 root=Path(models)
 family=str(setting('flux3_comfy_family','base'))
 precision=str(setting('flux3_comfy_precision','int8_convrot'))
 encoder_precision=str(setting('flux3_comfy_encoder_precision','int8_convrot'))
 stem='flux3-image-klein-base' if family=='base' else 'flux3-image-klein'
 suffix='int8-convrot' if precision=='int8_convrot' else 'bf16'
 enc='qwen3vl_4b_int8_convrot.safetensors' if encoder_precision=='int8_convrot' else 'qwen3vl_4b_bf16.safetensors'
 profile=validate_profile(root,checkpoint=root/'Stable-diffusion/FLUX-3'/f'{stem}-{suffix}.safetensors',encoder=pinned_file(root/'text_encoder/FLUX-3',PINNED[('encoder',encoder_precision)]),vae=root/'VAE/FLUX-3/FLUX.2-VAE--For-FLUX-3.safetensors',support_dir=root/'FLUX3/flux-3-image-klein/text_encoder',quantization=precision,encoder_quantization=encoder_precision,experimental=True,model_family=family)
 manifest=dict(require_profile(profile))
 manifest.update(comfy_profile=profile,model_dir=str(root),display_name=f'FLUX 3 Image Klein {family.title()} ({precision.upper()} + {encoder_precision.upper()} TE; experimental)')
 return manifest

def generate(request,progress=None,cancelled=None):
 global _MODEL,_KEY
 from .comfy_profiles import require_profile
 from .comfy_runtime import ExperimentalRuntime
 profile=request['comfy_profile'];require_profile(profile)
 key=(profile.identities,request.get('memory_mode','balanced'))
 with _LOCK:
  if _MODEL is None or _KEY!=key:
   unload();_MODEL=ExperimentalRuntime(profile,key[1],allow_unverified=True);_KEY=key
  try:return _MODEL.generate(request,progress,cancelled)
  except Exception:
   unload();raise

def unload():
 global _MODEL,_KEY
 with _LOCK:
  if _MODEL is not None:_MODEL.unload()
  _MODEL=None;_KEY=None
