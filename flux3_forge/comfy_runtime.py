"""Explicit experimental runtime, staged only. Native routing requires the explicit experimental profile setting."""
import gc
from collections import OrderedDict
import torch
from .comfy_profiles import require_profile
from .comfy_pipeline import build_pipeline
from .runtime import RuntimeModel
from .memory import bound_vae_convolutions

class ExperimentalRuntime(RuntimeModel):
    @property
    def dit(self):
        return self.pipeline.dits[self.manifest.get('model_family','distilled')]
    def __init__(self, profile, memory_mode='balanced', *, allow_unverified=False):
        if allow_unverified is not True:
            raise ValueError('Real-weight Comfy acceptance is missing; explicit experimental opt-in is required')
        manifest = require_profile(profile)
        if memory_mode not in ('balanced','resident'):
            raise ValueError('Use balanced or explicit resident memory policy')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Experimental Comfy inference requires a CUDA GPU with BF16 support')
        self.manifest = dict(manifest); self.memory_mode = memory_mode
        self.pipeline = None; self._cpu_backing = {}; self._prompt_cache = OrderedDict(); self._prompt_cache_bytes = 0
        failure = None
        try:
            self.pipeline = build_pipeline(profile, device='cuda' if memory_mode=='resident' else 'cpu', offload=memory_mode=='balanced')
            bound_vae_convolutions(self.pipeline.autoencoder)
        except Exception as error:
            import traceback
            failure = traceback.format_exc()
        if failure:
            self.unload()
            raise RuntimeError('Experimental profile construction failed and was released: '+failure)

    def generate(self, request, progress=None, cancelled=None):
        if request.get('adapters'):
            raise ValueError('Comfy LoRA/LoKR full-profile acceptance is missing; adapters are disabled on this experimental route')
        if request.get('spectrum'):
            raise ValueError('Spectrum is not validated for experimental Comfy profiles')
        try:
            if self.manifest.get('model_family')=='base' or request.get('exact_workflow') is True:
                from .exact_base_runtime import generate_exact
                image, metadata = generate_exact(self,request,progress,cancelled)
            else:
                image, metadata = super().generate(request, progress, cancelled)
        except torch.OutOfMemoryError:
            self.unload()
            raise RuntimeError('Experimental Comfy profile OOM; model released. No smaller-card support is implied') from None
        metadata.update(experimental=True, real_weight_acceptance=False,
                        encoder_quantization=self.manifest['encoder_quantization'],
                        source_revision=self.manifest['source_revision'])
        return image, metadata

    def __enter__(self): return self
    def __exit__(self, *_): self.unload()
