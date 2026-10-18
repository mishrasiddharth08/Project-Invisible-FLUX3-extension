"""Forge-owned runtime for BFL's official FLUX 3 Image Klein pipeline."""

from __future__ import annotations

import platform
import gc
import threading
import sys
from collections import OrderedDict

import torch

from . import adapters as adapter_support
from .assets import validate_model_dir
from .memory import bound_vae_convolutions
from . import staging
from .spectrum import SpectrumConfig, SpectrumState
from .vendor.flux3.autoencoder import latent_ids, latent_shape
from .vendor.flux3.dit import IN_CHANNELS
from .vendor.flux3.pipeline import (
    Pipeline,
    pack,
    prepare_multi_reference,
    prepare_reference,
    scaled_ids,
    timesteps,
    to_pil,
    to_tensor,
)


class Cancelled(RuntimeError):
    """Generation was cancelled by Forge."""


def _cancel(cancelled) -> None:
    if cancelled and cancelled():
        raise Cancelled("FLUX 3 generation cancelled")


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RuntimeModel:
    """One validated model package, loaded once and reused."""

    def __init__(self, manifest: dict, memory_mode: str):
        if memory_mode not in ("balanced", "resident"):
            raise ValueError("memory_mode must be 'balanced' or 'resident'")
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX 3 Image Klein requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16 FLUX 3 inference")
        if manifest["quantization"] == "fp8":
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) < (8, 9) or not hasattr(torch, "_scaled_mm"):
                raise RuntimeError("Official rowwise FP8 needs Ada, Hopper, or Blackwell and torch._scaled_mm")
        self.manifest = manifest
        self.memory_mode = memory_mode
        device = "cuda" if memory_mode == "resident" else "cpu"
        self.pipeline = Pipeline(
            {"distilled": manifest["checkpoint_path"]},
            manifest["text_encoder_path"],
            manifest["autoencoder_path"],
            device=device,
            offload=memory_mode == "balanced",
        )
        bound_vae_convolutions(self.pipeline.autoencoder)
        self._cpu_backing = {}
        self._prompt_cache = OrderedDict()
        self._prompt_cache_bytes = 0

    @property
    def dit(self):
        """Stable access for Forge adapters and future extension-owned patches."""
        return self.pipeline.dits["distilled"]

    def _stage(self, module, device: str):
        backing = None
        if self.memory_mode == "balanced" and device == "cuda":
            backing = staging.capture(module)
            self._cpu_backing[id(module)] = backing
        module.to(device)
        if backing is not None:
            staging.mark_staged(module, backing)
        return module

    def _offload(self, module) -> None:
        if self.memory_mode == "balanced":
            backing = self._cpu_backing.pop(id(module), None)
            if backing is None or not staging.restore(module, backing):
                module.to("cpu")
            _empty_cache()

    def _encode_prompt(self, prompt: str, cancelled):
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return tuple(value.clone() for value in cached)
        _cancel(cancelled)
        encoder = self.pipeline.text_encoder
        inference_module = getattr(encoder, "inference_module", encoder.model)
        self._stage(inference_module, "cuda")
        encoder.device = "cuda"
        try:
            tokens, ids = encoder.encode(prompt)
            result = (tokens.cpu(), ids.cpu())
        finally:
            self._offload(inference_module)
        item_bytes = sum(value.numel() * value.element_size() for value in result)
        self._prompt_cache[prompt] = result
        self._prompt_cache_bytes += item_bytes
        while len(self._prompt_cache) > 2 or self._prompt_cache_bytes > 384 * 1024 * 1024:
            _, removed = self._prompt_cache.popitem(last=False)
            self._prompt_cache_bytes -= sum(value.numel() * value.element_size() for value in removed)
        return tuple(value.clone() for value in result)

    def _encode_references(self, images, cancelled):
        if not images:
            return None
        if len(images) > 10:
            raise ValueError("FLUX 3 accepts at most 10 reference images")
        prepared = ([prepare_multi_reference(image) for image in images] if len(images) > 1
                    else [prepare_reference(images[0])[0]])
        ae = self._stage(self.pipeline.autoencoder, "cuda")
        tokens, ids = [], []
        try:
            for index, image in enumerate(prepared):
                _cancel(cancelled)
                values, positions = ae.encode(to_tensor(image).to("cuda"))
                positions = scaled_ids(positions)
                positions[:, 3] = -(1 + index)
                tokens.append(values.to(torch.bfloat16).cpu())
                ids.append(positions.cpu())
        finally:
            self._offload(ae)
        return torch.cat(tokens), torch.cat(ids)

    @torch.inference_mode()
    def generate(self, request: dict, progress=None, cancelled=None):
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt must be an expanded FLUX 3 caption")
        width, height = int(request.get("width", 1024)), int(request.get("height", 1024))
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise ValueError("width and height must each be from 64 to 4096 pixels")
        if width % 16 or height % 16:
            raise ValueError("width and height must be divisible by 16")
        h, w = latent_shape(height, width)
        steps = int(request.get("steps", 16))
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        seed = int(request.get("seed", 0)) & 0xFFFFFFFF
        images = list(request.get("images") or [])
        notify = progress if callable(progress) else lambda step, total: None
        notify(0, steps)

        cond = self._encode_references(images, cancelled)
        text = self._encode_prompt(prompt, cancelled)
        _cancel(cancelled)
        text = tuple(value.to("cuda") for value in text)
        cond = tuple(value.to("cuda") for value in cond) if cond else None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=generator)
        noise = noise.to(torch.bfloat16).float().to("cuda")
        ids = latent_ids(h, w, "cuda")
        packed = pack([[text, cond, (noise, scaled_ids(ids))]])
        schedule = timesteps(steps, float(request.get("alpha", 6.93)))
        adapter_specs = list(request.get("adapters") or [])
        if len(adapter_specs) > 8:
            raise ValueError("at most 8 stacked adapters are supported")
        dit = self.dit
        attention_mode = request.get("attention_backend", "auto")
        if attention_mode not in ("auto", "forge", "torch"):
            raise ValueError("attention_backend must be auto, forge, or torch")
        native_attention = sys.modules.get("backend.attention") if attention_mode != "torch" else None
        dit._forge_attention = getattr(native_attention, "attention_function", None)
        native_vae = getattr(native_attention, "attention_function_vae", None)
        for module in self.pipeline.autoencoder.modules():
            if module.__class__.__name__ == "AttnBlock":
                module._forge_attention = native_vae
        attention_used = getattr(dit._forge_attention, "__name__", "torch-sdpa")
        spectrum_config = SpectrumConfig.from_request(request.get("spectrum"))
        spectrum = SpectrumState(spectrum_config) if spectrum_config.enabled else None
        spectrum_stats = {}
        cache_reclaims = 0
        attached = adapter_support.attach(dit, adapter_specs) if adapter_specs else []
        try:
            dit._forge_spectrum = spectrum
            dit._forge_total = steps
            self._stage(dit, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for index in range(steps):
                    _cancel(cancelled)
                    t = schedule[index].item()
                    dt = (schedule[index + 1] - schedule[index]).item()
                    packed.timesteps["image"].fill_(t)
                    target = schedule[index + 1].reshape(1).to("cuda")
                    dit._forge_step = index
                    dit._forge_target_time = schedule[index + 1].item()
                    packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
                    notify(index + 1, steps)
                    if self.memory_mode == "balanced" and width * height >= 4 * 1024 * 1024:
                        reserved = torch.cuda.memory_reserved()
                        allocated = torch.cuda.memory_allocated()
                        total = torch.cuda.get_device_properties(0).total_memory
                        if reserved > total * 0.72 and reserved - allocated > total * 0.12:
                            _empty_cache()
                            cache_reclaims += 1
            tokens = packed.data["image"].cpu()
            ids = ids.cpu()
        finally:
            try:
                self._offload(dit)
            finally:
                adapter_support.detach(attached)
                if spectrum is not None:
                    spectrum_stats = spectrum.diagnostics()
                    spectrum_stats["history_observations"] = spectrum_stats["full_steps"]
                    spectrum_stats["full_steps"] = steps-spectrum_stats["forecast_steps"]
                    spectrum.release()
                dit._forge_spectrum = None
        del packed, text, cond, noise, dit
        _empty_cache()
        _cancel(cancelled)
        if request.get("latent_checkpoint"):
            from safetensors.torch import save_file
            save_file({"tokens": tokens.contiguous(), "ids": ids.contiguous()},
                      str(request["latent_checkpoint"]))
        ae = self._stage(self.pipeline.autoencoder, "cuda")
        try:
            image = to_pil(ae.decode(tokens.to("cuda"), ids.to("cuda")))
        finally:
            self._offload(ae)
        return image, {
            "model": "flux-3-image-klein",
            "quantization": self.manifest["quantization"],
            "text_encoder_mode": "Qwen3-VL language hidden states only; visual/logits unused",
            "compute": ("fp8_scalar_gemm_rowwise_output_scale" if any(
                getattr(module, "rowwise_mode", "native") == "scalar_output_scale" for module in self.dit.modules()
            ) else self.manifest["quantization"]),
            "prompt": prompt,
            "prompt_upsampled": False,
            "prompt_note": "Prompt was used directly; supply an expanded PUP-style caption for intended quality.",
            "width": image.width,
            "height": image.height,
            "steps": steps,
            "seed": seed,
            "references": len(images),
            "adapters": len(adapter_specs),
            "memory_mode": self.memory_mode,
            "attention_backend": attention_used,
            "spectrum": spectrum_stats,
            "spectrum_forecast_steps": spectrum_stats.get("forecast_steps", 0),
            "pressure_cache_reclaims": cache_reclaims,
            "vae_attention": "fp32-query-chunked" if image.width * image.height // 64 > 16384 else "fp32-sdpa",
            "vae_convolutions": "bounded-row-workspace",
            "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        }

    def unload(self):
        self._prompt_cache.clear()
        self._prompt_cache_bytes = 0
        getattr(self, '_cpu_backing', {}).clear()
        self.pipeline = None
        # Some transformer module graphs retain cycles. Release unreachable
        # tensors before asking CUDA to return unused allocator blocks.
        gc.collect()
        _empty_cache()


_LOCK = threading.RLock()
_MODEL: RuntimeModel | None = None
_MODEL_KEY = None


def load(request: dict) -> RuntimeModel:
    """Validate and load the requested package; exposes `.dit` for adapters."""
    global _MODEL, _MODEL_KEY
    manifest = validate_model_dir(request["model_dir"], request.get("quantization", "bf16"))
    memory_mode = request.get("memory_mode", "balanced")
    fingerprint = tuple((item["path"], item["size"], item["mtime_ns"]) for item in manifest["fingerprint"])
    key = (manifest["checkpoint_path"], memory_mode, fingerprint)
    with _LOCK:
        if _MODEL is None or _MODEL_KEY != key:
            unload()
            _MODEL = RuntimeModel(manifest, memory_mode)
            _MODEL_KEY = key
        return _MODEL


def generate(request: dict, progress=None, cancelled=None):
    """Return ``(PIL.Image, metadata)`` using official Klein components."""
    try:
        with _LOCK:
            return load(request).generate(request, progress, cancelled)
    except torch.OutOfMemoryError:
        unload()
        raise RuntimeError("FLUX 3 ran out of GPU memory; use FP8 or balanced memory mode") from None


def unload() -> None:
    global _MODEL, _MODEL_KEY
    with _LOCK:
        if _MODEL is not None:
            _MODEL.unload()
        _MODEL = None
        _MODEL_KEY = None
