# Validation status — 2026-10-18

This is a compatibility bridge, not full blueprint certification. Earlier real generation used the original rowwise FP8 DiT + matching BF16 HF Qwen3-VL-4B + FLUX2 VAE on RTX 5090. Matching baseline images, repeated release, OOM recovery and bounded allocator runs were recorded locally. Original native FP8 GEMM needed a compatibility fallback on the tested Torch build. 67 regression checks and later 72 checks including preset discovery passed before the training pause. No compatible trained local adapter was found among the inspected headers; mathematical adapter tests are not real-adapter visual validation.

BF16 DiT, canonical Comfy INT8/BF16 combinations and physical smaller cards remain unverified. An RTX 5090 allocator cap is not a physical-card test. Image editing can change geometry and unmasked detail. Supplied-mask compositing is not native masked inference.

GPU validation is paused while AI Toolkit training runs. No Forge restart, CUDA import/allocation, dependency installation or model download was performed for this publication. Resume only after confirmed training completion and safe Forge idle checks; elapsed time or a single utilization sample is insufficient.

## Post-training acceptance and second publication

After training: one job at a time. MOSS first: cold/warm fixed-seed generation, dropdown/AutoLink selection, output/gallery/save metadata, cancellation/recovery, release and repeated-run memory. FLUX3 second: preserve original FP8 baseline, then opt-in workflow alpha/64-bit seeds/layout with official captions, references and failure cases. BF16/INT8 real-weight profiles require matching available assets and remain blocked when absent. Refiner comes third: valid ordered target profiles with explicit matching TE/VAE, callback/negative-prompt boundaries, release and restore. Record cold/warm wall time, peak GPU and host RAM, actual outputs/hashes, failure recovery and repeated-run leaks. Allocator budgets on RTX 5090 are simulations, not physical 6/8/10/12/16/20/24/32 GB card certification. No numerical visual/performance threshold has been supplied. Publish subsequent reviewed fixes and measured results to these same repositories after testing; never include private prompts/images/logs or weights.

## Earlier implementation record (historical, superseded above)

# Validation record

Date: 2026-10-17. Status: compatibility implementation and code/kernel checks passed; full model acceptance is pending model access.

## Passed checks

- 21 tests passed in Forge's existing Python: asset validation/cache/cancellation, adapter numerical equivalence and rejection, bridge batches/save/progress, scoped dispatch and rollback, and GPU kernels.
- Two tests ran real kernels on the RTX 5090: packed attention with separate samples, and FP8 matrix multiplication with an unmerged LoRA branch. They do not load FLUX3 weights.
- LoRA, full/factorized LoKr and LoHa matched explicit CPU matrix calculations. Stacking, detach, foreign-family rejection and duplicate alias rejection passed.
- The official native rowwise FP8 call failed on this installed GPU/PyTorch combination. The adapted path retains FP8 matrix multiplication and applies the same per-row scales to its FP32 result before BF16 conversion. This path passed the actual GPU test; full-model speed and image quality remain unmeasured.
- A public FLUX3 adapter's small header was inspected without downloading its full weights. AI Toolkit's `diffusion_model.content_mode_blocks...lora_A/B.weight` target naming matches the implemented original-model mapping. This is format evidence only.

## Checked environment

- Installed Forge: Python 3.13.15; PyTorch 2.13.0+cu130; Transformers 4.57.6; Diffusers 0.37.1; Accelerate 1.14.0; Safetensors 0.8.0.
- Physical GPU: NVIDIA GeForce RTX 5090, 31.82 GiB. CUDA, BF16 and the Qwen3VL Transformers API are available. This is an API probe, not model inference proof.
- No FLUX3 weights found in the inspected Forge models folder or default Hugging Face cache. No saved Hugging Face token was found.
- Official source commit: `e53f4ea697fe45db8f38fca91a90f5623f7f8f37`.
- Original model package metadata revision inspected: `337337621b8526e120f2275d250242f4a6bedd75`.

## Blueprint deviation

The user explicitly approved a compatibility extension after being informed that this Forge build has no FLUX3 native engine/flow-map registration contract. Native checkpoint selection, Generate and gallery are UI integration only; native pipeline equivalence is not claimed. The extension owns conditioning, sampling, component memory and result assembly for its checkpoint.

## Not yet verified

Real model loading; fixed-seed image generation; edit quality; real-adapter visual quality; actual saved image/gallery parity in the running UI; measured full-model speed/VRAM; restart/removal in a live Forge session; third-party extensions; physical GPUs other than the RTX 5090. These need complete model files and live tests. NVFP4, ComfyUI int8/ConvRot, GGUF/NF4 and arbitrary adapter algorithms are unsupported.
