# Project Invisible FLUX3 for Forge Neo

An **explicit compatibility bridge**, approved separately from the supplied native-integration blueprint. Uses Forge's normal checkpoint selector, Generate button, progress, saved images and gallery. It runs the actual Black Forest Labs FLUX3 image model in the existing Forge Python environment.

This is not Forge-native engine registration. This Forge snapshot has no public registration contract for FLUX3's three-stream transformer and target-time flow-map sampler. The extension uses checkpoint-scoped, reversible runtime wrappers. No Forge source file, Python environment, GPU library, application server or model weight is installed or changed automatically.

## Setup

1. Put this folder under `extensions/project-invisible-flux3` and restart Forge.
2. Obtain access to `black-forest-labs/flux-3-image-klein` on Hugging Face and accept its model terms yourself. Download the **official original-format distilled package**, including `model.safetensors`, configuration/manifest, `ae.safetensors`, and the complete `text_encoder/` folder. FP8 additionally needs `variants/fp8r/`. Weights are not included.
3. In Forge Settings, find FLUX3, set the package folder and memory policy, then apply settings and refresh the checkpoint list. Select the FLUX3 compatibility entry.
4. Use a complete scene caption, CFG 1, dimensions divisible by 16, and start with 512×512 and 16 steps. Generate normally. For editing, use the native img2img image input; this is reference editing, so denoising strength is not an editing-strength control.

FLUX3's authors require expanded scene captions for intended quality. Run their prompt upsampler or another suitable language model first and paste its resulting caption into Forge. This version does not silently download or run the separate 9B upsampler. Short prompts may give generic results. Confirm the caption setting before generation.

After you accept access terms and sign in locally with `hf auth login`, `tools/download_model.py --destination <folder> --quantization fp8` downloads only the original components and selected variant. Run it with Forge's Python. This is a separate explicit download action, never a Forge startup step. Downloads and access terms require their separate explicit approval; the pending BF16/INT8 bundle is not approved by general testing/repair requests.

## Scope

- Distilled FLUX3 original-format BF16 and rowwise FP8; prompt-to-image and reference editing, subject to local component validation.
- The real flow-map sampler receives both the current and target time. CFG has no meaning beyond 1 for this distilled checkpoint.
- Balanced memory moves inactive components to system RAM. Resident mode retains components for warm runs and uses more VRAM. Prompt embeddings are bounded and cached only for matching components and text.
- In Forge, attention can follow the host's selected Sage/Flash/xFormers/PyTorch backend; choose `torch` for the reference SDPA path. FLUX2 model classes and incompatible weight formats are never substituted.
- Large images stream VAE normalization, convolution, nearest-neighbor upsampling and residual projections with full-image normalization statistics and proper convolution halos. Attention retains all spatial keys. This is not independent whole-image VAE tiling.
- Spectrum is an optional, experimental adaptation of spectral hidden-feature forecasting. It preserves current/target flow-map conditioning and recomputes the output head, bounds its cache, checks forecasts and resets every request. Upstream FLUX3 support is not claimed. The initial 512px/16-step test showed only 1.02x speedup, so it is **off by default**. It can change image details; do not assume the paper's FLUX1 speedups apply to FLUX3.
- Fixed/random seeds, sequential batches, cancellation between steps and standard saving are bridged explicitly.
- Original FLUX3 linear LoRA, LoKr and LoHa adapters use native `<lora:name:strength>` tags. Multiple adapters stack without replacing or dequantizing the base weights. Only matching original layer names and shapes are accepted; unrelated families and unimplemented adapter types fail clearly. Adapter mathematics has been tested; visual results with real FLUX3 adapters remain unverified.
- Native FLUX2 checkpoints, ComfyUI int8/ConvRot/NVFP4 files, GGUF/NF4, base FLUX3, ControlNet, native masked inpainting, Hires fix and variation seeds are not supported by this initial bridge. This is not universal quantization or universal adapter compatibility.
- Some extensions depend on intermediate native conditioning or latent callbacks which this compatibility route cannot supply. Standard finishing callbacks supported by the bridge do not establish universal extension compatibility.

No claim of lower-VRAM hardware support, maximum speed, image quality or successful model loading follows from unit tests. See `docs/VALIDATION.md` for the actual evidence.

## Check your setup

Run `tools/check_environment.py` using Forge's Python. It reads environment information; `--model-dir` additionally validates local components. It never installs or downloads anything. Do not share access tokens in problem reports.

Disable this extension and restart Forge to remove its integration. Model files remain untouched. Changing back to another checkpoint releases this extension's model memory.

## Attribution and sources

Original inference implementation: https://github.com/black-forest-labs/flux-3-image-klein

Official model, demo, ComfyUI package and collection:

- https://huggingface.co/black-forest-labs/flux-3-image-klein
- https://huggingface.co/spaces/black-forest-labs/flux-3-image-klein
- https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
- https://huggingface.co/collections/black-forest-labs/flux-3-image-klein

Preserved third-party source licenses are under `flux3_forge/vendor/flux3/`. Extension-owned code is MIT licensed; this does not relicense model weights or third-party code. This project is not sponsored or endorsed by Black Forest Labs.

Spectrum algorithm reference: https://github.com/hanjq17/Spectrum at `4e41f91c93ff66b045332530dd7bcb4c60eafbab`; its MIT notice is preserved in `flux3_forge/vendor/SPECTRUM_LICENSE`.


## Supplied-mask source protection (explicit opt-in)

In img2img, open **FLUX.3 source protection** and enable **Composite supplied mask after generation (not native masked inpainting)**. Supply your own grayscale edit mask: white takes generated RGB, black copies source RGB exactly, gray blends. Nothing is inferred. Source, output and mask dimensions must match. Disable mask inversion and inpaint crop/full-res mode. This mode uses raw mask alpha; native blur and denoising-strength controls do not control the compositor. PNG is required when saving, and final saved pixels are verified.

The model still generates a full frame conditioned on its reference. The mask is applied after finishing callbacks; this is not mask-conditioned diffusion. It protects black regions, not original geometry or texture within white regions. A poor selection or generated lighting mismatch can leave a seam. No automatic person/garment mask is created. Without a supplied mask, enabling this mode fails clearly. With the mode off, masks remain unsupported.

API: pass `alwayson_scripts: {"FLUX3 source protection": {"args": [true]}}`, your ordinary img2img `init_images` and `mask`, matching `width`/`height`, `inpaint_full_res: false`, `inpainting_mask_invert: 0`, and PNG saving. Other model routes remain unaffected; the switch defaults off.

## Balanced staging and measured limits

Balanced mode now keeps original CPU backing for read-only inference weights while staged on GPU and restores it instead of repeatedly copying unchanged weights back from GPU. Tracked version/structure changes fall back to normal copying. This is not support for arbitrary in-place weight-mutating third-party patches. Resident mode remains explicit opt-in.

In the matched ten-picture FP8 workload on the RTX5090/Windows test host, all ten original-prompt outputs remained pixel-identical. Sampled process-private peak changed73.9GB to31.1GB and working-set peak49.2GB to18.3GB. These are host process measurements, not VRAM, guaranteed memory caps, minimum-RAM specifications or physical low-card validation. Current lower-card/BF16/INT8 acceptance remains open.

## Checkpoint selector versus UI preset

Select **FLUX 3 Image Klein (FP8)** from Forge's common **Checkpoint** dropdown; configure the local package under **Settings > Project Invisible FLUX.3**. The package owns its matching encoder/VAE. The separate legacy **UI Preset > flux** is not a FLUX3 engine/preset registration and can restore older remembered selections. This revision registers a separate `flux3` preset when a valid local package is discovered. It does not turn the legacy `flux` preset into FLUX3. Final clicked-browser acceptance remains pending. Actual browser interaction was unavailable during this audit; API generation does not certify mouse-driven UI generation.

Strict color-only editing is not guaranteed: the bounded preservation trial still showed skin/detail regeneration and object redesign; no prompt was silently changed or promoted as a universal cure. Supplied-mask protection requires an explicit selection and does not fix shapes inside the editable area. No native masked-inpainting claim is made.


## Focused support boundary

Only the existing original rowwise FP8 + BF16 HF Qwen3VL package is real-tested here. BF16 DiT is code-supported but unavailable; Comfy INT8/NVFP4/GGUF/NF4 and quantized conditioning TEs are unsupported by this route. Quantization config/storage is now rejected explicitly. The text-only encoder avoids unused vision/logit work and matched the same hidden states/output. Allocator simulations on an RTX5090 generated the baseline at8/10/12/16/20/24/full available GiB limits, but do not certify physical smaller cards or other architectures; nonallocator/context overhead remains.67 regressions passed; compatible linear LoRA/LoKr have synthetic math checks, but no genuine matching trained local adapter was found. No universal latest-format/adapter guarantee or automatic per-card tuning.

## Publication status (2026-10-18)

The earlier training pause has ended; see the dated post-training validation record. Historical pause note: No Forge restart, CUDA import/allocation, dependency installation or model download was performed for this publication. Resume only after confirmed training completion and safe Forge idle checks; elapsed time or a single utilization sample is insufficient. See [validation](docs/VALIDATION.md) for current evidence, known limits and the sequential acceptance plan. New workflow controls, when present on the experimental branch, default off; they do not activate Base, Comfy ConvRot, NVFP4 or PUP inference.

Current evidence: [2026-10-18 post-training validation](docs/POST-TRAINING-20261018.md). Bounded real generation and backend pipeline/gallery requests passed; clicked-browser acceptance remains open. New workflow controls remain default off on the [experimental branch](https://github.com/mishrasiddharth08/Project-Invisible-FLUX3-extension/tree/experimental/workflow-controls-20261018), with their tested scope documented separately.
