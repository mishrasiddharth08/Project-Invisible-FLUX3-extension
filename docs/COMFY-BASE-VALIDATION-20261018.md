# Matched Comfy Base profile: bounded validation, 2026-10-18

Experimental branch only. Native TXT2IMG and reference editing were exercised with Base INT8 ConvRot, matching Qwen3-VL-4B INT8 ConvRot and FLUX2 VAE on one RTX 5090. This is not full Blueprint certification or support for every precision/hardware combination. The original FP8 route remains separate; the new profile is off by default.

## Activation and assets

In Settings > Project Invisible FLUX.3, enable Experimental matched Comfy profiles, choose Base, checkpoint/encoder `int8_convrot`, alpha16 and64-bit seeds. Acknowledge full captions and reference behavior when editing. Refresh model registration/restart after initial model placement. Select the real Base checkpoint in the normal checkpoint dropdown or the FLUX3 preset. Use Euler/Simple, Base CFG3.5 and25steps for the validated workflow. No extra Generate tab/server/venv or Forge core edits are used.

Canonical files, relative to Forge models:

| Asset | Path | Bytes | SHA256 |
|---|---|---:|---|
| Base | Stable-diffusion/FLUX-3/flux3-image-klein-base-int8-convrot.safetensors | 5807427339 | 8949ecfc1bf3871b08be02cf43ec7276a4914e1a86eade6b5eb8f4204426bc92 |
| Encoder | text_encoder/FLUX-3/Qwen3-VL-4B-TextEncoder-int8_convrot--For-FLUX-3.safetensors | 4858559240 | 32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c |
| VAE | VAE/FLUX-3/FLUX.2-VAE--For-FLUX-3.safetensors | 336211292 | 868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3 |

Publisher revision: feef60b0aaf63a4773ace96636ebbcf43b881312. The two approved new files total10665986579bytes; the VAE was reused by a verified hardlink. No optional PUP/distilled/BF16 weights or new dependencies were downloaded. The extension never downloads assets or accepts license gates automatically. Model terms: [FLUX Kommunity](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/feef60b0aaf63a4773ace96636ebbcf43b881312/LICENSE.md).

## Repairs and measured evidence

- Actual Comfy DiT uses fused QKV plus quantization metadata. Loading now splits row views without changing original INT8 storage; native fused encoder layers remain fused. Schema, payload hashes and unknown metadata fail closed.
- Meta-created Qwen RoPE buffers are rebuilt through installed Transformers constructors; checkpoint parameters still load strictly. A parity test checks the derived buffers without materializing weights.
- AutoLink renamed the unlabelled encoder to a Krea2-labelled filename, invalidating registration. Native discovery now finds exactly one size/SHA-pinned encoder within its canonical folder, independent of basename. Duplicate matching files and missing/changed components fail clearly. A local companion policy also gives AutoLink a FLUX3 family and matching component preference.
- An initial basename override fell through to QWEN; identity assertions rejected that image. Registered-basename resolution and missing managed-profile guards prevent that fallback. Failed attempts are preserved privately and excluded from passing counts.
- The supplied active graph produced five separate1776x1776 images,25steps/CFG3.5/alpha16 and the same64-bit seed per caption. Repeat and cancellation-at-step2 recovery reproduced exact pixels. Comfy10000-entry Simple schedule/BCHW FP32 noise and original Base CFG versus distilled flow-map loops have source-parity CPU checks. No independent full Comfy-render image baseline is claimed.
- Ten distinct1024x1280 reference photographs were edited, plus exact repeat/cancel/recovery. Raw and protected outputs were saved separately. All ten composites preserved every black-mask source pixel. One actual uploaded-mask native Generate/gallery/save call preserved1220016black pixels; all five native/refined galleries independently matched real Forge-saved PNG pixels.
- Native Gradio Generate/gallery/save passed, with QWEN2.1->Base, Base->MOSS and MOSS->Base refinement. Unsupported negative prompts, foreign encoder overrides and native masks without explicit compositing returned clear failures; a subsequent valid native output matched the baseline exactly. Validation used served UI configuration and actual Gradio backend functions, **not clicked-browser interaction**. `/sdapi` parity was not tested; API enablement was unchanged.

First process request at1776x1776:80.116s; following requests approximately64-66s. This includes conditioning, sampling and decode but excludes download/profile preflight; OS/model caches were already warm. Peak Torch allocated10.001GiB/reserved13.602GiB; sampled host private memory reached29.007GiB. After explicit unload, Torch allocated0.0315GiB/reserved0.0332GiB. Short runs are not long-duration leak certification. Native512x512/25steps:11.901s first,7.182s recovery. First reference edit52.628s, subsequent edits approximately44.3s. Refiner combined source/target timings:27.721/34.238/20.846s respectively; these are different workloads, not model speed rankings.

Allocator-only simulations on RTX5090:6/8/10GiB caps OOM and released the model;12/16/20/24/32GiB caps completed with identical baseline pixels. Full-budget recovery also matched. These are not physical smaller-card tests: contexts, other applications and nonallocator memory are outside the cap. No physical6-24GB-card guarantee is made. Runtime requires BF16-capable CUDA and working Comfy-Kitchen ConvRot kernels; only this5090 architecture was validated.

Versions: Python3.13.15, Torch2.13.0+cu130, Transformers4.57.6, Comfy-Kitchen0.2.38, safetensors0.8.0, Gradio4.40.0, Diffusers0.37.1.80 staged runtime/native CPU tests passed with CUDA initialization forbidden;54 dependency-free/mock tests form public CI. The suites overlap and must not be added as independent counts. Local Refiner70 and AutoLink22 checks are separate companion results.

## Visual and capability limits

Four supplied captions matched their main scene content. The trading card's small ability text/credit was garbled or misspelled: typography fidelity is a partial failure. Raw reference edits can reconstruct skin, face, lighting or background outside the requested region. No guaranteed quality improvement or identity/pixel lock is claimed.

Source protection is explicit final-frame compositing, **not native masked inpainting**. Broad audit ellipses include background and can leave visible boundaries or unrecolored long hair beyond paint. The old audit-mask generator also clipped blurred masks at roughly half-opacity at the edge; separate inward-feather diagnostic variants removed that discontinuity without expanding paint or changing original masks. Geometric coverage/model drift can still remain. Production honors supplied raw alpha; it never silently changes the user's mask or performs hair segmentation.

Conventional denoising-strength img2img, native masked editing/outpainting, Hires, adapters/Spectrum on this profile, enabled PUP/custom comparison/cleanup behavior, multi-reference GPU acceptance, the complete BF16/mixed-profile matrix, all Refiner profile pairs, full browser64-bit integer fidelity, clicked-browser interaction and physical smaller-card coverage remain unsupported or untested. The metadata acceptance flag stays conservative while this wider matrix is incomplete. Existing model-branded source-protection UI has not completed strict Blueprint control consolidation.

## Companion deployment boundary

The tested machine has local AutoLink and Refiner companion fixes. **They are not published or bundled by this FLUX3 update.** The local Refiner distinguishes Base CFG0..20/default25steps from distilled CFG1/default16steps and retains matched target routing/release/restore. Public Refiner main30e283ce1fd960a1358540ca237be2185e7a8040 is unchanged; its Base integration must not be inferred from these local tests. The AutoLink policy adds the real Base header fingerprint, Qwen3-VL-4B/FLUX2 VAE slots, canonical naming and component preference. Standalone Base inference uses pinned components directly; automatic family selection requires that companion policy or an equivalent supported AutoLink update. Canonical naming before activation avoids first-start rename invalidation; refresh registration after any rename.

Private prompts, source photographs, generated media, raw logs, credentials and weights are excluded from this repository. Main/default support is not promoted by this experimental branch.
