# Post-training validation — 2026-10-18

The specific AI Toolkit job completed at step6200. Its final checkpoint/optimizer and2/2 samples completed; database status was completed, queue stopped, and training workers exited before testing began. All model/test jobs ran sequentially. No models or dependencies were downloaded.

## Actual tested build and route

Installed experimental workflow build based on branch `experimental/workflow-controls-20261018` at `3b433d97275edcb95a273503606aa3b9e5f10fec`; controls default off. Original rowwise FP8 DiT + matching BF16 HF Qwen3-VL-4B + FLUX2 VAE on physical RTX5090, Torch2.13.0+cu130. This does not validate Base or Comfy ConvRot profiles.

72 installed regressions passed. Real512x512,16-step runs: default cold/warm, alpha16 + seed478920109938706 +64-bit width + TXT2IMG bbox layout twice, one-reference conditioning, cancellation after step2 and subsequent recovery. Default repeats and recovery match historical pixel SHA256 `a219dbab4db611136d453bdf8d9933b6b23f94cf4ec189c40b6bcbce73b89edf` exactly; workflow repeats also match exactly. Visual inspection of the workflow output showed the expected coherent red teapot scene. No numerical quality threshold or exact bbox-placement guarantee is claimed.

Default cold 42.696s; warm 2.963s. Workflow first/repeat 4.132s/2.452s. Peak Torch allocation across this sequence 7.617GiB; peak sampled process RSS 15.919GiB, private memory 26.647GiB. After explicit unload: 0.03125GiB allocated/reserved. Short repeated-run checks do not establish long-duration leak freedom. Windows system overhead and nonallocator device memory are outside the Torch allocation metric.

## Boundaries still open

Exact Comfy Simple scheduler parity, complete official gated workflows, native prompt upsampling, Base CFG inference, canonical BF16/INT8 real-weight combinations, compatible trained adapter visual tests, every Refiner ordered pair, actual clicked-browser interaction, and physical smaller-card testing remain open. RTX5090 allocator budgets remain simulations. Missing assets remain blocked; no contact-sharing gate was accepted. Controls stay experimental and off by default. Private prompts, images and raw logs are excluded from this publication.

## Native Generate and Refiner follow-up

These were backend pipeline requests through native Gradio. No browser clicking was performed; clicked-browser acceptance remains open.

An actual native Gradio Generate/gallery/save request completed in31.116s with the original FP8 route, Forge SageAttention, correct PNG sampler/component metadata and unchanged saved configuration. After a deliberately rejected unsupported target-negative Refiner request, normal FLUX3 generation recovered. Qwen2.1→FLUX3 refinement also succeeded in67.831s using explicit caption/empty negative/16steps/CFG1 and full-reference flow-map inference. This is one tested pair, not all ordered profiles.
