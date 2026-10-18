# Exact supplied workflow: experimental V3

**Not registered, activated or validated with real matching weights. No end-to-end success claim.** This branch preserves the existing FP8 bridge and adds isolated modules for a different checkpoint contract. No import hook enables them.

The supplied active branch is Base INT8 ConvRot,25steps/CFG3.5/alpha16,1776x1776. Prompt-list output1 expands to five separate literal captions; the same64-bit scalar seed broadcasts to each. The alternate distilled branch and both PUP branches are bypassed. Image nodes feed comparison only, not inference references. Supplied prompts/workflow JSON and generated images are excluded from this repository.

Publisher Git-LFS pointer identities prove the historical no-sd alias is current `flux3-image-klein-base-int8-convrot.safetensors` (5,807,427,339bytes, Git pointer object450f47e2f4811aaa396fe646ba76741f1c2f31b8); the historical sd alias is distilled. Source revisions: HF current feef60b0aaf63a4773ace96636ebbcf43b881312; historical5c9b714851dc072b04a23ef6d2d36421f5ccb3a0. Payload SHA256 is masked anonymously: future validation reconstructs the pinned publisher LFS pointer from actual payload SHA256 and size instead of inventing a checksum.

## Implemented, with bounded tests

- Explicit Base/distilled profiles; correct meta architecture and wrong-family rejection; pinned component checks.
- Base unconditional/conditional CFG and distilled target-time stepping. Comfy10000-entry sampled schedule and BCHW CPU FP32 noise preserve the graph contract instead of silently reusing native BF16-rounded token-major noise.
- ResolutionSelector and canonical Easy-Use promptList semantics, ordered five-request plans and64-bit seed handling. Original encrypted-package metadata does not prove identical fork behavior; the port implements the reviewed canonical contract.
- Disabled GUI preview/native-control update specification, escaped summary and gallery-result adapter. This is staged backend adapter code, not installed GUI wiring or clicked-browser acceptance.
- Explicit-opt-in sequential execution adapter with actual-metadata checks, cancellation, owned-engine release and atomic no-overwrite PNG saving. PNG pixel chunks are retained; UTF8 workflow metadata and standard parameters infotext are added. This is not a claim of identical Comfy frontend/API workflow metadata or native Forge save-callback authority.

Local evidence:13 semantic source-parity checks and54 staged CPU regressions passed. Final runtime suite forbade CUDA initialization. The preview/save adapter adds13 standard-library tests. Public CI runs the standard-library/mocked subset only; it does not install Torch, download weights or run models. Full runtime tests require the user's already-installed offline ML dependencies. Test reference excerpts retain Comfy GPL3 and BFL Apache2 notices under semantic-sources.

## Remaining acceptance boundary

Exact Base DiT and matching INT8 Qwen3VL4B conditioning are missing. The matching FLUX2 VAE is hash-verified and reused locally by a canonical hardlink, with no new weight bytes. Real-profile constructor/load, five-image generation, quality, memory/performance, GUI/API/save integration, model cancellation/OOM/recovery and physical-card coverage remain open. PUP-enabled execution and unrecognized custom cleanup/timer/compare semantics are not implemented or validated. Do not promote this route to main/default support on CPU evidence.

## Minimal active-branch assets: approval required

Only two additional files are proposed, at immutable HF revision feef60b0aaf63a4773ace96636ebbcf43b881312:

| File | Bytes | Forge destination |
|---|---:|---|
| diffusion_models/flux3-image-klein-base-int8-convrot.safetensors | 5,807,427,339 | models/Stable-diffusion/FLUX-3/flux3-image-klein-base-int8-convrot.safetensors |
| text_encoders/qwen3vl_4b_int8_convrot.safetensors | 4,858,559,240 | models/text_encoder/FLUX-3/qwen3vl_4b_int8_convrot.safetensors |

Total10,665,986,579bytes. No PUP/distilled weights are in this minimum. No download or gate acceptance occurred.

Model source: https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy . Model license: https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/feef60b0aaf63a4773ace96636ebbcf43b881312/LICENSE.md (FLUX Kommunity). Conditioning encoder has separate Apache2 terms per the official model card. Access is auto-gated: Hugging Face shares the requesting account's username/email with BFL; access reports can include public profile fullname and request timestamps. No extra gated fields were exposed in public metadata; the actual signed-in form was not inspected. The user must review it and consent separately. Gate documentation: https://huggingface.co/docs/hub/models-gated .
