"""Explicit, resumable official-package download; run only after approving it."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "black-forest-labs/flux-3-image-klein"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--quantization", choices=("bf16","fp8"), default="bf16")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit("First accept the model's Hugging Face terms and sign in locally with `hf auth login`. Never paste tokens into this chat.")
    api = HfApi(token=token)
    try:
        api.auth_check(repo_id=REPO, repo_type="model")
        revision = api.model_info(REPO).sha
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise SystemExit("Model access is not enabled. Accept its Hugging Face terms using the account you signed in with.") from None
        raise SystemExit(f"Model connection failed ({type(exc).__name__}); retry. Your login may still be valid.") from None
    except Exception as exc:
        raise SystemExit(f"Model connection failed ({type(exc).__name__}); retry. Your login may still be valid.") from None
    target = Path(args.destination).expanduser().resolve()
    if (target/"model.safetensors").exists() or (target/"variants/fp8r/model.safetensors").exists():
        print("Existing files will be reused when their repository hashes match.")
    patterns = ["ae.safetensors","text_encoder/*","README.md","LICENSE.md","NOTICE.md"]
    patterns += (["model.safetensors","config.json","manifest.json"] if args.quantization == "bf16" else ["variants/fp8r/*"])
    print(f"Downloading {REPO}@{revision}, {args.quantization}; no duplicate Diffusers transformer or other variants.")
    snapshot_download(REPO,revision=revision,local_dir=str(target),allow_patterns=patterns,token=token,max_workers=4)
    print(f"Downloaded to {target}. Set this folder in Forge's FLUX3 Settings.")


if __name__ == "__main__":
    main()
