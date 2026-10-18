"""Read-only FLUX3 environment check; never installs or downloads anything."""
import argparse
import importlib.metadata
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--quantization", choices=("bf16", "fp8"), default="bf16")
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "packages": {}, "checks": {},
              "generation_verified": False}
    for name in ("torch", "transformers", "safetensors", "Pillow", "numpy"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        report["checks"]["qwen3vl_api"] = True
        report["checks"]["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = {"name": torch.cuda.get_device_name(0),
                             "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                             "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
                             "bf16": torch.cuda.is_bf16_supported()}
    except Exception as exc:
        report["checks"]["runtime_error"] = f"{type(exc).__name__}: {exc}"
    if args.model_dir:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        try:
            from flux3_forge.assets import validate_model_dir
            report["model"] = validate_model_dir(args.model_dir, args.quantization)
        except Exception as exc:
            report["checks"]["model_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["checks"]["model_dir"] = "Not supplied; model loading and generation remain untested."
    print(json.dumps(report, indent=2, default=str))
    return int(any(key.endswith("error") for key in report["checks"]))


if __name__ == "__main__":
    raise SystemExit(main())
