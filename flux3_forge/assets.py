"""Offline validation for official FLUX 3 Image Klein model packages."""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_QUANTIZATIONS = ("bf16", "fp8")
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"


def _fail(message: str) -> ValueError:
    return ValueError(f"Invalid FLUX 3 Image Klein package: {message}")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _fail(f"cannot read {path.name}: {error}") from error


def _text_encoder_files(text: Path, config: dict) -> list[Path]:
    architecture = config.get("architectures") or []
    if "Qwen3VLForConditionalGeneration" not in architecture:
        raise _fail("text encoder architecture must be Qwen3VLForConditionalGeneration")
    index = text / "model.safetensors.index.json"
    single = text / "model.safetensors"
    if single.is_file():
        weights = [single]
    elif index.is_file():
        mapping = _read_json(index).get("weight_map") or {}
        names = sorted(set(mapping.values()))
        if not names:
            raise _fail("text encoder shard index has no weights")
        weights = []
        for name in names:
            candidate = (text / name).resolve()
            if candidate.parent != text.resolve() or not candidate.is_file():
                raise _fail(f"text encoder shard is missing or unsafe: {name}")
            weights.append(candidate)
        weights.append(index)
    else:
        raise _fail("text encoder model.safetensors or shard index is missing")
    support = [text / "config.json"]
    for names in (("tokenizer.json", "tokenizer.model"),
                  ("processor_config.json", "preprocessor_config.json")):
        present = next((text / name for name in names if (text / name).is_file()), None)
        if present is None:
            raise _fail(f"text encoder needs one of {names}")
        support.append(present)
    for optional in ("tokenizer_config.json", "chat_template.json", "generation_config.json"):
        if (text / optional).is_file():
            support.append(text / optional)
    return weights + support


def _fingerprint(root: Path, files: list[Path]) -> list[dict]:
    result = []
    for path in sorted(set(files)):
        stat = path.stat()
        result.append({"path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def _shape(reader, key: str, expected: tuple[int, ...], label: str) -> None:
    try:
        actual = tuple(reader.get_slice(key).get_shape())
    except KeyError as error:
        raise _fail(f"{label} is missing tensor {key}") from error
    if actual != expected:
        raise _fail(f"{label} tensor {key} has shape {actual}, expected {expected}")


def _inspect_safetensors(path: Path, label: str):
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise _fail("safetensors is required to inspect model files") from error
    try:
        return safe_open(str(path), framework="pt", device="cpu")
    except Exception as error:
        raise _fail(f"cannot read {label} header: {error}") from error



def _validate_storage_dtypes(reader, label: str, allowed: set[str]) -> None:
    for key in reader.keys():
        if "weight" not in key and not key.endswith(".bias"):
            continue
        if key.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        dtype = reader.get_slice(key).get_dtype()
        if dtype not in allowed:
            raise _fail(f"{label} tensor {key} uses unsupported storage dtype {dtype}; this loader does not dequantize arbitrary formats")


def _validate_text_storage(files: list[Path], config: dict) -> None:
    if config.get("quantization_config") or config.get("text_config", {}).get("quantization_config"):
        raise _fail("quantized text-encoder configurations are not supported; use the local ordinary floating-point Qwen3-VL package loaded as BF16")
    for path in files:
        if path.suffix != ".safetensors":
            continue
        with _inspect_safetensors(path, "text encoder") as reader:
            if any(any(marker in key for marker in ("weight_scale", "quant_state", "qweight", "qzeros")) for key in reader.keys()):
                raise _fail("quantized text-encoder tensors are not supported by the original Qwen3-VL loader")
            _validate_storage_dtypes(reader, "text encoder", {"BF16", "F16", "F32"})

def validate_model_dir(path, quantization: str = "bf16") -> dict:
    """Validate local official assets without loading tensor payloads."""
    root = Path(path).expanduser().resolve()
    if quantization == "nvfp4":
        raise _fail("NVFP4 is not supported by this Forge route; use BF16 or official rowwise FP8")
    if quantization in ("int8", "fp8_scaled", "comfy"):
        raise _fail("ComfyUI int8/scaled formats are not compatible with the official Klein loader used by Forge")
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise _fail(f"unsupported quantization {quantization!r}; validated modes are BF16 and official rowwise FP8")
    if not root.is_dir():
        raise _fail(f"folder does not exist: {root}")

    checkpoint = root / ("model.safetensors" if quantization == "bf16" else "variants/fp8r/model.safetensors")
    ae = root / "ae.safetensors"
    text = root / "text_encoder"
    config_path = text / "config.json"
    for required in (checkpoint, ae, config_path):
        if not required.is_file():
            raise _fail(f"missing {required.relative_to(root)}")

    config = _read_json(config_path)
    text_config = config.get("text_config", config)
    if config.get("model_type") not in ("qwen3_vl", "qwen3-vl"):
        raise _fail("text_encoder/config.json is not Qwen3-VL")
    if int(text_config.get("hidden_size", 0)) != 2560:
        raise _fail("Qwen3-VL text hidden size must be 2560")
    if int(text_config.get("num_hidden_layers", 0)) < 32:
        raise _fail("Qwen3-VL text encoder needs at least 32 layers")
    text_files = _text_encoder_files(text, config)

    with _inspect_safetensors(checkpoint, "DiT") as reader:
        keys = set(reader.keys())
        for key, expected in {
            "emb_in.image.weight": (3072, 128),
            "emb_in.image_cond.weight": (3072, 128),
            "txt_in.weight": (3072, 20480),
            "single_blocks.27.q_proj.weight": (3072, 3072),
            FLOW_MAP_KEY: (3072, 256),
        }.items():
            _shape(reader, key, expected, "DiT")
        scaled = any(key.endswith(".weight_scale") for key in keys)
        if quantization == "fp8" and not scaled:
            raise _fail("FP8 checkpoint has no rowwise weight scales")
        if quantization == "bf16" and scaled:
            raise _fail("BF16 selection points to a quantized checkpoint")
        allowed = {"BF16", "F16", "F32"} | ({"F8_E4M3"} if quantization == "fp8" else set())
        _validate_storage_dtypes(reader, "DiT " + quantization, allowed)

    with _inspect_safetensors(ae, "autoencoder") as reader:
        for key, expected in {
            "encoder.conv_in.weight": (128, 3, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3),
            "bn.running_mean": (128,),
        }.items():
            _shape(reader, key, expected, "autoencoder")

    _validate_text_storage(text_files, config)
    fingerprint = _fingerprint(root, [checkpoint, ae, *text_files])
    return {
        "model_dir": str(root),
        "checkpoint_path": str(checkpoint),
        "autoencoder_path": str(ae),
        "text_encoder_path": str(text),
        "quantization": quantization,
        "model": "distilled",
        "display_name": f"FLUX 3 Image Klein ({quantization.upper()})",
        "inventory": inventory_model_dir(root),
        "fingerprint": fingerprint,
    }


def inventory_model_dir(path) -> dict:
    """Report components and known variants without claiming compatibility."""
    root = Path(path).expanduser().resolve()
    return {
        "bf16_dit": (root / "model.safetensors").is_file(),
        "fp8r_dit": (root / "variants/fp8r/model.safetensors").is_file(),
        "nvfp4_dit": (root / "variants/nvfp4/model.safetensors").is_file(),
        "autoencoder": (root / "ae.safetensors").is_file(),
        "text_encoder": (root / "text_encoder/config.json").is_file(),
        "supported": list(SUPPORTED_QUANTIZATIONS),
    }


def discover_model_packages(root) -> list[dict]:
    """Find valid packages at root or one directory below it."""
    base = Path(root).expanduser()
    candidates = [base]
    if base.is_dir():
        candidates.extend(path for path in base.iterdir() if path.is_dir())
    found = []
    for candidate in candidates:
        for quantization in SUPPORTED_QUANTIZATIONS:
            try:
                found.append(validate_model_dir(candidate, quantization))
            except ValueError:
                pass
    return found
