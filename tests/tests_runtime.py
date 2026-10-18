import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flux3_forge import assets, runtime


class FakeReader:
    def __init__(self, shapes, keys=None):
        self.shapes = shapes
        self._keys = keys or shapes.keys()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return list(self._keys)

    def get_slice(self, key):
        shape = self.shapes[key]
        return SimpleNamespace(get_shape=lambda: shape, get_dtype=lambda: "BF16")


class AssetsTests(unittest.TestCase):
    def test_validates_official_component_shapes_and_fingerprint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            text = root / "text_encoder"
            text.mkdir()
            for name in ("model.safetensors", "ae.safetensors"):
                (root / name).write_bytes(b"x")
            for name in ("model.safetensors", "tokenizer.json", "processor_config.json"):
                (text / name).write_bytes(b"x")
            (text / "config.json").write_text(json.dumps({
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"hidden_size": 2560, "num_hidden_layers": 36},
            }), encoding="utf-8")
            dit = FakeReader({
                "emb_in.image.weight": (3072, 128),
                "emb_in.image_cond.weight": (3072, 128),
                "txt_in.weight": (3072, 20480),
                "single_blocks.27.q_proj.weight": (3072, 3072),
                assets.FLOW_MAP_KEY: (3072, 256),
            })
            ae = FakeReader({
                "encoder.conv_in.weight": (128, 3, 3, 3),
                "decoder.conv_out.weight": (3, 128, 3, 3),
                "bn.running_mean": (128,),
            })
            with patch.object(assets, "_inspect_safetensors", side_effect=[dit, ae, FakeReader({"model.language_model.embed_tokens.weight": (151936,2560)})]):
                result = assets.validate_model_dir(root, "bf16")
            self.assertEqual(result["checkpoint_path"], str(root / "model.safetensors"))
            self.assertTrue(result["fingerprint"])

    def test_rejects_unverified_comfy_quantization(self):
        with self.assertRaisesRegex(ValueError, "ComfyUI"):
            assets.validate_model_dir(".", "int8")


class RuntimeTests(unittest.TestCase):
    def test_cancelled_callback_raises(self):
        with self.assertRaises(runtime.Cancelled):
            runtime._cancel(lambda: True)

    def test_prompt_cache_is_bounded_to_two(self):
        model = runtime.RuntimeModel.__new__(runtime.RuntimeModel)
        model.memory_mode = "resident"
        model._prompt_cache = runtime.OrderedDict()
        model._prompt_cache_bytes = 0
        encoder = SimpleNamespace(
            model=SimpleNamespace(to=lambda device: None),
            device="cpu",
            encode=lambda prompt: (torch.ones(2, 3), torch.ones(2, 4)),
        )
        model.pipeline = SimpleNamespace(text_encoder=encoder)
        model._stage = lambda module, device: module
        model._offload = lambda module: None
        for prompt in ("one", "two", "three"):
            model._encode_prompt(prompt, None)
        self.assertEqual(list(model._prompt_cache), ["two", "three"])

    def test_size_limits_fail_before_cuda_work(self):
        model = runtime.RuntimeModel.__new__(runtime.RuntimeModel)
        with self.assertRaisesRegex(ValueError, "64 to 4096"):
            model.generate({"prompt": "expanded", "width": 32, "height": 1024})


if __name__ == "__main__":
    unittest.main()
