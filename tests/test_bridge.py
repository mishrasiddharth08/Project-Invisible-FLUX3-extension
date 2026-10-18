import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flux3_forge import bridge
import flux3_forge.runtime as runtime_module


class P:
    negative_prompt = ""
    cfg_scale = 1.0
    enable_hr = False
    image_mask = None
    mask = None
    prompt = "a complete caption"
    extra_network_data = None
    subseed_strength = 0.0
    seed = 7
    batch_size = 2
    n_iter = 1
    init_images = []
    steps = 4
    width = 64
    height = 64
    styles = []
    extra_generation_params = {}
    do_not_save_samples = False
    outpath_samples = "."
    comments = {}


class BridgeTests(unittest.TestCase):
    def test_native_adapter_tag_resolves_without_activation(self):
        params = types.SimpleNamespace(positional=["style", "0.75"], named={})
        extra = types.SimpleNamespace(parse_prompt=lambda prompt: ("clean", {"lora": [params]}))
        modules = types.ModuleType("modules")
        modules.extra_networks = extra
        networks = types.ModuleType("networks")
        networks.available_networks = {"style": types.SimpleNamespace(filename=str(ROOT / "style.safetensors"))}
        networks.available_network_aliases = {}
        networks.list_available_networks = lambda: None
        with patch.dict(sys.modules, {"modules": modules, "networks": networks}):
            clean, specs = bridge._resolve_adapters("<lora:style:0.75> caption")
        self.assertEqual(clean, "clean")
        self.assertEqual(specs[0]["strength"], 0.75)
        self.assertTrue(specs[0]["path"].endswith("style.safetensors"))

    def test_rejects_unsupported_negative_before_loading(self):
        p = P()
        p.negative_prompt = "bad"
        with patch.object(bridge, "_setting", return_value=True):
            with self.assertRaisesRegex(ValueError, "negative prompts"):
                bridge._validate_processing(p)

    def test_batch_seed_progress_save_and_processed(self):
        saved, calls, callback_indices = [], [], []
        state = types.SimpleNamespace(interrupted=False, skipped=False, job_count=0,
            sampling_steps=0, sampling_step=0, textinfo="", assign_current_image=lambda image: None,
            nextjob=lambda: None)
        shared = types.SimpleNamespace(state=state,
            prompt_styles=types.SimpleNamespace(apply_styles_to_prompt=lambda prompt, styles: prompt,
                                                apply_negative_styles_to_prompt=lambda prompt, styles: prompt),
            opts=types.SimpleNamespace(samples_save=True, samples_format="png"))
        class Processed:
            def __init__(self, p, images, seed, info, **kwargs):
                self.images, self.seed, self.info, self.kwargs = images, seed, info, kwargs
        modules = types.ModuleType("modules")
        modules.shared = shared
        modules.images = types.SimpleNamespace(save_image=lambda *args, **kwargs: saved.append(args[0]))
        modules.processing = types.SimpleNamespace(Processed=Processed)
        scripts_module = types.ModuleType("modules.scripts")
        class PostprocessImageArgs:
            def __init__(self, image, index):
                self.image, self.index = image, index
        scripts_module.PostprocessImageArgs = PostprocessImageArgs
        modules.extra_networks = types.SimpleNamespace(parse_prompt=lambda prompt: (prompt, {}))
        backend = types.ModuleType("backend")
        backend.memory_management = types.SimpleNamespace(unload_all_models=lambda: None)
        def generate(request, progress, cancelled):
            calls.append(request)
            progress(1, request["steps"])
            return object(), {"route": "official"}
        p = P()
        p.extra_generation_params = {}
        p.scripts = types.SimpleNamespace(postprocess_image=lambda request, pp: callback_indices.append(pp.index))
        with patch.dict(sys.modules, {"modules": modules, "modules.scripts": scripts_module, "backend": backend}), \
             patch.object(runtime_module, "generate", generate), \
             patch.object(runtime_module, "unload", lambda: None), \
             patch.object(bridge, "_setting", side_effect=lambda key, default=None: True if key.endswith("acknowledged") else default):
            result = bridge.generate(p, {"checkpoint_path": "model.safetensors", "model_dir": "pkg", "quantization": "bf16"})
        self.assertEqual([item["seed"] for item in calls], [7, 8])
        self.assertEqual(len(saved), 2)
        self.assertEqual(len(result.images), 2)
        self.assertEqual(result.seed, 7)
        self.assertEqual(callback_indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
