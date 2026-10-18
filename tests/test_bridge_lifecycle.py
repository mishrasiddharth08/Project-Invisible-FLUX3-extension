import sys
import types
import unittest
from unittest.mock import patch
from flux3_forge import bridge


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        module = types.ModuleType("modules")
        class Runner:
            def run(self,p,*args,**kwargs):
                return "native-script"
        module.scripts = types.SimpleNamespace(ScriptRunner=Runner)
        module.processing = types.SimpleNamespace(process_images=lambda p:"native-process")
        module.sd_models = types.SimpleNamespace(list_models=lambda:None,checkpoints_list={},checkpoint_aliases={})
        module.script_callbacks = types.SimpleNamespace(on_script_unloaded=lambda callback:None)
        module.shared = types.SimpleNamespace(opts=types.SimpleNamespace(data_labels={}))
        return module

    def test_scoped_dispatch_idempotency_and_uninstall(self):
        modules = self.fixture()
        original_process,original_run = modules.processing.process_images,modules.scripts.ScriptRunner.run
        p = object()
        with patch.dict(sys.modules,{"modules":modules}), patch.object(bridge,"_release"), patch.object(bridge,"register_checkpoint"), patch.object(bridge,"selected",return_value=None) as selected, patch.object(bridge,"generate",return_value="flux3"):
            try:
                self.assertTrue(bridge.install())
                wrapped = modules.processing.process_images
                self.assertTrue(bridge.install())
                self.assertIs(modules.processing.process_images,wrapped)
                self.assertEqual(modules.processing.process_images(p),"native-process")
                self.assertEqual(modules.scripts.ScriptRunner().run(p),"native-script")
                selected.return_value = {"model_dir":"local"}
                self.assertEqual(modules.processing.process_images(p),"flux3")
                self.assertEqual(modules.scripts.ScriptRunner().run(p),"flux3")
            finally:
                bridge.uninstall()
            self.assertIs(modules.processing.process_images,original_process)
            self.assertIs(modules.scripts.ScriptRunner.run,original_run)

    def test_failed_install_rolls_back(self):
        modules = self.fixture()
        original = modules.processing.process_images
        with patch.dict(sys.modules,{"modules":modules}), patch.object(bridge,"_release"), patch.object(bridge,"register_checkpoint",side_effect=RuntimeError("host drift")):
            self.assertFalse(bridge.install())
            self.assertIs(modules.processing.process_images,original)
            self.assertFalse(bridge._installed)

    def test_late_api_alias_keeps_native_chain_and_routes_flux3(self):
        modules = self.fixture()
        api = types.ModuleType("modules.api.api")
        api.process_images = lambda p: "late-native-alias"
        original = api.process_images
        with patch.dict(sys.modules,{"modules":modules,"modules.api.api":api}), patch.object(bridge,"_release"), patch.object(bridge,"register_checkpoint"), patch.object(bridge,"selected",return_value=None) as selected, patch.object(bridge,"generate",return_value="flux3"):
            try:
                self.assertTrue(bridge.install())
                bridge.bind_entrypoints()
                wrapped = api.process_images
                bridge.bind_entrypoints()
                self.assertIs(api.process_images,wrapped)
                self.assertEqual(api.process_images(object()),"late-native-alias")
                selected.return_value={"model_dir":"local"}
                self.assertEqual(api.process_images(object()),"flux3")
            finally:
                bridge.uninstall()
            self.assertIs(api.process_images,original)


if __name__ == "__main__":
    unittest.main()
