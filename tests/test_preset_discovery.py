import sys,types,unittest,importlib.util
from pathlib import Path
from unittest.mock import patch
from flux3_forge import presets as integration

ROOT=Path(r'G:\FORGE UI NEO\sd-webui-forge-classic')
class Info:
 def __init__(self,default=None,*args,**kwargs):self.default=default
class Options:
 def __init__(self):
  self.data={'forge_preset':'moss','sd_model_checkpoint':'current-MOSS','forge_additional_modules':['current-TE'],'forge_checkpoint_flux':'legacyFLUX'};self.data_labels={}
 def add_option(self,key,info):self.data_labels[key]=info
class PresetDiscoveryTests(unittest.TestCase):
 def setup(self,checkpoint=True):
  spec=importlib.util.spec_from_file_location('_fresh_forge_presets',ROOT/'modules_forge/presets.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
  opts=Options();modules=types.ModuleType('modules');modules.shared=types.SimpleNamespace(opts=opts)
  forge=types.ModuleType('modules_forge');forge.presets=native
  options=types.ModuleType('modules.options');options.OptionInfo=Info;options.OptionRow=Info;options.options_section=lambda section,data:data
  items=types.ModuleType('modules.shared_items');items.list_samplers=lambda:[];items.list_schedulers=lambda:[]
  gradio=types.ModuleType('gradio');gradio.Dropdown=object;gradio.Slider=object
  card=types.SimpleNamespace(name='FLUX 3 Image Klein (FP8)',title='FLUX 3 Image Klein (FP8)',filename='real/model.safetensors',ids=['real/model.safetensors']) if checkpoint else None
  context=patch.dict(sys.modules,{'modules':modules,'modules_forge':forge,'modules.options':options,'modules.shared_items':items,'gradio':gradio});context.start();self.addCleanup(context.stop)
  context=patch('flux3_forge.bridge.register_checkpoint',return_value=card);context.start();self.addCleanup(context.stop)
  return native,opts
 def test_real_native_enum_choices_and_options_add_distinct_flux3(self):
  native,opts=self.setup();self.assertNotIn('flux3',native.PresetArch.choices());self.assertTrue(integration.register());self.assertIn('flux3',native.PresetArch.choices())
  self.assertEqual(opts.data['forge_checkpoint_flux3'],'FLUX 3 Image Klein (FP8)');self.assertEqual(opts.data['forge_additional_modules_flux3'],[])
  self.assertEqual(opts.data_labels['flux3_t2i_step'].default,16);self.assertEqual(opts.data_labels['flux3_i2i_cfg'].default,1.0)
 def test_current_selection_and_legacy_flux_are_untouched(self):
  native,opts=self.setup();before=opts.data.copy();integration.register()
  self.assertTrue(all(opts.data[k]==v for k,v in before.items()))
 def test_missing_package_does_not_advertise_fake_checkpoint_or_preset(self):
  native,opts=self.setup(False);before=opts.data.copy();self.assertFalse(integration.register());self.assertEqual(opts.data,before);self.assertNotIn('flux3',native.PresetArch.choices())
 def test_repeated_registration_is_idempotent_and_keeps_user_defaults(self):
  native,opts=self.setup();integration.register();arch=native.PresetArch.flux3;info=opts.data_labels['flux3_t2i_step'];opts.data['flux3_t2i_step']=24;integration.register()
  self.assertIs(native.PresetArch.flux3,arch);self.assertEqual(native.PresetArch.choices().count('flux3'),1);self.assertIs(opts.data_labels['flux3_t2i_step'],info);self.assertEqual(opts.data['flux3_t2i_step'],24)
 def test_stale_foreign_saved_preset_uses_validated_package(self):
  native,opts=self.setup();opts.data['forge_checkpoint_flux3']='oldFLUX2';opts.data['forge_additional_modules_flux3']=['foreignTE'];integration.register()
  self.assertEqual(opts.data['forge_checkpoint_flux3'],'FLUX 3 Image Klein (FP8)');self.assertEqual(opts.data['forge_additional_modules_flux3'],[])
if __name__=='__main__':unittest.main()
