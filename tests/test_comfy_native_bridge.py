import unittest,types,sys
from unittest.mock import patch,Mock
from flux3_forge import bridge,comfy_bridge as c
class NativeProfileTests(unittest.TestCase):
 def tearDown(self):c.unload()
 def test_registered_basename_resolves_without_core_fallback(self):
  asset={'checkpoint_path':'models/FLUX-3/base.safetensors'};sd=types.SimpleNamespace(checkpoint_aliases={},checkpoints_list={});modules=types.ModuleType('modules');modules.sd_models=sd;modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint='unrelated'))
  with patch.dict(sys.modules,{'modules':modules}),patch.dict(bridge._registered,{'canonical':asset},clear=True):
   self.assertIs(bridge.selected(name='base.safetensors'),asset)
   self.assertIsNone(bridge.selected(name='unrelated.safetensors'))
 def test_pinned_encoder_survives_autolink_rename_and_rejects_wrong_copy(self):
  import tempfile,hashlib
  from pathlib import Path
  scratch=Path('lightweight-test-tmp');scratch.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=scratch) as folder:
   root=Path(folder);file=root/'Qwen3-VL-4B-TextEncoder-int8_convrot--For-FLUX-3.safetensors';file.write_bytes(b'verified')
   pin=(8,hashlib.sha256(b'verified').hexdigest());self.assertEqual(c.pinned_file(root,pin),file.resolve())
   file.rename(root/'another-canonical-name.safetensors');self.assertEqual(c.pinned_file(root,pin).name,'another-canonical-name.safetensors')
   (root/'wrong.safetensors').write_bytes(b'not-pin!');self.assertEqual(c.pinned_file(root,pin).name,'another-canonical-name.safetensors')
   (root/'duplicate.safetensors').write_bytes(b'verified')
   with self.assertRaisesRegex(ValueError,'found 2'):c.pinned_file(root,pin)
 def test_missing_managed_profile_never_falls_through_to_unrelated_model(self):
  modules=types.ModuleType('modules');modules.sd_models=types.SimpleNamespace(checkpoint_aliases={},checkpoints_list={});modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint='unrelated'))
  with patch.dict(sys.modules,{'modules':modules}),patch.dict(bridge._registered,{},clear=True),patch.object(bridge,'_setting',return_value=True):
   with self.assertRaisesRegex(ValueError,'profile is unavailable'):bridge.selected(name='flux3-image-klein-base-int8-convrot.safetensors')
 def test_refiner_exact_raw_checkpoint_uses_verified_profile_without_global_enable(self):
  from pathlib import Path
  raw=Path('models/Stable-diffusion/FLUX-3/flux3-image-klein-base-int8-convrot.safetensors').resolve()
  asset={'checkpoint_path':str(raw),'comfy_profile':object()};paths=types.ModuleType('modules.paths_internal');paths.models_path=str(Path('models').resolve())
  with patch.object(bridge,'selected',return_value=None),patch.object(c,'configured_asset_for_checkpoint',return_value=asset) as configured,patch.dict(sys.modules,{'modules.paths_internal':paths}):
   self.assertIs(bridge.selected_refiner(filename=str(raw)),asset)
  configured.assert_called_once();self.assertFalse(bridge._setting('flux3_comfy_enabled',False))
 def test_refiner_never_substitutes_different_verified_checkpoint(self):
  from pathlib import Path
  chosen=Path('models/Stable-diffusion/FLUX-3/flux3-image-klein-base-int8-convrot.safetensors').resolve();other=chosen.with_name('flux3-image-klein-int8-convrot.safetensors')
  paths=types.ModuleType('modules.paths_internal');paths.models_path=str(Path('models').resolve())
  with patch.object(bridge,'selected',return_value=None),patch.object(c,'configured_asset_for_checkpoint',return_value={'checkpoint_path':str(other)}),patch.dict(sys.modules,{'modules.paths_internal':paths}):
   with self.assertRaisesRegex(ValueError,'no different model was substituted'):bridge.selected_refiner(filename=str(chosen))
 def test_default_off_keeps_legacy_validation(self):
  assets=types.ModuleType('flux3_forge.assets');assets.validate_model_dir=Mock(return_value={'legacy':True})
  with patch.dict(sys.modules,{'flux3_forge.assets':assets}),patch.object(bridge,'_setting',side_effect=lambda k,d=None:'legacy' if k=='flux3_model_dir' else d):
   self.assertEqual(bridge.configured_asset(),{'legacy':True})
  assets.validate_model_dir.assert_called_once_with('legacy','bf16')
 def test_canonical_klein_folder_preferred_with_legacy_fallback(self):
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);new=root/'Stable-diffusion'/'FLUX3 KLEIN';old=root/'Stable-diffusion'/'FLUX-3';new.mkdir(parents=True);old.mkdir()
   name='flux3-image-klein-base-int8-convrot.safetensors';(old/name).touch()
   self.assertEqual(c._profile_file(root,name),old/name)
   (new/name).touch();self.assertEqual(c._profile_file(root,name),new/name)
 def test_selected_distilled_profile_ignores_different_ui_variant(self):
  import tempfile
  from pathlib import Path
  from flux3_forge import comfy_profiles
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);checkpoint=root/'Stable-diffusion'/'FLUX3 KLEIN'/'flux3-image-klein-int8-convrot.safetensors';checkpoint.parent.mkdir(parents=True);checkpoint.touch()
   profile=object();settings=lambda key,default=None:{'flux3_comfy_family':'base','flux3_comfy_precision':'bf16','flux3_comfy_encoder_precision':'int8_convrot'}.get(key,default)
   with patch.object(c,'pinned_file',return_value=root/'encoder.safetensors'),patch.object(comfy_profiles,'validate_profile',return_value=profile) as validate,patch.object(comfy_profiles,'require_profile',return_value={'checkpoint_path':str(checkpoint),'model_family':'distilled','quantization':'int8_convrot'}):
    asset=c.configured_asset_for_checkpoint(settings,root,checkpoint)
   self.assertEqual(asset['model_family'],'distilled');self.assertEqual(asset['quantization'],'int8_convrot')
   self.assertEqual(validate.call_args.kwargs['model_family'],'distilled')
 def test_base_cfg_is_distinct_from_distilled(self):
  p=types.SimpleNamespace(negative_prompt='',cfg_scale=3.5,enable_hr=False,image_mask=None,mask=None,subseed_strength=0)
  with patch.object(bridge,'_setting',return_value=True):
   bridge._validate_processing(p,'base')
   with self.assertRaisesRegex(ValueError,'CFG scale 1'):bridge._validate_processing(p,'distilled')
 def test_invalid_base_cfg_rejected(self):
  for cfg in [float('nan'),float('inf'),-1,21]:
   p=types.SimpleNamespace(negative_prompt='',cfg_scale=cfg)
   with patch.object(bridge,'_setting',return_value=True),self.assertRaisesRegex(ValueError,'Base CFG'):bridge._validate_processing(p,'base')
 def mocks(self):
  profiles=types.ModuleType('flux3_forge.comfy_profiles');profiles.require_profile=Mock(return_value={})
  runtime=types.ModuleType('flux3_forge.comfy_runtime');runtime.ExperimentalRuntime=Mock()
  profile=types.SimpleNamespace(identities=(('a',1,2),));return profiles,runtime,profile
 def test_reuses_only_same_profile_and_memory_policy(self):
  profiles,runtime,p=self.mocks();runtime.ExperimentalRuntime.return_value.generate.return_value=('image',{})
  with patch.dict(sys.modules,{'flux3_forge.comfy_profiles':profiles,'flux3_forge.comfy_runtime':runtime}):
   for _ in range(2):self.assertEqual(c.generate({'comfy_profile':p}),('image',{}))
   runtime.ExperimentalRuntime.assert_called_once_with(p,'balanced',allow_unverified=True)
   c.generate({'comfy_profile':p,'memory_mode':'resident'})
   self.assertEqual(runtime.ExperimentalRuntime.call_count,2)
 def test_failure_releases_owned_model(self):
  profiles,runtime,p=self.mocks();runtime.ExperimentalRuntime.return_value.generate.side_effect=ValueError('unsupported')
  with patch.dict(sys.modules,{'flux3_forge.comfy_profiles':profiles,'flux3_forge.comfy_runtime':runtime}):
   with self.assertRaisesRegex(ValueError,'unsupported'):c.generate({'comfy_profile':p})
   self.assertIsNone(c._MODEL);runtime.ExperimentalRuntime.return_value.unload.assert_called_once()
 def test_profile_validation_precedes_construction(self):
  profiles,runtime,p=self.mocks();profiles.require_profile.side_effect=ValueError('changed asset')
  with patch.dict(sys.modules,{'flux3_forge.comfy_profiles':profiles,'flux3_forge.comfy_runtime':runtime}):
   with self.assertRaisesRegex(ValueError,'changed asset'):c.generate({'comfy_profile':p})
   runtime.ExperimentalRuntime.assert_not_called()
class NativeGenerateRouting(unittest.TestCase):
 def fixture(self):
  state=types.SimpleNamespace(interrupted=False,skipped=False,assign_current_image=lambda im:None,nextjob=lambda:None)
  shared=types.SimpleNamespace(state=state,opts=types.SimpleNamespace(samples_save=False,samples_format='png'),prompt_styles=types.SimpleNamespace(apply_styles_to_prompt=lambda p,s:p,apply_negative_styles_to_prompt=lambda p,s:p))
  modules=types.ModuleType('modules');modules.shared=shared;modules.images=types.SimpleNamespace(save_image=Mock())
  class Processed:
   def __init__(self,p,images,seed,info,**kw):self.images=images;self.info=info
  modules.processing=types.SimpleNamespace(Processed=Processed);modules.extra_networks=types.SimpleNamespace(parse_prompt=lambda p:(p,{}))
  backend=types.ModuleType('backend');backend.memory_management=types.SimpleNamespace(unload_all_models=Mock())
  p=types.SimpleNamespace(prompt='complete literal caption',negative_prompt='',cfg_scale=3.5,batch_size=1,n_iter=1,width=64,height=64,steps=25,seed=478920109938706,styles=[],init_images=[],extra_generation_params={},sampler_name='Euler',scheduler='Simple',override_settings={})
  profile=object();asset=dict(checkpoint_path='base.safetensors',model_dir='models',quantization='int8_convrot',model_family='base',comfy_profile=profile)
  return modules,backend,p,asset
 def test_base_generate_forwards_cfg_large_seed_schedule_and_profile(self):
  modules,backend,p,asset=self.fixture()
  with patch.dict(sys.modules,{'modules':modules,'backend':backend}),patch.object(bridge,'_setting',side_effect=lambda k,d=None:True if k.endswith('acknowledged') else d),patch.object(c,'generate',return_value=(object(),{'task':'txt2img'})) as generate:
   result=bridge.generate(p,asset)
  self.assertEqual(len(result.images),1);r=generate.call_args.args[0]
  self.assertIs(r['comfy_profile'],asset['comfy_profile']);self.assertEqual((r['cfg'],r['alpha'],r['seed_bits'],r['seed']),(3.5,16.,64,478920109938706));self.assertFalse(r['reference_edit'])
 def test_foreign_encoder_override_rejected_before_engine(self):
  modules,backend,p,asset=self.fixture();p.override_settings={'forge_additional_modules':['foreign-encoder.safetensors']}
  with patch.dict(sys.modules,{'modules':modules,'backend':backend}),patch.object(bridge,'_setting',side_effect=lambda k,d=None:True if k.endswith('acknowledged') else d),patch.object(c,'generate') as generate:
   with self.assertRaisesRegex(RuntimeError,'pinned Qwen3-VL'):bridge.generate(p,asset)
   generate.assert_not_called();backend.memory_management.unload_all_models.assert_not_called()
if __name__=='__main__':unittest.main()
