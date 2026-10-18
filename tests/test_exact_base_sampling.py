import ast,copy,math,types,unittest
from pathlib import Path
import torch
from unittest.mock import patch
from flux3_forge import exact_base_sampling as s
def function(path,name):
 tree=ast.parse(Path(path).read_text(encoding='utf-8'));fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name));fn.decorator_list=[];fn.returns=None
 for arg in fn.args.args:arg.annotation=None
 env={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'reviewed-upstream-sampling','exec'),env);return env[name]
class TinyBase:
 flow_map=False
 def __call__(self,packed):
  x=packed.data['image'].chunk(2)[0];return torch.cat([x*.1+1,x*.1+2])
def packed():
 value=torch.arange(12,dtype=torch.float32).reshape(4,3)
 return types.SimpleNamespace(data={'image':torch.cat([value,value])},timesteps={'image':torch.zeros(8)})
class Sampling(unittest.TestCase):
 def test_flow_map_matches_official_target_time_loop(self):
  class Flow:
   flow_map=True
   def __call__(self,p,timesteps_ref):return p.data['image']*.1+timesteps_ref.item()
  ref=function('semantic-sources/bfl_sampling.py','denoise_flow_map');a=packed();b=packed();schedule=s.simple_schedule(16,16)
  self.assertTrue(torch.equal(s.run_flow_map_steps(Flow(),a,schedule),ref(Flow(),b,schedule)))
 def test_flow_map_rejects_base_family(self):
  with self.assertRaises(ValueError):s.run_flow_map_steps(TinyBase(),packed(),s.simple_schedule(4,16))
 def test_cfg_matches_official_base_loop(self):
  ref=function('semantic-sources/bfl_sampling.py','denoise')
  for cfg in [0,1,3.5,10]:
   a=packed();b=packed();schedule=s.simple_schedule(7,16)
   self.assertTrue(torch.equal(s.run_base_steps(TinyBase(),a,schedule,cfg),ref(TinyBase(),b,schedule,cfg)))
 def test_comfy_schedule_source_exact(self):
  ref=function('semantic-sources/comfy_samplers.py','simple_scheduler')
  for alpha in [1,6.93,16]:
   t=torch.arange(1,10001,dtype=torch.float32)/10000;shift=math.exp(math.log(alpha));table=shift/(shift+(1/t-1))
   for steps in [1,7,16,25,100]:self.assertTrue(torch.equal(s.simple_schedule(steps,alpha),ref(types.SimpleNamespace(sigmas=table),steps)))
 def test_comfy_noise_source_exact(self):
  ref=function('semantic-sources/comfy_sample.py','prepare_noise_inner');seed=478920109938706
  latent=torch.zeros(1,128,2,3);noise=ref(latent,torch.Generator().manual_seed(seed));tokens=noise.permute(0,2,3,1).reshape(6,128)
  self.assertTrue(torch.equal(s.latent_noise(seed,2,3),tokens))
 def test_not_original_native_noise_layout(self):
  seed=123;old=torch.randn((6,128),generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).float()
  self.assertFalse(torch.equal(s.latent_noise(seed,2,3),old))
 def test_cancel_and_recovery(self):
  progress=[]
  with self.assertRaises(InterruptedError):s.run_base_steps(TinyBase(),packed(),s.simple_schedule(8,16),3.5,lambda i,total:progress.append(i),lambda:len(progress)>=2)
  self.assertEqual(progress,[1,2]);a=s.run_base_steps(TinyBase(),packed(),s.simple_schedule(8,16),3.5);b=s.run_base_steps(TinyBase(),packed(),s.simple_schedule(8,16),3.5);self.assertTrue(torch.equal(a,b))
 def test_flowmap_wrong_sampler_rejected(self):
  dit=TinyBase();dit.flow_map=True
  with self.assertRaises(ValueError):s.run_base_steps(dit,packed(),s.simple_schedule(4,16),3.5)
 def test_invalid_cfg(self):
  for value in [float('nan'),float('inf'),-1,21,True]:
   with self.assertRaises(ValueError):s.run_base_steps(TinyBase(),packed(),s.simple_schedule(4,16),value)
 def test_invalid_schedule(self):
  for steps,alpha in [(0,16),(True,16),(16,0),(16,float('nan'))]:
   with self.assertRaises(ValueError):s.simple_schedule(steps,alpha)
 def test_no_cuda_initialization(self):self.assertFalse(torch.cuda.is_initialized())
 def test_real_meta_architectures_distinguish_family(self):
  from flux3_forge.vendor.flux3.dit import DiT
  with torch.device('meta'):base=DiT(False);distilled=DiT(True)
  self.assertNotIn('ref_time_in.in_layer.weight',base.state_dict());self.assertIn('ref_time_in.in_layer.weight',distilled.state_dict())
  self.assertFalse(base.flow_map);self.assertTrue(distilled.flow_map)
 def test_builder_rejects_wrong_flowmap_before_allocation(self):
  from flux3_forge import comfy_pipeline as builder
  manifest=dict(model_family='base',quantization='bf16',encoder_quantization='bf16',checkpoint_path='dit',text_encoder_path='support',autoencoder_path='vae',text_encoder_weights='encoder')
  with patch.object(builder,'require_profile',return_value=manifest):
   with self.assertRaisesRegex(ValueError,'flow-map'):builder.build_pipeline(object(),factories={'flow_map':lambda _:True})
 def test_unknown_profile_family_rejected_before_file_access(self):
  from flux3_forge import comfy_profiles
  with self.assertRaisesRegex(ValueError,'family'):comfy_profiles.validate_profile('.',checkpoint='missing',encoder='missing',vae='missing',support_dir='missing',quantization='int8_convrot',encoder_quantization='int8_convrot',experimental=True,model_family='unknown')
 def test_base_runtime_rejects_reference_comparison_input(self):
  from flux3_forge.exact_base_runtime import generate_base
  owner=types.SimpleNamespace(manifest={'model_family':'base'},dit=TinyBase())
  with self.assertRaisesRegex(ValueError,'comparison'):generate_base(owner,{'images':[object()]})
if __name__=='__main__':unittest.main(verbosity=2)
