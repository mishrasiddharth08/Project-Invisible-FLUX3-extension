import json,sys,tempfile,unittest,hashlib,copy
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux3_forge import comfy_profiles as profiles

def descriptor(**extra):
 return torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':4,**extra}).encode()),dtype=torch.uint8)
def module():return nn.Sequential(nn.Linear(16,4),nn.LayerNorm(4))
def state(precision):
 data={k:v.to(torch.bfloat16).contiguous() for k,v in module().state_dict().items()}
 if precision=='int8_convrot':
  data['0.weight']=torch.ones((4,16),dtype=torch.int8);data['0.weight_scale']=torch.ones((4,1));data['0.comfy_quant']=descriptor()
 return data

class ProfilesTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.support=self.root/'support';self.support.mkdir()
  self.config={'model_type':'qwen3_vl','text_config':dict(profiles.TEXT_CONTRACT,rope_scaling={'rope_type':'default','mrope_section':[24,20,20],'mrope_interleaved':True})}
  (self.support/'config.json').write_text(json.dumps(self.config));self.catalog={'files':{'config.json':profiles.sha256(self.support/'config.json')}}
  self.catalog_path=self.root/'pins.json';self.catalog_path.write_text(json.dumps(self.catalog))
  self.paths={};self.pins={}
  for role,parent in [('dit','Stable-diffusion'),('encoder','text_encoder')]:
   for precision in ['bf16','int8_convrot']:
    path=self.root/parent/'FLUX-3'/(precision+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
    data=state(precision)
    if role=='encoder':data={'model.'+k:v for k,v in data.items()}
    save_file(data,str(path));self.paths[(role,precision)]=path;self.pins[(role,precision)]=(path.stat().st_size,profiles.sha256(path))
  vae=self.root/'VAE/FLUX-3/vae.safetensors';vae.parent.mkdir(parents=True);save_file(module().state_dict(),str(vae));self.vae=vae;self.pins[('vae','fp32')]=(vae.stat().st_size,profiles.sha256(vae))
  # Only the test replaces source pins/architecture with tiny manufactured fixtures.
  for target,value in [('PINNED',self.pins),('_schemas',lambda config:{r:module() for r in ['dit','encoder','vae']})]:
   context=patch.object(profiles,target,value);context.start();self.addCleanup(context.stop)
  original=Path.read_text
  def read(path,*args,**kwargs):
   if path.name=='comfy_support_pins.json':return json.dumps(self.catalog)
   return original(path,*args,**kwargs)
  context=patch.object(Path,'read_text',read);context.start();self.addCleanup(context.stop)
 def request(self,dit='bf16',enc='bf16',**extra):
  data=dict(checkpoint=self.paths[('dit',dit)],encoder=self.paths[('encoder',enc)],vae=self.vae,support_dir=self.support,quantization=dit,encoder_quantization=enc,experimental=True);data.update(extra)
  return profiles.validate_profile(self.root,**data)
 def test_all_four_combinations_receive_immutable_offline_receipts(self):
  for dit in ['bf16','int8_convrot']:
   for encoder in ['bf16','int8_convrot']:
    receipt=self.request(dit,encoder);data=profiles.require_profile(receipt)
    self.assertFalse(data['real_weight_acceptance']);self.assertEqual(data['quantization'],dit)
    with self.assertRaises(TypeError):data['quantization']='nvfp4'
 def test_default_off_and_foreign_precisions_reject(self):
  for kwargs in [{'experimental':False},{'quantization':'nvfp4'},{'encoder_quantization':'fp8'}]:
   with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.request(**kwargs)
 def test_caller_boolean_is_not_a_receipt(self):
  with self.assertRaisesRegex(ValueError,'caller-supplied'):profiles.require_profile({'profile_validated':True})
 def test_missing_and_noncanonical_paths_reject_before_construction(self):
  for path in [self.root/'missing.safetensors',self.root/'Stable-diffusion/FLUX-3/missing.safetensors',self.paths[('encoder','bf16')]]:
   with self.subTest(path=path),self.assertRaises(ValueError):self.request(checkpoint=path)
 def test_exact_pin_rejects_changed_size_and_same_size_changed_payload(self):
  path=self.paths[('dit','bf16')];original=path.read_bytes();path.write_bytes(original+b'x')
  with self.assertRaisesRegex(ValueError,'size differs'):self.request()
  path.write_bytes(original[:-1]+bytes([original[-1]^1]))
  with self.assertRaisesRegex(ValueError,'SHA256'):self.request()
 def test_receipt_detects_changed_or_missing_weights(self):
  receipt=self.request();path=self.paths[('dit','bf16')];path.write_bytes(path.read_bytes()+b'x')
  with self.assertRaisesRegex(ValueError,'changed'):profiles.require_profile(receipt)
  path.unlink()
  with self.assertRaisesRegex(ValueError,'missing'):profiles.require_profile(receipt)
 def test_support_file_is_pinned_and_receipt_tracks_it(self):
  receipt=self.request();(self.support/'config.json').write_text('{}')
  with self.assertRaises(ValueError):profiles.require_profile(receipt)
  with self.assertRaisesRegex(ValueError,'offline tokenizer'):self.request()
 def test_exact_qwen_config_rejects_wrong_rope_size_vocab_and_remote_code(self):
  for change in [('hidden_size',4096),('num_hidden_layers',32),('vocab_size',64),('rope_theta',10000)]:
   config=copy.deepcopy(self.config);config['text_config'][change[0]]=change[1];(self.support/'config.json').write_text(json.dumps(config))
   with self.assertRaisesRegex(ValueError,'matching'):profiles._config(self.support)
  for config in [dict(self.config,auto_map={'x':'y'}),dict(self.config,quantization_config={'x':1})]:
   (self.support/'config.json').write_text(json.dumps(config))
   with self.assertRaisesRegex(ValueError,'Foreign'):profiles._config(self.support)
 def inspect(self,data,precision='int8_convrot',role='dit'):
  path=self.root/'fixture.safetensors';save_file(data,str(path));return profiles.inspect_contract(path,module(),role,precision)
 def test_full_shape_key_contract_and_storage_not_filename(self):
  for mutation in ['missing','extra','wrong_shape','float8','integer_without_descriptor']:
   data=state('bf16')
   if mutation=='missing':data.pop('1.weight')
   elif mutation=='extra':data['foreign.weight']=torch.ones(1)
   elif mutation=='wrong_shape':data['0.weight']=torch.zeros(4,15,dtype=torch.bfloat16)
   elif mutation=='float8':data['0.weight']=data['0.weight'].to(torch.float8_e4m3fn)
   else:data['0.weight']=torch.ones(4,16,dtype=torch.int8)
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.inspect(data,'bf16')
 def test_strict_metadata_scale_and_convrot_failures(self):
  for mutation in ['nan','negative','bf16_scale','wrong_scale','unknown_conf','not_convrot','bad_group','oversized']:
   data=state('int8_convrot')
   if mutation=='nan':data['0.weight_scale'][0]=float('nan')
   elif mutation=='negative':data['0.weight_scale'][0]=-1
   elif mutation=='bf16_scale':data['0.weight_scale']=data['0.weight_scale'].to(torch.bfloat16)
   elif mutation=='wrong_scale':data['0.weight_scale']=torch.ones(1)
   elif mutation=='unknown_conf':data['0.comfy_quant']=descriptor(unknown=True)
   elif mutation=='not_convrot':data['0.comfy_quant']=descriptor(convrot=False)
   elif mutation=='bad_group':data['0.comfy_quant']=descriptor(convrot_groupsize=64)
   else:data['0.comfy_quant']=torch.ones(65537,dtype=torch.uint8)
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.inspect(data)
 def test_attention_policy_only_known_bounded_metadata(self):
  data=state('int8_convrot');data['0.comfy_attention.config']=torch.tensor(list(b'{"attention":"comfy_kitchen_int8"}'),dtype=torch.uint8)
  self.assertEqual(self.inspect(data),1)
  data['0.comfy_attention.config']=torch.tensor(list(b'{"attention":"other"}'),dtype=torch.uint8)
  with self.assertRaisesRegex(ValueError,'Unknown attention'):self.inspect(data)
 def test_duplicate_normalized_encoder_keys_reject(self):
  data=state('bf16');data.update({'model.'+key:value.clone() for key,value in list(data.items())})
  with self.assertRaisesRegex(ValueError,'Ambiguous'):self.inspect(data,'bf16','encoder')
 def test_exact_two_bf16_shards_reuse_without_merge(self):
  source=self.paths[('encoder','bf16')];from safetensors.torch import load_file
  data=load_file(str(source));keys=list(data);parts=[dict((k,data[k]) for k in keys[::2]),dict((k,data[k]) for k in keys[1::2])];paths=[]
  for i,part in enumerate(parts):
   path=source.with_name(f'shard-{i}.safetensors');save_file(part,str(path));paths.append(path);self.pins[(f'encoder_shard_{i+1}','bf16')]=(path.stat().st_size,profiles.sha256(path))
  receipt=self.request(encoder=paths);self.assertEqual(len(profiles.require_profile(receipt)['text_encoder_weights']),2)
  with self.assertRaisesRegex(ValueError,'Duplicate tensor'):profiles.inspect_contract([paths[0],paths[0]],module(),'encoder','bf16')

class RealArchitectureSchemaTests(unittest.TestCase):
 def test_full_original_meta_contract_contains_flow_map_and_qwen4b(self):
  config=json.loads((Path(__file__).parent/'fixtures/real-qwen-config.json').read_text())
  schemas=profiles._schemas(config);dit=schemas['dit'].state_dict();encoder=schemas['encoder'].state_dict()
  self.assertEqual(tuple(dit['ref_time_in.in_layer.weight'].shape),(3072,256))
  self.assertEqual(tuple(dit['single_blocks.27.q_proj.weight'].shape),(3072,3072))
  self.assertEqual(tuple(encoder['language_model.embed_tokens.weight'].shape),(151936,2560))
  self.assertIn('language_model.layers.35.self_attn.q_proj.weight',encoder)
  self.assertTrue(all(t.is_meta for m in schemas.values() for t in m.state_dict().values()))

if __name__=='__main__':unittest.main()
