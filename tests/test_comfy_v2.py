import sys,unittest,json,types
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux3_forge import comfy_int8 as packed, comfy_pipeline as builder, comfy_runtime as runtime
from flux3_forge.vendor.flux3.text_encoder import TextEncoder,OUTPUT_LAYERS
from flux3_forge import staging

def descriptor(**extra):return torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':4,**extra}).encode()),dtype=torch.uint8)
class NewLoaderTests(unittest.TestCase):
 def fixture(self):
  with torch.device('meta'):module=nn.Sequential(nn.Linear(16,4,bias=False))
  data={'0.weight':torch.ones(4,16,dtype=torch.int8),'0.weight_scale':torch.ones(4,1),'0.comfy_quant':descriptor()}
  return module,data
 def test_conflicting_nested_settings_reject(self):
  with self.assertRaisesRegex(ValueError,'Conflicting'):packed.configuration(descriptor(params={'convrot':False}))
 def test_foreign_fp8_plain_leaf_cannot_silently_cast_to_bf16(self):
  module,data=self.fixture();module.add_module('1',nn.Linear(4,4,bias=False,device='meta'));data['1.weight']=torch.ones(4,4).to(torch.float8_e4m3fn)
  with self.assertRaisesRegex(ValueError,'foreign'):packed.load_state(module,data)
 def test_encoder_full_precision_override_keeps_integer_storage(self):
  module,data=self.fixture();loaded=packed.load_state(module,data,compute_dtype=torch.float32,full_precision_matrix_mult=True)
  self.assertTrue(loaded[0].conf['full_precision_matrix_mult']);self.assertEqual(loaded[0].qdata.dtype,torch.int8)
  self.assertTrue(torch.equal(loaded[0].qdata,data['0.weight']))
 def test_cpu_backing_restores_qdata_and_scale_identity(self):
  module,data=self.fixture();loaded=packed.load_state(module,data,compute_dtype=torch.float32)
  backing=staging.capture(loaded);original_q=loaded[0].qdata;original_scale=loaded[0].scale
  loaded[0].qdata=loaded[0].qdata.clone();loaded[0].scale=loaded[0].scale.clone();staging.mark_staged(loaded,backing)
  self.assertTrue(staging.restore(loaded,backing));self.assertEqual(loaded[0].qdata.data_ptr(),original_q.data_ptr());self.assertEqual(loaded[0].scale.data_ptr(),original_scale.data_ptr())
 def test_wrong_override_type_rejects(self):
  module,data=self.fixture()
  with self.assertRaises(ValueError):packed.load_state(module,data,full_precision_matrix_mult='yes')

class ExperimentalLifecycleTests(unittest.TestCase):
 def test_default_off_rejects_before_cuda_or_builder(self):
  with patch.object(runtime,'build_pipeline') as build,patch.object(torch.cuda,'is_available') as cuda:
   with self.assertRaisesRegex(ValueError,'opt-in'):runtime.ExperimentalRuntime({})
   build.assert_not_called();cuda.assert_not_called()
 def test_raw_manifest_cannot_opt_in_past_validation(self):
  with self.assertRaisesRegex(ValueError,'caller-supplied'):runtime.ExperimentalRuntime({'profile_validated':True},allow_unverified=True)
 def fake(self):
  obj=runtime.ExperimentalRuntime.__new__(runtime.ExperimentalRuntime);obj.manifest={'encoder_quantization':'int8_convrot','source_revision':'synthetic'};obj.pipeline=object();obj._prompt_cache={};obj._prompt_cache_bytes=0;obj._cpu_backing={};return obj
 def test_adapters_and_spectrum_disabled_before_base_sampling(self):
  obj=self.fake()
  with patch.object(runtime.RuntimeModel,'generate') as generate:
   for request in [{'adapters':[{}]},{'spectrum':True}]:
    with self.assertRaises(ValueError):obj.generate(request)
   generate.assert_not_called()
 def test_oom_releases_model_and_cache_with_clear_error(self):
  obj=self.fake();obj._cpu_backing['x']=object();obj._prompt_cache['x']=object()
  with patch.object(runtime.RuntimeModel,'generate',side_effect=torch.OutOfMemoryError('synthetic')):
   with self.assertRaisesRegex(RuntimeError,'model released'):obj.generate({})
  self.assertIsNone(obj.pipeline);self.assertFalse(obj._prompt_cache);self.assertFalse(obj._cpu_backing)
 def test_context_exit_releases_and_metadata_stays_experimental(self):
  obj=self.fake()
  with obj,patch.object(runtime.RuntimeModel,'generate',return_value=(object(),{})):
   _,metadata=obj.generate({});self.assertTrue(metadata['experimental']);self.assertFalse(metadata['real_weight_acceptance'])
  self.assertIsNone(obj.pipeline)
 def test_constructor_failure_is_released_without_fallback(self):
  with patch.object(runtime,'require_profile',return_value={}),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'is_bf16_supported',return_value=True),patch.object(runtime,'build_pipeline',side_effect=ValueError('wrong layout')),patch.object(runtime.ExperimentalRuntime,'unload') as unload:
   with self.assertRaisesRegex(RuntimeError,'wrong layout'):runtime.ExperimentalRuntime(object(),allow_unverified=True)
   unload.assert_called_once()

class EncoderWiringTests(unittest.TestCase):
 def test_single_file_base_wrapper_matches_all_output_tokens_and_ids(self):
  from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
  config=Qwen3VLConfig(text_config={'vocab_size':64,'hidden_size':32,'intermediate_size':64,'num_hidden_layers':36,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,'rope_scaling':{'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}},vision_config={'hidden_size':32,'intermediate_size':64,'depth':1,'num_heads':4,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':2,'out_hidden_size':32,'num_position_embeddings':64,'deepstack_visual_indexes':[]},image_token_id=61,video_token_id=62)
  torch.manual_seed(12);full=Qwen3VLForConditionalGeneration(config).eval()
  class Tokenizer:
   def __call__(self,text,**kwargs):
    length=kwargs.get('max_length',4) if kwargs.get('padding')=='max_length' else 4
    ids=torch.zeros(1,length,dtype=torch.long);ids[:,:4]=torch.tensor([1,2,3,4]);mask=torch.zeros_like(ids);mask[:,:4]=1
    return dict(input_ids=ids,attention_mask=mask)
  processor=types.SimpleNamespace(tokenizer=Tokenizer(),apply_chat_template=lambda *a,**k:'caption')
  encoder=builder.wrap_encoder(full.model,processor,'cpu',TextEncoder)
  self.assertIs(encoder.inference_module,full.model.language_model)
  tokens,ids=encoder.encode('caption');inputs=processor.tokenizer('caption',padding='max_length',max_length=80);mask=inputs['attention_mask'];pos=(mask.cumsum(-1)-1).masked_fill(mask==0,1)
  with torch.no_grad():reference=full(**inputs,position_ids=pos[None].expand(4,-1,-1),output_hidden_states=True,use_cache=False)
  expected=torch.stack([reference.hidden_states[k] for k in OUTPUT_LAYERS],dim=1).permute(0,2,1,3).reshape(80,-1).to(torch.bfloat16)
  torch.testing.assert_close(tokens,expected,atol=0,rtol=0);self.assertEqual(tokens.shape,(80,256));self.assertEqual(ids.shape,(80,4));self.assertTrue(torch.equal(ids[:,3],torch.arange(80)))

if __name__=='__main__':unittest.main()
