import unittest
from collections import OrderedDict
from types import SimpleNamespace as NS
import torch
from flux3_forge import runtime
from flux3_forge.vendor.flux3.text_encoder import TextEncoder

class TextOnlyStaging(unittest.TestCase):
 def test_runtime_stages_only_explicit_language_module(self):
  language=object();full=object();calls=[];encoder=NS(model=full,inference_module=language,encode=lambda p:(torch.zeros(80,16),torch.zeros(80,4)))
  model=runtime.RuntimeModel.__new__(runtime.RuntimeModel);model.pipeline=NS(text_encoder=encoder);model._prompt_cache=OrderedDict();model._prompt_cache_bytes=0;model._stage=lambda m,d:calls.append(('stage',m));model._offload=lambda m:calls.append(('offload',m))
  model._encode_prompt('test',None);self.assertEqual(calls,[('stage',language),('offload',language)]);model._encode_prompt('test',None);self.assertEqual(len(calls),2)
 def test_encoder_requests_hidden_states_without_calling_logits_wrapper(self):
  calls=[]
  class Base:
   language_model=object()
   def __call__(self,**kwargs):calls.append(kwargs);return NS(hidden_states=[torch.zeros(1,80,2)]*33)
  class Full:
   model=Base()
   def __call__(self,**kwargs):raise AssertionError('Logits wrapper must not be called')
  class Tokenizer:
   def __call__(self,*a,**kw):return {'input_ids':torch.ones(1,80,dtype=torch.long),'attention_mask':torch.ones(1,80,dtype=torch.long)}
  encoder=TextEncoder.__new__(TextEncoder);encoder.model=Full();encoder.device='cpu';encoder.processor=NS(tokenizer=Tokenizer(),apply_chat_template=lambda *a,**kw:'test')
  values,ids=encoder.encode('test');self.assertEqual(values.shape,(80,16));self.assertEqual(ids.shape,(80,4));self.assertTrue(calls[0]['output_hidden_states']);self.assertFalse(calls[0]['use_cache'])
