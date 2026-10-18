import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from pathlib import Path
from flux3_forge import assets

class FormatGuards(unittest.TestCase):
 def test_quantized_te_config_rejected_without_tensor_loading(self):
  with self.assertRaisesRegex(ValueError,'quantized text-encoder configurations'):
   assets._validate_text_storage([],{'quantization_config':{'quant_method':'bitsandbytes','load_in_4bit':True}})
 def test_te_int8_scaled_storage_rejected(self):
  reader=NS(__unused__=None)
  class Header:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def keys(self):return ['model.layer.weight','model.layer.weight_scale']
  with patch.object(assets,'_inspect_safetensors',return_value=Header()):
   with self.assertRaisesRegex(ValueError,'quantized text-encoder tensors'):assets._validate_text_storage([Path('model.safetensors')],{})
 def test_integer_weight_cannot_masquerade_as_fp8(self):
  reader=NS(keys=lambda:['layer.weight'],get_slice=lambda key:NS(get_dtype=lambda:'I8'))
  with self.assertRaisesRegex(ValueError,'unsupported storage dtype I8'):assets._validate_storage_dtypes(reader,'DiT fp8',{'BF16','F8_E4M3'})
 def test_unscaled_fp8_cannot_masquerade_as_bf16_or_te(self):
  reader=NS(keys=lambda:['layer.weight'],get_slice=lambda key:NS(get_dtype=lambda:'F8_E4M3'))
  with self.assertRaisesRegex(ValueError,'unsupported storage dtype'):assets._validate_storage_dtypes(reader,'ordinary weights',{'BF16','F16','F32'})
