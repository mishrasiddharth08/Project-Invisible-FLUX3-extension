import json,types,unittest
import torch
from torch import nn
from flux3_forge import comfy_checkpoint_layout as layout,comfy_int8
class Split(nn.Module):
 def __init__(self,bias=False):
  super().__init__();self.q_proj=nn.Linear(16,2,bias=bias);self.k_proj=nn.Linear(16,2,bias=bias);self.v_proj=nn.Linear(16,2,bias=bias)
def data():
 desc=torch.tensor(list(json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=4)).encode()),dtype=torch.uint8)
 return {'qkv.weight':torch.arange(96,dtype=torch.int8).reshape(6,16),'qkv.weight_scale':torch.ones(6,1),'qkv.comfy_quant':desc}
class Layout(unittest.TestCase):
 def test_fused_rows_preserve_storage(self):
  original=data();out=layout.expand_state(original,Split())
  for i,key in enumerate(['q','k','v']):
   value=out[key+'_proj.weight'];self.assertTrue(torch.equal(value,original['qkv.weight'][i*2:(i+1)*2]));self.assertEqual(value.untyped_storage().data_ptr(),original['qkv.weight'].untyped_storage().data_ptr());self.assertEqual(out[key+'_proj.comfy_quant'].data_ptr(),original['qkv.comfy_quant'].data_ptr())
 def test_reader_shapes_and_dtype(self):
  values=data()
  class Reader:
   def keys(self):return values.keys()
   def get_slice(self,key):return types.SimpleNamespace(get_shape=lambda:tuple(values[key].shape),get_dtype=lambda:str(values[key].dtype))
   def get_tensor(self,key):return values[key]
  view=layout.ExpandedReader(Reader(),Split());self.assertEqual(view.get_slice('v_proj.weight').get_shape(),(2,16));self.assertEqual(view.get_slice('q_proj.comfy_quant').get_shape(),(values['qkv.comfy_quant'].numel(),));self.assertTrue(torch.equal(view.get_tensor('k_proj.weight'),values['qkv.weight'][2:4]))
 def test_bad_fused_shape_rejected(self):
  values=data();values['qkv.weight']=torch.zeros(5,16,dtype=torch.int8)
  with self.assertRaises(ValueError):layout.expand_state(values,Split())
 def test_conflicting_separate_keys_rejected(self):
  values=data();values['q_proj.weight']=torch.zeros(2,16)
  with self.assertRaises(ValueError):layout.expand_state(values,Split())
 def test_bias_architecture_rejected(self):
  with self.assertRaises(ValueError):layout.expand_state(data(),Split(True))
 def test_native_fused_encoder_layer_is_preserved(self):
  model=nn.Module();model.qkv=nn.Linear(16,6,bias=False);values=data();self.assertEqual(set(layout.expand_state(values,model)),set(values))
 def test_packed_load_handles_fusion_and_known_attention_metadata(self):
  values=data();values['block.comfy_attention.config']=torch.tensor(list(b'{"attention":"comfy_kitchen_int8"}'),dtype=torch.uint8)
  model=comfy_int8.load_state(Split(),values,device='cpu',compute_dtype=torch.bfloat16)
  self.assertTrue(all(isinstance(getattr(model,k+'_proj'),comfy_int8.Int8Linear) for k in ['q','k','v']));self.assertTrue(torch.equal(model.v_proj.qdata,values['qkv.weight'][4:]));self.assertEqual(model.v_proj.scale.dtype,torch.float32)
 def test_unknown_attention_metadata_rejected(self):
  with self.assertRaises(ValueError):layout.attention_metadata({'a.comfy_attention.config':torch.tensor(list(b'{"attention":"unknown"}'),dtype=torch.uint8)})
if __name__=='__main__':unittest.main(verbosity=2)
