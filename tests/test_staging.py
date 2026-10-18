import unittest
import torch
from torch import nn
from flux3_forge.staging import capture,mark_staged,restore

class CPUBackingTests(unittest.TestCase):
 def test_restores_original_cpu_storage_and_registered_buffer(self):
  model=nn.Linear(8,4);model.register_buffer('counter',torch.ones(2));backing=capture(model);pointers={n:t.data_ptr() for n,t in model.state_dict().items()}
  # Simulate device allocation with fresh storage, preserving values.
  for p in model.parameters():p.data=p.detach().clone()
  model.counter=model.counter.clone();mark_staged(model,backing)
  self.assertTrue(restore(model,backing));self.assertEqual(pointers,{n:t.data_ptr() for n,t in model.state_dict().items()})
 def test_in_place_update_requests_copying_fallback(self):
  model=nn.Linear(2,2);backing=capture(model);mark_staged(model,backing)
  with torch.no_grad():model.weight.add_(1)
  self.assertFalse(restore(model,backing))
 def test_structure_change_requests_copying_fallback(self):
  model=nn.Linear(2,2);backing=capture(model);mark_staged(model,backing);model.register_buffer('new',torch.zeros(1));self.assertFalse(restore(model,backing))
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_real_gpu_roundtrip_reuses_cpu_backing_and_keeps_outputs(self):
  torch.manual_seed(7);model=nn.Linear(16,8).eval();model.register_buffer('test_buffer',torch.ones(2));backing=capture(model);pointer=model.weight.data_ptr();x=torch.randn(3,16,device='cuda');model.to('cuda');mark_staged(model,backing);expected=model(x).detach().cpu();self.assertTrue(restore(model,backing));self.assertEqual(model.weight.device.type,'cpu');self.assertEqual(model.weight.data_ptr(),pointer);model.to('cuda');actual=model(x).detach().cpu();self.assertTrue(torch.equal(expected,actual));model.cpu()
