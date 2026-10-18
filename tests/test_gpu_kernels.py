"""Small real GPU checks: these do not prove full FLUX3 generation."""
import unittest
import torch
from torch.nn import functional as F
from flux3_forge.adapters import AdapterLinear
from flux3_forge.vendor.flux3.fp8 import FP8Linear, quantize_rows


@unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
class GPUKernels(unittest.TestCase):
    @torch.inference_mode()
    def test_rowwise_fp8_and_lora_keep_packed_weights(self):
        torch.manual_seed(13)
        weight = torch.randn(256,256,device="cuda",dtype=torch.bfloat16)*.04
        packed,scale = quantize_rows(weight)
        layer = FP8Linear(256,256,device="cuda")
        layer.weight.copy_(packed)
        layer.weight_scale.copy_(scale)
        x = torch.randn(128,256,device="cuda",dtype=torch.bfloat16)
        baseline = layer(x)
        expected = F.linear(x,weight)
        relative = (baseline.float()-expected.float()).norm()/expected.float().norm()
        self.assertLess(float(relative),.08)
        down = torch.randn(4,256,dtype=torch.bfloat16)*.02
        up = torch.randn(256,4,dtype=torch.bfloat16)*.02
        wrapper = AdapterLinear(layer,{"lora_A.weight":down,"lora_B.weight":up},.7).to("cuda")
        actual = wrapper(x)
        branch = F.linear(F.linear(x,down.cuda()),up.cuda())*.7
        torch.testing.assert_close(actual,baseline+branch,rtol=.02,atol=.01)
        self.assertEqual(layer.weight.dtype,torch.float8_e4m3fn)

    @torch.inference_mode()
    def test_official_packed_attention_separates_samples(self):
        from flux3_forge.vendor.flux3.dit import packed_attention
        q,k,v = [torch.randn(1,2,16,128,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
        bounds = [(0,7),(7,16)]
        actual = packed_attention(q,k,v,bounds)
        expected = torch.cat([F.scaled_dot_product_attention(q[:,:,a:b],k[:,:,a:b],v[:,:,a:b])[0].transpose(0,1).reshape(b-a,256) for a,b in bounds])
        torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)

    @torch.inference_mode()
    def test_chunked_fp8_keeps_row_scales(self):
        torch.manual_seed(29)
        weight=torch.randn(256,256,device="cuda",dtype=torch.bfloat16)*.04
        packed,scale=quantize_rows(weight)
        layer=FP8Linear(256,256,device="cuda")
        layer.weight.copy_(packed)
        layer.weight_scale.copy_(scale)
        x=torch.randn(4353,256,device="cuda",dtype=torch.bfloat16)
        actual=layer(x)
        expected=F.linear(x,weight)
        relative=(actual.float()-expected.float()).norm()/expected.float().norm()
        self.assertLess(float(relative),.08)
        self.assertEqual(actual.shape,(4353,256))


if __name__ == "__main__":
    unittest.main()
