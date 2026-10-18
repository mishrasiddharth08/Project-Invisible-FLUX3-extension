import unittest
import torch
from torch.nn import functional as F
from flux3_forge.vendor.flux3.autoencoder import _spatial_attention, swish


class VAEAttentionTests(unittest.TestCase):
    def test_chunking_preserves_all_keys_and_output(self):
        torch.manual_seed(77)
        q,k,v=[torch.randn(1,1,31,32) for _ in range(3)]
        expected=F.scaled_dot_product_attention(q,k,v)
        actual=_spatial_attention(q,k,v,chunk_size=7,threshold=0)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_swish_reuses_temporary_without_changing_input(self):
        x=torch.linspace(-8,8,101)
        original=x.clone()
        torch.testing.assert_close(swish(x),x*torch.sigmoid(x),atol=0,rtol=0)
        self.assertTrue(torch.equal(x,original))


if __name__=="__main__":
    unittest.main()
