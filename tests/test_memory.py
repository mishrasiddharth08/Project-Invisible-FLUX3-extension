import unittest
import torch
from torch import nn
from flux3_forge.memory import RowChunkedConv2d,bound_vae_convolutions,norm_swish_conv,upsample_conv


class MemoryTests(unittest.TestCase):
    @torch.inference_mode()
    def test_chunked_convolutions_match_padding_stride_dilation(self):
        for stride,padding,dilation,groups in ((1,1,1,1),(2,1,1,1),(1,2,2,1),(1,0,1,2)):
            torch.manual_seed(8)
            reference=nn.Conv2d(4,6,3,stride=stride,padding=padding,dilation=dilation,groups=groups)
            model=nn.Sequential(reference)
            x=torch.randn(1,4,37,29)
            expected=model(x)
            bound_vae_convolutions(model)
            layer=model[0]
            layer.threshold=0
            layer.chunk_rows=7
            self.assertIsInstance(layer,RowChunkedConv2d)
            self.assertIs(layer.weight,reference.weight)
            torch.testing.assert_close(model(x),expected,atol=1e-6,rtol=1e-5)

    @torch.inference_mode()
    def test_streamed_groupnorm_keeps_global_statistics_and_halos(self):
        torch.manual_seed(19)
        for stride,padding in ((1,1),(2,1),(1,0)):
            x=torch.randn(2,4,37,29)
            norm=nn.GroupNorm(2,4)
            conv=nn.Conv2d(4,6,3,stride=stride,padding=padding)
            value=norm(x)
            expected=conv(value*torch.sigmoid(value))
            actual=norm_swish_conv(norm,conv,x,threshold=0,chunk_rows=7)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

    @torch.inference_mode()
    def test_streamed_residual_projection_matches_original_block(self):
        from flux3_forge.vendor.flux3.autoencoder import ResnetBlock
        torch.manual_seed(39)
        block=ResnetBlock(32,64).eval()
        x=torch.randn(1,32,1025,4)
        swish=lambda value:value*torch.sigmoid(value)
        h=block.conv1(swish(block.norm1(x)))
        h=block.conv2(swish(block.norm2(h)))
        expected=block.nin_shortcut(x)+h
        torch.testing.assert_close(block(x),expected,atol=3e-6,rtol=3e-5)

    @torch.inference_mode()
    def test_streamed_inplace_convolution_preserves_original_halos(self):
        torch.manual_seed(49)
        x=torch.randn(2,4,37,29)
        norm=nn.GroupNorm(2,4)
        conv=nn.Conv2d(4,4,3,padding=1)
        value=norm(x)
        expected=conv(value*torch.sigmoid(value))
        pointer=x.data_ptr()
        actual=norm_swish_conv(norm,conv,x,threshold=0,chunk_rows=7,inplace=True)
        self.assertEqual(actual.data_ptr(),pointer)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

    @torch.inference_mode()
    def test_streamed_nearest_upsampling_preserves_phase_and_halos(self):
        torch.manual_seed(59)
        x=torch.randn(2,4,19,13)
        for stride,padding,dilation in ((1,1,1),(2,1,1),(1,2,2),(1,0,1)):
            conv=nn.Conv2d(4,6,3,stride=stride,padding=padding,dilation=dilation)
            expected=conv(torch.nn.functional.interpolate(x,scale_factor=2,mode="nearest"))
            actual=upsample_conv(conv,x,threshold=0,chunk_rows=7)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


if __name__=="__main__":
    unittest.main()
