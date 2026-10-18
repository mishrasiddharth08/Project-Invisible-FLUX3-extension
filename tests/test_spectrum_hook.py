import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from flux3_forge.vendor.flux3 import dit


class SpectrumHookTests(unittest.TestCase):
    @torch.inference_mode()
    def test_forecast_skips_shared_blocks_but_recomputes_output_head(self):
        with patch.multiple(dit,HIDDEN=16,HEADS=2,DEPTH=1,DEPTH_SINGLE=2,
                            AXES_DIM=(2,2,2,2),IN_CHANNELS={"image":4,"image_cond":4,"text":8}):
            model=dit.DiT(flow_map=True).eval()
            counts={"shared":0,"head":0}
            def count_shared(*_):counts["shared"]+=1
            def count_head(*_):counts["head"]+=1
            for block in model.single_blocks:block.register_forward_hook(count_shared)
            model.final_layer["image"].register_forward_hook(count_head)
            packed=dit.Packed(data={"image":torch.randn(3,4),"image_cond":torch.zeros(0,4),"text":torch.randn(2,8)},
                ids={"image":torch.zeros(3,4),"image_cond":torch.zeros(0,4),"text":torch.zeros(2,4)},
                timesteps={"image":torch.ones(3),"image_cond":torch.zeros(0),"text":torch.zeros(2)},
                mode_indices={"image":[(0,3)],"image_cond":[(0,0)],"text":[(0,2)]})
            model(packed,timesteps_ref=torch.tensor([.5]))
            self.assertEqual(counts,{"shared":2,"head":1})
            controller=SimpleNamespace(active=True,should_compute=lambda *args:False,
                                       predict=lambda target:torch.zeros(5,16))
            model._forge_spectrum=controller
            model._forge_target_time=.4
            model._forge_step=4
            model._forge_total=16
            actual=model(packed,timesteps_ref=torch.tensor([.4]))
            self.assertEqual(counts,{"shared":2,"head":2})
            self.assertEqual(actual.shape,(3,4))
            self.assertTrue(torch.isfinite(actual).all())
