import pathlib
import tempfile
import unittest
import torch
from torch import nn
from safetensors.torch import save_file
from flux3_forge.adapters import AdapterLinear, attach, detach


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = nn.Linear(6,6,bias=False)
        self.x = torch.randn(2,3,6)

    def check_delta(self, layer, delta, scale=1.):
        torch.testing.assert_close(layer(self.x),self.base(self.x)+self.x @ delta.T*scale)
        self.assertIs(layer.base,self.base)

    def test_lora_alpha_and_strength(self):
        down,up = torch.randn(2,6),torch.randn(6,2)
        layer = AdapterLinear(self.base,{"lora_down.weight":down,"lora_up.weight":up,"alpha":torch.tensor(4.)},.5)
        self.check_delta(layer,up@down)

    def test_lokr_full(self):
        a,b = torch.randn(2,3),torch.randn(3,2)
        self.check_delta(AdapterLinear(self.base,{"lokr_w1":a,"lokr_w2":b},.25),torch.kron(a,b),.25)

    def test_lokr_factorized(self):
        a,b,c,d = torch.randn(2,1),torch.randn(1,3),torch.randn(3,2),torch.randn(2,2)
        layer = AdapterLinear(self.base,{"lokr_w1_a":a,"lokr_w1_b":b,"lokr_w2_a":c,"lokr_w2_b":d,"alpha":torch.tensor(2.)},1.)
        self.check_delta(layer,torch.kron(a@b,c@d))

    def test_loha_without_full_weight_materialization(self):
        a,b,c,d = torch.randn(6,2),torch.randn(2,6),torch.randn(6,3),torch.randn(3,6)
        layer = AdapterLinear(self.base,{"hada_w1_a":a,"hada_w1_b":b,"hada_w2_a":c,"hada_w2_b":d,"alpha":torch.tensor(2.)},.8)
        self.check_delta(layer,(a@b)*(c@d),.8)

    def test_atomic_file_validation_and_detach(self):
        model = nn.ModuleDict({"q_proj":self.base})
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as folder:
            path = str(pathlib.Path(folder)/"adapter.safetensors")
            save_file({"q_proj.lora_A.weight":torch.randn(2,6),"q_proj.lora_B.weight":torch.randn(6,2)},path)
            records = attach(model,[{"path":path,"strength":1.}])
            self.assertIsInstance(model["q_proj"],AdapterLinear)
            detach(records)
            self.assertIs(model["q_proj"],self.base)
            with self.assertRaises(ValueError):
                attach(model,[{"path":path},{"path":path+"missing"}])
            self.assertIs(model["q_proj"],self.base)

    def test_stacking(self):
        model = nn.ModuleDict({"q_proj":self.base})
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as folder:
            path = str(pathlib.Path(folder)/"adapter.safetensors")
            a,b = torch.randn(2,6),torch.randn(6,2)
            save_file({"q_proj.lora_A.weight":a,"q_proj.lora_B.weight":b},path)
            records = attach(model,[{"path":path,"strength":.25},{"path":path,"strength":.5}])
            torch.testing.assert_close(model["q_proj"](self.x),self.base(self.x)+self.x@(b@a).T*.75)
            detach(records)
            self.assertIs(model["q_proj"],self.base)

    def test_reject_foreign_shapes_and_nan(self):
        for factors in ({"lora_A.weight":torch.ones(2,4),"lora_B.weight":torch.ones(6,2)},
                        {"lokr_w1":torch.full((2,3),float("nan")),"lokr_w2":torch.ones(3,2)}):
            with self.assertRaises(ValueError):
                AdapterLinear(self.base,factors,1.)

    def test_reject_foreign_family_metadata(self):
        model = nn.ModuleDict({"q_proj":self.base})
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as folder:
            path = str(pathlib.Path(folder)/"foreign.safetensors")
            save_file({"q_proj.lora_A.weight":torch.randn(2,6),"q_proj.lora_B.weight":torch.randn(6,2)},path,metadata={"ss_base_model_version":"sdxl"})
            with self.assertRaisesRegex(ValueError,"incompatible model family"):
                attach(model,[{"path":path}])

    def test_reject_duplicate_target_aliases(self):
        model = nn.ModuleDict({"q_proj":self.base})
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as folder:
            path = str(pathlib.Path(folder)/"duplicate.safetensors")
            tensors = {f"{prefix}q_proj.{suffix}":value for prefix in ("","diffusion_model.") for suffix,value in (("lora_A.weight",torch.randn(2,6)),("lora_B.weight",torch.randn(6,2)))}
            save_file(tensors,path)
            with self.assertRaisesRegex(ValueError,"Duplicate adapter target"):
                attach(model,[{"path":path}])
            self.assertIs(model["q_proj"],self.base)


if __name__ == "__main__":
    unittest.main()
