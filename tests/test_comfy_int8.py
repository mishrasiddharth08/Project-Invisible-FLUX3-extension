import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,unittest,sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from comfy_kitchen.tensor import TensorWiseINT8Layout
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux3_forge import comfy_int8 as loader

H=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float32)/2
def descriptor(**kwargs):return torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':4,**kwargs}).encode()),dtype=torch.uint8)
def unpack(q,scale):return ((q.float()*scale).reshape(q.shape[0],-1,4)@H).reshape(q.shape)
def linear_reference(x,q,scale,bias):
    rotated=(x.reshape(*x.shape[:-1],-1,4)@H).reshape(x.shape)
    xscale=(rotated.abs().amax(-1,keepdim=True)/127).clamp(min=1e-30)
    qx=(rotated/xscale).round().clamp(-127,127).to(torch.int32)
    return (qx@q.to(torch.int32).T).float()*xscale*scale.T+bias

class ConvRotLoaderTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(11);model=nn.Sequential(nn.Linear(16,4),nn.LayerNorm(4));state=dict(model.state_dict())
        q,params=TensorWiseINT8Layout.quantize(state['0.weight'],per_channel=True,convrot=True,convrot_groupsize=4)
        state.update({'0.weight':q,'0.weight_scale':params.scale,'0.comfy_quant':descriptor()})
        return state,q.clone(),params.scale.clone()
    def fresh(self):
        with torch.device('meta'):return nn.Sequential(nn.Linear(16,4),nn.LayerNorm(4))
    def test_cpu_linear_matches_independent_rotated_integer_reference(self):
        state,q,scale=self.fixture();x=torch.randn(2,3,16)
        expected=F.layer_norm(linear_reference(x,q,scale,state['0.bias']),(4,),state['1.weight'],state['1.bias'])
        model=loader.load_state(self.fresh(),state,compute_dtype=torch.float32)
        torch.testing.assert_close(model(x),expected,atol=1e-5,rtol=1e-5)
        self.assertEqual(model[0].qdata.dtype,torch.int8);self.assertTrue(torch.equal(model[0].qdata,q))
    def test_full_precision_compute_override_preserves_integer_storage(self):
        state,q,scale=self.fixture();state['0.comfy_quant']=descriptor(full_precision_matrix_mult=True)
        model=loader.load_state(self.fresh(),state,compute_dtype=torch.float32)
        self.assertTrue(model[0].conf['full_precision_matrix_mult']);self.assertTrue(torch.equal(model[0].qdata,q))
        x=torch.randn(2,16)
        expected=F.layer_norm(F.linear(x,unpack(q,scale),state['0.bias']),(4,),state['1.weight'],state['1.bias'])
        torch.testing.assert_close(model(x),expected,atol=1e-5,rtol=1e-5)
    def test_bf16_compute_does_not_convert_integer_storage_or_scale_dtype(self):
        state,q,scale=self.fixture();model=loader.load_state(self.fresh(),state)
        self.assertEqual(model[0].weight.dtype,torch.bfloat16);self.assertEqual(model[0].qdata.dtype,torch.int8)
        original_scale=model[0].scale.clone()
        model.to(dtype=torch.bfloat16);self.assertEqual(model[0].scale.dtype,torch.float32)
        self.assertTrue(torch.equal(model[0].scale,original_scale))
    def test_embedding_gather_matches_independent_reference(self):
        torch.manual_seed(9);weight=torch.randn(12,16);q,p=TensorWiseINT8Layout.quantize(weight,per_channel=True,convrot=True,convrot_groupsize=4)
        with torch.device('meta'):model=nn.Sequential(nn.Embedding(12,16))
        model=loader.load_state(model,{'0.weight':q,'0.weight_scale':p.scale,'0.comfy_quant':descriptor()},compute_dtype=torch.float32)
        ids=torch.tensor([[0,5,5,11]]);torch.testing.assert_close(model(ids),F.embedding(ids,unpack(q,p.scale)),atol=1e-5,rtol=1e-5)
    def test_safetensors_file_roundtrip_load(self):
        state,q,scale=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic-int8-convrot.safetensors';save_file(state,str(path))
            model=loader.load_checkpoint(self.fresh(),path,compute_dtype=torch.float32)
            self.assertTrue(torch.equal(model[0].qdata,q));self.assertFalse(any(p.is_meta for p in model.parameters()))
    def test_bad_scale_fails_before_model_replacement(self):
        for invalid in [torch.tensor(float('nan')),torch.tensor(-1.),torch.ones(2,2)]:
            state,_,_=self.fixture();state['0.weight_scale']=invalid;model=self.fresh()
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):loader.load_state(model,state)
                self.assertIsInstance(model[0],nn.Linear)
    def test_foreign_quantization_or_unknown_layer_fails_closed(self):
        state,_,_=self.fixture();state['0.comfy_quant']=descriptor(format='nvfp4')
        with self.assertRaisesRegex(ValueError,'no format substitution'):loader.load_state(self.fresh(),state)
        state,_,_=self.fixture();state['foreign.comfy_quant']=state.pop('0.comfy_quant')
        with self.assertRaisesRegex(ValueError,'target architecture'):loader.load_state(self.fresh(),state)
    def test_unannotated_integer_weight_and_bad_group_rejected(self):
        state,_,_=self.fixture();state['1.weight']=torch.ones(4,dtype=torch.int8)
        with self.assertRaisesRegex(ValueError,'Unannotated integer'):loader.load_state(self.fresh(),state)
        state,_,_=self.fixture();state['0.comfy_quant']=descriptor(convrot_groupsize=8)
        with self.assertRaisesRegex(ValueError,'power of four'):loader.load_state(self.fresh(),state)
    def test_no_quantized_profile_not_mislabeled_int8(self):
        with self.assertRaisesRegex(ValueError,'no validated ConvRot'):loader.load_state(self.fresh(),nn.Sequential(nn.Linear(16,4),nn.LayerNorm(4)).state_dict())

if __name__=='__main__':unittest.main()
