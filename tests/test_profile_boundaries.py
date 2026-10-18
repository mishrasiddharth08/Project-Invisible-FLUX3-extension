"""CPU structural boundary checks; these do not establish real weight compatibility."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from flux3_forge import assets

class Reader:
    def __init__(self, shapes): self.shapes=shapes
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def keys(self): return self.shapes.keys()
    def get_slice(self,key): return SimpleNamespace(get_shape=lambda:self.shapes[key],get_dtype=lambda:"BF16")

class ProfileBoundaries(unittest.TestCase):
    def fixture(self,root,mode='bf16'):
        text=root/'text_encoder';text.mkdir()
        for name in ['model.safetensors','tokenizer.json','processor_config.json']:(text/name).write_bytes(b'x')
        (text/'config.json').write_text(json.dumps({'model_type':'qwen3_vl','architectures':['Qwen3VLForConditionalGeneration'],'text_config':{'hidden_size':2560,'num_hidden_layers':36}}))
        checkpoint=root/('model.safetensors' if mode=='bf16' else 'variants/fp8r/model.safetensors')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'x');(root/'ae.safetensors').write_bytes(b'x')
        return {'emb_in.image.weight':(3072,128),'emb_in.image_cond.weight':(3072,128),'txt_in.weight':(3072,20480),'single_blocks.27.q_proj.weight':(3072,3072),assets.FLOW_MAP_KEY:(3072,256)}
    def test_missing_flow_map_target_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shapes=self.fixture(root);del shapes[assets.FLOW_MAP_KEY]
            with patch.object(assets,'_inspect_safetensors',return_value=Reader(shapes)):
                with self.assertRaisesRegex(ValueError,'ref_time_in'): assets.validate_model_dir(root,'bf16')
    def test_bf16_profile_rejects_quantized_weight_scales(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shapes=self.fixture(root);shapes['layer.weight_scale']=(3072,)
            with patch.object(assets,'_inspect_safetensors',return_value=Reader(shapes)):
                with self.assertRaisesRegex(ValueError,'quantized checkpoint'): assets.validate_model_dir(root,'bf16')
    def test_fp8_profile_requires_real_scale_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shapes=self.fixture(root,'fp8')
            with patch.object(assets,'_inspect_safetensors',return_value=Reader(shapes)):
                with self.assertRaisesRegex(ValueError,'no rowwise weight scales'): assets.validate_model_dir(root,'fp8')
    def test_foreign_text_encoder_rejected_before_weight_inspection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);(root/'text_encoder/config.json').write_text(json.dumps({'model_type':'qwen3','text_config':{'hidden_size':2560,'num_hidden_layers':36}}))
            with patch.object(assets,'_inspect_safetensors') as inspect:
                with self.assertRaisesRegex(ValueError,'not Qwen3-VL'): assets.validate_model_dir(root,'bf16')
                inspect.assert_not_called()
    def test_shard_escape_or_absence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            text=Path(folder)
            for name in ['../outside.safetensors','missing.safetensors']:
                with self.subTest(name=name):
                    (text/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':name}}))
                    with self.assertRaisesRegex(ValueError,'missing or unsafe'):
                        assets._text_encoder_files(text,{'architectures':['Qwen3VLForConditionalGeneration']})

if __name__=='__main__':unittest.main()
