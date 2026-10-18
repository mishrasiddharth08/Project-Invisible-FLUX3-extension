import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from flux3_forge.comfy_pipeline import build_pipeline

class Pipeline:pass
class DiT(nn.Linear):
    def __init__(self,flow_map):super().__init__(4,4);self.flow_map=flow_map
class TextEncoder:
    def __init__(self,path,device):self.path=path;self.device=device
class Qwen(nn.Module):
    def __init__(self):
        super().__init__();self.language_model=nn.Linear(4,4)

class ProfileRoutingTests(unittest.TestCase):
    def fixture(self,checkpoint='int8_convrot',encoder='int8_convrot'):
        manifest={'quantization':checkpoint,'encoder_quantization':encoder,'profile_validated':True,'checkpoint_path':'dit.safetensors','text_encoder_path':'support','autoencoder_path':'vae.safetensors'}
        manifest['text_encoder_weights']='encoder.safetensors'
        factories={'pipeline':Pipeline,'dit':DiT,'vae':lambda:object(),'flow_map':Mock(return_value=True),
            'packed_load':Mock(side_effect=lambda module,*a,**k:module),'dense_load':Mock(side_effect=lambda module,*a,**k:module),
            'text_encoder':TextEncoder,'config':S(from_pretrained=Mock(return_value=S(model_type='qwen3_vl',text_config=S(hidden_size=2560,num_hidden_layers=36)))),
            'processor':S(from_pretrained=Mock(return_value=S(tokenizer=S(padding_side='left')))),
            'qwen_base':lambda config:Qwen(),'encoder_load':Mock(side_effect=lambda model,*a:model.to_empty(device='cpu'))}
        return manifest,factories
    def test_all_four_explicit_stored_profile_combinations_route_separately(self):
        for checkpoint in ['bf16','int8_convrot']:
            for encoder in ['bf16','int8_convrot']:
                with self.subTest(checkpoint=checkpoint,encoder=encoder):
                    manifest,f=self.fixture(checkpoint,encoder)
                    with patch('flux3_forge.comfy_pipeline.require_profile',return_value=manifest):
                        pipe=build_pipeline(manifest,device='cpu',factories=f)
                    self.assertTrue(pipe.dits['distilled'].flow_map)
                    self.assertEqual(f['packed_load'].call_count,checkpoint=='int8_convrot')
                    self.assertEqual(f['encoder_load'].call_count,1)
                    if encoder in ('bf16','int8_convrot'):
                        f['config'].from_pretrained.assert_called_once_with('support',local_files_only=True)
                        f['processor'].from_pretrained.assert_called_once_with('support',local_files_only=True)
                        self.assertEqual(pipe.text_encoder.processor.tokenizer.padding_side,'right')
                    self.assertEqual(f['dense_load'].call_args.args[-1],torch.float32)
    def test_unvalidated_foreign_precision_or_missing_component_stops_before_load(self):
        for update in [{'profile_validated':False},{'quantization':'nvfp4'},{'text_encoder_weights':''}]:
            manifest,f=self.fixture();manifest.update(update)
            with self.assertRaises(ValueError):build_pipeline(manifest,factories=f)
            f['packed_load'].assert_not_called();f['dense_load'].assert_not_called()
    def test_missing_target_time_flow_map_rejected(self):
        manifest,f=self.fixture();f['flow_map'].return_value=False
        with patch('flux3_forge.comfy_pipeline.require_profile',return_value=manifest):
            with self.assertRaisesRegex(ValueError,'target-time'):build_pipeline(manifest,factories=f)
        f['packed_load'].assert_not_called()

class QwenSelectedTapParityTests(unittest.TestCase):
    def test_original_causal_wrapper_and_base_match_all_eight_used_taps(self):
        from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
        config=Qwen3VLConfig(text_config={'vocab_size':64,'hidden_size':32,'intermediate_size':64,'num_hidden_layers':36,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,'rope_scaling':{'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}},vision_config={'hidden_size':32,'intermediate_size':64,'depth':1,'num_heads':4,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':2,'out_hidden_size':32,'num_position_embeddings':64,'deepstack_visual_indexes':[]},image_token_id=61,video_token_id=62)
        torch.manual_seed(12);model=Qwen3VLForConditionalGeneration(config).eval();ids=torch.tensor([[1,2,3,4]])
        kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(4).reshape(1,1,4).expand(4,1,4),output_hidden_states=True,use_cache=False)
        with torch.no_grad():full=model(**kwargs);base=model.model(**kwargs)
        for layer in [4,8,12,16,20,24,28,32]:torch.testing.assert_close(full.hidden_states[layer],base.hidden_states[layer],atol=0,rtol=0)
        # Final-layer capture normalization differs; FLUX does not request it.
        self.assertGreater((full.hidden_states[36]-base.hidden_states[36]).abs().max().item(),0)


class DerivedRotaryBufferTests(unittest.TestCase):
    def test_meta_rotary_matches_original_without_materializing_weights(self):
        from transformers import Qwen3VLConfig,Qwen3VLModel
        from flux3_forge.comfy_pipeline import materialize_qwen_rotary
        config=Qwen3VLConfig(text_config={'vocab_size':64,'hidden_size':32,'intermediate_size':64,'num_hidden_layers':1,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,'rope_scaling':{'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}},vision_config={'hidden_size':32,'intermediate_size':64,'depth':1,'num_heads':4,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':2,'out_hidden_size':32,'num_position_embeddings':64,'deepstack_visual_indexes':[]})
        original=Qwen3VLModel(config)
        with torch.device('meta'): model=Qwen3VLModel(config)
        materialize_qwen_rotary(model,config)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertFalse(any(b.is_meta for b in model.buffers()))
        for name,value in original.named_buffers():torch.testing.assert_close(dict(model.named_buffers())[name],value,rtol=0,atol=0)
        self.assertIs(model.language_model.rotary_emb.original_inv_freq,model.language_model.rotary_emb.inv_freq)

if __name__=='__main__':unittest.main()
