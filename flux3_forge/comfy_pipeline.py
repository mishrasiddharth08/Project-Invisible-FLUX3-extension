"""Experimental complete-profile builder. Not enabled in the live bridge.

The official DiT, target-time sampler and encoder tap selection are preserved.
Requires an independently validated manifest; no download/install fallback.
"""
from __future__ import annotations
import torch
from .comfy_int8 import load_checkpoint as load_packed
from .comfy_profiles import require_profile

ALLOWED={'bf16','int8_convrot'}

def wrap_encoder(model, processor, device, encoder_class):
    encoder=encoder_class.__new__(encoder_class)
    wrapper=torch.nn.Module();wrapper.model=model
    encoder.model=wrapper;encoder.device=device;encoder.processor=processor
    processor.tokenizer.padding_side='right'
    model.language_model.to(device)
    return encoder

def _factories():
    from .vendor.flux3.pipeline import Pipeline
    from .vendor.flux3.dit import DiT
    from .vendor.flux3.autoencoder import AutoEncoder
    from .vendor.flux3.checkpoint import load_checkpoint,is_flow_map
    from .vendor.flux3.text_encoder import TextEncoder
    from transformers import AutoConfig,AutoProcessor,Qwen3VLModel
    return dict(pipeline=Pipeline,dit=DiT,vae=AutoEncoder,dense_load=load_checkpoint,
                flow_map=is_flow_map,text_encoder=TextEncoder,config=AutoConfig,
                processor=AutoProcessor,qwen_base=Qwen3VLModel,packed_load=load_packed)

def build_pipeline(profile,*,device='cpu',offload=True,factories=None):
    """Build one original distilled pipeline using explicit checkpoint/TE profiles.

    BF16 and INT8 ConvRot are storage labels, not aliases for each other.
    The caller must validate hashes/header contract and canonical model paths.
    """
    manifest=require_profile(profile)
    precision=manifest['quantization'];encoder_precision=manifest['encoder_quantization']
    if precision not in ALLOWED or encoder_precision not in ALLOWED:
        raise ValueError('Explicit BF16/INT8 ConvRot checkpoint and encoder profiles are required')
    required=['checkpoint_path','text_encoder_path','autoencoder_path','text_encoder_weights']
    if any(not manifest.get(key) for key in required):raise ValueError('Complete checkpoint, encoder/support and VAE paths are required')
    f=factories or _factories()
    family=manifest.get('model_family','distilled')
    if family not in ('base','distilled'):raise ValueError('Unknown FLUX3 model family')
    if bool(f['flow_map'](manifest['checkpoint_path'])) != (family=='distilled'):
        raise ValueError('Checkpoint target-time flow-map keys do not match the explicit Base/distilled family')
    pipeline=f['pipeline'].__new__(f['pipeline']);pipeline.device=device;pipeline.offload=offload
    with torch.device('meta'):dit=f['dit'](flow_map=family=='distilled')
    target='cpu' if offload else device
    loader=f['packed_load'] if precision=='int8_convrot' else f['dense_load']
    if precision=='int8_convrot':dit=loader(dit,manifest['checkpoint_path'],device=target,compute_dtype=torch.bfloat16)
    else:dit=loader(dit,manifest['checkpoint_path'],target,torch.bfloat16)
    pipeline.dits={family:dit}
    if encoder_precision=='bf16' and not manifest.get('text_encoder_weights'):
        # Existing verified original shards: original inference wrapper, offline.
        pipeline.text_encoder=f['text_encoder'](manifest['text_encoder_path'],device)
    else:
        config=f['config'].from_pretrained(manifest['text_encoder_path'],local_files_only=True)
        text=getattr(config,'text_config',None)
        if (getattr(config,'model_type',None)!='qwen3_vl' or text is None
                or text.hidden_size!=2560 or text.num_hidden_layers!=36):
            raise ValueError('Matching Qwen3-VL-4B (2560 hidden,36 layers) configuration is required')
        with torch.device('meta'):model=f['qwen_base'](config)
        # The published encoder has tied/no LM-head weights. FLUX reads taps
        #4,8,...32, all before the final layer36 normalization; base vs causal
        #wrapper tap parity is tested separately. No logits are required.
        encoder_loader=f.get('encoder_load')
        if encoder_loader is None:
            from safetensors.torch import load_file
            from .comfy_int8 import load_state
            paths=manifest['text_encoder_weights']
            paths=paths if isinstance(paths,(list,tuple)) else [paths]
            state={}
            for path in paths:
                shard=load_file(str(path),device='cpu')
                if set(state)&set(shard):raise ValueError('Duplicate encoder keys across shards')
                state.update(shard)
            state={key.removeprefix('model.'):value for key,value in state.items() if key!='lm_head.weight'}
            if encoder_precision=='int8_convrot':
                # Text computation dequantizes per layer; packed INT8 storage stays intact.
                model=load_state(model,state,device='cpu',compute_dtype=torch.bfloat16,full_precision_matrix_mult=True)
            else:
                if any(key.endswith('.comfy_quant') for key in state):raise ValueError('BF16 encoder profile points to packed weights')
                model.load_state_dict({key:value.to(dtype=torch.bfloat16) if value.is_floating_point() else value for key,value in state.items()},strict=True,assign=True)
                model=model.to('cpu').eval()
        else:model=encoder_loader(model,manifest,device)
        # Match the original conditional wrapper's .model base without an LM head.
        # This also preserves the updated TextEncoder.inference_module property.
        processor=f['processor'].from_pretrained(manifest['text_encoder_path'],local_files_only=True)
        encoder=wrap_encoder(model,processor,device,f['text_encoder'])
        pipeline.text_encoder=encoder
    pipeline.autoencoder=f['dense_load'](f['vae'](),manifest['autoencoder_path'],device,torch.float32)
    require_profile(profile)  # detect file changes during construction
    return pipeline
