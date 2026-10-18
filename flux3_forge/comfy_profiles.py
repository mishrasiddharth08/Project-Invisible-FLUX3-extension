"""Offline, pinned Comfy profiles. Experimental receipts are not GPU acceptance.

No download, conversion, installation or live Forge registration occurs here.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from contextlib import ExitStack
from types import MappingProxyType
import torch
from safetensors import safe_open
from .comfy_int8 import configuration

REVISION = 'feef60b0aaf63a4773ace96636ebbcf43b881312'
PINNED = {
    ('dit', 'bf16'): (11097404512, 'c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c'),
    ('dit', 'int8_convrot'): (5827874779, '847d8933548afd93df4af2cac2498e70891bf28d9dae06d695f536992e168fe2'),
    ('encoder', 'bf16'): (8875719408, 'f0b24fc5c8d7c0ea6ca909da14c20af5cac53f22506be942c6c956f838b9630f'),
    ('encoder', 'int8_convrot'): (4858559240, '32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c'),
    ('vae', 'fp32'): (336211292, '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3'),
    ('encoder_shard_1', 'bf16'): (4967229296, '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9'),
    ('encoder_shard_2', 'bf16'): (3908490048, '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6'),
}
TEXT_CONTRACT = dict(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                     num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                     vocab_size=151936, rope_theta=5000000)
_SEAL = object()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''): digest.update(chunk)
    return digest.hexdigest()

def _identity(path):
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)

@dataclass(frozen=True)
class ValidatedProfile:
    manifest: object
    identities: tuple
    _seal: object = field(repr=False)

def require_profile(profile):
    if type(profile) is not ValidatedProfile or profile._seal is not _SEAL:
        raise ValueError('Use validate_profile; a caller-supplied profile_validated flag is not validation')
    for path, size, mtime in profile.identities:
        try: identity = _identity(Path(path))
        except OSError as error: raise ValueError(f'Validated profile asset is missing: {path}') from error
        if identity != (path, size, mtime):
            raise ValueError(f'Validated profile changed; revalidate before loading: {path}')
    return profile.manifest

def _canonical(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path.suffix != '.safetensors' or not path.is_relative_to(parent):
        raise ValueError(f'Asset must remain inside canonical {parent}: {path}')
    if not path.is_file(): raise ValueError(f'Missing local profile asset: {path}; no download fallback')
    return path

def _pinned(path, role, precision):
    expected_size, expected_sha = PINNED[(role, precision)]
    if path.stat().st_size != expected_size: raise ValueError(f'{role} size differs from pinned {REVISION}')
    if sha256(path) != expected_sha: raise ValueError(f'{role} SHA256 differs from pinned {REVISION}')

def _config(support):
    config = json.loads((support/'config.json').read_text(encoding='utf-8'))
    text = config.get('text_config', {})
    if config.get('model_type') != 'qwen3_vl' or any(text.get(k) != v for k, v in TEXT_CONTRACT.items()):
        raise ValueError('Exact matching Qwen3-VL-4B configuration is required')
    rope = text.get('rope_scaling', {})
    if rope.get('mrope_section') != [24,20,20] or rope.get('mrope_interleaved') is not True or rope.get('rope_type') != 'default':
        raise ValueError('Qwen3-VL rotary configuration differs from the pinned encoder')
    if config.get('quantization_config') or text.get('quantization_config') or config.get('auto_map'):
        raise ValueError('Foreign quantizer/remote code configuration is not accepted')
    return config

def _schemas(config, model_family='distilled'):
    from .vendor.flux3.dit import DiT
    from .vendor.flux3.autoencoder import AutoEncoder
    from transformers import Qwen3VLConfig, Qwen3VLModel
    with torch.device('meta'):
        dit, encoder, vae = DiT(model_family=='distilled'), Qwen3VLModel(Qwen3VLConfig(**config)), AutoEncoder()
    return {'dit': dit, 'encoder': encoder, 'vae': vae}

def _normal(key, role):
    # These are explicit known container names, not arbitrary key guessing.
    return key.removeprefix('model.') if role == 'encoder' else key

def inspect_contract(path, module, role, precision):
    expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    layers = dict(module.named_modules()); packed = 0
    paths = path if isinstance(path, (list,tuple)) else [path]
    with ExitStack() as stack:
        sources = {}; readers = [stack.enter_context(safe_open(str(p), framework='pt', device='cpu')) for p in paths]
        for item in readers:
            for key in item.keys():
                if key in sources: raise ValueError('Duplicate tensor across encoder shards')
                sources[key] = item
        class Reader:
            def keys(self): return list(sources)
            def get_slice(self,key): return sources[key].get_slice(key)
            def get_tensor(self,key): return sources[key].get_tensor(key)
        reader = Reader()
        names = {_normal(k, role): k for k in reader.keys()}
        if len(names) != len(reader.keys()): raise ValueError('Ambiguous duplicate normalized tensor keys')
        if role == 'encoder' and 'lm_head.weight' in names:
            head = reader.get_slice(names.pop('lm_head.weight'))
            if tuple(head.get_shape()) != (151936,2560) or head.get_dtype() != 'BF16':
                raise ValueError('Unsupported optional Qwen tied LM-head layout')
        consumed = set()
        for name, shape in expected.items():
            if name not in names: raise ValueError(f'{role} missing architecture tensor: {name}')
            value = reader.get_slice(names[name]); dtype = value.get_dtype()
            if tuple(value.get_shape()) != shape: raise ValueError(f'{role} shape mismatch: {name}')
            consumed.add(name)
            stem = name.removesuffix('.weight'); qkey = stem+'.comfy_quant'; scale = stem+'.weight_scale'
            if name.endswith('.weight') and qkey in names:
                if precision != 'int8_convrot' or not isinstance(layers.get(stem), (torch.nn.Linear, torch.nn.Embedding)):
                    raise ValueError(f'Packed tensor not allowed for {role}/{precision}: {name}')
                descriptor = reader.get_slice(names[qkey]); scale_slice = reader.get_slice(names.get(scale, names[name]))
                if descriptor.get_dtype() != 'U8' or len(descriptor.get_shape()) != 1 or descriptor.get_shape()[0] > 65536:
                    raise ValueError('Invalid bounded comfy_quant descriptor')
                conf = configuration(reader.get_tensor(names[qkey]))
                if dtype != 'I8' or len(shape) != 2 or scale not in names:
                    raise ValueError(f'Packed weight/scale contract mismatch: {name}')
                scale_shape = tuple(scale_slice.get_shape())
                if scale_slice.get_dtype() != 'F32' or scale_shape != (shape[0],1) or shape[0]*4 > 16*1024**2:
                    raise ValueError(f'Expected bounded FP32 per-row scale: {name}')
                values = reader.get_tensor(names[scale])
                if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
                    raise ValueError(f'Invalid scale values: {name}')
                if not conf['convrot'] or shape[1] % conf['group_size']:
                    raise ValueError(f'Profile requires compatible ConvRot layers: {name}')
                consumed.update([qkey, scale]); packed += 1
            else:
                allowed = {'F32','I64'} if role == 'vae' else {'BF16'}
                if dtype not in allowed: raise ValueError(f'{role}/{precision} unsupported plain storage {dtype}: {name}')
        extras = set(names)-consumed
        for key in list(extras):
            if key.endswith('.comfy_attention.config'):
                item = reader.get_slice(names[key])
                if item.get_dtype() != 'U8' or len(item.get_shape()) != 1 or item.get_shape()[0] > 4096:
                    raise ValueError('Invalid bounded attention metadata')
                data = json.loads(bytes(reader.get_tensor(names[key]).tolist()).decode())
                if data != {'attention':'comfy_kitchen_int8'}: raise ValueError('Unknown attention metadata')
                extras.remove(key)
        if extras: raise ValueError(f'{role} unexpected tensors: {sorted(extras)[:4]}')
        if precision == 'int8_convrot' and not packed: raise ValueError('INT8 profile has no ConvRot layers')
    return packed

def validate_profile(models_root, *, checkpoint, encoder, vae, support_dir,
                     quantization, encoder_quantization, experimental=False, model_family='distilled'):
    if experimental is not True: raise ValueError('Comfy profiles remain experimental and disabled by default')
    if model_family not in ('base','distilled'):raise ValueError('Use an explicit verified Base or distilled family')
    if quantization not in ('bf16','int8_convrot') or encoder_quantization not in ('bf16','int8_convrot'):
        raise ValueError('Only explicit BF16/INT8 ConvRot profiles are accepted')
    root = Path(models_root).resolve(); support = Path(support_dir).resolve()
    encoder_paths = encoder if isinstance(encoder,(list,tuple)) else [encoder]
    if len(encoder_paths) not in (1,2) or (len(encoder_paths)==2 and encoder_quantization!='bf16'):
        raise ValueError('Use one pinned encoder file or the exact two original BF16 shards')
    encoder_paths = [_canonical(p, root/'text_encoder/FLUX-3') for p in encoder_paths]
    paths = {'dit':_canonical(checkpoint, root/'Stable-diffusion/FLUX-3'),
             'encoder':encoder_paths,
             'vae':_canonical(vae, root/'VAE/FLUX-3')}
    # Pins are independent of a caller's manifest. Fail on missing weights before meta-model construction.
    if model_family=='base':
        from .exact_workflow_semantics import verify_lfs_payload
        size,pointer={'bf16':(11076957072,'d14d35da5f5cc4f47127a504cc5dfcb53a8a774d'),'int8_convrot':(5807427339,'450f47e2f4811aaa396fe646ba76741f1c2f31b8')}[quantization]
        verify_lfs_payload(paths['dit'],dict(size_bytes=size,publisher_pointer_git_oid=pointer))
        _pinned(paths['vae'],'vae','fp32')
    else:
        for role, precision in [('dit',quantization),('vae','fp32')]: _pinned(paths[role], role, precision)
    for i,path in enumerate(encoder_paths):
        _pinned(path, 'encoder' if len(encoder_paths)==1 else f'encoder_shard_{i+1}', encoder_quantization)
    catalog = json.loads((Path(__file__).parent/'comfy_support_pins.json').read_text(encoding='utf-8'))
    files = [paths['dit'],paths['vae'],*encoder_paths]
    for name, digest in catalog['files'].items():
        item = (support/name).resolve()
        if not item.is_relative_to(support) or not item.is_file() or sha256(item) != digest:
            raise ValueError(f'Pinned offline tokenizer/config support differs or is missing: {name}')
        files.append(item)
    config = _config(support); schemas = _schemas(config,model_family) if model_family=='base' else _schemas(config)
    for role, precision in [('dit',quantization),('encoder',encoder_quantization),('vae','fp32')]:
        inspect_contract(paths[role], schemas[role], role, precision)
    manifest = dict(checkpoint_path=str(paths['dit']), text_encoder_weights=tuple(str(p) for p in encoder_paths),
                    text_encoder_path=str(support), autoencoder_path=str(paths['vae']),
                    quantization=quantization, encoder_quantization=encoder_quantization,
                    experimental=True, source_revision=REVISION, real_weight_acceptance=False,
                    model_family=model_family,
                    encoder_source='original BF16 shards337337621b8526e120f2275d250242f4a6bedd75' if len(encoder_paths)==2 else 'pinned Comfy single file')
    return ValidatedProfile(MappingProxyType(manifest), tuple(_identity(p) for p in files), _SEAL)
