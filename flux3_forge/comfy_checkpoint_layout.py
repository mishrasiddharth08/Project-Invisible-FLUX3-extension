"""Strict, storage-preserving mapping of published Comfy fused-QKV checkpoints."""
import json
def qkv_aliases(keys,module):
 keys=set(keys);aliases={};fused=set()
 for name in sorted(keys):
  if not name.endswith('qkv.weight'):continue
  stem=name[:-len('qkv.weight')]
  try:module.get_submodule(stem+'qkv');continue  # Native fused layer, e.g. Qwen vision.
  except AttributeError:pass
  try:layers=[module.get_submodule(stem+part+'_proj') for part in ['q','k','v']]
  except AttributeError as error:raise ValueError('Fused QKV has no matching explicit architecture: '+stem) from error
  shapes=[tuple(layer.weight.shape) for layer in layers]
  if len(set(shapes))!=1 or len(shapes[0])!=2 or any(getattr(layer,'bias',None) is not None for layer in layers):raise ValueError('Only equal, bias-free Q/K/V projections are supported')
  rows=shapes[0][0]
  for suffix in ['weight','weight_scale','comfy_quant']:
   original=stem+'qkv.'+suffix
   if original not in keys:continue
   fused.add(original)
   for index,part in enumerate(['q','k','v']):
    virtual=stem+part+'_proj.'+suffix
    if virtual in keys:raise ValueError('Both fused and separate QKV exist; no ambiguous precedence')
    aliases[virtual]=(original,index,rows,suffix)
 return aliases,fused
class ExpandedReader:
 def __init__(self,reader,module):self.reader=reader;self.aliases,self.hidden=qkv_aliases(reader.keys(),module)
 def keys(self):return [k for k in self.reader.keys() if k not in self.hidden]+list(self.aliases)
 def get_slice(self,key):
  if key not in self.aliases:return self.reader.get_slice(key)
  original,index,rows,suffix=self.aliases[key];value=self.reader.get_slice(original)
  shape=tuple(value.get_shape())
  if suffix!='comfy_quant' and (not shape or shape[0]!=rows*3):raise ValueError('Fused QKV row count differs from architecture')
  class Slice:
   def get_shape(self):return shape if suffix=='comfy_quant' else (rows,*shape[1:])
   def get_dtype(self):return value.get_dtype()
  return Slice()
 def get_tensor(self,key):
  if key not in self.aliases:return self.reader.get_tensor(key)
  original,index,rows,suffix=self.aliases[key];value=self.reader.get_tensor(original)
  if suffix=='comfy_quant':return value
  if value.shape[0]!=rows*3:raise ValueError('Fused QKV row count differs from architecture')
  return value.narrow(0,index*rows,rows)
def expand_state(state,module):
 aliases,hidden=qkv_aliases(state,module);out={k:v for k,v in state.items() if k not in hidden}
 for key,(original,index,rows,suffix) in aliases.items():
  value=state[original]
  if suffix!='comfy_quant' and value.shape[0]!=rows*3:raise ValueError('Fused QKV row count differs from architecture')
  out[key]=value if suffix=='comfy_quant' else value.narrow(0,index*rows,rows)
 return out
def attention_metadata(state):
 out={}
 for key,value in state.items():
  if key.endswith('.comfy_attention.config'):
   if str(value.dtype)!='torch.uint8' or value.ndim!=1 or value.numel()>4096:raise ValueError('Invalid bounded uint8 attention metadata')
   data=json.loads(bytes(value.cpu().tolist()).decode('utf-8'))
   if data!={'attention':'comfy_kitchen_int8'}:raise ValueError('Unknown attention metadata')
  else:out[key]=value
 return out
def load_dense_dit(module,path,device,dtype):
 from safetensors.torch import load_file
 state=attention_metadata(expand_state(load_file(str(path),device='cpu'),module))
 if any(k.endswith('.comfy_quant') for k in state):raise ValueError('Packed file cannot be loaded as dense BF16')
 module.load_state_dict({k:v.to(dtype=dtype) if v.is_floating_point() else v for k,v in state.items()},strict=True,assign=True)
 return module.to(device).eval()
