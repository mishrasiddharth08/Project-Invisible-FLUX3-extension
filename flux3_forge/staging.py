"""CPU backing for read-only inference weights; avoid repeated D2H copies."""
from dataclasses import dataclass
import torch

def version(tensor):
    try:return tensor._version
    except RuntimeError:return None

@dataclass
class Backing:
    parameters: dict
    buffers: dict
    versions: dict

def capture(module):
    return Backing({n:p.detach() for n,p in module.named_parameters(remove_duplicate=False)},
                   {n:b.detach() for n,b in module.named_buffers(remove_duplicate=False)}, {})

def mark_staged(module,backing):
    backing.versions={('p',n):version(p) for n,p in module.named_parameters(remove_duplicate=False)}
    backing.versions.update({('b',n):version(b) for n,b in module.named_buffers(remove_duplicate=False)})

def restore(module,backing):
    """Return False for changed structure/versions so caller preserves updates via normal offload."""
    params=dict(module.named_parameters(remove_duplicate=False));buffers=dict(module.named_buffers(remove_duplicate=False))
    if set(params)!=set(backing.parameters) or set(buffers)!=set(backing.buffers):return False
    for kind,items,originals in [('p',params,backing.parameters),('b',buffers,backing.buffers)]:
        for name,tensor in items.items():
            old=originals[name];before=backing.versions.get((kind,name))
            if tensor.shape!=old.shape or tensor.dtype!=old.dtype:return False
            if before is not None and version(tensor)!=before:return False
    with torch.no_grad():
        for name,tensor in params.items():tensor.data=backing.parameters[name]
        for name,tensor in backing.buffers.items():
            parent,_,leaf=name.rpartition('.');module.get_submodule(parent)._buffers[leaf]=tensor
    return True
