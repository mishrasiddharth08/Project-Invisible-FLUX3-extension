"""Strict linear LoRA/LoKr/LoHa branches; base quantized weights stay intact."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

SUFFIXES = ("lora_down.weight", "lora_up.weight", "lora_A.weight", "lora_B.weight",
            "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_w1", "lokr_w2",
            "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha", "scale")


class AdapterLinear(nn.Module):
    def __init__(self, base, tensors, strength):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        if not math.isfinite(float(strength)):
            raise ValueError("Adapter strength must be finite")
        self.strength = float(strength)
        self.names = {}
        for index, (name, tensor) in enumerate(tensors.items()):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("Adapter tensors must be finite floating point values")
            key = f"tensor_{index}"
            self.register_buffer(key, tensor)
            self.names[name] = key
        self.kind = "lora" if any(k.startswith("lora_") for k in tensors) else "lokr" if any(k.startswith("lokr_") for k in tensors) else "loha"
        allowed = {"alpha", "scale"}
        if self.kind == "lora":
            self.down = "lora_down.weight" if "lora_down.weight" in tensors else "lora_A.weight"
            self.up = "lora_up.weight" if "lora_up.weight" in tensors else "lora_B.weight"
            allowed.update((self.down, self.up))
            down, up = self.get(self.down), self.get(self.up)
            if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError("LoRA factors have incompatible shapes")
            shape, rank = (up.shape[0], down.shape[1]), down.shape[0]
        elif self.kind == "lokr":
            allowed.update(k for k in SUFFIXES if k.startswith("lokr_"))
            w1, rank1 = self.matrix("lokr_w1")
            w2, rank2 = self.matrix("lokr_w2")
            shape = (w1.shape[0]*w2.shape[0], w1.shape[1]*w2.shape[1])
            rank = rank2 or rank1
        else:
            allowed.update(k for k in SUFFIXES if k.startswith("hada_"))
            a,b,c,d = (self.get(k) for k in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
            if any(t.ndim != 2 for t in (a,b,c,d)) or a.shape[1] != b.shape[0] or c.shape[1] != d.shape[0] or (a.shape[0],b.shape[1]) != (c.shape[0],d.shape[1]):
                raise ValueError("LoHa factors have incompatible shapes")
            shape, rank = (a.shape[0], b.shape[1]), b.shape[0]
        if set(tensors) - allowed:
            raise ValueError("Mixed or unsupported adapter factor types")
        if shape != (self.out_features, self.in_features):
            raise ValueError(f"Adapter shape {shape} does not match layer {(self.out_features,self.in_features)}")
        alpha, scale = tensors.get("alpha"), tensors.get("scale")
        if any(t is not None and t.numel() != 1 for t in (alpha,scale)):
            raise ValueError("Adapter alpha/scale must be scalar")
        self.multiplier = self.strength * (float(alpha.item())/rank if alpha is not None and rank else 1.) * (float(scale.item()) if scale is not None else 1.)

    def get(self, name, value=None):
        if name not in self.names:
            raise ValueError(f"Missing adapter factor: {name}")
        tensor = getattr(self, self.names[name])
        return tensor.to(device=value.device, dtype=value.dtype) if value is not None else tensor

    def matrix(self, stem, value=None):
        if stem in self.names:
            if stem+"_a" in self.names or stem+"_b" in self.names:
                raise ValueError("LoKr cannot mix full and factorized matrices")
            tensor = self.get(stem, value)
            if tensor.ndim != 2:
                raise ValueError("Only linear LoKr matrices are supported")
            return tensor, None
        a,b = self.get(stem+"_a",value), self.get(stem+"_b",value)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("LoKr factors have incompatible shapes")
        return a @ b, b.shape[0]

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base,"bias",None)

    def forward(self, x):
        output = self.base(x)
        if self.multiplier == 0:
            return output
        if self.kind == "lora":
            branch = F.linear(F.linear(x,self.get(self.down,x)),self.get(self.up,x))
        elif self.kind == "lokr":
            w1,_ = self.matrix("lokr_w1",x)
            w2,_ = self.matrix("lokr_w2",x)
            branch = x.reshape(*x.shape[:-1],w1.shape[1],w2.shape[1])
            branch = F.linear(F.linear(branch,w2).transpose(-1,-2),w1).transpose(-1,-2).reshape(*x.shape[:-1],self.out_features)
        else:
            a,b,c,d = (self.get(k,x) for k in ("hada_w1_a","hada_w1_b","hada_w2_a","hada_w2_b"))
            # Bound activation memory by processing rank pairs individually.
            branch = torch.zeros_like(output)
            for r in range(b.shape[0]):
                hidden = F.linear(x,b[r:r+1]*d)
                branch = branch + F.linear(hidden,a[:,r:r+1]*c)
        return output + branch.to(output.dtype)*self.multiplier


def attach(model, specs):
    """Atomically validate/attach local safetensors adapters; return undo records."""
    aliases = {}
    for name, module in model.named_modules():
        if name and hasattr(module,"in_features") and hasattr(module,"out_features"):
            for alias in (name,"diffusion_model."+name,"model.diffusion_model."+name,"transformer."+name,"base_model.model."+name,"lora_unet_"+name.replace(".","_")):
                aliases.setdefault(alias,[]).append((name,module))
    plans, current = [], {}
    for spec in specs:
        path = Path(spec["path"])
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            raise ValueError("Adapters require existing local .safetensors files")
        groups = {}
        with safe_open(str(path),framework="pt",device="cpu") as handle:
            family = str((handle.metadata() or {}).get("ss_base_model_version", "")).lower()
            if family and family not in ("flux3", "flux-3", "flux_3", "flux3-klein", "flux-3-image-klein"):
                raise ValueError(f"Adapter declares incompatible model family: {family}")
            for key in handle.keys():
                suffix = next((s for s in SUFFIXES if key.endswith("."+s)),None)
                if suffix is None:
                    raise ValueError(f"Unsupported adapter tensor: {key}")
                group = key[:-len(suffix)-1]
                groups.setdefault(group,{})[suffix] = handle.get_tensor(key)
        if not groups:
            raise ValueError("Adapter contains no tensors")
        seen = set()
        for group,tensors in groups.items():
            targets = aliases.get(group,[])
            if len(targets) != 1:
                raise ValueError(f"Adapter target missing or ambiguous: {group}; only FLUX3 original layer names are supported")
            name,base = targets[0]
            if name in seen:
                raise ValueError(f"Duplicate adapter target through different aliases: {name}")
            seen.add(name)
            wrapper = AdapterLinear(current.get(name,base),tensors,spec.get("strength",1.))
            current[name] = wrapper
    for name,wrapper in current.items():
        parent,_,leaf = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        plans.append((owner,leaf,getattr(owner,leaf),wrapper))
    for owner,leaf,base,wrapper in plans:
        setattr(owner,leaf,wrapper)
    return plans


def detach(plans):
    for owner,leaf,base,wrapper in reversed(plans):
        if getattr(owner,leaf) is wrapper:
            setattr(owner,leaf,base)
