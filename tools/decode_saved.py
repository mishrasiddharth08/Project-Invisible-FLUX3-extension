"""Decode a saved FLUX3 latent with only the genuine model autoencoder loaded."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--latents",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from PIL import PngImagePlugin
    from flux3_forge.memory import bound_vae_convolutions
    from flux3_forge.vendor.flux3.autoencoder import AutoEncoder
    from flux3_forge.vendor.flux3.checkpoint import load_checkpoint
    from flux3_forge.vendor.flux3.pipeline import to_pil
    torch.cuda.set_per_process_memory_fraction(.82)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    model=load_checkpoint(AutoEncoder(),str(Path(args.model_dir)/"ae.safetensors"),"cuda",torch.float32)
    bound_vae_convolutions(model)
    values=load_file(args.latents,device="cuda")
    with torch.inference_mode():
        image=to_pil(model.decode(values["tokens"],values["ids"]))
    path=Path(args.output).resolve()
    png=PngImagePlugin.PngInfo()
    png.add_text("parameters","FLUX3 original sampled latents; streamed full-spatial FP32 VAE decode")
    image.save(path,pnginfo=png)
    torch.cuda.synchronize()
    result=dict(file=str(path),size=image.size,seconds=round(time.perf_counter()-start,3),
                peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3),
                physical_gpu=torch.cuda.get_device_name(0),soft_process_fraction=.82)
    path.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
