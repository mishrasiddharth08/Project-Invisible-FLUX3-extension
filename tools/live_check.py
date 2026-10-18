"""Explicit local-model generation check; produces images and measured evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAPTION = ("A professional studio photograph of a single red ceramic teapot on a pale oak table. "
           "The teapot is centered with its short spout pointing left and its curved handle on the right. "
           "A matching round lid rests securely on top. Soft window light from the left reveals the glossy red glaze, "
           "a gentle highlight, and a natural shadow beneath it. The background is an uncluttered warm cream wall. "
           "Eye-level camera, realistic proportions, crisp ceramic detail, calm minimal composition, no writing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantization", default="fp8",choices=("fp8","bf16"))
    parser.add_argument("--memory-mode", default="balanced",choices=("balanced","resident"))
    parser.add_argument("--runs",type=int,default=2)
    parser.add_argument("--width",type=int,default=512)
    parser.add_argument("--height",type=int,default=512)
    parser.add_argument("--prompt-file")
    parser.add_argument("--keep-latents",action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image, PngImagePlugin
    from flux3_forge import runtime
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    prompt = json.loads(Path(args.prompt_file).read_text(encoding="utf8"))["prompt"] if args.prompt_file else CAPTION
    request = dict(model_dir=args.model_dir, quantization=args.quantization,
                   memory_mode=args.memory_mode,prompt=prompt,width=args.width,height=args.height,steps=16,seed=123456,images=[])
    if args.keep_latents:
        request["latent_checkpoint"]=str(output/"sampled-latents.safetensors")
    record = dict(request={k:v for k,v in request.items() if k!="images"},runs=[],
                  gpu=torch.cuda.get_device_name(0),torch=torch.__version__,full_forge_path_tested=False)
    try:
        for run in range(args.runs):
            print(f"RUN {run+1}/{args.runs}",flush=True)
            torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            def progress(step,total):
                if args.width*args.height>1024*1024 or step%4==0 or step==total:
                    print(f"Step {step}/{total}; {time.perf_counter()-started:.1f}s; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)
            image,metadata = runtime.generate(request,progress,lambda:False)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            arr=np.asarray(image)
            assert image.size==(args.width,args.height) and image.mode=="RGB"
            assert float(arr.std())>3,"Generated image appears constant"
            path=output/f"flux3-fp8-run-{run+1}.png"
            png=PngImagePlugin.PngInfo()
            png.add_text("parameters",prompt+f"\nSteps: 16, Seed: 123456, Size: {args.width}x{args.height}, FLUX3: fp8 direct runtime acceptance")
            image.save(path,pnginfo=png)
            with Image.open(path) as saved:
                assert np.array_equal(np.asarray(saved),arr),"Saved pixels differ"
                assert saved.info.get("parameters"),"PNG metadata missing"
            result=dict(file=str(path),seconds=round(elapsed,3),
                        peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                        peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3),
                        pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),metadata=metadata)
            record["runs"].append(result)
            print(json.dumps(result,default=str),flush=True)
            (output/"live-evidence.json").write_text(json.dumps(record,indent=2,default=str),encoding="utf8")
        if args.runs>1:
            record["same_seed_pixels_equal"]=len({r["pixel_sha256"] for r in record["runs"]})==1
        (output/"live-evidence.json").write_text(json.dumps(record,indent=2,default=str),encoding="utf8")
    finally:
        runtime.unload()


if __name__=="__main__":
    main()
