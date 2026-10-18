"""Explicit A/B benchmark of experimental Spectrum on a complete local model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_check import CAPTION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    import torch
    import numpy as np
    from flux3_forge import runtime
    folder=Path(args.output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    request=dict(model_dir=args.model_dir,quantization="fp8",memory_mode="balanced",prompt=CAPTION,
                 width=512,height=512,steps=16,seed=123456,images=[],attention_backend="torch")
    results=[]
    pixels={}
    try:
        for name,enabled in (("warmup",False),("baseline",False),("spectrum",True),("spectrum-repeat",True)):
            print(name,flush=True)
            request["spectrum"]=enabled
            torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            image,metadata=runtime.generate(request,None,lambda:False)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            image.save(folder/f"{name}.png")
            array=np.asarray(image)
            pixels[name]=array
            result=dict(name=name,seconds=round(elapsed,3),peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                        pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),spectrum=metadata.get("spectrum",{}))
            results.append(result)
            print(json.dumps(result),flush=True)
        baseline=next(x for x in results if x["name"]=="baseline")
        accelerated=next(x for x in results if x["name"]=="spectrum-repeat")
        record=dict(results=results,speedup=round(baseline["seconds"]/accelerated["seconds"],3),
                    pixel_mae=float(np.abs(pixels["baseline"].astype(float)-pixels["spectrum"].astype(float)).mean()),
                    spectrum_repeat_pixels_equal=bool(np.array_equal(pixels["spectrum"],pixels["spectrum-repeat"])))
        (folder/"spectrum-evidence.json").write_text(json.dumps(record,indent=2),encoding="utf8")
        print(json.dumps({k:v for k,v in record.items() if k!="results"}),flush=True)
    finally:
        runtime.unload()


if __name__=="__main__":
    main()
