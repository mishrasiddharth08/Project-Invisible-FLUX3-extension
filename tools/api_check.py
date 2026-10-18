"""Explicit Forge API acceptance check; restores temporary save settings."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image
import numpy as np
from live_check import CAPTION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--url",default="http://127.0.0.1:7860")
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    def api(endpoint,payload=None):
        request=urllib.request.Request(args.url+"/sdapi/v1/"+endpoint,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=600) as response:
            return json.load(response)
    options=api("options")
    keys=("outdir_samples","outdir_txt2img_samples","samples_save","samples_format")
    previous={key:options.get(key) for key in keys}
    (root/"save-settings-before.json").write_text(json.dumps(previous,indent=2),encoding="utf8")
    models=api("sd-models")
    selected=next(m for m in models if m["title"].startswith("FLUX 3 Image Klein (FP8)"))
    try:
        api("options",dict(outdir_samples="",outdir_txt2img_samples=str(root/"saved"),samples_save=True,samples_format="png"))
        payload=dict(prompt=CAPTION,negative_prompt="",seed=123456,steps=16,width=512,height=512,
                     cfg_scale=1,batch_size=1,n_iter=1,sampler_name="Euler",save_images=True,
                     override_settings={"sd_model_checkpoint":selected["title"]},
                     override_settings_restore_afterwards=True)
        started=time.perf_counter()
        response=api("txt2img",payload)
        elapsed=time.perf_counter()-started
        (root/"response-info.json").write_text(response.get("info",""),encoding="utf8")
        assert len(response.get("images",[]))==1,response.get("info")
        image=Image.open(io.BytesIO(base64.b64decode(response["images"][0])))
        image.save(root/"api-returned.png")
        pixels=np.asarray(image.convert("RGB"))
        files=list((root/"saved").rglob("*.png"))
        assert files,"Forge did not save a PNG"
        matches=[]
        for path in files:
            with Image.open(path) as saved:
                if saved.size==image.size and np.array_equal(np.asarray(saved.convert("RGB")),pixels):
                    assert "FLUX.3" in saved.info.get("parameters",""),"Native saved metadata missing FLUX3"
                    matches.append(str(path))
        assert matches,"Native saved pixels differ from API output"
        info=json.loads(response["info"])
        assert info["all_seeds"]==[123456],info
        record=dict(seconds=round(elapsed,3),size=image.size,seed=123456,
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    native_saved_files=matches,api_saved_pixels_equal=True,
                    model=selected["title"],info=info)
        (root/"native-evidence.json").write_text(json.dumps(record,indent=2),encoding="utf8")
        print(json.dumps({k:v for k,v in record.items() if k!="info"}),flush=True)
    finally:
        api("options",previous)


if __name__=="__main__":
    main()
