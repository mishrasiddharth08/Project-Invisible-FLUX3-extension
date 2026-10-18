"""Explicit final-frame source compositing, never latent/native inpainting."""
from dataclasses import dataclass
import hashlib
import numpy as np
from PIL import Image

def enabled(p):
    for script in getattr(getattr(p,'scripts',None),'alwayson_scripts',[]):
        if getattr(script,'flux3_source_protection_control',False):
            args=getattr(p,'script_args',())[script.args_from:script.args_to]
            value=args[0] if args else False
            if not isinstance(value,bool):raise ValueError('FLUX.3 source protection must be true or false.')
            return value
    return False

@dataclass
class Plan:
    source: Image.Image
    mask: Image.Image
    mask_sha256: str
    protected_pixels: int

def prepare(p):
    if not enabled(p):return None
    refs=list(getattr(p,'init_images',[]) or [])
    if len(refs)!=1:raise ValueError('Source protection needs exactly one img2img source reference.')
    source=refs[0].convert('RGB').copy()
    mask=getattr(p,'image_mask',None)
    if mask is None:mask=getattr(p,'mask',None)
    if not isinstance(mask,Image.Image):raise ValueError('Source protection requires an explicitly supplied grayscale paint mask; no region is inferred.')
    if getattr(p,'inpaint_full_res',False):raise ValueError('Source protection requires the full source canvas; disable inpaint crop/full-res mode.')
    if getattr(p,'inpainting_mask_invert',0):raise ValueError('Source protection uses white=generated, black=source; disable mask inversion.')
    if source.size!=(int(p.width),int(p.height)) or mask.size!=source.size:raise ValueError('Source protection requires source, output and mask dimensions to match exactly; no resizing is applied.')
    rgb=np.asarray(mask.convert('RGB'));alpha=np.asarray(mask.getchannel('A')) if 'A' in mask.getbands() else np.full(rgb.shape[:2],255,dtype=np.uint8)
    if np.any((np.ptp(rgb,axis=2)>0)&(alpha>0)):raise ValueError('Source protection requires a grayscale mask, not a colored photograph or colored paint.')
    values=((np.asarray(mask.convert('L'),dtype=np.uint16)*alpha.astype(np.uint16)+127)//255).astype('uint8')
    if not np.any(values):raise ValueError('Source protection mask has no editable pixels.')
    return Plan(source,Image.fromarray(values),hashlib.sha256(values.tobytes()).hexdigest(),int(np.count_nonzero(values==0)))

def apply(image,plan):
    if image.size!=plan.source.size:raise ValueError('A finishing callback changed the canvas; source protection cannot safely align it.')
    output=Image.composite(image.convert('RGB'),plan.source,plan.mask)
    black=np.asarray(plan.mask)==0
    if not np.array_equal(np.asarray(output)[black],np.asarray(plan.source)[black]):raise RuntimeError('Source protection verification failed.')
    return output


def verify_saved(path, expected):
    try:
        with Image.open(path) as saved:
            equal = saved.format == "PNG" and saved.size == expected.size and np.array_equal(np.asarray(saved.convert("RGB")), np.asarray(expected))
    except Exception as error:
        raise RuntimeError("Could not verify the protected PNG saved by Forge.") from error
    if not equal:
        raise RuntimeError("A save callback changed protected output pixels or format; saved output is not validated.")
