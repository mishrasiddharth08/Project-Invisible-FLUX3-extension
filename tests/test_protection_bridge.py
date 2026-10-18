import sys,types,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
from flux3_forge import bridge,runtime
from test_bridge import P

class ProtectedBridge(unittest.TestCase):
 def test_final_protection_after_all_callbacks_and_saved_parity(self):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);saved=[]
  def save(image,*args,**kwargs):
   saved.append(image.copy());path=Path(temp.name)/'output.png';image.save(path);return str(path),None
  state=types.SimpleNamespace(interrupted=False,skipped=False,job_count=0,sampling_steps=0,sampling_step=0,textinfo='',assign_current_image=lambda im:None,nextjob=lambda:None)
  shared=types.SimpleNamespace(state=state,prompt_styles=types.SimpleNamespace(apply_styles_to_prompt=lambda p,s:p,apply_negative_styles_to_prompt=lambda p,s:p),opts=types.SimpleNamespace(samples_save=True,samples_format='png'))
  class Processed:
   def __init__(self,p,images,seed,info,**kwargs):self.images=images
  modules=types.ModuleType('modules');modules.shared=shared;modules.processing=types.SimpleNamespace(Processed=Processed);modules.images=types.SimpleNamespace(save_image=save);modules.extra_networks=types.SimpleNamespace(parse_prompt=lambda p:(p,{}))
  backend=types.ModuleType('backend');backend.memory_management=types.SimpleNamespace(unload_all_models=lambda:None)
  p=P();p.batch_size=1;p.width=p.height=64;p.init_images=[Image.new('RGB',(64,64),(11,22,33))];p.image_mask=Image.new('L',(64,64),0);p.image_mask.paste(255,(16,16,48,48));p.script_args=[True];p.extra_generation_params={}
  control=types.SimpleNamespace(flux3_source_protection_control=True,args_from=0,args_to=1)
  p.scripts=types.SimpleNamespace(alwayson_scripts=[control],postprocess=lambda p,result:setattr(result,'images',[Image.new('RGB',(64,64),'red')]))
  with patch.dict(sys.modules,{'modules':modules,'backend':backend}),patch.object(runtime,'generate',return_value=(Image.new('RGB',(64,64),'blue'),{})),patch.object(runtime,'unload'),patch.object(bridge,'_setting',side_effect=lambda key,default=None:True if key.endswith('acknowledged') else default):
   result=bridge.generate(p,dict(checkpoint_path='model.safetensors',model_dir='existing',quantization='fp8'))
  self.assertEqual(result.images[0].getpixel((0,0)),(11,22,33));self.assertEqual(result.images[0].getpixel((20,20)),(255,0,0));self.assertEqual(len(saved),1);self.assertTrue(np.array_equal(np.asarray(saved[0]),np.asarray(result.images[0])));self.assertIn('not native inpainting',p.extra_generation_params['FLUX.3 source protection'])
