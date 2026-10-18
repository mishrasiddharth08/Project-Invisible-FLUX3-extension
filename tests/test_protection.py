import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from PIL import Image
import numpy as np
from flux3_forge import protection

class ProtectionTests(unittest.TestCase):
 def p(self,mask=None,**kw):
  script=NS(flux3_source_protection_control=True,args_from=0,args_to=1)
  args=dict(scripts=NS(alwayson_scripts=[script]),script_args=[True],init_images=[Image.new('RGB',(16,16),(12,34,56))],image_mask=mask,width=16,height=16);args.update(kw);return NS(**args)
 def test_explicit_opt_in_required(self):self.assertFalse(protection.enabled(self.p(script_args=[False])))
 def test_no_mask_is_not_inferred(self):
  with self.assertRaisesRegex(ValueError,'explicitly supplied'):protection.prepare(self.p())
 def test_binary_and_soft_mask_preserve_black_exactly(self):
  mask=Image.new('L',(16,16),0);mask.putpixel((3,4),255);mask.putpixel((5,6),128);p=self.p(mask);plan=protection.prepare(p);result=protection.apply(Image.new('RGB',(16,16),'white'),plan)
  self.assertEqual(result.getpixel((3,4)),(255,255,255));self.assertEqual(result.getpixel((0,0)),(12,34,56));self.assertNotEqual(result.getpixel((5,6)),(255,255,255));self.assertEqual(plan.protected_pixels,254)
 def test_transparent_white_does_not_expand_edit_area(self):
  mask=Image.new('RGBA',(16,16),(255,255,255,0));mask.putpixel((2,3),(255,255,255,255));plan=protection.prepare(self.p(mask));self.assertEqual(plan.protected_pixels,255)
 def test_photo_mask_rejected(self):
  with self.assertRaisesRegex(ValueError,'colored photograph'):protection.prepare(self.p(Image.new('RGB',(16,16),'red')))
 def test_geometry_is_never_invented_or_resized(self):
  with self.assertRaisesRegex(ValueError,'dimensions'):protection.prepare(self.p(Image.new('L',(8,8),255)))
  plan=protection.prepare(self.p(Image.new('L',(16,16),255)))
  with self.assertRaisesRegex(ValueError,'changed the canvas'):protection.apply(Image.new('RGB',(32,32)),plan)
 def test_invert_and_crop_rejected(self):
  for key in ['inpainting_mask_invert','inpaint_full_res']:
   with self.assertRaises(ValueError):protection.prepare(self.p(Image.new('L',(16,16),255),**{key:True}))

 def test_saved_pixel_changes_are_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'result.png';image=Image.new('RGB',(16,16),'red');image.save(path);protection.verify_saved(path,image)
   with self.assertRaisesRegex(RuntimeError,'save callback'):protection.verify_saved(path,Image.new('RGB',(16,16),'blue'))
