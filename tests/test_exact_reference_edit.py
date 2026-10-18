import unittest
from PIL import Image
from flux3_forge.exact_base_runtime import validate_reference_request
class ReferenceContract(unittest.TestCase):
 def test_graph_comparison_images_need_explicit_optin(self):
  with self.assertRaisesRegex(ValueError,'explicit'):validate_reference_request({'images':[Image.new('RGB',(64,64))]})
 def test_native_mask_or_denoising_is_never_silently_ignored(self):
  for key in ['mask','image_mask','denoising_strength']:
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Native masked'):validate_reference_request({key:0})
 def test_empty_reference_edit_rejected(self):
  with self.assertRaisesRegex(ValueError,'at least'):validate_reference_request({'reference_edit':True})
 def test_reference_count_type_and_size_validated(self):
  for images in ['path',list(range(11)),[Image.new('RGB',(16,64))],['unsafe-path']]:
   with self.subTest(images=type(images)),self.assertRaises(ValueError):validate_reference_request({'reference_edit':True,'images':images})
 def test_valid_one_and_multi_references_preserved_in_order(self):
  images=[Image.new('RGB',(64,64)),Image.new('RGB',(128,64))]
  self.assertIs(validate_reference_request({'reference_edit':True,'images':images}),images)
 def test_txt2img_remains_empty(self):self.assertEqual(validate_reference_request({}),[])
if __name__=='__main__':unittest.main()
