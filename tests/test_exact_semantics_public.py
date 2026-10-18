import copy,unittest
from flux3_forge import exact_workflow_semantics as s
def document():
 fields=['prompt','width','height','batch_size','steps','alpha','seed','sampler_name','scheduler','unet_name','clip_name','vae_name','cfg']
 values=['cached',1024,1024,1,25,16,478920109938706,'euler','simple','flux3_image_klein_int8_convrot.safetensors','qwen3vl_4b_int8_convrot.safetensors','flux2-vae.safetensors',3.5]
 return {'nodes':[{'id':128,'type':'diff','mode':0,'widgets_values':values,'inputs':[{'link':3},{'link':1},{'link':2}]},{'id':19,'type':'ResolutionSelector','widgets_values':['1:1 (Square)',3,16]},{'id':184,'type':'easy promptList','widgets_values':['one','two','three','four','five']}],'definitions':{'subgraphs':[{'id':'diff','inputs':[{'name':k} for k in fields]}]},'links':[[1,19,0,128,1,'INT'],[2,19,1,128,2,'INT'],[3,184,1,128,0,'STRING']]}
class PublicSemantics(unittest.TestCase):
 def test_resolution(self):self.assertEqual(s.resolution('1:1 (Square)',3,16),(1776,1776))
 def test_literal_prompt_whitespace(self):self.assertEqual(s.prompt_list(['a','',' ','a','b'],['first']),['first','a',' ','a','b'])
 def test_bad_resolution(self):
  with self.assertRaises(ValueError):s.resolution('1:1 (Square)',float('nan'),16)
 def test_known_base_alias(self):self.assertEqual(s.checkpoint_alias('flux3_image_klein_int8_convrot.safetensors')['model_family'],'base')
 def test_no_alias_guessing(self):
  with self.assertRaises(ValueError):s.checkpoint_alias('renamed.safetensors')
 def test_five_seed_broadcast_requests(self):
  p=s.execution_plan(document());self.assertEqual(len(p['requests']),5);self.assertEqual({r['seed'] for r in p['requests']},{478920109938706});self.assertTrue(all(r['exact_workflow'] for r in p['requests']))
 def test_plan_immutable(self):
  d=document();before=copy.deepcopy(d);s.execution_plan(d);self.assertEqual(d,before)
 def test_bypassed_not_implicitly_enabled(self):
  d=document();d['nodes'][0]['mode']=4
  with self.assertRaises(ValueError):s.execution_plan(d)
