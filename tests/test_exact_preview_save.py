import copy,json,struct,tempfile,unittest,zlib
from pathlib import Path
from types import SimpleNamespace
from flux3_forge import exact_workflow_adapter as a
def png():
 header=struct.pack('>IIBBBBB',2,2,8,2,0,0,0);rows=b'\0'+b'\xff\0\0'*2+b'\0'+b'\0\xff\0'*2
 return a.SIGNATURE+a._chunk(b'IHDR',header)+a._chunk(b'IDAT',zlib.compress(rows))+a._chunk(b'IEND',b'')
def plan():
 request=dict(prompt='literal __import__("os") <caption>',seed=478920109938706,steps=25,cfg=3.5,alpha=16,width=2,height=2)
 return dict(experimental=True,model={'model_family':'base'},width=2,height=2,steps=25,cfg=3.5,requests=[request,dict(request,prompt='second')])
class Engine:
 def __init__(self):self.calls=[];self.released=0
 def generate(self,request,cancelled=None):
  self.calls.append(request);return SimpleNamespace(save=lambda buffer,format:buffer.write(png())),dict(request,model_family='base')
 def unload(self):self.released+=1
class Adapter(unittest.TestCase):
 def setUp(self):
  root=Path('lightweight-test-tmp');root.mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
 def test_preview_disabled_and_immutable(self):
  p=plan();v=a.gui_preview(p);self.assertFalse(v['can_execute']);v['prompts'][0]['prompt']='changed';self.assertNotEqual(p['requests'][0]['prompt'],'changed')
 def test_large_seed_never_float_coerced(self):
  p=plan();p['requests'][0]['seed']=2**63+123;v=a.gui_preview(p);self.assertNotIn('txt2img_seed',v['native_control_values']);self.assertEqual(v['seed_text'],str(2**63+123))
 def test_native_update_spec(self):
  values=a.gradio_updates(a.gui_preview(plan()),SimpleNamespace(update=lambda **kwargs:kwargs));self.assertEqual(values['txt2img_steps'],{'value':25})
 def test_png_payload_preserved_and_metadata_roundtrip(self):
  source=png();meta={'seed':2**63+123,'caption':'Unicode: \u4e2d\u6587'};out=a.annotate_png(source,meta,(2,2));old=[v for k,v in a.png_chunks(source) if k==b'IDAT'];new=[v for k,v in a.png_chunks(out) if k==b'IDAT'];self.assertEqual(old,new)
  payload=next(v for k,v in a.png_chunks(out) if k==b'iTXt');self.assertEqual(json.loads(payload.split(b'\0',5)[5]),meta)
 def test_bad_crc_rejected(self):
  data=bytearray(png());data[-1]^=1
  with self.assertRaises(ValueError):a.annotate_png(bytes(data),{},(2,2))
 def test_dimension_mismatch(self):
  with self.assertRaises(ValueError):a.annotate_png(png(),{},(4,4))
 def test_nan_metadata_rejected(self):
  with self.assertRaises(ValueError):a.annotate_png(png(),{'bad':float('nan')},(2,2))
 def test_path_traversal_rejected(self):
  for prefix in ['../escape','C:\\escape','a/b','']:
   with self.assertRaises(ValueError):a.save_png(png(),self.folder,prefix,{},(2,2))
 def test_no_overwrite(self):
  first=self.folder/'test_00001_.png';first.write_bytes(b'existing');row=a.save_png(png(),self.folder,'test',{},(2,2));self.assertEqual(first.read_bytes(),b'existing');self.assertTrue(row['path'].endswith('test_00002_.png'))
 def test_disabled_run_has_no_engine_effect(self):
  engine=Engine()
  with self.assertRaises(ValueError):a.run_plan(plan(),engine,self.folder)
  self.assertFalse(engine.calls);self.assertEqual(engine.released,0)
 def test_sequential_save_and_metadata(self):
  p=plan();original=copy.deepcopy(p);engine=Engine();rows=a.run_plan(p,engine,self.folder,allow_experimental=True)
  self.assertEqual(len(rows),2);self.assertEqual([r['seed'] for r in engine.calls],[478920109938706]*2);self.assertEqual(engine.released,1);self.assertEqual(p,original)
  self.assertEqual(len(a.gallery_results(rows)),2)
  chunks=a.png_chunks(Path(rows[0]['path']).read_bytes());self.assertTrue(any(k==b'iTXt' and v.startswith(b'parameters\0') for k,v in chunks))
 def test_cancel_before_second_save_keeps_completed_first(self):
  engine=Engine()
  with self.assertRaises(a.PlanCancelled):a.run_plan(plan(),engine,self.folder,allow_experimental=True,cancelled=lambda:len(engine.calls)>=2)
  self.assertEqual(len(list(self.folder.glob('*.png'))),1);self.assertEqual(engine.released,1);self.assertFalse(list(self.folder.glob('*.partial')))
 def test_runtime_metadata_mismatch_fails(self):
  engine=Engine();generate=engine.generate
  def bad(request,cancelled=None):
   image,metadata=generate(request,cancelled);metadata['cfg']=7;return image,metadata
  engine.generate=bad
  with self.assertRaises(ValueError):a.run_plan(plan(),engine,self.folder,allow_experimental=True)
  self.assertEqual(engine.released,1);self.assertFalse(list(self.folder.glob('*.png')))
if __name__=='__main__':unittest.main(verbosity=2)
