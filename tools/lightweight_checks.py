"""Source/control and mocked bridge checks; never load inference frameworks."""
import ast,importlib.abc,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise ImportError('Torch prohibited in lightweight checks')
sys.meta_path.insert(0,NoTorch())
for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
pkg=types.ModuleType('flux3_forge');pkg.__path__=[str(ROOT/'flux3_forge')];sys.modules['flux3_forge']=pkg
runtime=types.ModuleType('flux3_forge.runtime')
class Cancelled(Exception):pass
runtime.Cancelled=Cancelled;runtime.generate=lambda *args:None;runtime.unload=lambda:None
sys.modules['flux3_forge.runtime']=runtime
from flux3_forge import bridge,workflow_support as w
class Integration(unittest.TestCase):
 def test_default_seed_path(self):
  with patch.object(bridge,'_setting',side_effect=lambda key,default=None:default):self.assertEqual(bridge._seeds(types.SimpleNamespace(seed=2**32-1),2),[2**32-1,0])
 def test_enabled_seed_path(self):
  with patch.object(bridge,'_setting',side_effect=lambda key,default=None:{'flux3_workflow_controls':True,'flux3_seed_bits':'64'}.get(key,default)):
   self.assertEqual(bridge._seeds(types.SimpleNamespace(seed=478920109938706),2),[478920109938706,478920109938707])
 def test_runtime_preflight_before_load(self):
  tree=ast.parse((ROOT/'flux3_forge/runtime.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
  namespace={'workflow_support':w,'load':lambda _:self.fail('Invalid controls must fail before weight loading')}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'runtime-preflight','exec'),namespace)
  for request in [{'alpha':0},{'seed_bits':16},{'seed':float('nan')}]:
   with self.assertRaises(ValueError):namespace['generate'](request)
 def test_prompt_layout_after_adapter_resolution(self):
  tree=ast.parse((ROOT/'flux3_forge/bridge.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
  calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)]
  resolve=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='_resolve_adapters')
  append=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='caption')
  self.assertLess(resolve,append)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(Integration)
for name,path in [('bridge_checks',ROOT/'tests/test_bridge.py'),('workflow_checks',ROOT/'tests/test_workflow_support.py')]:
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
result=unittest.TextTestRunner(verbosity=1).run(suite)
assert 'torch' not in sys.modules
sys.exit(not result.wasSuccessful())
