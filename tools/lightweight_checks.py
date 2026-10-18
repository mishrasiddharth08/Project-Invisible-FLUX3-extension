"""Standard-library source and mocked-bridge checks; no inference imports."""
import ast,importlib.abc,importlib.util,sys,types,unittest
from pathlib import Path
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
spec=importlib.util.spec_from_file_location('bridge_checks',ROOT/'tests/test_bridge.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
assert 'torch' not in sys.modules
sys.exit(not result.wasSuccessful())
