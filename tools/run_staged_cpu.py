"""Explicitly isolate legacy CUDA capability probes; prohibit CUDA initialization."""
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));(ROOT/'scratch').mkdir(exist_ok=True)
assert (Path.cwd()/'tests/test_exact_base_sampling.py').is_file(),'Run from the staged V3 root; never discover ambient site-packages tests'
tempfile.tempdir=str(Path('scratch').resolve())
def forbidden(*args,**kwargs):raise AssertionError('CUDA initialization is forbidden in this CPU-only suite')
with patch.object(torch.cuda,'is_available',return_value=False),patch.object(torch.cuda,'is_bf16_supported',return_value=True),patch.object(torch.cuda,'empty_cache',return_value=None),patch.object(torch.cuda,'_lazy_init',side_effect=forbidden):
 suite=unittest.TestSuite()
 for pattern in ['test_comfy_int8.py','test_comfy_profiles.py','test_comfy_pipeline.py','test_comfy_v2.py','test_exact_base_sampling.py']:
  suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized(),'CPU-only test isolation failed'
print('CUDA initialized: false; no real weights loaded')
sys.exit(not result.wasSuccessful())
