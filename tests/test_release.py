import gc
import unittest
import weakref
from unittest.mock import patch
from flux3_forge import runtime

class ReleaseTests(unittest.TestCase):
 def test_unload_collects_unreachable_pipeline_cycles_before_cache_release(self):
  class Pipeline:
   pass
  model=runtime.RuntimeModel.__new__(runtime.RuntimeModel)
  model._prompt_cache={'one':object()};model._prompt_cache_bytes=100
  pipeline=Pipeline();pipeline.cycle=pipeline;reference=weakref.ref(pipeline)
  model.pipeline=pipeline;del pipeline
  enabled=gc.isenabled();gc.disable()
  try:
   def empty():self.assertIsNone(reference())
   with patch.object(runtime,'_empty_cache',side_effect=empty):model.unload()
   self.assertEqual(model._prompt_cache,{});self.assertEqual(model._prompt_cache_bytes,0);self.assertIsNone(model.pipeline)
  finally:
   if enabled:gc.enable()
