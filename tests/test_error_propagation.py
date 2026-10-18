import sys
import types
import unittest
from unittest.mock import patch
from flux3_forge import bridge

class ErrorPropagation(unittest.TestCase):
 def test_invalid_request_releases_and_propagates_instead_of_empty_success(self):
  modules=types.ModuleType('modules');modules.images=object();modules.processing=object();modules.shared=object()
  with patch.dict(sys.modules,{'modules':modules}), patch.object(bridge,'_validate_processing',side_effect=ValueError('masks and inpainting are not supported')), patch.object(bridge,'_release') as release:
   with self.assertRaisesRegex(RuntimeError,'masks and inpainting are not supported'):
    bridge.generate(object(),{})
   release.assert_called_once()
