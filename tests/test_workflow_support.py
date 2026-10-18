"""Standard-library checks: no inference package or CUDA imports."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
import importlib.util
_spec=importlib.util.spec_from_file_location('workflow_support',Path(__file__).resolve().parents[1]/'flux3_forge/workflow_support.py')
w=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(w)

class Controls(unittest.TestCase):
    def test_defaults(self): self.assertEqual(w.controls({}),dict(alpha=6.93,seed_bits=32,seed=0))
    def test_alpha(self):
        self.assertEqual(w.alpha(16),16)
        for v in [True,0,-1,1001,float('nan'),float('inf')]:
            with self.subTest(v=v),self.assertRaises(ValueError): w.alpha(v)
    def test_seed_width(self):
        for v in [True,16,'64.0']:
            with self.assertRaises(ValueError): w.seed_bits(v)
    def test_legacy(self): self.assertEqual(w.seeds(2**32-1,3),[2**32-1,0,1])
    def test_large_seed(self): self.assertEqual(w.seeds(478920109938706,2,64),[478920109938706,478920109938707])
    def test_unsafe_float(self):
        for v in [float(2**54),1.5,float('inf'),True]:
            with self.assertRaises(ValueError):w.normalize_seed(v,64)
    def test_random(self):self.assertEqual(w.seeds([-1,6],2,64,lambda bits:bits),[64,6])
    def test_short(self):
        with self.assertRaises(ValueError):w.seeds([1],2)

class Layout(unittest.TestCase):
    def row(self):return {'id':'subject','bbox':[0,10,1000,900],'desc':'literal <lora:no:1>', 'origin':'user'}
    def test_roundtrip(self):
        row=self.row();out=w.elements([row]);self.assertEqual(out,[row]);self.assertIsNot(out[0],row)
    def test_empty(self):self.assertEqual(w.caption('caption  ',[]),'caption  ')
    def test_caption(self):self.assertTrue(w.caption('caption',[self.row()]).startswith('caption [{'))
    def test_invalid_bbox(self):
        for box in [[0,0,0,1],[0,0,1,1001],[0,False,1,1],[0,0,float('nan'),1]]:
            row=self.row();row['bbox']=box
            with self.assertRaises(ValueError):w.elements([row])
    def test_duplicate(self):
        with self.assertRaises(ValueError):w.elements([self.row(),self.row()])
    def test_limits(self):
        for v in ['a'*65537,[self.row()]*65,{},[{'bbox':[0,0,1,1]}]]:
            with self.assertRaises((ValueError,TypeError)):w.elements(v)
    def test_auxiliary_nan(self):
        r=self.row();r['origin']=float('nan')
        with self.assertRaises(ValueError):w.elements([r])

if __name__=='__main__':unittest.main(verbosity=2)
