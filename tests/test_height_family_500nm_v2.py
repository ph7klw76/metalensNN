import sys,copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import export_height_family_attached_slc_500nm_v2 as q
import export_height_family_attached_slc_v1 as e
import audit_metalens_slc as a
class Tests(unittest.TestCase):
 def test_rounding_ties_and_bounds(self):
  d=dict(H_um=2.,cylinder_diameter_um=6.,units='um',heights_um=[1.,.75,.7,.3,.25,.2,0.]);before=copy.deepcopy(d)
  out,m=q.quantize(d);self.assertEqual(d,before);self.assertEqual(out['heights_um'],[1.,1.,.5,.5,.5,0.,0.]);self.assertLessEqual(m['maximum_height_change_um'],.25)
 def test_invalid_source_grid(self):
  with self.assertRaises(ValueError):q.quantize(dict(H_um=2.,cylinder_diameter_um=6.,heights_um=[1.,.123,0.]))
 def test_500nm_independent_fixture_and_original_mismatch(self):
  d=dict(H_um=2.,cylinder_diameter_um=6.,units='um',heights_um=[1.,.7,0.]);effective,_=q.quantize(d)
  core,assembly=e.load_helpers();builder=core.load_builder(e.BUILDER)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'source.json';source.write_text(json.dumps(effective))
   m=e.export_component(effective,source,root/'out',builder,core,assembly,'T01.slc',.5,False)
   raw=Path(m['final_slc']).read_bytes();result=a.audit(raw,source.read_bytes(),.5,128,'assembly')
   self.assertEqual(result['status'],'PASS',result['failed_checks']);self.assertEqual(m['layer_count'],6)
   self.assertEqual(a.audit(raw,json.dumps(d).encode(),.5,128,'assembly')['status'],'FAIL')
if __name__=='__main__':unittest.main(verbosity=2)
