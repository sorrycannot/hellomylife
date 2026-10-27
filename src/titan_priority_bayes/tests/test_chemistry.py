from pathlib import Path
import unittest
import importlib.util
import pandas as pd
from priority_bayes.protocol import read_json

HAS_RDKIT=importlib.util.find_spec('rdkit') is not None


@unittest.skipUnless(HAS_RDKIT,'RDKit needed for actual rule matching tests')
class ChemistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from priority_bayes.chemistry import Matcher
        cls.reference=read_json(Path(__file__).parents[1]/'priority_bayes/reference_rules.json')
        cls.matcher=Matcher(cls.reference)

    def test_all_17_discovery_memberships_and_primary_identity_count(self):
        from priority_bayes.chemistry import calibrate,row_audit
        f=pd.DataFrame(self.reference['validation_rows'],columns=['SMILES','label'])
        r=calibrate(self.matcher,self.reference,f)
        self.assertTrue(r['exact_member_sets_reproduced'])
        self.assertEqual(r['n_rules'],17)
        a=row_audit(self.matcher,f)
        self.assertEqual(int(a.primary_unit.sum()),723)

    def test_stereoisomers_remain_separate_and_conflicts_unweighted(self):
        from priority_bayes.chemistry import row_audit
        f=pd.DataFrame({'SMILES':['C[C@H](O)CC','C[C@@H](O)CC','CC(=O)O','CC(=O)O','CCN.Cl'],
                        'label':[1,1,0,1,0]})
        a=row_audit(self.matcher,f)
        self.assertNotEqual(a.canonical_smiles[0],a.canonical_smiles[1])
        self.assertEqual(a.connectivity_group[0],a.connectivity_group[1])
        self.assertTrue(a.primary_unit[0]);self.assertTrue(a.primary_unit[1])
        self.assertFalse(a.V01[2]);self.assertFalse(a.V01[3])
        self.assertEqual(a.eligibility_status[4],'multiple_components')


if __name__=='__main__':unittest.main()
