from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from priority_bayes.inputs import mechanism,array,align_frame


class InputTests(unittest.TestCase):
    def test_mixed_score_table_loads_only_selected_rows_and_named_score_order(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'shared_scores.csv'
            pd.DataFrame({'partition':['val','test','val'],
                'SMILES':['CCO','FORBIDDEN_TEST_STRUCTURE','CCN'],
                'original_row_index':[1,999,0],
                'score_influx':[.2,'NOT_NUMERIC',.3],
                'score_efflux':[.8,'NOT_NUMERIC',.7],
                'score_pampa':[.4,'NOT_NUMERIC',.6],
                'label':['PRIVATE','DO_NOT_LOAD','PRIVATE']}).to_csv(path,index=False)
            rec=dict(path=str(path),columns=['score_influx','score_efflux','score_pampa'],
                mechanism_order=['influx','efflux','pampa'],smiles_column='SMILES',
                row_index_column='original_row_index',split_column='partition')
            original=pd.read_csv
            def guarded(*a,**kw):
                self.assertNotIn('label',kw.get('usecols',[]))
                return original(*a,**kw)
            with mock.patch('priority_bayes.inputs.pd.read_csv',side_effect=guarded):
                result,mode=mechanism(rec,pd.DataFrame({'SMILES':['CCN','CCO']}),'val',('efflux','influx','pampa'))
            np.testing.assert_allclose(result,[[.7,.3,.6],[.8,.2,.4]])
            self.assertEqual(mode,'explicit_original_row_index')

    def test_duplicate_smiles_cannot_be_reordered_without_indices(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            align_frame(pd.DataFrame({'SMILES':['CCO','CCN','CCO']}),{'smiles_column':'SMILES'},['CCO','CCO','CCN'])

    def test_npz_does_not_materialize_other_partition_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'shared.npz'
            np.savez(p,train_H=np.ones((2,2048)),test_unread=np.array([object()],dtype=object))
            # Loading test_unread would fail under allow_pickle=False.
            result=array(dict(path=str(p),key='train_H'))
            self.assertEqual(result.shape,(2,2048))


if __name__=='__main__':unittest.main()
