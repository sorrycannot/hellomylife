"""Exercise extraction and LightGBM call contracts with explicit fake backends.

These are not substitutes for real Torch autograd or real LightGBM training.
"""
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from titan_classifier_comparison import common, data as d, training as t
from test_pipeline import PipelineTests


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,dtype=np.float32)
    @property
    def shape(self):return self.value.shape
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class ExtractionTests(PipelineTests):
    # Reuse fixture helpers but do not inherit duplicate test methods in discovery.
    def test_forward_hook_captures_pooled_vectors_and_reproduces_val(self):
        reference, arrays, frames, _ = self.inputs()
        instances=[]
        class Layer:
            callback=None
            def register_forward_pre_hook(self,callback):
                self.callback=callback
                return SimpleNamespace(remove=lambda:setattr(self,'callback',None))
        class Model:
            def __init__(self,*args,**kwargs):
                self.classifier=[Layer()];self.frozen=False;self.evaluation=False;instances.append(self)
            def to(self,device):return self
            def load_state_dict(self,state,strict):assert strict
            def eval(self):self.evaluation=True
            def parameters(self):return [SimpleNamespace(requires_grad_=lambda enabled:setattr(self,'frozen',not enabled))]
            def __call__(self,tab,img,txt):
                assert self.evaluation and self.frozen
                h=np.repeat(np.tanh(tab.value[:,:1]*.01),2048,axis=1).astype(np.float32)
                self.classifier[0].callback(self.classifier[0],(Tensor(h),))
                return Tensor(h[:,0])
        fake=SimpleNamespace(load=lambda *a,**kw:{'classifier.0.weight':np.empty((2048,2048),dtype=np.float32)},
                             as_tensor=lambda x,device:Tensor(x),no_grad=contextlib.nullcontext,
                             sigmoid=lambda x:Tensor(1/(1+np.exp(-x.value))),from_numpy=Tensor)
        expected=1/(1+np.exp(-np.tanh(arrays['val']['tabular'][:,0]*.01)))
        pd.DataFrame(dict(record=np.arange(1,len(expected)+1),y_true=arrays['val']['labels'],
                          probability=expected)).to_csv(reference['folder']/'val_predictions.csv',index=False)
        import sys
        with patch.dict(sys.modules,{'torch':fake}),patch.object(common,'import_file',return_value=SimpleNamespace(SMILESModel=Model)),\
             contextlib.redirect_stdout(__import__('io').StringIO()):
            folder=self.root/'hook_check'
            report=d.extract_features(reference,arrays,frames,folder,SimpleNamespace(type='cpu'),2)
            cached=d.read_cached_features(folder,report,arrays)
            for split in common.SPLITS:
                expected_h=np.repeat(np.tanh(arrays[split]['tabular'][:,:1]*.01),2048,axis=1)
                np.testing.assert_array_equal(cached[split]['pooled'],expected_h)
            self.assertTrue(report['frozen'])
            self.assertFalse(report['mechanism_scores_entered_encoder'])
            self.assertIsNone(instances[0].classifier[0].callback)
            wrong=pd.read_csv(reference['folder']/'val_predictions.csv');wrong['probability']=.999
            wrong.to_csv(reference['folder']/'val_predictions.csv',index=False)
            with self.assertRaisesRegex(ValueError,'do not reproduce'):
                d.extract_features(reference,arrays,frames,self.root/'hook_bad',SimpleNamespace(type='cpu'),2)
            self.assertIsNone(instances[1].classifier[0].callback)


# Avoid duplicating the fixture class's tests in this module's unittest discovery.
for _name in list(PipelineTests.__dict__):
    if _name.startswith('test_'):
        setattr(ExtractionTests,_name,None)
del PipelineTests


class LightGBMContracts(unittest.TestCase):
    def test_binary_probabilities_weighted_metric_and_best_model(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='lgb_contract_') as tmp:
            folder=Path(tmp)/'fit';seen={}
            class Dataset:
                def __init__(self,x,label,feature_name,reference=None):
                    self.x,self.y,self.names=x,label,feature_name
                def get_label(self):return self.y
            class Booster:
                best_iteration=2
                def __init__(self,model_file=None):
                    if model_file:assert Path(model_file).read_text()=='BEST_ROUNDS=2'
                def save_model(self,path,num_iteration):
                    assert num_iteration==2;Path(path).write_text('BEST_ROUNDS=2')
                def predict(self,x):return np.linspace(.2,.8,len(x))
                def feature_importance(self,kind):return np.zeros(2051)
            def train(params,train,num_boost_round,valid_sets,valid_names,feval,callbacks):
                self.assertEqual(params['objective'],'binary')
                self.assertEqual(params['metric'],'None')
                self.assertEqual(valid_names,['val'])
                self.assertEqual(train.x.shape[1],2051)
                self.assertEqual(train.names[-3:],common.SCORE_COLUMNS)
                val=valid_sets[0];p=np.linspace(.2,.8,len(val.y))
                name,loss,higher=feval(p,val)
                self.assertEqual(name,'weighted_bce');self.assertFalse(higher)
                self.assertAlmostEqual(loss,t.weighted_bce(val.y,p,params['scale_pos_weight']))
                for cb in callbacks:
                    if isinstance(cb,tuple) and cb[0]=='record':cb[1]['val']={'weighted_bce':[.8,.6,.7]}
                seen['called']=True
                return Booster()
            fake=SimpleNamespace(Dataset=Dataset,Booster=Booster,train=train,
                 early_stopping=lambda rounds,first_metric_only,verbose:('early',rounds),
                 record_evaluation=lambda hist:('record',hist),log_evaluation=lambda period:None)
            arrays={s:dict(pooled=np.zeros((n,2048),dtype=np.float32),
                           mechanism=np.full((n,3),.4,dtype=np.float32),labels=(np.arange(n)%2).astype(np.float32))
                    for s,n in [('train',8),('val',4)]}
            args=SimpleNamespace(seed=42,num_threads=4,lgbm_rounds=2000,lgbm_patience=50)
            with patch.object(t,'lgb',fake,create=True),contextlib.redirect_stdout(__import__('io').StringIO()):
                config,lock=t.fit_group('lightgbm_mechanism',args,arrays,folder,None)
            self.assertTrue(seen['called'])
            self.assertEqual(lock['best_iteration'],2)
            # Returned probabilities are saved directly, without another sigmoid.
            np.testing.assert_array_equal(np.load(folder/'val_probability.npy'),np.linspace(.2,.8,4))
            self.assertTrue((folder/'feature_importance.csv').exists())
            self.assertEqual(len(pd.read_csv(folder/'history.csv')),3)
            self.assertEqual(config['input_dim'],2051)


if __name__=='__main__':unittest.main()
