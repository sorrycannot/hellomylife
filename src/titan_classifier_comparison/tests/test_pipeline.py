"""Real NumPy/sklearn file checks + mocked training orchestration, no Torch needed.

Run from TITAN's src directory:
  python -m unittest discover -s titan_classifier_comparison/tests -v
These checks never assert real Torch training or real LightGBM performance.
"""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from titan_classifier_comparison import data as d
from titan_classifier_comparison import training as t
from titan_classifier_comparison import run_classifier_comparison as r
from titan_classifier_comparison.common import (SCORE_COLUMNS, SPLITS, choose_threshold,
    classification_metrics, file_record, read_json, sha256, write_json)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='classifier_qa_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.counts = dict(train=8,val=4,test=4)
        self.dims = dict(tabular=5,image=7,text=9)
        for attr, value in [('COUNTS',self.counts),('DIMS',self.dims)]:
            patcher = patch.object(d,attr,value)
            patcher.start(); self.addCleanup(patcher.stop)
        for folder in ('dataset','features','mechanism_features','src/utils','titan_attention_ablation_seed42_results'):
            (self.root/folder).mkdir(parents=True)
        (self.root/'src/utils/model.py').write_text('# synthetic model source for file checks only\n')
        self.run = self.root/'titan_attention_ablation_seed42_results'
        self.model = self.run/'experiments/baseline/attempt_001/baseline_seed42'
        self.model.mkdir(parents=True)
        checkpoint = self.model/'best_model.pth';checkpoint.write_bytes(b'SYNTHETIC_ONLY')
        config = dict(mode='baseline',seed=42,pooled_dim=2048,classifier_input_dim=2048,
                      d_tab=5,d_img=7,d_txt=9,projection_dim=2048,dropout=.1)
        write_json(self.model/'model_config.json',config)
        write_json(self.model/'selection_lock.json',dict(threshold=.7,checkpoint=file_record(checkpoint)))
        inputs, datasets, splits, scalers, self.raw = {}, {}, {}, {}, {}
        for i,split in enumerate(SPLITS):
            n = self.counts[split]
            frame = pd.DataFrame(dict(SMILES=[f'QA_{split}_{k}' for k in range(n)],label=np.arange(n)%2))
            path = self.root/f'dataset/{split}_classification.csv';frame.to_csv(path,index=False)
            datasets[split] = file_record(path)
            m = np.random.default_rng(42+i).uniform(.01,.99,(n,3)).astype(np.float32)
            score = self.root/f'mechanism_features/{split}-mechanism-classification.npy';np.save(score,m)
            audit = pd.DataFrame(dict(record=np.arange(1,n+1),original_smiles=frame.SMILES))
            for j,col in enumerate(SCORE_COLUMNS):audit[col]=m[:,j]
            ap = self.root/f'mechanism_features/{split}-mechanism-audit.csv';audit.to_csv(ap,index=False)
            splits[split] = dict(scores=file_record(score),audit=file_record(ap))
            inputs[split] = dict(dataset=datasets[split],mechanism=file_record(score),row_audit=file_record(ap),n=n)
            for mod,width in self.dims.items():
                # Strong shift: a train-only scaler can be distinguished from val/test fits.
                x = (np.arange(n*width).reshape(n,width)+i*10000).astype(np.float32)
                self.raw[split,mod] = x
                fp = self.root/f'features/{split}-{mod}-classification.npy';np.save(fp,x)
                inputs[split][mod] = file_record(fp)
                if split=='train':scalers[mod]=StandardScaler().fit(x)
        joblib.dump(scalers,self.run/'titan_modality_scalers.joblib')
        p = .13+.72*(np.arange(self.counts['val'])%2)
        pd.DataFrame(dict(record=np.arange(1,len(p)+1),y_true=np.arange(len(p))%2,
                          probability=p)).to_csv(self.model/'val_predictions.csv',index=False)
        write_json(self.root/'mechanism_features/manifest.json',dict(status='complete',task='classification',
            score_columns=SCORE_COLUMNS,dataset_files=datasets,splits=splits,
            existing_TITAN_split_identity_overlaps=dict(train__val=1,train__test=2,val__test=0)))
        write_json(self.run/'run_manifest.json',dict(status='complete',definition=dict(seed=42,
            data=dict(inputs=inputs),sources=[file_record(self.root/'src/utils/model.py')]),
            fitted=dict(baseline=dict(folder=str(self.model.relative_to(self.run)),
                config=file_record(self.model/'model_config.json'),lock=file_record(self.model/'selection_lock.json')))))

    def inputs(self):
        ref = d.locate_reference(self.root,None,42)
        arrays, frames, fingerprints = d.load_inputs(self.root,self.root/'mechanism_features',ref)
        return ref, arrays, frames, fingerprints

    def test_input_provenance_and_scalers(self):
        _, arrays, _, _ = self.inputs()
        for split in SPLITS:
            np.testing.assert_array_equal(arrays[split]['labels'],np.arange(self.counts[split])%2)
            for mod in self.dims:
                expected = StandardScaler().fit(self.raw['train',mod]).transform(self.raw[split,mod]).astype(np.float32)
                np.testing.assert_array_equal(arrays[split][mod],expected)
        bad = self.root/'mechanism_features/train-mechanism-classification.npy'
        np.save(bad,np.zeros((8,3),dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'changed'):self.inputs()

    def test_wrong_checkpoint_and_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed=42'):d.locate_reference(self.root,self.run,2022)
        # Even a consistently re-hashed altered manifest cannot turn 2051-D into a baseline.
        p = self.model/'model_config.json';cfg=read_json(p);cfg['classifier_input_dim']=2051;write_json(p,cfg)
        mp=self.run/'run_manifest.json';meta=read_json(mp)
        meta['fitted']['baseline']['config']=file_record(p);write_json(mp,meta)
        with self.assertRaisesRegex(ValueError,'2048-D'):self.inputs()

    def test_check_only_is_read_only(self):
        args = r.parse_args(['--project-root',str(self.root),'--check-only'])
        before = {str(p):sha256(p) for p in self.root.rglob('*') if p.is_file()}
        with patch.object(t,'load_backends',side_effect=AssertionError('No heavy backend in preflight')):
            with contextlib.redirect_stdout(io.StringIO()):r.run(args)
        after = {str(p):sha256(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_four_locks_before_test_resume_and_integrity(self):
        ref, source_arrays, frames, _ = self.inputs()
        out = self.root/'titan_classifier_comparison_seed42_results'
        events = []
        def fake_extract(reference, data, frame, folder, device, batch_size):
            events.append(('prepare',None));folder.mkdir()
            files = {}
            for split in SPLITS:
                h = np.random.default_rng(1).normal(size=(len(frame[split]),2048)).astype(np.float32)
                hp=folder/f'{split}-pooled.npy';np.save(hp,h)
                pp=folder/f'{split}-original-baseline-probability.npy';np.save(pp,.13+.72*data[split]['labels'])
                files[split]=dict(pooled=file_record(hp),original_probability=file_record(pp))
            report=dict(status='complete',files=files,frozen=True)
            write_json(folder/'preparation.json',report)
            return report
        def fake_fit(group,args,subset,folder,device):
            self.assertEqual(set(subset),{'train','val'})
            self.assertFalse(any(e[0]=='test' for e in events))
            events.append(('fit',group));folder.mkdir()
            for split in ('train','val'):
                x=t.features(subset[split],group)
                self.assertEqual(x.shape[1],2051 if group.endswith('_mechanism') else 2048)
                if group.endswith('_mechanism'):
                    np.testing.assert_array_equal(x[:,-3:],source_arrays[split]['mechanism'])
            cfg=dict(experiment=group,seed=42,input_dim=x.shape[1]);write_json(folder/'model_config.json',cfg)
            ck=folder/'best_model.pth';ck.write_bytes(b'SYNTHETIC_'+group.encode())
            p=.13+.72*subset['val']['labels'];np.save(folder/'val_probability.npy',p)
            th=choose_threshold(subset['val']['labels'],p)
            lock=dict(experiment=group,seed=42,threshold=th,checkpoint=file_record(ck),
                config=file_record(folder/'model_config.json'),val_probability=file_record(folder/'val_probability.npy'))
            write_json(folder/'selection_lock.json',lock)
            return cfg,lock
        def fake_predict(group,folder,config,lock,arrays,device):
            self.assertTrue((out/'all_selections_locked.json').exists())
            self.assertEqual(sum(e[0]=='fit' for e in events),4)
            self.assertEqual(set(read_json(out/'all_selections_locked.json')['groups']),set(t.GROUPS))
            events.append(('test',group))
            return .13+.72*arrays['labels']
        before={str(p):sha256(p) for p in self.root.rglob('*') if p.is_file()}
        args=r.parse_args(['--project-root',str(self.root)])
        with patch.object(t,'load_backends',return_value=(SimpleNamespace(type='cpu'),dict(torch='MOCK',lightgbm='MOCK'))),\
             patch.object(d,'extract_features',fake_extract),patch.object(t,'fit_group',fake_fit),\
             patch.object(t,'predict_group',fake_predict),contextlib.redirect_stdout(io.StringIO()):
            r.run(args)
            self.assertEqual(events,[('prepare',None)]+[('fit',g) for g in t.GROUPS]+[('test',g) for g in t.GROUPS])
            for split in ('val','test'):
                selected=pd.read_csv(out/f'{split}_comparison.csv')
                self.assertEqual(len(selected),4)
                self.assertTrue(selected.threshold.eq(.3).all())
                self.assertTrue(set(t.METRICS).issubset(selected.columns))
                self.assertEqual(len(pd.read_csv(out/f'{split}_all_thresholds.csv')),12)
                for name,th in [('05',.5),('07',.7)]:
                    self.assertTrue(pd.read_csv(out/f'{split}_fixed_threshold{name}.csv').threshold.eq(th).all())
            self.assertEqual(len(pd.read_csv(out/'contrast_summary.csv')),24)
            self.assertEqual(read_json(out/'run_manifest.json')['status'],'complete')
            args.resume=True;r.run(args)
            self.assertEqual(sum(e[0]=='fit' for e in events),4)
            self.assertEqual(sum(e[0]=='prepare' for e in events),1)
            self.assertEqual(sum(e[0]=='test' for e in events),8)
            # Every original dataset, baseline checkpoint, scaler and score stays unchanged.
            for path,digest in before.items():self.assertEqual(sha256(path),digest)
            prep=read_json(out/'run_manifest.json')['preparation']
            hp=out/prep['folder']/'train-pooled.npy';np.save(hp,np.zeros((8,2048),dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'changed'):r.run(args)


class MetricTests(unittest.TestCase):
    def test_strict_threshold_and_first_tie(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.9,.2,.8])
        self.assertEqual(choose_threshold(y,p),.3)
        metrics=classification_metrics(y,np.array([.5,.5,.6,.9]),.5)
        self.assertEqual((metrics['TN'],metrics['FP'],metrics['FN'],metrics['TP']),(1,1,1,1))
        for bad in (np.array([.2,.3,np.nan,.5]),np.array([.2,.3,1.1,.5])):
            with self.assertRaises(ValueError):choose_threshold(y,bad)

    def test_weighted_loss_mean_over_n(self):
        y=np.array([0.,1.,1.]);p=np.array([.2,.7,.8]);spw=.5
        expected=-(np.log(.8)+.5*np.log(.7)+.5*np.log(.8))/3
        self.assertAlmostEqual(t.weighted_bce(y,p,spw),expected)
        # Denominator sum(weights) would give a different, incorrect selection metric.
        self.assertNotAlmostEqual(t.weighted_bce(y,p,spw),expected*3/2)


if __name__ == '__main__':
    unittest.main()
