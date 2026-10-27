import unittest
import numpy as np
from priority_bayes.protocol import (GRID,ANCHORS,initial_plan,pair_weights,build_weights,
    metrics,choose_threshold,choose_best,next_pair,log_expected_improvement)


def record(pair,auc):
    return {'indices':list(pair),'validation_metrics':{'fixed_0p5':{'AUROC':auc}}}


class ProtocolTests(unittest.TestCase):
    def test_grid_and_random_startup(self):
        self.assertEqual(len(GRID),1681)
        self.assertEqual(pair_weights((0,40)),(1.,3.))
        self.assertEqual(pair_weights((1,10)),(1.05,1.5))
        p=initial_plan();self.assertEqual(p[:4],list(ANCHORS))
        self.assertEqual(len(p),40);self.assertEqual(len(set(p)),40)
        self.assertEqual(p,initial_plan())

    def test_both_labels_same_rule_and_overlap_is_max(self):
        w=build_weights([0,1,0,1],[0,0,1,1],(10,30))
        np.testing.assert_array_equal(w,[1,1.5,2.5,2.5])

    def test_auroc_only_and_conservative_exact_tie(self):
        a=record((40,40),.90);b=record((0,0),.89)
        a['validation_metrics']['fixed_0p5']['BACC']=.1
        b['validation_metrics']['fixed_0p5']['BACC']=.99
        self.assertIs(choose_best([a,b]),a)
        c=record((0,0),.90)
        self.assertIs(choose_best([a,c]),c)

    def test_threshold_strict_greater_and_grid(self):
        y=np.array([0,0,1,1]);p=np.array([.05,.5,.5,.95])
        m=metrics(y,p,.5)
        self.assertEqual((m['TN'],m['FP'],m['FN'],m['TP']),(2,0,1,1))
        t=choose_threshold(y,p)
        self.assertIn(t,(.1,.3,.5,.7,.9))
        self.assertEqual(metrics(y,p,.1)['AUROC'],metrics(y,p,.9)['AUROC'])
        self.assertIsNone(metrics(np.array([1,1]),np.array([.2,.8]),.5)['AUROC'])

    def test_log_ei_extreme_tail_accuracy_without_underflow(self):
        from scipy.special import ndtr
        from scipy.integrate import quad
        z=np.array([-5.,-1.,0.,1.,5.])
        expected=z*ndtr(z)+np.exp(-z*z/2)/np.sqrt(2*np.pi)
        np.testing.assert_allclose(np.exp(log_expected_improvement(z,np.ones_like(z))),expected,rtol=1e-9)
        for a in (10.,40.,1000.):
            integral=quad(lambda v:v*np.exp(-v-v*v/(2*a*a)),0,np.inf)[0]
            expected=-a*a/2-.5*np.log(2*np.pi)-2*np.log(a)+np.log(integral)
            got=log_expected_improvement(np.array([-a]),np.ones(1))[0]
            self.assertTrue(np.isfinite(got))
            self.assertAlmostEqual(got,expected,places=6)

    def test_gp_selection_is_unseen_deterministic_and_resume_equivalent(self):
        rows=[]
        for pair in initial_plan():
            x,y=pair_weights(pair);value=.9-.015*((x-1.8)**2+(y-2.2)**2)
            rows.append(record(pair,value))
        seen={tuple(r['indices']) for r in rows}
        first,meta=next_pair(rows)
        again,_=next_pair(list(rows))
        self.assertEqual(first,again);self.assertNotIn(first,seen)
        self.assertTrue(meta['method'].startswith('GP_'))
        self.assertTrue(np.isfinite(meta['expected_improvement']))
        for _ in range(3):
            pair,_=next_pair(rows)
            self.assertNotIn(pair,seen);seen.add(pair)
            x,y=pair_weights(pair)
            rows.append(record(pair,.9-.015*((x-1.8)**2+(y-2.2)**2)))
        self.assertEqual(len(seen),43)


if __name__=='__main__':unittest.main()
