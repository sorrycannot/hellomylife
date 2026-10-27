"""Execute actual make_head() with a NumPy-backed forward API, not Torch training."""
import contextlib
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from titan_classifier_comparison import training as t


class Tensor:
    def __init__(self,value):self.v=np.asarray(value,dtype=np.float32)
    def __getitem__(self,key):return Tensor(self.v[key])
    def copy_(self,other):self.v[...]=other.v;return self
    def zero_(self):self.v[...]=0;return self


class Linear:
    def __init__(self,inputs,outputs):
        limit=1/math.sqrt(inputs)
        self.weight=Tensor(np.random.uniform(-limit,limit,(outputs,inputs)))
        self.bias=Tensor(np.random.uniform(-limit,limit,outputs))
    def __call__(self,x):return Tensor(x.v@self.weight.v.T+self.bias.v)


class ReLU:
    def __call__(self,x):return Tensor(np.maximum(x.v,0))


class Dropout:
    def __init__(self,p):self.p=p
    def __call__(self,x):return x  # Evaluation mode only.


class Sequential:
    def __init__(self,*layers):self.layers=list(layers)
    def __getitem__(self,i):return self.layers[i]
    def __setitem__(self,i,v):self.layers[i]=v
    def __call__(self,x):
        for layer in self.layers:x=layer(x)
        return x


class HeadMathTests(unittest.TestCase):
    def test_common_weights_rng_zero_columns_and_effect(self):
        fake_torch=SimpleNamespace(get_rng_state=np.random.get_state,set_rng_state=np.random.set_state,
                                   no_grad=contextlib.nullcontext)
        fake_nn=SimpleNamespace(Sequential=Sequential,Linear=Linear,ReLU=ReLU,Dropout=Dropout)
        with patch.object(t,'torch',fake_torch,create=True),patch.object(t,'nn',fake_nn,create=True):
            for width in (12,2048):
                np.random.seed(42);a=t.make_head(False,width,width);next_a=np.random.random()
                np.random.seed(42);b=t.make_head(True,width,width);next_b=np.random.random()
                self.assertEqual(next_a,next_b)
                np.testing.assert_array_equal(a[0].weight.v,b[0].weight.v[:,:width])
                np.testing.assert_array_equal(a[0].bias.v,b[0].bias.v)
                np.testing.assert_array_equal(a[3].weight.v,b[3].weight.v)
                self.assertFalse(b[0].weight.v[:,width:].any())
                x=np.sin(np.arange(4*width)*.13).reshape(4,width).astype(np.float32)
                m=np.array([[.1,.2,.9],[.8,.3,.2],[.2,.7,.6],[.8,.6,.4]],dtype=np.float32)
                xp=np.concatenate([x,m],axis=1)
                np.testing.assert_allclose(a(Tensor(x)).v,b(Tensor(xp)).v,rtol=1e-5,atol=1e-6)
                b[0].weight.v[:,width:]=.1
                delta=b(Tensor(xp)).v-b(Tensor(np.concatenate([x,np.zeros_like(m)],axis=1))).v
                self.assertGreater(float(np.max(np.abs(delta))),1e-6)


if __name__=='__main__':unittest.main()
