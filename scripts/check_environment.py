#!/usr/bin/env python3
import importlib,importlib.metadata,json,platform,sys,tempfile,traceback
from pathlib import Path
EXPECTED = {'numpy': '2.2.6', 'pandas': '2.3.2', 'scipy': '1.15.3', 'scikit-learn': '1.7.2', 'lightgbm': '4.7.0', 'rdkit': '2025.3.6', 'joblib': '1.6.0', 'openpyxl': '3.1.5', 'optuna': '4.5.0', 'torch': '2.7.1+cu118', 'torchvision': '0.22.1+cu118', 'transformers': '4.56.2', 'tokenizers': '0.22.2', 'huggingface-hub': '0.36.2', 'Pillow': '12.3.0', 'click': '8.3.0', 'tqdm': '4.67.1', 'safetensors': '0.8.0'}
report={'python':platform.python_version(),'executable':sys.executable,'versions':{},'errors':[],
        'smoke_tests':{},'cuda_available':False,'pipeline_ready':False}
if report['python']!='3.10.18':report['errors'].append('Python must be 3.10.18')
for name,wanted in EXPECTED.items():
    try:actual=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:actual='MISSING'
    report['versions'][name]={'expected':wanted,'actual':actual,'matches':actual==wanted}
    if actual!=wanted:report['errors'].append(name+': expected '+wanted+', actual '+actual)
def check(name,action):
    try:
        detail=action()
        report['smoke_tests'][name]={'ok':True,'detail':detail}
    except Exception:
        detail=traceback.format_exc()
        report['smoke_tests'][name]={'ok':False,'detail':detail}
        report['errors'].append(name+': '+detail.splitlines()[-1])
for module in ['numpy','pandas','scipy','sklearn','lightgbm','rdkit','joblib','openpyxl','optuna',
               'torch','torchvision','transformers','tokenizers','huggingface_hub','PIL','click','tqdm','safetensors']:
    check('import_'+module,lambda m=module: str(importlib.import_module(m).__file__))
def chemistry():
    from rdkit import Chem
    from rdkit.Chem import Draw,rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol=Chem.MolFromSmiles('CCO')
    assert mol is not None and Chem.MolToInchiKey(mol)
    image=Draw.MolToImage(mol,size=(100,100));assert image.size==(100,100)
    assert rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol).GetNumBits()==2048
    assert rdMolStandardize.FragmentParent(mol) is not None
    return 'RDKit parsing, InChI, image rendering, Morgan and standardization work'
def lgb():
    import numpy as np,lightgbm as lgb
    x=np.arange(128,dtype=np.float32).reshape(32,4);y=np.tile([0,1],16)
    model=lgb.train({'objective':'binary','verbosity':-1,'num_threads':1,'seed':42,
        'min_data_in_leaf':2,'feature_pre_filter':False},lgb.Dataset(x,label=y),num_boost_round=2)
    assert np.isfinite(model.predict(x)).all()
    return 'Native LightGBM binary training works'
def vision():
    import torch,torchvision
    boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,2.,2.]])
    keep=torchvision.ops.nms(boxes,torch.tensor([.9,.8]),.5)
    assert keep.numel()>0
    return 'torchvision compiled operators work'
def text():
    from transformers import AutoModel,RobertaConfig
    model=AutoModel.from_config(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
        num_attention_heads=2,intermediate_size=32))
    import torch
    with torch.no_grad():value=model(input_ids=torch.tensor([[0,3,2]])).last_hidden_state
    assert tuple(value.shape)==(1,3,16)
    return 'Transformers Roberta forward works; no model weights downloaded'
def excel():
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'test.xlsx';pd.DataFrame({'value':[1]}).to_excel(path,index=False)
        assert pd.read_excel(path).iloc[0,0]==1
    return 'Excel read/write works'
def cuda():
    import torch
    report['torch_cuda_runtime']=torch.version.cuda
    report['cuda_available']=torch.cuda.is_available()
    if torch.version.cuda!='11.8':raise RuntimeError('Expected Torch CUDA runtime 11.8')
    if not report['cuda_available']:raise RuntimeError('CUDA unavailable: check NVIDIA driver, GPU assignment/container access')
    x=torch.ones(16,device='cuda',requires_grad=True);(x*x).sum().backward();torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    report['gpu_name']=torch.cuda.get_device_name(0)
    return 'CUDA allocation, kernel and backward work'
for name,action in [('chemistry',chemistry),('lightgbm_native',lgb),('torchvision_ops',vision),
                    ('transformers_roberta',text),('excel',excel),('cuda',cuda)]:check(name,action)
report['pipeline_ready']=not report['errors']
path=Path(sys.argv[1] if len(sys.argv)>1 else "environment_check.json");path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if report['pipeline_ready'] else 2)
