#!/usr/bin/env python3
"""Create/check the exact Python package environment required by TITAN/Deep-B3.

Run from a machine with Python >=3.8 and Conda installed:
  python build_titan_bbb_env.py
  python build_titan_bbb_env.py --name titan_bbb_new
  python build_titan_bbb_env.py --name titan_bbb --check-only
  python build_titan_bbb_env.py --dry-run

Linux x86_64 + NVIDIA GPU is the pipeline target. No driver installer is run.
An existing environment is checked, never automatically changed. Resume a
partially completed environment created by THIS script with --resume-build.
ChemBERTa and ResNet50 pretrained weights are separate model assets.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, shlex, shutil, subprocess, sys
from pathlib import Path

PYTHON_VERSION = '3.10.18'
PYTORCH_INDEX = 'https://download.pytorch.org/whl/cu118'
PYPI_INDEX = 'https://pypi.org/simple'
EXPECTED = {'numpy': '2.2.6', 'pandas': '2.3.2', 'scipy': '1.15.3', 'scikit-learn': '1.7.2', 'lightgbm': '4.7.0', 'rdkit': '2025.3.6', 'joblib': '1.6.0', 'openpyxl': '3.1.5', 'optuna': '4.5.0', 'torch': '2.7.1+cu118', 'torchvision': '0.22.1+cu118', 'transformers': '4.56.2', 'tokenizers': '0.22.2', 'huggingface-hub': '0.36.2', 'Pillow': '12.3.0', 'click': '8.3.0', 'tqdm': '4.67.1', 'safetensors': '0.8.0'}
DEPENDENCY_CONSTRAINTS = {'alembic': '1.20.1', 'certifi': '2026.7.22', 'charset-normalizer': '3.5.2', 'click': '8.3.0', 'cloudpickle': '3.1.2', 'colorlog': '6.12.0', 'et_xmlfile': '2.0.0', 'filelock': '4.1.0', 'fsspec': '2026.9.0', 'greenlet': '3.5.6', 'hf-xet': '1.7.0', 'huggingface_hub': '0.36.2', 'idna': '3.20', 'Jinja2': '3.1.6', 'joblib': '1.6.0', 'lightgbm': '4.7.0', 'Mako': '1.4.3', 'MarkupSafe': '3.0.4', 'mpmath': '1.3.0', 'narwhals': '2.27.1', 'networkx': '3.4.2', 'numpy': '2.2.6', 'nvidia-cublas-cu11': '11.11.3.6', 'nvidia-cuda-cupti-cu11': '11.8.87', 'nvidia-cuda-nvrtc-cu11': '11.8.89', 'nvidia-cuda-runtime-cu11': '11.8.89', 'nvidia-cudnn-cu11': '9.1.0.70', 'nvidia-cufft-cu11': '10.9.0.58', 'nvidia-curand-cu11': '10.3.0.86', 'nvidia-cusolver-cu11': '11.4.1.48', 'nvidia-cusparse-cu11': '11.7.5.86', 'nvidia-nccl-cu11': '2.21.5', 'nvidia-nvtx-cu11': '11.8.86', 'openpyxl': '3.1.5', 'optuna': '4.5.0', 'packaging': '26.3', 'pandas': '2.3.2', 'pillow': '12.3.0', 'pip': '25.2', 'python-dateutil': '2.9.0.post0', 'pytz': '2026.5', 'PyYAML': '6.0.3', 'rdkit': '2025.3.6', 'regex': '2026.9.29', 'requests': '2.34.2', 'safetensors': '0.8.0', 'scikit-learn': '1.7.2', 'scipy': '1.15.3', 'setuptools': '84.0.0', 'six': '1.17.0', 'SQLAlchemy': '2.0.55', 'sympy': '1.14.0', 'threadpoolctl': '3.7.0', 'tokenizers': '0.22.2', 'tomli': '2.5.0', 'torch': '2.7.1+cu118', 'torchaudio': '2.7.1+cu118', 'torchvision': '0.22.1+cu118', 'tqdm': '4.67.1', 'transformers': '4.56.2', 'triton': '3.3.1', 'typing_extensions': '4.16.0', 'tzdata': '2026.5', 'urllib3': '2.8.0', 'wheel': '0.48.0'}
TORCH_PACKAGES = ('torch', 'torchvision')
INSTALLER_VERSION = 1


def lock_digest():
    payload = {'python': PYTHON_VERSION, 'packages': EXPECTED,
               'constraints': DEPENDENCY_CONSTRAINTS, 'installer_version': INSTALLER_VERSION}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


def run(command, capture=False):
    print('\nRUN:',shlex.join([str(x) for x in command]),flush=True)
    return subprocess.run([str(x) for x in command],check=True,text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None)


def conda_prefix(conda,name):
    data=json.loads(run([conda,'env','list','--json'],capture=True).stdout)
    matches=[Path(x).resolve() for x in data['envs'] if Path(x).name==name]
    if len(matches)>1:raise RuntimeError('Multiple Conda prefixes share this name; choose a unique --name')
    return matches[0] if matches else None


def target_command(conda,name,*args):
    return [conda,'run','--no-capture-output','-n',name,'python',*map(str,args)]


def verify_source():
    # Executed by the TARGET interpreter, not the bootstrap/base Python.
    return '''import importlib,importlib.metadata,json,platform,sys,tempfile,traceback
from pathlib import Path
EXPECTED = '''+repr(EXPECTED)+'''
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
path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if report['pipeline_ready'] else 2)
'''


def verify(conda,name,work,current=False):
    work.mkdir(parents=True,exist_ok=True)
    checker=work/'verify_environment.py';checker.write_text(verify_source(),encoding='utf-8')
    report=work/'environment_check.json'
    command=[sys.executable,str(checker),str(report)] if current else target_command(conda,name,checker,report)
    checked=subprocess.run(command,check=False)
    # Dependency metadata consistency is checked separately as well.
    pip_cmd=[sys.executable,'-m','pip','check'] if current else target_command(conda,name,'-m','pip','check')
    dependency_check=subprocess.run(pip_cmd,check=False)
    if checked.returncode or dependency_check.returncode:
        print('\nEnvironment check failed. Details:',report,flush=True)
        return 2
    print('\nEnvironment ready. Activate with: conda activate '+name,flush=True)
    print('Model assets remain separate: bundle ChemBERTa + Torch cached ResNet50.',flush=True)
    return 0


def show_plan(name,work):
    print('Target: Linux x86_64, Python '+PYTHON_VERSION+', Conda environment '+name)
    print('Conda: python='+PYTHON_VERSION+', pip=25.2, libgcc-ng, libstdcxx-ng, libgomp (conda-forge)')
    print('PyTorch wheel index:',PYTORCH_INDEX)
    print('Other wheels/dependencies:',PYPI_INDEX)
    for name,version in EXPECTED.items():print('  '+name+'=='+version)
    print('Recorded transitive versions are supplied as pip constraints.')
    print('An existing environment is only checked; no package installation occurs there.')
    print('Build log/verification directory:',work)
    print('No environment creation or package installation was performed by this plan.')


def install(conda,name,work):
    work.mkdir(parents=True,exist_ok=True)
    constraints=work/'recorded_constraints.txt'
    constraints.write_text('\n'.join(name+'=='+version for name,version in DEPENDENCY_CONSTRAINTS.items())+'\n')
    requirements=work/'requirements_pipeline.txt'
    requirements.write_text('\n'.join(name+'=='+version for name,version in EXPECTED.items() if name not in TORCH_PACKAGES)+'\n')
    wheels=work/'pytorch_wheels';wheels.mkdir(exist_ok=True)
    # Fetch only the two pinned local-version CUDA wheels from the official index.
    run(target_command(conda,name,'-m','pip','--isolated','download','--only-binary=:all:','--no-deps',
        '--index-url',PYTORCH_INDEX,'--dest',wheels,*[p+'=='+EXPECTED[p] for p in TORCH_PACKAGES]))
    selected=[]
    for package in TORCH_PACKAGES:
        matches=sorted(wheels.glob(package+'-'+EXPECTED[package]+'-*.whl'))
        if len(matches)!=1:raise RuntimeError('Expected one exact CUDA wheel for '+package+', found '+str(len(matches)))
        selected.append(matches[0])
    # One resolver transaction pins both CUDA wheels and the full pipeline core.
    run(target_command(conda,name,'-m','pip','--isolated','install','--only-binary=:all:',
        '--index-url',PYPI_INDEX,'--constraint',constraints,'--requirement',requirements,*selected))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name',default='titan_bbb',help='Conda environment name; existing names are checked without modifying them')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true',help='only check an existing named Conda environment')
    mode.add_argument('--check-current',action='store_true',help='only check the interpreter currently running this script')
    mode.add_argument('--resume-build',action='store_true',help='resume only an unfinished environment created by this exact installer lock')
    parser.add_argument('--dry-run',action='store_true',help='show the plan; no commands run and no files written')
    parser.add_argument('--list-packages',action='store_true',help='print required pinned packages and exit')
    parser.add_argument('--work-dir',type=Path,help='download cache and reports; default ./titan_bbb_env_build/ENV_NAME')
    parser.add_argument('--conda',help='path to conda executable when not found in CONDA_EXE/PATH')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,100}',args.name):parser.error('Use a simple environment name, not a path')
    work=(args.work_dir or Path.cwd()/'titan_bbb_env_build'/args.name).expanduser().resolve()
    if args.dry_run or args.list_packages:show_plan(args.name,work);return 0
    if args.check_current:return verify(None,args.name,work,current=True)
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        raise RuntimeError('This recorded pipeline targets Linux x86_64 with NVIDIA CUDA; no silent CPU/platform substitution')
    conda=args.conda or os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda:raise RuntimeError('Conda was not found. Run on your Miniconda/Anaconda server, or pass --conda /path/to/conda')
    prefix=conda_prefix(conda,args.name)
    marker_name='titan_bbb_environment_builder.json'
    if prefix is not None:
        if not args.resume_build:
            print('Environment already exists; checking only:',prefix,flush=True)
            return verify(conda,args.name,work)
        marker=prefix/marker_name
        if not marker.is_file() or json.loads(marker.read_text()).get('lock_sha256')!=lock_digest():
            raise RuntimeError('--resume-build is allowed only for an unfinished environment created by this installer lock; use a new --name')
        if json.loads(marker.read_text()).get('installation_complete'):
            return verify(conda,args.name,work)
    else:
        if args.check_only or args.resume_build:raise RuntimeError('Named environment does not exist: '+args.name)
        if args.name=='base':raise RuntimeError('Choose a new environment name, not base')
        run([conda,'create','-y','-n',args.name,'--override-channels','-c','conda-forge',
            'python='+PYTHON_VERSION,'pip=25.2','libgcc-ng','libstdcxx-ng','libgomp'])
        prefix=conda_prefix(conda,args.name)
        if prefix is None:raise RuntimeError('Conda creation finished without a discoverable target prefix')
        write_json(prefix/marker_name,{'lock_sha256':lock_digest(),'installation_complete':False})
    install(conda,args.name,work)
    write_json(prefix/marker_name,{'lock_sha256':lock_digest(),'installation_complete':True})
    return verify(conda,args.name,work)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,OSError,subprocess.CalledProcessError) as error:
        print('\nEnvironment build stopped:',error,file=sys.stderr)
        print('No alternate versions were selected. For an interrupted build created by this script, retry with --resume-build.',file=sys.stderr)
        raise SystemExit(1)
