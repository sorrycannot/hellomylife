#!/usr/bin/env python3
"""Fresh raw-data rebuild, or separately labelled exact archived-model replay.

Rebuild: python run_pipeline.py --assets-root /path/to/original/TITAN-BBB-main --output /new/run
Replay:  python run_pipeline.py --mode replay --assets-root /path/to/original/TITAN-BBB-main --output /new/replay
Resume: repeat the identical command with --resume. Historical output differences
are reported by default; --strict-history makes them fatal. Data integrity,
alignment and current-run checkpoint checks always remain mandatory.
"""
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata, importlib.util, json, os
from pathlib import Path
import platform, shutil, subprocess, sys, time
from types import SimpleNamespace

BUNDLE=Path(__file__).resolve().parent
REF=BUNDLE/'reference'
SPLITS=('train','val','test')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def check(p,digest):
 p=Path(p)
 if not p.is_file() or sha(p)!=digest:raise ValueError(f'Missing or different historical bytes: {p}')
 return p

def history_compare(a,name,actual,expected):
 # Only comparisons with PRIOR RUN outputs go through here. Never use this
 # helper for input integrity, alignment, leakage or current checkpoint locks.
 # JSON normalization makes this stable across resume (e.g. tuple/list weights).
 actual=json.loads(json.dumps(actual,allow_nan=False))
 expected=json.loads(json.dumps(expected,allow_nan=False))
 matched=actual==expected
 target=a.output/'historical_comparison.json'
 report=read(target) if target.is_file() else dict(comparisons={})
 report['policy']='strict' if a.strict_history else 'report_only'
 report['comparisons'][name]=dict(matches_history=matched,actual=actual,historical=expected)
 report['all_recorded_comparisons_match']=all(x['matches_history'] for x in report['comparisons'].values())
 write(target,report)
 if not matched:
  print('HISTORICAL DIFFERENCE:',name,'(see historical_comparison.json)',flush=True)
  if a.strict_history:raise ValueError('Strict historical comparison failed: '+name)
 return matched

def history_file(a,path,digest):
 path=Path(path)
 if not path.is_file():raise FileNotFoundError(f'Current-run output missing: {path}')
 history_compare(a,str(path.relative_to(a.output)),sha(path),digest)
 return path
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def call(args,cwd=None,allowed=(0,)):
 print('RUN:', ' '.join(map(str,args)),flush=True)
 p=subprocess.run([str(x) for x in args],cwd=cwd,check=False)
 if p.returncode not in allowed:raise RuntimeError(f'Command exited {p.returncode}: {args}')

def versions():
 observed=read(REF/'environment_observed.json')['packages'];actual={};bad=[]
 for name,wanted in observed.items():
  try:value=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:value='MISSING'
  actual[name]=value
  if value!=wanted:bad.append(f'{name}: current={value}, recorded={wanted}')
 if platform.python_version()!='3.10.18':bad.append('Python must be 3.10.18')
 if bad:raise RuntimeError('Use the original titan_bbb environment; do not silently upgrade:\n'+'\n'.join(bad))
 return actual

def initialize(a):
 # Validate packaged bytes before importing any scientific source.
 manifest=read(BUNDLE/'bundle_sha256.json')
 for name,digest in manifest.items():check(BUNDLE/name,digest)
 a.assets_root=a.assets_root.expanduser().resolve();a.output=a.output.expanduser().resolve()
 if a.output==a.assets_root or a.output.is_relative_to(a.assets_root):
  raise ValueError('--output must be outside the original project to protect all old results')
 if a.output==BUNDLE or BUNDLE.is_relative_to(a.output):raise ValueError('Output must not contain the code bundle')
 if a.mode!='replay' and a.structure_source=='discover' and a.anchor_policy=='archived':raise ValueError('Newly discovered rules require refit; archived anchors require --structure-source fixed')
 if not 1<=a.search_startup<=a.search_budget<=1681:raise ValueError('Require 1 <= search-startup <= search-budget <= 1681')
 env=versions()
 reuse=None
 if a.reuse_upstream_from is not None:
  reuse=a.reuse_upstream_from.expanduser().resolve(strict=True)
  if a.output==reuse or a.output.is_relative_to(reuse) or reuse.is_relative_to(a.output):raise ValueError('Use a separate new output directory for upstream reuse')
  source_lock=read(reuse/'pipeline_lock.json')
  if source_lock.get('mode')!='rebuild' or source_lock.get('versions')!=env or source_lock.get('device')!=a.device:
   raise ValueError('Upstream reuse requires a completed rebuild in the same environment/device')
  # Only selection code changed. Numerical upstream implementation must match.
  source_paths=list((BUNDLE/'src').glob('generate_*.py'))+[BUNDLE/'src/mechanism_common.py',BUNDLE/'src/train_auxiliary_titan.py']
  for folder in ('utils','titan_attention_seed2022','titan_classifier_comparison'):
   source_paths.extend((BUNDLE/'src'/folder).glob('*.py'))
  source_paths.extend((BUNDLE/'s2').glob('*.py'))
  for path in source_paths:check(reuse/path.relative_to(BUNDLE),sha(path))
  for split in SPLITS:check(reuse/'dataset'/f'{split}_classification.csv',sha(BUNDLE/'dataset'/f'{split}_classification.csv'))
  a.reuse_upstream_from=reuse
 payload=dict(mode=a.mode,anchor_policy=a.anchor_policy,structure_source=a.structure_source,search_budget=a.search_budget,search_startup=a.search_startup,strict_history=a.strict_history,assets_root=str(a.assets_root),device=a.device,
  reuse_upstream_from=str(reuse) if reuse else None,reuse_source_lock_sha256=sha(reuse/'pipeline_lock.json') if reuse else None,
  resnet_weights=str(a.resnet_weights.resolve()) if a.resnet_weights else None,
  versions=env,bundle_sha256=sha(BUNDLE/'bundle_sha256.json'))
 if a.output.exists():
  if not a.resume:raise FileExistsError('Output exists. Use the identical command with --resume.')
  if read(a.output/'pipeline_lock.json')!=payload:raise ValueError('Resume inputs/code/versions/device differ')
 else:
  a.output.mkdir(parents=True);write(a.output/'pipeline_lock.json',payload)
 return payload

def step(a,name,action,roots):
 out=a.output;done=out/'steps'/f'{name}.json'
 if done.exists():
  record=read(done)
  for path,digest in record['outputs'].items():check(out/path,digest)
  print('VERIFIED COMPLETED STEP:',name,flush=True);return
 started=time.monotonic();print('\nSTART STEP:',name,flush=True)
 action()
 outputs={str(p.relative_to(out)):sha(p) for r in roots for p in ([r] if r.is_file() else sorted(r.rglob('*')))
          if p.is_file() and '__pycache__' not in p.parts and p.name!='run.lock'}
 write(done,dict(step=name,seconds=time.monotonic()-started,outputs=outputs))

def prepare(a):
 out=a.output
 for name in ('src','dataset','s2','raw'):
  shutil.copytree(BUNDLE/name,out/name,dirs_exist_ok=True)
 for name in ('legacy_structure_review.json','structure_selection_config.json'):
  shutil.copyfile(BUNDLE/name,out/name)
 for split in SPLITS:
  expected=read(REF/'mechanism.json')['dataset_files'][split]['sha256']
  check(out/'dataset'/f'{split}_classification.csv',expected)
 # Reproduction starts from the bundled exact raw CSV, not a downloaded newer release.
 raw=out/'raw/Cornelissen_20220103.csv'
 before=read(REF/'raw_sources.json')
 check(raw,before['cornelissen_sha256'])
 write(out/'raw/input_check.json',dict(datasets='original bytes verified',cornelissen_sha256=sha(raw)))

def adopt_upstream(a,name,roots):
 source=a.reuse_upstream_from;record=read(source/'steps'/f'{name}.json')
 wanted={}
 for relative,digest in record['outputs'].items():
  path=Path(relative)
  if path.is_absolute() or '..' in path.parts or not any(path.is_relative_to(Path(r)) for r in roots):
   raise ValueError('Unsafe or unexpected upstream step path: '+relative)
  if (source/path).is_symlink():raise ValueError('Upstream artifact must be a real file: '+relative)
  check(source/path,digest);wanted[relative]=digest
 if not wanted:raise ValueError('No completed upstream outputs for '+name)
 # Only uncommitted destinations inside this new run can be replaced.
 for root in roots:
  target=a.output/root
  if target.exists():shutil.rmtree(target)
 for relative,digest in wanted.items():
  target=a.output/relative;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(source/relative,target);check(target,digest)
 write(a.output/'upstream_reuse'/f'{name}.json',dict(source=str(source),step=name,
  source_step_sha256=sha(source/'steps'/f'{name}.json'),files_verified=len(wanted),new_training_performed=False))
 print('REUSED VERIFIED UPSTREAM STEP:',name,flush=True)

def s2_and_exclusion(a):
 out=a.output;verify=out/'s2_verification';aux=out/'auxiliary_titan'
 for p in (verify,aux):
  if p.exists():shutil.rmtree(p) # Only this fresh run's uncommitted step outputs.
 call([sys.executable,out/'s2/verify_cornelissen_s2.py','--source',out/'raw/Cornelissen_20220103.csv',
       '--s2',out/'s2/S2_Cornelissen_Ingestion.py','--s9',out/'s2/S9_Stage1_Final_Refit.py',
       '--chem',out/'s2/chem.py','--output-dir',verify],allowed=(0,2))
 helper=module('_repro_s2_check',out/'s2/verify_cornelissen_s2.py')
 expected=helper.read_s9_expected(out/'s2/S9_Stage1_Final_Refit.py');report=[]
 for task in ('efflux','influx','pampa'):
  p=verify/'data_processed'/f'cornelissen_{task}.csv'
  # Original historical match was CRLF, no BOM, final newline. Only serialize;
  # do not reorder, relabel, or change chemical content to obtain a match.
  raw=p.read_bytes();canonical=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
  digest=hashlib.sha256(canonical).hexdigest()
  history_compare(a,'s2_CRLF_'+task,digest,expected[task][0])
  report.append(dict(task=task,actual_lf_sha256=sha(p),CRLF_sha256=digest,expected_sha256=expected[task][0]))
 write(verify/'CRLF_content_verification.json',report)
 call([sys.executable,out/'s2/deduplicate_auxiliary_titan.py','--source',out/'raw/Cornelissen_20220103.csv',
       '--titan-dir',out/'dataset','--chem',out/'s2/chem.py','--output-dir',aux])
 history=read(REF/'mechanism.json')
 for task,r in history['auxiliary_pools'].items():history_file(a,aux/f'cornelissen_{task}_titan_excluded.csv',r['sha256'])
 history_file(a,aux/'titan_identity_map.csv',history['identity_map']['sha256'])

def auxiliary(a):
 out=a.output
 for name in ('auxiliary_stage1','mechanism_features'):
  p=out/name
  if p.exists():shutil.rmtree(p)
 call([sys.executable,out/'src/train_auxiliary_titan.py','--data-dir',out/'auxiliary_titan',
       '--output-dir',out/'auxiliary_stage1','--num-threads','4'])
 actual=read(out/'auxiliary_stage1/locked_iterations.json')
 wanted={'efflux':598,'influx':711,'pampa':884}
 history_compare(a,'stage1_selected_rounds',{k:v['rounds'] for k,v in actual.items()},wanted)
 call([sys.executable,out/'src/generate_mechanism_features.py','--project-root',out,'--num-threads','4'])
 # Scores, not path/timestamp-bearing manifests, must reproduce exact old bytes.
 for split,r in read(REF/'mechanism.json')['splits'].items():
  history_file(a,out/'mechanism_features'/f'{split}-mechanism-classification.npy',r['scores']['sha256'])
  history_file(a,out/'mechanism_features'/f'{split}-mechanism-audit.csv',r['audit']['sha256'])

def modalities(a):
 import torch
 from torchvision import models
 out=a.output;features=out/'features';features.mkdir(exist_ok=True)
 if a.device!='cuda':raise ValueError('Historical feature extraction used cuda:0. Use --device cuda.')
 source=a.assets_root/'pretrained/ChemBERTa-100M-MLM'
 if not (source/'model.safetensors').is_file():raise FileNotFoundError(f'Keep original offline ChemBERTa files at {source}')
 resnet=a.resnet_weights or Path(torch.hub.get_dir())/'checkpoints/resnet50-0676ba61.pth'
 if not resnet.is_file():raise FileNotFoundError(f'Original offline ResNet50 weights missing: {resnet}; pass --resnet-weights')
 tab=module('_raw_tab',out/'src/generate_tabular_features.py')
 image=module('_raw_image',out/'src/generate_image_features.py')
 text=module('_raw_text',out/'src/generate_text_features.py')
 # Preserve original numerical functions; adapt only absolute file resolution.
 # Tabular/image original functions use cwd-relative ../dataset and ../features.
 previous=Path.cwd();os.chdir(out/'src')
 try:
  tab.main.callback('classification')
  def encoder(device):
   model=models.resnet50(weights=None)
   model.load_state_dict(torch.load(resnet,map_location='cpu',weights_only=True),strict=True)
   return torch.nn.Sequential(*list(model.children())[:-1]).eval().to(device)
  image.get_resnet50_encoder=encoder
  image.main.callback('classification')
  tokenizer=text.AutoTokenizer.from_pretrained(str(source),local_files_only=True)
  model=text.AutoModel.from_pretrained(str(source),output_hidden_states=True,local_files_only=True).to('cuda:0').eval()
  import pandas as pd
  for split in SPLITS:
   frame=pd.read_csv(out/'dataset'/f'{split}_classification.csv')
   text.process_data(frame,tokenizer,model,torch.device('cuda:0'),split,'classification')
 finally:os.chdir(previous)
 write(features/'pretrained_files.json',dict(chemberta={p.name:sha(p) for p in source.iterdir() if p.is_file()},resnet_path=str(resnet),resnet_sha256=sha(resnet)))
 expected=read(REF/'modalities.json')['definition']['data']['inputs']
 for split in SPLITS:
  for kind in ('tabular','image','text'):history_file(a,features/f'{split}-{kind}-classification.npy',expected[split][kind]['sha256'])

def fusion(a):
 import torch,joblib
 out=a.output;sys.path.insert(0,str(out/'src'))
 from titan_attention_seed2022 import trainer
 from titan_classifier_comparison import data as feature_source
 arrays,frames,scalers,_=trainer.load_inputs(out,out/'mechanism_features')
 trainer.load_dependencies(out)
 folder=out/'fusion';folder.mkdir(exist_ok=True)
 existing=folder/'baseline_seed42'
 if existing.exists():shutil.rmtree(existing)
 # No new smoke test/RNG/optimizer/seed behavior: execute the actual baseline trainer.
 baseline,config,lock=trainer.train_one('baseline',SimpleNamespace(seed=42,epochs=1000),
                                    {s:arrays[s] for s in ('train','val')},folder,torch.device(a.device))
 joblib.dump(scalers,folder/'modality_scalers.joblib')
 output=out/'pooled'
 if output.exists():shutil.rmtree(output)
 reference=dict(model_source=out/'src/utils/model.py',config=config,
                checkpoint=baseline/'best_model.pth',folder=baseline)
 report=feature_source.extract_features(reference,arrays,frames,output,torch.device(a.device),32)
 wanted=read(REF/'pooled.json')['files']
 for split in SPLITS:history_file(a,output/f'{split}-pooled.npy',wanted[split]['pooled']['sha256'])
 write(folder/'numeric_checkpoint_verification.json',dict(pooled_arrays_byte_identical=all(sha(output/f'{s}-pooled.npy')==wanted[s]['pooled']['sha256'] for s in SPLITS),
   checkpoint_sha256=sha(baseline/'best_model.pth'),original_checkpoint_sha256=read(REF/'pooled.json')['source_checkpoint']['sha256'],
   note='Current baseline validation predictions are checked internally. Historical H2048 differences are reported; strict-history makes them fatal.'))

def fresh_context(a,with_structures=True):
 import numpy as np,pandas as pd
 out=a.output;gm=module('_pipeline_grouped',out/'src/run_grouped_multiobjective_bayes.py')
 core=gm.core_modules(out);protocol=read(REF/'grouped_protocol.json');inputs=read(REF/'input_audit.json')
 gm.grouped_runtime();frames={};X={};original=dict(datasets={},selected_arrays={});files={}
 # Only training and validation files enter discovery and model selection.
 for split in ('train','val'):
  ds=out/'dataset'/f'{split}_classification.csv';check(ds,inputs['datasets'][split]['sha256'])
  h=out/'pooled'/f'{split}-pooled.npy';history_file(a,h,inputs['selected_arrays'][split]['H']['sha256'])
  score=out/'mechanism_features'/f'{split}-mechanism-audit.csv';history_file(a,score,inputs['selected_arrays'][split]['M']['sha256'])
  original['datasets'][split]=dict(path=str(ds),sha256=sha(ds))
  original['selected_arrays'][split]=dict(H=dict(path=str(h),sha256=sha(h)),
   M=dict(path=str(score),sha256=sha(score),columns=['p_efflux','p_influx','p_pampa'],
          smiles_column='original_smiles',mechanism_order=['efflux','influx','pampa']))
  for path in (ds,h,score):files[str(path)]=sha(path)
  frames[split]=pd.read_csv(ds,keep_default_na=False)
  m=pd.read_csv(score,keep_default_na=False)
  if m.original_smiles.tolist()!=frames[split].SMILES.tolist():raise ValueError('M3 SMILES row order changed')
  H=np.load(h,allow_pickle=False);M=m[['p_efflux','p_influx','p_pampa']].to_numpy(dtype=float)
  if H.shape!=(len(frames[split]),2048) or not np.isfinite(H).all():raise ValueError('Invalid current H2048 shape/values')
  if M.shape!=(len(frames[split]),3) or not np.isfinite(M).all() or ((M<0)|(M>1)).any():raise ValueError('Invalid current mechanism scores')
  X[split]=np.column_stack([H,M]).astype(np.float32)
  if not np.isfinite(X[split]).all():raise ValueError('Nonfinite float32 model input')
 ctx=dict(root=out,gm=gm,core=core,files=files,frames=frames,X=X,warm={},
  locked=dict(source=out,original=original),params=protocol['params'],max_rounds=protocol['max_rounds'],
  patience=protocol['patience'],class_weight=protocol['class_weight'])
 if not with_structures:return ctx
 discovered=a.structure_source=='discover'
 rules_path=out/'structure_discovery/reference_rules.json' if discovered else out/'src/titan_priority_bayes/priority_bayes/reference_rules.json'
 reference=read(rules_path);files[str(rules_path)]=sha(rules_path)
 if discovered:
  if reference.get('origin')!='generated_from_current_validation' or not reference.get('no_test_data_used'):
   raise ValueError('New discovery reference is missing its origin or test-exclusion declaration')
  complete=out/'structure_discovery/COMPLETE.json'
  for name,digest in read(complete)['files_sha256'].items():check(complete.parent/name,digest)
  files[str(complete)]=sha(complete)
  gm.CORE_RULES=tuple(reference['core_rules']);gm.RESERVE_RULES=tuple(reference['reserve_rules'])
  gm.EXPECTED_RESERVE_COVERAGE={rule:reference['expected_training_coverage'][rule] for rule in gm.RESERVE_RULES}
  # Empty tiers are valid: fix their inactive weight to 1, shrinking the grid.
  gm.GRID=tuple((i,j) for i in (range(41) if gm.CORE_RULES else (0,))
                         for j in (range(41) if gm.RESERVE_RULES else (0,)))
 matches,audit=core['chemistry'].prepare(reference,frames)
 if not audit['training_coverage_matches_reported_audit']:raise ValueError('Current rule training coverage changed')
 matcher=core['chemistry'].Matcher(reference);coverage=[];group_coverage=[]
 for split in ('train','val'):
  table=matches[split];eligible=table.eligibility_status.isin(['primary_representative','consistent_duplicate'])
  hits=[set(matcher.graph(s)['hits']) if ok else set() for s,ok in zip(frames[split].SMILES,eligible)]
  for rule in gm.CORE_RULES+gm.RESERVE_RULES:table[rule]=[rule in h for h in hits]
  table['core_group']=table[list(gm.CORE_RULES)].any(axis=1)
  table['reserve_group']=table[list(gm.RESERVE_RULES)].any(axis=1)
  table['eligible_for_structure_weight']=eligible
  for rule in gm.CORE_RULES+gm.RESERVE_RULES:
   primary=table[rule].to_numpy() & table.primary_unit.to_numpy()
   coverage.append(dict(split=split,candidate=rule,units=int(primary.sum()),
    positive=int((primary & frames[split].label.eq(1).to_numpy()).sum()),negative=int((primary & frames[split].label.eq(0).to_numpy()).sum()),
    connectivity_groups=int(table.loc[primary,'connectivity_group'].nunique())))
  c,r=table.core_group.to_numpy(),table.reserve_group.to_numpy()
  for name,mask in dict(core=c,reserve=r,intersection=c&r,union=c|r,reserve_only=r&~c,neither=~(c|r)).items():
   primary=mask & table.primary_unit.to_numpy()
   group_coverage.append(dict(split=split,cohort=name,raw_rows=int(mask.sum()),units=int(primary.sum())))
 coverage=pd.DataFrame(coverage,columns=['split','candidate','units','positive','negative','connectivity_groups'])
 train=coverage.loc[coverage.split.eq('train')].set_index('candidate')
 for rule,expected in gm.EXPECTED_RESERVE_COVERAGE.items():
  if any(int(train.loc[rule,k])!=v for k,v in expected.items()):raise ValueError('Reserve coverage changed: '+rule)
 path=out/'stage2';path.mkdir(exist_ok=True);(path/'figures').mkdir(exist_ok=True)
 budget=min(a.search_budget,len(gm.GRID));startup=min(a.search_startup,budget)
 signature=core['protocol'].object_sha(dict(pipeline_lock=read(out/'pipeline_lock.json'),files=files,
  params=protocol['params'],core_rules=gm.CORE_RULES,reserve_rules=gm.RESERVE_RULES,budget=budget,startup=startup))
 write(path/'group_definition.json',dict(source=a.structure_source,core_rules=gm.CORE_RULES,reserve_rules=gm.RESERVE_RULES,
  active_grid_size=len(gm.GRID),budget=budget,startup=startup,empty_group_weight=1))
 ctx.update(out=path,matches=matches,signature=signature,budget=budget,startup=startup,
  coverage=coverage,group_coverage=pd.DataFrame(group_coverage))
 return ctx

def structures(a):
 out=a.output;folder=out/'structure_discovery'
 if folder.exists():shutil.rmtree(folder)
 if a.structure_source=='fixed':
  folder.mkdir();write(folder/'summary.json',dict(source='fixed_historical_rules',new_discovery_performed=False));return
 ctx=fresh_context(a,with_structures=False)
 helper=module('_current_structure_discovery',out/'src/discover_structures.py')
 helper.discover(ctx,read(BUNDLE/'structure_selection_config.json'),folder)

def register_test_inputs(a,ctx):
 # Called only after the four validation choices and matched controls are locked.
 out=a.output;inputs=read(REF/'input_audit.json');split='test'
 ds=out/'dataset/test_classification.csv';check(ds,inputs['datasets'][split]['sha256'])
 h=out/'pooled/test-pooled.npy';history_file(a,h,inputs['selected_arrays'][split]['H']['sha256'])
 m=out/'mechanism_features/test-mechanism-audit.csv';history_file(a,m,inputs['selected_arrays'][split]['M']['sha256'])
 original=ctx['locked']['original'];original['datasets'][split]=dict(path=str(ds),sha256=sha(ds))
 original['selected_arrays'][split]=dict(H=dict(path=str(h),sha256=sha(h)),
  M=dict(path=str(m),sha256=sha(m),columns=['p_efflux','p_influx','p_pampa'],
    smiles_column='original_smiles',mechanism_order=['efflux','influx','pampa']))
 for path in (ds,h,m):ctx['files'][str(path)]=sha(path)

def stage2(a):
 import pandas as pd
 ctx=fresh_context(a);gm=ctx['gm'];out=ctx['out'];core=ctx['core']
 # Replay uses true saved anchor checkpoints in the historical search. A rebuild
 # refits all anchors; warm reuse is optional only with validated archived heads.
 # Preserve original anchors exactly when present, but label their reuse explicitly.
 anchor_manifest=read(REF/'anchor_checkpoints.json') if a.anchor_policy=='archived' else []
 for entry in anchor_manifest:
  if a.anchor_policy=='refit':continue
  p=a.assets_root/entry['relative_path']
  if not p.is_file():raise FileNotFoundError(f'Archived anchor missing: {p}. Do not silently mix anchor policies.')
  check(p,entry['sha256'])
  import lightgbm as lgb
  model=lgb.Booster(model_file=str(p));mode=entry['mode'];pair=tuple(entry['indices'])
  import numpy as np
  mask=None if mode=='H_only' else gm.MODEL_MASKS[mode]
  probability=model.predict(gm.make_model_input(ctx['X']['val'],mask))
  wanted=read(REF/'anchor_metrics.json')[entry['key']]
  for policy,threshold in wanted['thresholds'].items():
   actual=core['protocol'].metrics(ctx['frames']['val'].label.to_numpy(),probability,threshold)
   for k,v in actual.items():
    if v!=wanted['validation_metrics'][policy][k]:raise ValueError('Historical anchor validation differs')
  ctx['warm'][mode,pair]=dict(path=p,model=model,model_sha256=entry['sha256'])
 # Warm reuse is recorded, not described as refitting these ten old checkpoints.
 core['protocol'].write_json(out/'anchor_reuse.json',dict(reused=[dict(mode=m,indices=list(k)) for m,k in ctx['warm']],
  policy=a.anchor_policy,reason='Explicit policy: all anchors refitted, or all ten historical anchors validated and reused. No test score used.'))
 core['protocol'].write_json(out/'protocol_lock.json',dict(run_signature=ctx['signature'],params=ctx['params'],budget=ctx['budget'],startup=ctx['startup'],
  selection='Original Optuna4.5.0 MOTPE, Pareto plus three-metric gate and fixed S',max_rounds=ctx['max_rounds'],patience=ctx['patience'],core_rules=gm.CORE_RULES,reserve_rules=gm.RESERVE_RULES))
 histories={mode:gm.optimize_mode(ctx,mode,ctx['budget'],ctx['startup']) for mode in gm.MODEL_MASKS}
 registry,val=gm.select_and_lock(ctx,histories);gm.export_search_workbook(ctx)
 # Check the independently obtained validation selections BEFORE reading test outcomes.
 actual={mode:gm.pair_weights(tuple(v['selected']['indices'])) for mode,v in registry['comparisons'].items()}
 expected={mode:(read(REF/f'selection_{mode}.json')['core_weight'],read(REF/f'selection_{mode}.json')['reserve_weight']) for mode in gm.MODEL_MASKS}
 matches_history=history_compare(a,'validation_selected_weights',actual,expected)
 write(out/'historical_weight_comparison.json',dict(current_validation_selected_weights=actual,
  historical_weights=expected,matches_history=matches_history,comparison_only=not a.strict_history))
 register_test_inputs(a,ctx)
 gm.evaluate_grouped_test(ctx)
 ab=module('_pipeline_ablation',a.output/'src/run_four_weight_ablation.py')
 selections={mode:read(out/'selected_models'/mode/'selection_lock.json') for mode in gm.MODEL_MASKS}
 plan=ab.plan_from_locks(selections,gm);warm={}
 for group,conditions in registry['comparisons'].items():
  for cond in ('selected','H_only'):
   spec=conditions[cond];mode='H_only' if cond=='H_only' else group
   record=read(out/spec['record_path']);p=out/spec['model_path']
   import lightgbm as lgb
   warm[mode,tuple(spec['indices'])]=dict(path=p,model=lgb.Booster(model_file=str(p)),record=record,record_path=out/spec['record_path'])
 folder=a.output/'ablation';folder.mkdir(exist_ok=True)
 ac=dict(ctx,out=folder,gm=gm,warm=warm,source_registry=registry,plan=plan,
  weights_table=pd.DataFrame([dict(weight_group=m,core_weight=w[0],reserve_weight=w[1]) for m,w in actual.items()]))
 records=[ab.fit_arm(ac,spec) for spec in plan]
 locked,validation=ab.lock_validation(ac,records);ab.evaluate_test(ac,records,locked,validation)
 verify_results(a.output/'ablation/all_test_comparison.csv',strict=a.strict_history)

def verify_results(csv_path,strict=False):
 import pandas as pd
 actual=pd.read_csv(csv_path);expected=pd.read_csv(REF/'expected_all_test.csv')
 keys=['weight_group','experiment','threshold_policy'];metrics=['threshold','AUROC','AUPRC_AP','ACC','BACC','SN','SP','F1','MCC','FN','FP']
 metrics += [k for k in ('core_weight','reserve_weight','n','TN','TP') if k in expected.columns]
 if len(actual)!=72 or actual.duplicated(keys).any():raise ValueError('Expected 36 heads x two threshold policies, unique keys')
 # Complete current-run result schema remains mandatory even in report mode.
 import numpy as np
 if not set(keys+metrics).issubset(actual.columns):raise ValueError('Current results lack required columns')
 if not np.isfinite(actual[metrics].to_numpy(dtype=float)).all():raise ValueError('Current results contain nonfinite values')
 actual=actual.set_index(keys);expected=expected.set_index(keys);differences=[]
 for key,ref in expected.iterrows():
  if key not in actual.index:raise ValueError('Missing arm '+str(key))
  for k in metrics:
   if f'{float(actual.loc[key,k]):.6f}'!=f'{float(ref[k]):.6f}':differences.append(dict(key=list(key),metric=k,actual=float(actual.loc[key,k]),expected=float(ref[k])))
 if set(actual.index)!=set(expected.index):raise ValueError('Current result arm/policy keys are incomplete or unexpected')
 report=dict(all_72_rows_match_at_report_precision=not differences,differences=differences,
             policy='strict' if strict else 'report_only',
             precision='six decimals; does not imply model file or probability byte identity')
 write(Path(csv_path).parent/'reported_result_verification.json',report)
 if differences:
  if strict:raise ValueError(f'{len(differences)} reported values differ; see reported_result_verification.json')
  print(f'COMPLETED: 36 heads, 72 result rows; {len(differences)} historical differences reported.',flush=True)
 else:print('VERIFIED: all 36 heads, both threshold policies, all reported metrics match at six decimals.',flush=True)
 return report

def replay(a):
 """Native archived checkpoints + exact historical H/M; no refit/search claim."""
 import numpy as np,pandas as pd,lightgbm as lgb
 root=a.assets_root;out=a.output;rows=[];probability_map={}
 oldroot=Path(read(REF/'grouped_protocol.json')['project_root'])
 inputs=read(REF/'input_audit.json');X={};frames={}
 for split in ('val','test'):
  ds=check(root/'dataset'/f'{split}_classification.csv',inputs['datasets'][split]['sha256']);frames[split]=pd.read_csv(ds,keep_default_na=False)
  h=inputs['selected_arrays'][split]['H'];m=inputs['selected_arrays'][split]['M']
  hp=check(root/Path(h['path']).relative_to(oldroot),h['sha256']);mp=check(root/Path(m['path']).relative_to(oldroot),m['sha256'])
  mf=pd.read_csv(mp,keep_default_na=False)
  if mf.original_smiles.tolist()!=frames[split].SMILES.tolist():raise ValueError('Replay M3 alignment failed')
  X[split]=np.column_stack([np.load(hp,allow_pickle=False),mf[m['columns']].to_numpy()]).astype(np.float32)
 gm=module('_replay_grouped',BUNDLE/'src/run_grouped_multiobjective_bayes.py');core=gm.core_modules(BUNDLE)
 records=read(REF/'archived_ablation_records.json')
 # All 36 model hashes/dimensions and validation-selected thresholds checked before test prediction.
 models={}
 for key,record in records.items():
  spec=record['spec'];path=check(root/record['source_relative_model_path'],record['files_sha256']['model.txt'])
  model=lgb.Booster(model_file=str(path));mask=spec['mechanism_mask']
  if model.num_feature()!=spec['input_dim'] or model.current_iteration()!=record['best_iteration']:raise ValueError('Replay model contract failed')
  pv=np.asarray(model.predict(gm.make_model_input(X['val'],mask)),dtype=float)
  if core['protocol'].choose_threshold(frames['val'].label.to_numpy(),pv)!=record['thresholds']['validation_selected']:raise ValueError('Replay validation threshold failed')
  for policy,threshold in record['thresholds'].items():
   got=core['protocol'].metrics(frames['val'].label.to_numpy(),pv,threshold)
   if any(abs(got[k]-record['validation_metrics'][policy][k])>1e-12 for k in got):raise ValueError('Replay validation probabilities/metrics failed')
  models[key]=model
 write(out/'replay_validation_lock.json',dict(models={k:r['files_sha256']['model.txt'] for k,r in records.items()},test_prediction_started=False,mode='archived_model_replay; no training'))
 for key,record in records.items():
  spec=record['spec'];p=np.asarray(models[key].predict(gm.make_model_input(X['test'],spec['mechanism_mask'])),dtype=float)
  for policy,threshold in record['thresholds'].items():
   metric=core['protocol'].metrics(frames['test'].label.to_numpy(),p,threshold)
   rows.append(dict(weight_group=spec['weight_group'],experiment=spec['experiment'],threshold_policy=policy,
    core_weight=spec['core_weight'],reserve_weight=spec['reserve_weight'],**metric))
   folder=out/'predictions'/spec['weight_group']/spec['experiment']/policy
   f=gm.outcome_frame(frames['test'],p,threshold);core['report'].csv(folder/'test_all.csv',f)
   for label in ('TP','TN','FP','FN'):core['report'].csv(folder/f'test_{label}.csv',f.loc[f.outcome.eq(label)])
   gm.confusion_svg(folder/'confusion_matrix.svg',key,metric)
 table=pd.DataFrame(rows);core['report'].csv(out/'all_test_comparison.csv',table)
 core['report'].workbook(out/'results.xlsx',dict(all_test=table))
 ab=module('_replay_ablation',BUNDLE/'src/run_four_weight_ablation.py')
 for group in ab.EXPECTED_WEIGHTS:
  for policy in ab.POLICIES:
   ab.table_svg(out/f'{group}_{policy}.svg',table.loc[table.weight_group.eq(group)&table.threshold_policy.eq(policy)],
    f'TITAN test: {group}',f'Archived model replay | {policy} | seed=42')
 verify_results(out/'all_test_comparison.csv',strict=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--assets-root',default=BUNDLE,type=Path,help='asset root; defaults to this bundle; archived/replay require the historical project')
 p.add_argument('--output',required=True,type=Path,help='NEW directory OUTSIDE the original project')
 p.add_argument('--mode',choices=['rebuild','replay'],default='rebuild')
 p.add_argument('--anchor-policy',choices=['refit','archived'],default='refit',help='refit starts every Stage-2 head fresh; archived explicitly reuses the ten original validation-verified anchors')
 p.add_argument('--structure-source',choices=['discover','fixed'],default='discover',help='discover current validation candidates (default), or explicitly reuse historical rules')
 p.add_argument('--search-budget',type=int,default=200);p.add_argument('--search-startup',type=int,default=40)
 p.add_argument('--reuse-upstream-from',type=Path,help='copy verified completed exclusion/auxiliary/modalities/fusion steps into a new run; no GPU retraining of those steps')
 p.add_argument('--strict-history',action='store_true',help='stop on differences from historical outputs; default records differences and continues')
 p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--resnet-weights',type=Path)
 p.add_argument('--resume',action='store_true');p.add_argument('--stage',choices=['all','prepare','exclusion','auxiliary','modalities','fusion','structures','stage2'],default='all')
 a=p.parse_args();initialize(a)
 import fcntl
 with (a.output/'process.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.mode=='replay':replay(a);return
  stages=[('prepare',prepare,['src','dataset','s2','raw','legacy_structure_review.json','structure_selection_config.json']),('exclusion',s2_and_exclusion,['s2_verification','auxiliary_titan']),
   ('auxiliary',auxiliary,['auxiliary_stage1','mechanism_features']),('modalities',modalities,['features']),
   ('fusion',fusion,['fusion','pooled']),('structures',structures,['structure_discovery']),('stage2',stage2,['stage2','ablation'])]
  try:
   for name,fn,roots in stages:
    action=(lambda name=name,roots=roots:adopt_upstream(a,name,roots)) if a.reuse_upstream_from and name in ('exclusion','auxiliary','modalities','fusion') else (lambda fn=fn:fn(a))
    step(a,name,action,[a.output/r for r in roots])
    if a.stage==name:break
  except BaseException as e:
   write(a.output/'FAILURE.json',dict(type=type(e).__name__,error=str(e),UTC=time.time(),
    instruction='Keep this run. Repeat same command with --resume. No mismatch tolerance was relaxed.'))
   raise
if __name__=='__main__':main()
