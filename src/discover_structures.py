#!/usr/bin/env python3
"""Current-run validation error enrichment. Never consumes test data or old rules.

The default legacy_review_v1 replays documented review roles and recomputes
original S1-S5 statistics. automatic_v2 remains an explicit alternative, not a
reproduction of the earlier manual identity/literature review. All rules retain their exact
Morgan environment or Murcko identity; no expansion to named functional groups.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

FAMILIES=('verified_environments','murcko_scaffolds')
CONTRASTS=(('FN','TP'),('FP','TN'))
TEST_COLUMNS=['family','feature_id','contrast','FN','TP','FP','TN','same_label_support',
              'error_rate','nonfeature_error_rate','risk_difference','p_value',
              'q_BH_global','q_BY_global','q_BH_family','q_BY_family']

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()

def fdr(values,method):
    values=np.asarray(values,dtype=float);n=len(values)
    if not n:return np.array([],dtype=float)
    order=np.argsort(values,kind='stable')
    factor=np.sum(1/np.arange(1,n+1)) if method=='by' else 1.
    adjusted=np.minimum.accumulate((values[order]*n*factor/np.arange(1,n+1))[::-1])[::-1]
    result=np.empty(n);result[order]=np.clip(adjusted,0,1)
    return result

class Graphs:
    def __init__(self):
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog('rdApp.*')
        self.Chem,self.Murcko=Chem,MurckoScaffold
        self.generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,
            includeChirality=False,useBondTypes=True,countSimulation=False)
        self.AdditionalOutput=rdFingerprintGenerator.AdditionalOutput
        self.cache={};self.metadata={}

    def graph(self,smiles):
        if smiles in self.cache:return self.cache[smiles]
        Chem=self.Chem;mol=Chem.MolFromSmiles(smiles)
        row=dict(status='parse_failure',canonical_smiles='',InChIKey='',connectivity_group='',features=set())
        if mol is not None and len(Chem.GetMolFrags(mol))!=1:row['status']='multiple_components'
        elif mol is not None:
            key=Chem.MolToInchiKey(mol)
            if not key:raise ValueError('A parsed single-component molecule has no InChIKey')
            present=set();ao=self.AdditionalOutput();ao.AllocateBitInfoMap()
            self.generator.GetSparseCountFingerprint(mol,additionalOutput=ao)
            for sparse,pairs in ao.GetBitInfoMap().items():
                for center,radius in pairs:
                    if radius==0:continue
                    bonds=list(Chem.FindAtomEnvironmentOfRadiusN(mol,int(radius),int(center)))
                    atoms={int(center)}
                    for index in bonds:
                        bond=mol.GetBondWithIdx(index);atoms.update((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
                    fragment=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),bondsToUse=bonds,
                        rootedAtAtom=int(center),canonical=True,isomericSmiles=False,
                        allHsExplicit=True,allBondsExplicit=True)
                    token=f'{int(sparse)}|r{int(radius)}|{fragment}'
                    feature='Env_'+hashlib.sha256(token.encode()).hexdigest()[:20]
                    identity=('verified_environments',feature);present.add(identity)
                    meta=dict(family=identity[0],feature_id=feature,sparse_id=int(sparse),radius=int(radius),fragment_smiles=fragment)
                    if identity in self.metadata and self.metadata[identity]!=meta:raise ValueError('Environment signature collision')
                    self.metadata[identity]=meta
            scaffold=self.Murcko.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
            if scaffold:
                identity=('murcko_scaffolds',scaffold);present.add(identity)
                self.metadata[identity]=dict(family=identity[0],feature_id=scaffold,scaffold_smiles=scaffold)
            row=dict(status='valid',canonical_smiles=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True),
                InChIKey=key,connectivity_group=key.split('-')[0],features=present)
        self.cache[smiles]=row
        return row

def units(frame,graphs,probability=None,identity='canonical_smiles'):
    if probability is not None:
        probability=np.asarray(probability,dtype=float)
        if probability.shape!=(len(frame),) or not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
            raise ValueError('Validation probabilities must align with all original rows')
    groups=defaultdict(list);audit=[]
    for index,(smiles,label) in enumerate(zip(frame.SMILES,frame.label)):
        g=graphs.graph(str(smiles))
        if int(label) not in (0,1):raise ValueError('Need original binary labels')
        row=dict(original_row_index=index,SMILES=str(smiles),label=int(label),
            status=g['status'],canonical_smiles=g['canonical_smiles'],connectivity_group=g['connectivity_group'])
        if g['status']=='valid':groups[g[identity]].append(index)
        audit.append(row)
    result=[]
    for indices in groups.values():
        if len({int(frame.label.iloc[i]) for i in indices})!=1:
            for i in indices:audit[i]['status']='conflicting_labels_same_identity'
            continue
        i=indices[0];g=graphs.graph(str(frame.SMILES.iloc[i]))
        for j in indices:audit[j]['status']='primary_representative' if j==i else 'consistent_duplicate'
        item=dict(structure_id=f'{"Stereo" if identity=="canonical_smiles" else "Conn"}_{len(result)+1:04d}',
            SMILES=str(frame.SMILES.iloc[i]),canonical_smiles=g['canonical_smiles'],
            connectivity_group=g['connectivity_group'],label=int(frame.label.iloc[i]),
            source_rows=list(indices),features=set().union(*(graphs.graph(str(frame.SMILES.iloc[j]))['features'] for j in indices)))
        if probability is not None:
            item['positive_probability']=float(probability[indices].mean())
            pred=item['positive_probability']>.5
            item['outcome']=('TP' if pred else 'FN') if item['label'] else ('FP' if pred else 'TN')
        result.append(item)
    return result,audit

def test_features(primary,min_support):
    present=defaultdict(set)
    for i,row in enumerate(primary):
        for feature in row['features']:present[feature].add(i)
    tests=[]
    for error,correct in CONTRASTS:
        e={i for i,r in enumerate(primary) if r['outcome']==error}
        c={i for i,r in enumerate(primary) if r['outcome']==correct}
        if not e or not c:continue
        for (family,fid),hits in sorted(present.items()):
            a,b=len(e&hits),len(c&hits)
            if a+b<min_support:continue  # same-label total support, never error-only filtering
            other_e,other_c=len(e)-a,len(c)-b
            if other_e+other_c==0:continue  # no unexposed same-label comparison
            risk=a/(a+b);rest=other_e/(other_e+other_c)
            counts={k:sum(primary[i]['outcome']==k for i in hits) for k in ('FN','TP','FP','TN')}
            tests.append(dict(family=family,feature_id=fid,contrast=error+'_vs_'+correct,
                **counts,same_label_support=a+b,error_rate=risk,nonfeature_error_rate=rest,
                risk_difference=risk-rest,p_value=float(fisher_exact([[a,other_e],[b,other_c]],alternative='greater').pvalue)))
    for method in ('bh','by'):
        for row,q in zip(tests,fdr([r['p_value'] for r in tests],method)):row['q_'+method.upper()+'_global']=float(q)
        for family in FAMILIES:
            rows=[r for r in tests if r['family']==family]
            for row,q in zip(rows,fdr([r['p_value'] for r in rows],method)):row['q_'+method.upper()+'_family']=float(q)
    return tests,present

def coverage(primary,hits):
    return dict(units=len(hits),positive=sum(primary[i]['label']==1 for i in hits),
        negative=sum(primary[i]['label']==0 for i in hits),
        connectivity_groups=len({primary[i]['connectivity_group'] for i in hits}))

def leave_one_direction(primary,feature,contrast):
    # A directional sensitivity check, NOT a re-estimated FDR-significance claim.
    error,correct=contrast.split('_vs_');side=[r for r in primary if r['outcome'] in (error,correct)]
    groups=sorted({r['connectivity_group'] for r in side if feature in r['features'] and r['outcome']==error})
    differences=[]
    for group in groups:
        rows=[r for r in side if r['connectivity_group']!=group]
        hit=[r for r in rows if feature in r['features']];other=[r for r in rows if feature not in r['features']]
        if not hit or not other:return False,None,len(groups)
        differences.append(sum(r['outcome']==error for r in hit)/len(hit)-sum(r['outcome']==error for r in other)/len(other))
    return bool(differences and min(differences)>0),min(differences) if differences else None,len(groups)

def select(primary,train,tests,present,connectivity_tests,config):
    train_present=defaultdict(set)
    for i,row in enumerate(train):
        for key in row['features']:train_present[key].add(i)
    sensitivity={(r['family'],r['feature_id'],r['contrast']):r for r in connectivity_tests}
    candidates=[]
    for result in tests:
        row=dict(result);key=(row['family'],row['feature_id']);cov=coverage(train,train_present[key])
        row.update({'train_'+k:v for k,v in cov.items()})
        row.update(tier='excluded',decision='no_primary_BH_support',selected=False)
        if row['q_BH_global']>config['fdr_alpha'] or row['risk_difference']<=0:
            candidates.append(row);continue
        if (cov['units']<config['min_training_units'] or cov['positive']<config['min_training_per_class']
            or cov['negative']<config['min_training_per_class'] or cov['connectivity_groups']<config['min_training_connectivity_groups']):
            row['decision']='insufficient_training_coverage';candidates.append(row);continue
        s=sensitivity.get((key[0],key[1],row['contrast']))
        stable,min_difference,error_groups=leave_one_direction(primary,key,row['contrast'])
        row.update(connectivity_q_BH=None if s is None else s['q_BH_global'],
            leave_one_error_group_direction_stable=stable,minimum_leave_one_risk_difference=min_difference,
            distinct_error_connectivity_groups=error_groups)
        core=(row['q_BY_global']<=config['fdr_alpha'] and s is not None and s['q_BH_global']<=config['fdr_alpha']
            and s['risk_difference']>0 and stable and error_groups>=config['min_core_error_connectivity_groups'])
        row.update(tier='core' if core else 'reserve',decision='eligible_for_weighting_experiment')
        candidates.append(row)
    eligible=sorted([r for r in candidates if r['tier']!='excluded'],key=lambda r:(
        r['tier']!='core',r['q_BY_global'],r['q_BH_global'],-r['risk_difference'],r['family'],r['feature_id'],r['contrast']))
    selected=[];seen=set()
    for row in eligible:
        key=(row['family'],row['feature_id'])
        if key in seen:row['decision']='same_exact_feature_already_represented';continue
        redundant=next((old for old in selected if row['contrast']==old['contrast']
            and present[key]<=present[(old['family'],old['feature_id'])]
            and train_present[key]<=train_present[(old['family'],old['feature_id'])]),None)
        if redundant is not None:
            row['decision']='nested_in_'+redundant['candidate'];continue
        row.update(candidate=f'D{len(selected)+1:03d}',selected=True)
        selected.append(row);seen.add(key)
    return candidates,selected,train_present

def serial_units(items):
    return [{k:v for k,v in row.items() if k!='features'} for row in items]

def legacy_units(frame,graphs,probability=None):
    """Original sorted isomeric identities; first consistent row, no averaging."""
    probability=None if probability is None else np.asarray(probability,dtype=float)
    if probability is not None and (probability.shape!=(len(frame),) or not np.isfinite(probability).all()
            or ((probability<0)|(probability>1)).any()):raise ValueError('Invalid validation probabilities')
    groups=defaultdict(list);audit=[]
    for i,(smiles,label) in enumerate(zip(frame.SMILES,frame.label)):
        if label not in (0,1):raise ValueError('Need original binary labels')
        g=graphs.graph(str(smiles))
        audit.append(dict(original_row_index=i,SMILES=str(smiles),label=int(label),status=g['status'],
            canonical_smiles=g['canonical_smiles'],connectivity_group=g['connectivity_group']))
        if g['status']=='valid':groups[g['canonical_smiles']].append(i)
    result=[]
    for canonical,indices in sorted(groups.items()):
        labels={int(frame.label.iloc[i]) for i in indices}
        if len(labels)>1:reason='conflicting_labels_same_identity'
        elif probability is not None and len({bool(probability[i]>.5) for i in indices})>1:reason='conflicting_locked_outcomes'
        else:reason=None
        if reason:
            for i in indices:audit[i]['status']=reason
            continue
        first=indices[0];g=graphs.graph(str(frame.SMILES.iloc[first]))
        for i in indices:audit[i]['status']='primary_representative' if i==first else 'consistent_duplicate'
        row=dict(structure_id=f'Struct_{len(result)+1:04d}',SMILES=str(frame.SMILES.iloc[first]),
            canonical_smiles=canonical,connectivity_group=g['connectivity_group'],label=int(frame.label.iloc[first]),
            source_rows=list(indices),features=g['features'])
        if probability is not None:
            row['positive_probability']=float(probability[first]);pred=probability[first]>.5
            row['outcome']=('TP' if pred else 'FN') if row['label'] else ('FP' if pred else 'TN')
        result.append(row)
    return result,audit

def legacy_select(primary,train,tests,present,config,graphs,review):
    """Recompute original scenarios; replay recorded review roles transparently."""
    family_alpha=config['fdr_alpha']
    catalog={(r['family'],r['feature_id']):r for r in review['feature_catalog']}
    by_name={r['candidate']:key for key,r in catalog.items()}
    s1={r['canonical_smiles'] for r in primary if r['canonical_smiles'] in review['unresolved_isomer_canonical_smiles']}
    prefix=graphs.graph(review['tianeptine_anchor_canonical_smiles'])['connectivity_group']
    s2={r['canonical_smiles'] for r in primary if r['connectivity_group']==prefix}
    scaffold=('murcko_scaffolds',review['v14_scaffold'])
    s3={r['canonical_smiles'] for r in primary if scaffold in r['features']}
    s5={r['canonical_smiles'] for r in primary if r['canonical_smiles']==review['fluorescein_canonical_smiles']}
    scenarios={'S0_original':set(),'S1_unresolved_isomer_out':s1,'S2_tianeptine_out':s2,
        'S3_V14_series_out':s3,'S4_combined':s1|s2|s3,'S5_fluorescein_out':s5}
    all_scenario_tests=[];lookup={};members=[]
    for scenario,removed in scenarios.items():
        retained=[r for r in primary if r['canonical_smiles'] not in removed]
        rows,_=test_features(retained,config['min_validation_support'])
        for row in rows:
            item=dict(row,scenario=scenario)
            all_scenario_tests.append(item)
            lookup[(scenario,row['family'],row['feature_id'],row['contrast'])]=row
        for row in primary:
            if row['canonical_smiles'] in removed:members.append(dict(scenario=scenario,structure_id=row['structure_id'],
                canonical_smiles=row['canonical_smiles'],outcome=row['outcome'],label=row['label']))
    train_present=defaultdict(set)
    for i,row in enumerate(train):
        for key in row['features']:train_present[key].add(i)
    decisions=[];unknown=0
    for result in tests:
        row=dict(result);key=(row['family'],row['feature_id']);entry=catalog.get(key)
        cov=coverage(train,train_present[key]);row.update({'train_'+k:v for k,v in cov.items()})
        row.update(tier='excluded',selected=False,decision='no_primary_family_BH_support',
            historical_candidate=None if entry is None else entry['candidate'],
            prior_review_policy=None if entry is None else entry['review_policy'])
        for scenario in scenarios:
            item=lookup.get((scenario,key[0],key[1],row['contrast']))
            row[scenario+'_q_BH_family']=None if item is None else item['q_BH_family']
        if row['q_BH_family']>=family_alpha or row['risk_difference']<=0:
            decisions.append(row);continue
        if entry is None:
            unknown+=1;row['decision']='new_supported_structure_requires_review';decisions.append(row);continue
        row['candidate']=entry['candidate']
        if cov['units']==0:
            row['decision']='zero_training_coverage';decisions.append(row);continue
        if entry['review_policy']=='nested_alternative':
            parent=entry['review_parent'];parent_key=by_name[parent]
            row['review_parent']=parent
            row['nested_in_parent_validation']=present[key]<=present[parent_key]
            row['extra_training_units_outside_parent']=len(train_present[key]-train_present[parent_key])
            row['decision']='recorded_nested_alternative' if row['nested_in_parent_validation'] else 'prior_nesting_changed_requires_review'
            decisions.append(row);continue
        if entry['review_policy']!='preferred_weighting_candidate':
            row['decision']='recorded_hold_or_exploratory_only';decisions.append(row);continue
        stable=True
        for scenario in ('S1_unresolved_isomer_out','S2_tianeptine_out','S3_V14_series_out','S4_combined'):
            item=lookup.get((scenario,key[0],key[1],row['contrast']))
            stable &= item is not None and item['q_BH_family']<family_alpha and item['risk_difference']>0
        row.update(tier='core' if stable else 'reserve',decision='recorded_review_plus_recomputed_sensitivity',selected=True)
        decisions.append(row)
    # One exact structure receives only one weight rule, even if both directions pass.
    eligible=sorted([r for r in decisions if r['selected']],key=lambda r:(r['tier']!='core',r['q_BH_family'],r['p_value'],r['candidate'],r['contrast']))
    selected=[];seen=set()
    for row in eligible:
        key=(row['family'],row['feature_id'])
        if key in seen:row.update(selected=False,decision='same_exact_feature_already_represented');continue
        seen.add(key);selected.append(row)
    diagnostic=dict(scenario_definitions={name:sorted(values) for name,values in scenarios.items()},
        scenario_removed_counts={name:len(values) for name,values in scenarios.items()},
        review_provenance=review['provenance'],unknown_supported_hypotheses=unknown,
        prior_review_replayed=True,new_literature_review_performed=False,
        merged_connectivity_significance_used_as_core_gate=False,
        manual_prior_roles_are_data_not_automated_identity_verification=True)
    return decisions,selected,train_present,all_scenario_tests,members,diagnostic

def discover(ctx,config,output):
    if not 0<float(config['fdr_alpha'])<1:raise ValueError('FDR alpha must be in (0,1)')
    for key in ('min_validation_support','min_training_units','min_training_per_class',
                'min_training_connectivity_groups','min_core_error_connectivity_groups'):
        if isinstance(config[key],bool) or not isinstance(config[key],int) or config[key]<1:
            raise ValueError('Positive integer setting required: '+key)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    core=ctx['core'];gm=ctx['gm'];frames=ctx['frames']
    if set(frames)!=set(('train','val')):raise ValueError('Discovery accepts only train and val; no test frame')
    write(output/'selection_config.json',config)
    # Current model is fitted independently, with no structural weights.
    model,history=core['training'].fit_head(ctx['X']['train'],frames['train'].label.to_numpy(),
        ctx['params'],ctx['max_rounds'],ctx['X']['val'],frames['val'].label.to_numpy(),weights=None,
        patience=ctx['patience'],scale_pos_weight=ctx['class_weight'],verbose=False)
    model.save_model(str(output/'unweighted_full_fusion.txt'));write(output/'fit_history.json',history)
    probability=np.asarray(model.predict(ctx['X']['val']),dtype=float)
    prediction=gm.outcome_frame(frames['val'],probability,.5)
    prediction.to_csv(output/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    for outcome in ('TP','TN','FP','FN'):
        prediction.loc[prediction.outcome.eq(outcome)].to_csv(output/f'validation_{outcome}.csv',index=False,encoding='utf-8-sig')
    legacy=config.get('protocol')=='legacy_review_v1'
    graphs=Graphs()
    primary,val_audit=legacy_units(frames['val'],graphs,probability) if legacy else units(frames['val'],graphs,probability)
    connectivity,_=units(frames['val'],graphs,probability,identity='connectivity_group')
    train,train_audit=legacy_units(frames['train'],graphs) if legacy else units(frames['train'],graphs)
    tests,present=test_features(primary,config['min_validation_support'])
    sensitive,_=test_features(connectivity,config['min_validation_support'])
    scenario_tests=[];scenario_members=[];legacy_diagnostic={}
    if legacy:
        review_path=Path(__file__).resolve().parent.parent/'legacy_structure_review.json'
        review=json.loads(review_path.read_text(encoding='utf-8'))
        decisions,selected,train_present,scenario_tests,scenario_members,legacy_diagnostic=legacy_select(primary,train,tests,present,config,graphs,review)
        write(output/'legacy_review_record.json',review)
        write(output/'legacy_protocol_audit.json',legacy_diagnostic)
        pd.DataFrame(scenario_tests,columns=TEST_COLUMNS+['scenario']).to_csv(output/'all_scenario_structure_tests.csv',index=False)
        pd.DataFrame(scenario_members,columns=['scenario','structure_id','canonical_smiles','outcome','label']).to_csv(output/'scenario_exclusions.csv',index=False)
    else:decisions,selected,train_present=select(primary,train,tests,present,sensitive,config)
    rules=[];expected_members={};expected_training={}
    for row in selected:
        key=(row['family'],row['feature_id']);candidate=row['candidate']
        rules.append(dict(graphs.metadata[key],candidate=candidate,tier=row['tier'],contrast=row['contrast']))
        expected_members[candidate]=[primary[i]['structure_id'] for i in sorted(present[key])]
        expected_training[candidate]=coverage(train,train_present[key])
        from rdkit.Chem import Draw
        examples=[primary[i] for i in sorted(present[key])[:12]]
        svg=Draw.MolsToGridImage([graphs.Chem.MolFromSmiles(r['SMILES']) for r in examples],
            legends=[r['structure_id']+' '+r['outcome'] for r in examples],molsPerRow=3,subImgSize=(300,240),useSVG=True)
        (output/'SVG').mkdir(exist_ok=True)
        (output/'SVG'/f'{candidate}_examples.svg').write_text(str(svg),encoding='utf-8')
    reference=dict(origin='generated_from_current_validation',schema_version=2,rules=rules,
        core_rules=[r['candidate'] for r in selected if r['tier']=='core'],
        reserve_rules=[r['candidate'] for r in selected if r['tier']=='reserve'],
        target_sparse_ids=sorted({r['sparse_id'] for r in rules if r['family']=='verified_environments'}),
        validation_rows=[[str(s),int(y)] for s,y in zip(frames['val'].SMILES,frames['val'].label)],
        validation_units=serial_units(primary),expected_members=expected_members,
        expected_training_coverage=expected_training,expected_validation_primary_units=len(primary),
        validation_excluded_canonical_smiles=sorted({r['canonical_smiles'] for r in val_audit if r['status']=='conflicting_locked_outcomes'}),
        selection_config=config,no_test_data_used=True,manual_literature_review_performed=False,
        prior_review_replayed=legacy,
        grouping_basis='recorded prior review roles + recomputed primary family BH and S1-S4 sensitivity' if legacy else 'automatic statistical evidence tiers')
    # New reference must reproduce its own exact calibration and coverage.
    matches,audit=core['chemistry'].prepare(reference,frames)
    if not audit['training_coverage_matches_reported_audit']:raise ValueError('New rule calibration failed')
    write(output/'reference_rules.json',reference)
    write(output/'structure_audit.json',audit)
    pd.DataFrame(tests,columns=TEST_COLUMNS).to_csv(output/'all_structure_tests.csv',index=False)
    pd.DataFrame(sensitive,columns=TEST_COLUMNS).to_csv(output/'connectivity_sensitivity_tests.csv',index=False)
    decision_frame=pd.DataFrame(decisions)
    if decision_frame.empty:decision_frame=pd.DataFrame(columns=TEST_COLUMNS+['tier','decision','selected','candidate'])
    decision_frame.to_csv(output/'candidate_decisions.csv',index=False)
    pd.DataFrame(val_audit).to_csv(output/'validation_identity_audit.csv',index=False)
    pd.DataFrame(train_audit).to_csv(output/'training_identity_audit.csv',index=False)
    write(output/'validation_primary_units.json',serial_units(primary))
    membership=[];overlaps=[]
    for split,group,sets in [('val',primary,present),('train',train,train_present)]:
        for rule in selected:
            key=(rule['family'],rule['feature_id'])
            for i in sorted(sets[key]):
                membership.append(dict(split=split,candidate=rule['candidate'],**{k:v for k,v in group[i].items() if k not in ('features','source_rows')}))
        for i,r in enumerate(selected):
            a=sets[(r['family'],r['feature_id'])]
            for s in selected[i+1:]:
                b=sets[(s['family'],s['feature_id'])]
                overlaps.append(dict(split=split,rule_a=r['candidate'],rule_b=s['candidate'],
                    intersection=len(a&b),union=len(a|b),a_subset_b=a<=b,b_subset_a=b<=a))
    pd.DataFrame(membership,columns=['split','candidate','structure_id','SMILES','canonical_smiles','connectivity_group','label','positive_probability','outcome']).to_csv(output/'rule_members.csv',index=False)
    pd.DataFrame(overlaps,columns=['split','rule_a','rule_b','intersection','union','a_subset_b','b_subset_a']).to_csv(output/'rule_overlap.csv',index=False)
    coverage_frame=pd.DataFrame(audit['coverage'],columns=['split','candidate','raw_training_or_validation_rows','units','positive','negative','connectivity_groups'])
    coverage_frame.to_csv(output/'training_validation_coverage.csv',index=False)
    core['report'].workbook(output/'structure_selection.xlsx',dict(all_tests=pd.DataFrame(tests,columns=TEST_COLUMNS),
        decisions=decision_frame,connectivity_tests=pd.DataFrame(sensitive,columns=TEST_COLUMNS),coverage=coverage_frame,
        scenario_tests=pd.DataFrame(scenario_tests,columns=TEST_COLUMNS+['scenario'])))
    summary=dict(status='complete',n_primary_validation_units=len(primary),n_connectivity_sensitivity_units=len(connectivity),
        tested_hypotheses=len(tests),n_core=len(reference['core_rules']),n_reserve=len(reference['reserve_rules']),
        no_supported_candidates=not rules,no_test_data_used=True,current_validation_prediction_sha256=sha(output/'validation_predictions.csv'),
        automatic_grouping_not_equivalent_to_manual_literature_review=True,protocol=config.get('protocol','automatic_v2'),
        prior_review_replayed=legacy,core_rules=reference['core_rules'],reserve_rules=reference['reserve_rules'])
    write(output/'summary.json',summary)
    write(output/'COMPLETE.json',dict(summary,files_sha256={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}))
    print('STRUCTURE DISCOVERY:',summary,flush=True)
    return reference
