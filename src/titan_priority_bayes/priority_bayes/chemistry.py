"""Exact discovery-rule calibration; original isomeric identities remain primary."""
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .protocol import ACTIVE


def environment(mol, center, radius, Chem):
    # Identical to the preceding successful training-coverage audit.
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)) if radius else []
    atoms = {center}
    for index in bonds:
        bond = mol.GetBondWithIdx(index)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds,
        rootedAtAtom=center, canonical=True, isomericSmiles=False,
        allHsExplicit=True, allBondsExplicit=True)
    return fragment, sorted(atoms), bonds


class Matcher:
    def __init__(self, reference):
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog('rdApp.*')
        self.Chem, self.Murcko = Chem, MurckoScaffold
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
            includeChirality=False, useBondTypes=True, countSimulation=False)
        self.AdditionalOutput = rdFingerprintGenerator.AdditionalOutput
        self.envs = {r['feature_id']:r['candidate'] for r in reference['rules']
                     if r['family'] == 'verified_environments'}
        self.scaffolds = {r['feature_id']:r['candidate'] for r in reference['rules']
                         if r['family'] == 'murcko_scaffolds'}
        self.sparse_ids = set(reference['target_sparse_ids'])
        self.cache = {}

    def graph(self, smiles):
        if smiles in self.cache:
            return self.cache[smiles]
        Chem = self.Chem
        mol = Chem.MolFromSmiles(smiles)
        result = dict(status='parse_failure', canonical_smiles='', InChIKey='',
                      connectivity_group='', scaffold='', hits=())
        if mol is not None:
            if len(Chem.GetMolFrags(mol)) != 1:
                result['status'] = 'multiple_components'
            else:
                key = Chem.MolToInchiKey(mol)
                if not key:
                    raise ValueError('A valid single-component molecule has no InChIKey')
                scaf = self.Murcko.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
                hits = set()
                if scaf in self.scaffolds:
                    hits.add(self.scaffolds[scaf])
                ao = self.AdditionalOutput();ao.AllocateBitInfoMap()
                self.gen.GetSparseCountFingerprint(mol,additionalOutput=ao)
                for sparse,pairs in ao.GetBitInfoMap().items():
                    if int(sparse) not in self.sparse_ids:
                        continue
                    for center,radius in pairs:
                        if int(radius) == 0:
                            continue
                        fragment,_,_ = environment(mol,int(center),int(radius),Chem)
                        token = f'{int(sparse)}|r{int(radius)}|{fragment}'
                        env = 'Env_' + hashlib.sha256(token.encode()).hexdigest()[:20]
                        if env in self.envs:
                            hits.add(self.envs[env])
                result = dict(status='valid',canonical_smiles=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True),
                    InChIKey=key,connectivity_group=key.split('-')[0],scaffold=scaf,hits=tuple(sorted(hits)))
        self.cache[smiles] = result
        return result


def calibrate(matcher, reference, frame):
    rows = [[s,int(y)] for s,y in zip(frame.SMILES,frame.label)]
    if rows != reference['validation_rows']:
        raise ValueError('Validation SMILES/labels/order differ from the discovery snapshot')
    observed = {r['candidate']:set() for r in reference['rules']}
    for r in reference['validation_units']:
        g = matcher.graph(r['SMILES'])
        if g['status'] != 'valid' or g['canonical_smiles'] != r['canonical_smiles']:
            raise ValueError('RDKit canonical identity does not reproduce the discovery snapshot')
        for c in g['hits']:
            observed[c].add(r['structure_id'])
    for c,actual in observed.items():
        expected = set(reference['expected_members'][c])
        if actual != expected:
            raise ValueError(f'{c}: validation rule calibration mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    return dict(n_rules=len(observed),n_validation_primary_units=len(reference['validation_units']),
                exact_member_sets_reproduced=True)


def row_audit(matcher, frame, active=None):
    active = ACTIVE if active is None else tuple(active)
    rows, grouped = [], defaultdict(list)
    for i,(s,y) in enumerate(zip(frame.SMILES,frame.label)):
        g = matcher.graph(s)
        row = dict(original_row_index=i,canonical_smiles=g['canonical_smiles'],
            InChIKey=g['InChIKey'],connectivity_group=g['connectivity_group'],
            eligibility_status=g['status'],primary_unit=False,**{c:c in g['hits'] for c in active})
        rows.append(row)
        if g['status'] == 'valid':
            grouped[g['canonical_smiles']].append(i)
    for indices in grouped.values():
        if len({int(frame.label.iloc[i]) for i in indices}) > 1:
            for i in indices:
                rows[i]['eligibility_status'] = 'conflicting_labels_same_isomeric_identity'
                for c in active:
                    rows[i][c] = False
        else:
            for k,i in enumerate(indices):
                rows[i]['primary_unit'] = k == 0
                rows[i]['eligibility_status'] = 'primary_representative' if k == 0 else 'consistent_duplicate'
    return pd.DataFrame(rows)


def prepare(reference, frames):
    dynamic = reference.get('origin') == 'generated_from_current_validation'
    active = tuple(r['candidate'] for r in reference['rules']) if dynamic else ACTIVE
    matcher = Matcher(reference)
    report = {'validation_calibration':calibrate(matcher,reference,frames['val'])}
    audits = {s:row_audit(matcher,f,active) for s,f in frames.items()}
    if dynamic:
        excluded=reference.get('validation_excluded_canonical_smiles',[])
        mask=audits['val'].canonical_smiles.isin(excluded)
        audits['val'].loc[mask,'primary_unit']=False
        audits['val'].loc[mask,'eligibility_status']='conflicting_locked_outcomes_in_discovery'
        for rule in active:audits['val'].loc[mask,rule]=False
    coverage = []
    for split,f in frames.items():
        a = audits[split]
        for c in active:
            mask = a[c].to_numpy(dtype=bool)
            primary = mask & a.primary_unit.to_numpy(dtype=bool)
            coverage.append(dict(split=split,candidate=c,raw_training_or_validation_rows=int(mask.sum()),
                units=int(primary.sum()),positive=int((primary & f.label.eq(1).to_numpy()).sum()),
                negative=int((primary & f.label.eq(0).to_numpy()).sum()),
                connectivity_groups=int(a.loc[primary,'connectivity_group'].nunique())))
    matches_expected = True
    for c in active:
        actual = next(r for r in coverage if r['split']=='train' and r['candidate']==c)
        expected = reference['expected_training_coverage'][c]
        matches_expected &= all(actual[k] == v for k,v in expected.items())
    report.update(coverage=coverage,training_coverage_matches_reported_audit=bool(matches_expected),
        active_rules=list(active),primary_identity='canonical isomeric SMILES; no stereochemistry pooling',
        matching='original calibrated sparse-ID + radius + atom-environment string; Morgan radius2 fpSize2048 no chirality',
        fit_rows='All original 7593 train rows stay aligned. Invalid, multicomponent, or conflicting-identity rows get structure weight 1; no rows are deleted from any arm.',
        normalization='none; existing scale_pos_weight stays fixed; class weight totals are reported',
        val_primary_units=int(audits['val'].primary_unit.sum()))
    wanted = reference['expected_validation_primary_units'] if dynamic else 723
    if report['val_primary_units'] != wanted:
        raise ValueError('Validation primary-unit audit does not reproduce the locked discovery snapshot')
    return audits,report
