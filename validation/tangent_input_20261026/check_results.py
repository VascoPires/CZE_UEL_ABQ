"""Verify retained compiled/solver results; does not submit or rerun jobs."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
FIRST=ROOT/'validation/campaign_20261025'


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assembly=[]
    for baseline,test in [('baseline_v13','default_v14'),('default_v14','explicit_default_v14'),('fixed_small_v13','explicit_small_v14')]:
        a,b=[np.genfromtxt(BASE/'assembly'/name/'state.csv',delimiter=',',names=True) for name in [baseline,test]]
        assert a.shape==b.shape
        assert all(np.array_equal(a[key],b[key]) for key in a.dtype.names),(baseline,test)
        assembly.append(dict(baseline=baseline,test=test,rows=len(a),exact_match=True))
    material=[]
    for test,reference in [('material_default','turon2018_K1e6_canonical'),('material_small','turon2018_K1e6_z8')]:
        a=BASE/test/'response.csv';b=FIRST/'material'/reference/'response.csv'
        assert a.read_bytes()==b.read_bytes(),test
        assert load(BASE/test/'manifest.json')['returncode']==0
        material.append(dict(test=test,reference=reference,exact_match=True))
    solver=[]
    for test,reference in [('legacy_default','element_dt001'),('explicit_default','element_dt001'),('explicit_small','element_z8_dt001')]:
        folder=BASE/'runs'/test;m=load(folder/'run.json');assert m['status']=='COMPLETED'
        assert sha(folder/'3D_CZM.f')==sha(ROOT/'src/3D_CZM.f')==m['uel_sha256']
        assert sha(folder/(test+'.inp'))==m['input_sha256']
        for name in ['nodal_response.csv','sdv.csv']:
            assert (folder/name).read_bytes()==(FIRST/'runs'/reference/name).read_bytes(),(test,name)
        assert len(load(folder/'assessment/metrics.json'))==16
        solver.append(dict(test=test,reference=reference,patch_histories=16,exact_response_and_state_match=True))
    invalid=BASE/'runs/invalid_zero';assert load(invalid/'run.json')['status']=='FAILED'
    text='\n'.join(p.read_text(errors='replace') for p in invalid.glob('*') if p.suffix in ['.msg','.dat','.txt'])
    assert 'CZE UEL: PROPS(8)' in text and 'must be finite and strictly positive' in text
    assert not list((BASE/'runs').glob('*/*.lck'))
    baseline=load(BASE/'baseline.json');assert sha(ROOT/baseline['source_archive'])==baseline['source_sha256']
    first_runs=load(FIRST/'results/run_inventory.json')
    for r in first_runs:
        p=ROOT/r['run_path'];m=load(p/'run.json')
        assert sha(p/(m['name']+'.inp'))==m['input_sha256']
        if m.get('uel_sha256'):assert sha(p/'3D_CZM.f')==m['uel_sha256']
    original=load(ROOT/'validation/audit/phase0_2026-10-25/baseline_files.json')
    for name,record in original.items():
        if name.startswith('examples/'):assert sha(ROOT/name)==record['sha256']
    workflow=load(BASE/'workflow_checks.json')
    result=dict(status='PASS',scope='Optional-input compatibility and tested tangent selection; not full specimen validation or geometric-energy correction',
        source_sha256=sha(ROOT/'src/3D_CZM.f'),baseline_sha256=baseline['source_sha256'],assembly=assembly,
        material=material,abaqus=solver,invalid_input_rejected_by_fortran=True,
        preparation_example_decks=len(workflow['example_decks']),invalid_workflow_cases=len(workflow['rejected_invalid_cases']),
        original_example_files_unchanged=True,first_campaign_run_decks_and_sources_preserved=len(first_runs))
    (BASE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: optional-input checks; 48 Abaqus patch histories; invalid-input guard; all first-campaign artifacts preserved')


if __name__=='__main__':main()
