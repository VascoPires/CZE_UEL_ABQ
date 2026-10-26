"""Check the new extractor and generated 3D baseline against retained evidence."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def read_deck(path):
    nodes, elements, equations, state = {}, [], [], ''
    count = None
    for line in path.read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'):
            state = line.split(',')[0]
            continue
        fields = [v.strip() for v in line.split(',') if v.strip()]
        if state=='*NODE': nodes[int(fields[0])] = tuple(float(v) for v in fields[1:])
        elif state=='*ELEMENT': elements.append(tuple(int(v) for v in fields))
        elif state=='*EQUATION':
            if count is None: count = int(fields[0])
            else:
                assert len(fields)==3*count
                equations.append(tuple(float(v) for v in fields)); count = None
    return nodes,elements,equations


def main():
    original = ROOT/'validation/campaign_20261025/runs/fixed_dcb_a35p0'
    archived = json.loads((original/'run.json').read_text())
    data = json.loads((HERE/'extraction_control/response.json').read_text())['last']
    rows = list(csv.DictReader((original/'nodal_response.csv').open()))
    upper = [r for r in rows if r['group']=='upper'][-1]
    lower = [r for r in rows if r['group']=='lower'][-1]
    P = (float(upper['rf2_N'])-float(lower['rf2_N']))/2*archived['force_scale']
    C = (float(upper['u2_mm'])-float(lower['u2_mm']))/P
    old = json.loads((original/'kinematic_checks.json').read_text())[-1]
    errors = dict(compliance_relative=data['compliance_mm_per_N']/C-1.,
                  force_relative=data['force_N']/P-1.,
                  GI_relative=data['GI_N_per_mm']/old['vcct_GI_N_per_mm']-1.)
    assert max(abs(v) for v in errors.values())<1e-12
    oldnodes, oldelements, oldequations = read_deck(original/'fixed_dcb_a35p0.inp')
    newnodes, newelements, newequations = read_deck(HERE/'runs/C3D8I_dx0p5_a35p0/C3D8I_dx0p5_a35p0.inp')
    assert oldnodes.keys()==newnodes.keys()
    coordinate_error = max(abs(v-w) for n in oldnodes for v,w in zip(oldnodes[n],newnodes[n]))
    assert coordinate_error<1e-9 and oldelements==newelements
    assert sorted(oldequations)==sorted(newequations)
    result = dict(status='PASS',extractor_errors=errors,baseline_compliance_mm_per_N=C,
                  baseline_stiffness_N_per_mm=1/C,baseline_GI_N_per_mm=data['GI_N_per_mm'],
                  baseline_max_coordinate_difference_mm=coordinate_error,
                  baseline_bulk_connectivity_identical=True,baseline_equations_identical=True,
                  scope='Existing completed C3D8I ODB extraction and mesh/equation compatibility; not a newly solved 3D job or native 2D equivalence proof')
    (HERE/'control_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
