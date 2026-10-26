"""Assess only actually completed matched jobs; missing results stay unavailable."""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    results = []
    groups = {}
    for name in plan['jobs']:
        folder = HERE/'runs'/name
        m = json.loads((folder/'run.json').read_text())
        row = {key:m[key] for key in ['name','status','element','representation','dx_mm','ny_per_arm','nz','a_mm']}
        if (folder/'response.json').exists() and m['status']=='COMPLETED':
            response = json.loads((folder/'response.json').read_text())
            last = response['last']
            row.update({k:last[k] for k in ['compliance_mm_per_N','force_N','GI_N_per_mm','GII_N_per_mm',
                'reaction_imbalance_fraction','tip_tie_jump_mm','strain_energy_relative_error']})
            row['stiffness_N_per_mm'] = 1/row['compliance_mm_per_N']
            row['compliance_spread_fraction'] = response['compliance_spread_fraction']
            row['G_total_N_per_mm'] = row['GI_N_per_mm']+row['GII_N_per_mm']
            if row['G_total_N_per_mm']>0:
                row['LEFM_initiation_N_from_VCCT'] = last['force_N']*math.sqrt(m['parameters']['G'][0]/row['G_total_N_per_mm'])
        key = (m['element'],m['representation'],m['dx_mm'],m['ny_per_arm'],m['nz'])
        groups.setdefault(key,{})[m['a_mm']] = row
        results.append(row)
    derivatives = []
    for key,cracks in groups.items():
        if not all('compliance_mm_per_N' in cracks[a] for a in [34.5,35.,35.5]):
            continue
        lo,mid,hi = [cracks[a] for a in [34.5,35.,35.5]]
        dCda = (hi['compliance_mm_per_N']-lo['compliance_mm_per_N'])/(35.5-34.5)
        G = mid['force_N']**2/(2*20.)*dCda
        derivatives.append(dict(element=key[0],representation=key[1],dx_mm=key[2],ny=key[3],nz=key[4],
                                dC_da_per_N=dCda,G_compliance_N_per_mm=G,
                                G_VCCT_N_per_mm=mid['G_total_N_per_mm'],
                                VCCT_vs_compliance_fraction=mid['G_total_N_per_mm']/G-1.,
                                initiation_N_from_compliance=math.sqrt(.26*2*20./dCda)))
    status_counts = {}
    for r in results: status_counts[r['status']] = status_counts.get(r['status'],0)+1
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/assessment.json').write_text(json.dumps(dict(status_counts=status_counts,runs=results,derivatives=derivatives),indent=2)+'\n')
    fields = list(dict.fromkeys(k for r in results for k in r))
    with (HERE/'results/runs.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(results)
    print(json.dumps(status_counts))


if __name__=='__main__': main()
