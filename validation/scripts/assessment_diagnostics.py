"""Reproducible onset, viscous-quadrature and fixed-crack independent checks."""
import csv
import json
from pathlib import Path

import numpy as np

from campaign import CAMPAIGN,ROOT
from reduce_structure import rows


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def material_diagnostics():
    folder=CAMPAIGN/'material/extended_diagnostics'
    data=np.genfromtxt(folder/'response.csv',delimiter=',',names=True);result=[]
    for c in json.loads((folder/'cases.json').read_text()):
        a=data[data['case_id']==c['id']]
        if c['label'].startswith('onset'):
            result.append(dict(label=c['label'],D_before=float(a['D'][1]),D_after=float(a['D'][2]),
                lambda_ratio_before=float(a['lambda'][1]/a['lambda0'][1]),lambda_ratio_after=float(a['lambda'][2]/a['lambda0'][2]),
                onset_bracket_pass=bool(a['D'][1]==0 and a['D'][2]>0)))
        elif c['label'].startswith('visc'):
            u=np.c_[a['ds1'],a['ds2'],a['dn']];force=np.c_[a['ts1'],a['ts2'],a['tn']]
            work=np.sum(.5*(force[1:]+force[:-1])*(u[1:]-u[:-1]));kn=c['props'][5];ks=kn*.187/.781*(52/38)**2
            defect=.25*np.sum(np.diff(a['Dv'])*(np.diff(u,axis=0)**2@np.array([ks,ks,kn])))
            result.append(dict(label=c['label'],work_Nmm=float(work),diss_Nmm=float(a['diss'][-1]),
                trapezoid_energy_excess=float(a['diss'][-1]+a['psi'][-1]-work),predicted_discrete_identity_excess=float(defect)))
    save(folder/'assessment/onset_viscous_diagnostics.json',result)
    original=np.genfromtxt(CAMPAIGN/'material/canonical/response.csv',delimiter=',',names=True)
    variant=np.genfromtxt(CAMPAIGN/'material/zeta1e8/response.csv',delimiter=',',names=True)
    columns=[name for name in original.dtype.names if name!='tangent_error']
    differences={name:float(np.max(np.abs(original[name]-variant[name]))) for name in columns}
    save(CAMPAIGN/'sources/material_patch_response_equivalence.json',dict(samples=len(original),
        maximum_absolute_differences=differences,all_non_tangent_columns_identical=all(v==0 for v in differences.values()),
        limitation='Extracted Fortran material histories only; full solver and full element assembly checks are separate'))
    return result


def fixed_crack_checks():
    result=[]
    for folder in sorted((CAMPAIGN/'runs').glob('fixed_*')):
        if not (folder/'run.json').exists():continue
        manifest=json.loads((folder/'run.json').read_text());record=dict(name=folder.name,status=manifest['status'])
        if (folder/'assessment/additional_metrics.json').exists():
            extra=json.loads((folder/'assessment/additional_metrics.json').read_text())
            record.update(extra.get('VCCT',{}));record['kinematic_checks']=extra.get('kinematics',{})
        if (folder/'assessment/force_displacement.csv').exists():
            rr=rows(folder/'assessment/force_displacement.csv');u=np.array([float(r['displacement_mm']) for r in rr]);f=np.array([float(r['total_force_N']) for r in rr])
            if len(f)>2 and f@f>0:
                C=float(f@u/(f@f));record['elastic_compliance_mm_per_N']=C
                record['elastic_fit_R2']=float(1-np.sum((u-C*f)**2)/np.sum((u-u.mean())**2))
                record['last_force_N']=float(f[-1]);record['last_displacement_mm']=float(u[-1])
        if manifest['material']=='krueger2012' and manifest['family']=='mmb':
            target={.2:[128.5,1.64],.5:[385.,1.34],.8:[751.,1.65]}[manifest['mixity_nominal']]
            record['published_annotated_critical_N_mm']=target
            if 'critical_load_N' in record:
                record['critical_load_relative_difference']=record['critical_load_N']/target[0]-1
                record['critical_displacement_relative_difference']=record['critical_displacement_mm']/target[1]-1
        result.append(record)
    by_name={r['name']:r for r in result}
    names=['fixed_dcb_a34p5','fixed_dcb_a35p0','fixed_dcb_a35p5']
    if all(name in by_name and by_name[name]['status']=='COMPLETED' and 'elastic_compliance_mm_per_N' in by_name[name] for name in names):
        left,centre,right=[by_name[name] for name in names]
        dcda=(right['elastic_compliance_mm_per_N']-left['elastic_compliance_mm_per_N'])/1.
        # Full-width forces/compliances; reference width is20mm for this case.
        gt=centre['last_force_N']**2/(2*20.)*dcda
        compare=dict(compliance_derivative_mm_per_N_per_mm=dcda,G_compliance_N_per_mm=gt,
                     G_VCCT_N_per_mm=centre.get('GT_N_per_mm'),
                     signed_relative_difference=None if not centre.get('GT_N_per_mm') else centre['GT_N_per_mm']/gt-1,
                     limitation='Central crack-length difference of1mm; VCCT closure length0.5mm. This is an independent total-G diagnostic, not a converged crack-front study.')
        save(CAMPAIGN/'results/dcb_compliance_vs_vcct.json',compare)
    save(CAMPAIGN/'results/fixed_crack_checks.json',result)
    return result


def area_cycle_checks():
    result=[]
    for name in ['area_cycle_canonical','area_cycle_canonical_fine','area_cycle_reference']:
        path=CAMPAIGN/'element_tangent'/name/'area_cycle.csv'
        if not path.exists():continue
        data=np.genfromtxt(path,delimiter=',',names=True)
        work=data['external_work_Nmm'];stored=data['recoverable_Nmm'];diss=data['dissipation_Nmm']
        scale=max(max(abs(work)),1e-12);error=float(max(abs(work-stored-diss))/scale)
        result.append(dict(name=name,samples=len(data),final_work_Nmm=float(work[-1]),final_stored_Nmm=float(stored[-1]),
                           final_dissipation_Nmm=float(diss[-1]),peak_absolute_work_Nmm=float(scale),
                           maximum_relative_energy_balance_error=error,classification='PASS' if error<=.01 else 'FAIL',
                           maximum_inplane_force_norm_N=float(max(data['inplane_force_norm_N'])),
                           scope='Actual full Fortran UEL with serial runtime stubs; elastic constant-gap area change and closed geometry cycle; not an Abaqus solver interface test'))
    save(CAMPAIGN/'results/area_cycle_checks.json',result)
    return result


def basis_cycle_checks():
    result=[]
    for name in ['basis_cycle_canonical','basis_cycle_reference']:
        path=CAMPAIGN/'element_tangent'/name/'basis_cycle.csv'
        if not path.exists():continue
        data=np.genfromtxt(path,delimiter=',',names=True)
        work=data['external_work_Nmm'];stored=data['recoverable_Nmm'];diss=data['dissipation_Nmm']
        scale=max(max(abs(work)),1e-12);error=float(max(abs(work-stored-diss))/scale)
        result.append(dict(name=name,samples=len(data),final_work_Nmm=float(work[-1]),final_stored_Nmm=float(stored[-1]),
            final_dissipation_Nmm=float(diss[-1]),maximum_relative_energy_balance_error=error,
            maximum_resultant_internal_moment_Nmm=float(max(data['moment_norm_Nmm'])),classification='PASS' if error<=.01 else 'FAIL',
            scope='Actual full Fortran UEL with runtime stubs; constant-area midsurface rotation0→45degrees at fixed small global gap, closure and restoration. Not a rigid-body rotation of the entire separated configuration.'))
    save(CAMPAIGN/'results/basis_cycle_checks.json',result);return result


def experimental_toughness_screen():
    from structured_specimen import MATERIALS
    p=MATERIALS['turon2006_dcb'];result=[]
    for case in ['dcb','mmb20','mmb50','mmb80','enf']:
        path=ROOT/('validation/cases/turon2006MM_exp_'+case+'/data/experimental_peak_sources.csv')
        observed=next(r for r in rows(path) if r['source']=='turon2006MM')
        B=float(observed['nominal_GII_GT']);measured=float(observed['measured_Gc_N_per_mm'])
        predicted=p['G'][0]+(p['G'][1]-p['G'][0])*B**p['eta']
        result.append(dict(case=case,nominal_B=B,measured_Gc_N_per_mm=measured,BK_Gc_N_per_mm=predicted,
            relative_energy_difference=predicted/measured-1,formal_same_geometry_LEFM_force_scaling_difference=(predicted/measured)**.5-1,
            source=str(path.relative_to(ROOT)),scope='Explanatory nominal-fraction energy screen; not a corrected load prediction or calibration. Actual global/local fractions and finite process zones differ.'))
    save(CAMPAIGN/'results/experimental_toughness_fit_screen.json',result)
    with (CAMPAIGN/'results/experimental_toughness_fit_screen.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)


def reference_area_comparison():
    unit=[];structural=[];runs=CAMPAIGN/'runs'
    paths=[runs/name/'assessment/metrics.json' for name in ['element_dt001','element_refarea_dt001']]
    if all(p.exists() for p in paths):
        base,test=[{r['label']:r for r in json.loads(p.read_text())} for p in paths]
        for label,a in base.items():
            b=test[label]
            unit.append(dict(history=label,baseline_dissipation_Nmm=a['final_dissipation_Nmm'],test_dissipation_Nmm=b['final_dissipation_Nmm'],
                dissipation_difference_Nmm=b['final_dissipation_Nmm']-a['final_dissipation_Nmm'],
                baseline_work_error=a['final_work_balance_relative_error'],test_work_error=b['final_work_balance_relative_error']))
    for baseline,test in [('dcb_ps03','dcb_ps03_refarea'),('t2018_mmb20_lefmlever','t2018_mmb20_lefmlever_refarea')]:
        paths=[runs/name/'assessment/additional_metrics.json' for name in [baseline,test]]
        if not all(p.exists() for p in paths):continue
        a,b=[json.loads(p.read_text()) for p in paths]
        ma,mb=[json.loads((runs/name/'assessment/metrics.json').read_text()) for name in [baseline,test]]
        both_peaks=a.get('peak_eligible') and b.get('peak_eligible')
        structural.append(dict(baseline=baseline,test=test,baseline_status=ma['termination'],test_status=mb['termination'],
            baseline_peak_N=ma['maximum_reached_force_N'] if a.get('peak_eligible') else None,
            test_peak_N=mb['maximum_reached_force_N'] if b.get('peak_eligible') else None,
            relative_peak_difference=mb['maximum_reached_force_N']/ma['maximum_reached_force_N']-1 if both_peaks else None,
            baseline_energy_error=a.get('work_balance_relative_error'),test_energy_error=b.get('work_balance_relative_error')))
    result=dict(unit_histories=unit,structural_comparisons=structural,
        scope='Restricted matched tests of a different area measure; both variants fail the independent constant-area basis cycle. No canonical replacement or general geometric-energy fix is established.')
    save(CAMPAIGN/'results/reference_area_comparison.json',result)


def main():
    (CAMPAIGN/'results').mkdir(exist_ok=True)
    material=material_diagnostics();fixed=fixed_crack_checks();area=area_cycle_checks();basis=basis_cycle_checks();experimental_toughness_screen();reference_area_comparison()
    print('Onset brackets passed',sum(v.get('onset_bracket_pass',False) for v in material),'/20; fixed-crack records',len(fixed))
    print('Elastic area cycles',[(r['name'],r['classification']) for r in area])


if __name__=='__main__':main()
