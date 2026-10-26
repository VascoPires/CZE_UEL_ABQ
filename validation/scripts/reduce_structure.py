"""Additional structural diagnostics from executed jobs, without replacing raw output.

Crack fronts use undeformed mesh coordinates and all four UEL sampling points.
Published critical equilibrium points and experimental tables remain distinct.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from campaign import CAMPAIGN, ROOT
from lefm_reference import dcb, enf, mmb


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith('#')))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def damage(folder, manifest):
    path=folder/'sdv.csv'
    if manifest.get('bonded_fixed_crack'):return dict(status='NOT RUN',reason='Fixed-crack tied elastic control; no damage law active')
    if not path.exists() and not manifest.get('native'):
        subprocess.run([sys.executable,str(ROOT/'examples/element/parse_elprint_sdv_by_increment.py'),
                        '--datFile',str(folder/(manifest['name']+'.dat')),
                        '--outCsv',str(path),'--sdvCount','80'],check=True,capture_output=True)
    if not path.exists():return dict(status='NOT RUN',reason='Native damage field not requested in this control')
    raw=rows(path);states={};previous={};minimum=0.
    delayed=manifest['props'][6]>0
    damage_indices=[11,31,51,71] if delayed else [8,28,48,68]
    mesh={e['label']:e for e in manifest['interface_elements']}
    for r in raw:
        if not r.get('SDV68'):continue
        key=(int(r['step']),int(r['increment']))
        label=int(r['element'])
        values=[float(r['SDV%d'%i]) for i in damage_indices]
        if label in previous:minimum=min(minimum,min(a-b for a,b in zip(values,previous[label])))
        previous[label]=values
        states.setdefault(key,{})[label]=values
    records=[]
    for (step,inc),state in sorted(states.items()):
        record=dict(step=step,increment=inc,sampled_elements=len(state),damage_variable='Dv' if delayed else 'D')
        onset=max([manifest['parameters']['a']]+[mesh[e]['x1'] for e,v in state.items() if max(v)>1e-6])
        record['damage_front_any_D_gt_1e6_mm']=onset
        for threshold in [.995,.999]:
            by_x={}
            for e,v in state.items():
                element=mesh[e];key=(element['x0'],element['x1'])
                by_x.setdefault(key,[]).append(min(v))
            front=manifest['parameters']['a']
            for (x0,x1),values in sorted(by_x.items()):
                if x0>front+1e-6 or min(values)<threshold:break
                front=x1
            record['contiguous_all_IP_D_ge_'+str(threshold)+'_mm']=front
            record['cohesive_zone_extent_'+str(threshold)+'_mm']=max(0.,onset-front)
        active=[float(r['SDV%d'%i]) for r in raw if int(r['step'])==step and int(r['increment'])==inc
                for i,j in zip([10,30,50,70],damage_indices) if r.get('SDV%d'%j) and .01<float(r['SDV%d'%j])<.99]
        record['active_local_B_min']=min(active) if active else None
        record['active_local_B_max']=max(active) if active else None
        records.append(record)
    if records:
        with (folder/'assessment/crack_front.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return dict(status='EXTRACTED' if records else 'INCONCLUSIVE',printed_states=len(records),minimum_sampled_damage_increment=minimum,front_damage_variable='Dv' if delayed else 'D',
                final_printed_state=records[-1] if records else None,
                limitation='D is rounded in DAT and printed every ten increments plus final output; fronts bracket mesh intervals and are not experimental crack observations')


def stable_error(u, force, reference):
    """Compare only monotonic stable postpeak domains; no interpolation across jumps."""
    ru,rf=reference[:,0],reference[:,1]
    ip=int(np.argmax(force));rp=int(np.argmax(rf))
    ru,rf=ru[rp:],rf[rp:]
    # Exclude a drawn finite jump and everything before its end. Comparisons
    # use the subsequent stable branch, never an interpolation across the jump.
    jumps=np.flatnonzero((np.diff(ru)<.1)&(-np.diff(rf)>.1*np.max(reference[:,1])))
    if len(jumps):ru,rf=ru[jumps[-1]+1:],rf[jumps[-1]+1:]
    if np.any(np.diff(u[ip:])<=0):return dict(status='INCONCLUSIVE',reason='Simulation includes displacement reversal; no interpolation across unstable branch')
    if len(ru)<3 or np.any(np.diff(ru)<=0):
        return dict(status='INCONCLUSIVE',reason='Reference branch includes duplicate displacement or snap-back; no control-specific jump interpolation')
    left=max(u[ip],ru[0]);right=min(u[-1],ru[-1])
    if right-left<.05:return dict(status='INCONCLUSIVE',reason='Insufficient common stable postpeak interval')
    left+=.05*(right-left)
    xx=np.linspace(left,right,200)
    err=np.interp(xx,u,force)-np.interp(xx,ru,rf)
    return dict(status='ASSESSED',interval_mm=[float(left),float(right)],
                NMAE=float(np.mean(abs(err))/reference[:,1].max()),max_abs_error_over_reference_peak=float(max(abs(err))/reference[:,1].max()))


def reference_for(manifest):
    name=manifest['name'];material=manifest['material'];mix=manifest['mixity_nominal'];family=manifest['family']
    if material=='turon2010' and family=='dcb' and manifest['parameters']['tn']==30:
        path=CAMPAIGN/'reference_data/turon2010CS_dcb/fig04_tau30_graphical.csv'
        return path,dict(kind='published numerical curve, graphical estimate',uncertainty_mm_N=[.03,.8])
    if material=='turon2010_enf':
        path=CAMPAIGN/'reference_data/turon2010CS_enf/fig05_tau60_graphical.csv'
        return path,dict(kind='published pure-II numerical curve, graphical estimate',uncertainty_mm_N=[.04,5.])
    if material=='turon2010' and family=='mmb':
        path=CAMPAIGN/('reference_data/turon2010CS_mmb_v2/fig12_B%d_tau30_58p9_graphical.csv'%int(round(mix*100)))
        return path,dict(kind='published adjusted-strength numerical curve; lower graphical envelope estimate',uncertainty_mm_N=[.05,3.],
                         limitation='Shared analytical/numerical colours; peak/jump neighbourhood ambiguous')
    if material=='krueger2012':
        suffix=str(int(round(mix*100)));figure={20:'10',50:'14',80:'19'}[int(round(mix*100))]
        path=ROOT/('validation/cases/krueger2012CR_mmb'+suffix+'/data/fig'+figure+'_critical_graphical.csv')
        return path,dict(kind='published critical equilibrium symbols; control-specific jump path not supplied',uncertainty_mm_N=[.01,2.])
    if material=='turon2018':
        suffix=str(int(round(mix*100)));figure={20:'02',50:'03',70:'04'}[int(round(mix*100))]
        path=ROOT/('validation/cases/turon2018CS_mmb'+suffix+'/data/fig'+figure+'_tau30_80_graphical.csv')
        return path,dict(kind='published new-law numerical curve, graphical estimate',uncertainty_mm_N=[.01,1. if mix==.2 else 2.])
    return None,None


def reduce(folder):
    manifest=json.loads((folder/'run.json').read_text())
    curve=rows(folder/'assessment/force_displacement.csv')
    if not curve:
        result=dict(name=manifest['name'],solver_status=manifest['status'],completed=False,
            peak_observed=False,peak_eligible=False,assessment_status='INCONCLUSIVE',
            energy_screen='INCONCLUSIVE',stabilization_screen='INCONCLUSIVE',
            reason='No selected response frames; no load, energy or damage inference is available',
            damage=dict(status='NOT RUN',reason='No response frames'))
        dump(folder/'assessment/additional_metrics.json',result);return result
    u=np.array([float(r['displacement_mm']) for r in curve]);f=np.array([float(r['total_force_N']) for r in curve])
    peak=int(np.argmax(f));metrics=json.loads((folder/'assessment/metrics.json').read_text())
    result=dict(name=manifest['name'],solver_status=manifest['status'],
                completed=manifest['status']=='COMPLETED' and metrics['target_reached'],
                peak_observed=bool(peak<len(f)-1 and f[-1]<.98*f[peak]),
                damage=damage(folder,manifest))
    positive=np.flatnonzero((u>1e-9)&(f>1e-8))
    result['first_accepted_secant_compliance_mm_per_N']=float(u[positive[0]]/f[positive[0]]) if len(positive) else None
    result['first_secant_scope']='Single accepted early-load secant; not a multi-point undamaged compliance fit'
    result['peak_eligible']=result['completed'] and result['peak_observed']
    result['peak_eligibility_scope']='Completed loading path with resolved postpeak; energy, contact, convergence and source-equivalence screens remain separate'
    kpath=folder/'kinematic_checks.json'
    if kpath.exists():
        k=json.loads(kpath.read_text());keys=sorted(set(key for r in k for key in r if key.startswith('max_') or key.startswith('lever_')))
        result['kinematics']={key:max(abs(r.get(key,0.)) for r in k) for key in keys}
        if manifest.get('bonded_fixed_crack'):
            final=k[-1]
            if 'vcct_GI_N_per_mm' in final:
                gi=final['vcct_GI_N_per_mm'];gii=final['vcct_GII_N_per_mm'];gt=gi+gii
                result['VCCT']=dict(GI_N_per_mm=gi,GII_N_per_mm=gii,GT_N_per_mm=gt,
                    closure_length_mm=manifest['vcct_closure_length_mm'],
                    evidence='Independent virtual crack closure using tied-front bulk NFORC and behind-front relative nodal displacements; first-order straight-front approximation')
                if gt>0 and gi>=-1e-10 and gii>=-1e-10:
                    B=max(0.,min(1.,gii/gt));p=manifest['parameters'];gc=p['G'][0]+(p['G'][1]-p['G'][0])*B**p['eta'];factor=(gc/gt)**.5
                    result['VCCT'].update(B_GII_over_GT=B,BK_Gc_N_per_mm=gc,linear_critical_scale=factor,
                                          critical_load_N=float(f[-1]*factor),critical_displacement_mm=float(u[-1]*factor))
    else:result['kinematics']=dict(status='NOT RUN',reason='Earlier ODB extraction predates additional strain and lever checks')
    result['energies_final_Nmm']={key:value[-1][1] for key,value in json.loads((folder/'energies.json').read_text()).items() if value}
    histories=json.loads((folder/'energies.json').read_text())
    energies=result['energies_final_Nmm'];denom=max([abs(v[1]) for v in histories.get('ALLWK',[])]+[1e-12])
    result['subset_ledger_final_relative_error']=abs(energies.get('ALLWK',0)-energies.get('ALLIE',0)-energies.get('ALLSD',0)-energies.get('ALLVD',0))/denom
    if histories.get('ETOTAL'):
        total=np.array(histories['ETOTAL'])[:,1];variation=total-total[0]
        result['final_work_balance_relative_error']=float(abs(variation[-1])/denom)
        result['work_balance_relative_error']=float(max(abs(variation))/denom)
        result['energy_balance_method']='Maximum reported ETOTAL variation / peak absolute ALLWK; includes host contact-energy accounting'
    else:
        result['final_work_balance_relative_error']=result['subset_ledger_final_relative_error']
        result['work_balance_relative_error']=result['subset_ledger_final_relative_error']
        result['energy_balance_method']='Subset ledger only; complete contact-energy accounting unavailable'
    result['separate_contact_energy_histories_available']=all(key in histories for key in ['ALLCCE','ALLCCDW'])
    result['stabilization_over_peak_work']=abs(energies.get('ALLSD',0))/denom
    result['artificial_energy_over_peak_work']=abs(energies.get('ALLAE',0))/denom
    result['energy_screen']='INCONCLUSIVE' if denom<=1e-8 else 'PASS' if result['work_balance_relative_error']<=.01 else 'FAIL'
    result['stabilization_screen']='INCONCLUSIVE' if denom<=1e-8 else 'PASS' if result['stabilization_over_peak_work']<=.01 else 'FAIL'
    if manifest['family']=='mmb' and manifest.get('precrack_contact_active') is False:
        overlap=result.get('kinematics',{}).get('max_precrack_projected_normal_penetration_mm')
        result['inactive_contact_check']=dict(maximum_projected_penetration_mm=overlap,
            status='NOT RUN' if overlap is None else 'PASS' if overlap<=1e-5 else 'FAIL',
            tolerance_mm=1e-5,limitation='Projection of upper nodes onto lower piecewise linear x-y traces at each original width station; finite-width3D is diagnostic')
    path,metadata=reference_for(manifest)
    if path and path.exists():
        r=rows(path);reference=np.array([[float(v) for v in row.values()] for row in r])
        rp=int(np.argmax(reference[:,1]));refpeak=reference[rp,1]
        metadata.update(path=str(path.relative_to(ROOT)),peak_graphical_N=float(refpeak),peak_graphical_displacement_mm=float(reference[rp,0]),
                        peak_relative_error=float(f[peak]/refpeak-1) if result['peak_eligible'] else None,
                        maximum_reached_relative_difference=float(f[peak]/refpeak-1))
        if result['completed']:metadata['stable_propagation']=stable_error(u,f,reference)
        else:metadata['stable_propagation']=dict(status='INCONCLUSIVE',reason='Incomplete loading path')
        result['numerical_reference']=metadata
    if manifest['material'].startswith('turon2006'):
        case='turon2006MM_exp_'+('mmb'+str(int(round(manifest['mixity_nominal']*100))) if manifest['family']=='mmb' else manifest['family'])
        observations=rows(ROOT/('validation/cases/'+case+'/data/experimental_peak_sources.csv'))
        result['experimental_comparisons']=[]
        for obs in observations:
            same_crack=abs(float(obs['a0_mm'])-manifest['parameters']['a'])<1e-6
            result['experimental_comparisons'].append(dict(source=obs['source'],experimental_peak_N=float(obs['experimental_Pmax_N']),
                predicted_peak_N=float(f[peak]) if result['peak_eligible'] else None,
                peak_relative_error=float(f[peak])/float(obs['experimental_Pmax_N'])-1 if result['peak_eligible'] and same_crack else None,
                observation_initial_crack_mm=float(obs['a0_mm']),model_initial_crack_mm=manifest['parameters']['a'],same_initial_crack=same_crack,
                maximum_reached_force_N=float(f[peak]),
                status='INCONCLUSIVE',reason=('Initial cracks differ; do not treat this as a matched-specimen prediction. ' if not same_crack else '')+'Baseline prediction only; apparatus reconstruction, numerical convergence, prior toughness calibration and missing experimental scatter limit independent validation'))
        marker_dir=CAMPAIGN/'reference_data/turon2006_experimental_markers'
        candidates=list(marker_dir.glob('fig*_'+case.replace('turon2006MM_exp_','')+'_experimental_markers_graphical.csv'))
        if len(candidates)==1:
            rr=rows(candidates[0]);obs=np.array([[float(r['displacement_mm']),float(r['load_N'])] for r in rr])
            selected=obs[(obs[:,0]>=u.min())&(obs[:,0]<=u.max())]
            comparison=dict(source=str(candidates[0].relative_to(ROOT)),available_markers=len(obs),common_markers=len(selected),
                            scope='Prediction at published marker coordinates only; no experimental curve interpolation or extrapolation; observations are not held-out calibration data')
            if len(selected) and np.all(np.diff(u)>0):
                prediction=np.interp(selected[:,0],u,f);difference=prediction-selected[:,1]
                comparison.update(NMAE_over_graphical_peak=float(np.mean(abs(difference))/max(obs[:,1])),
                                  maximum_relative_error_over_graphical_peak=float(max(abs(difference))/max(obs[:,1])))
                with (folder/'assessment/experimental_marker_comparison.csv').open('w',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(['displacement_mm','experimental_graphical_N','predicted_N','difference_N']);writer.writerows(zip(selected[:,0],selected[:,1],prediction,difference))
            result['experimental_marker_comparison']=comparison
    p=manifest['parameters'];family=manifest['family']
    # Formal equilibrium only. Stop before a reaches the midspan for ENF/MMB.
    aa=np.linspace(p['a'],min(p['L']-.001,p['a']*1.8) if family!='dcb' else min(p['length']-.001,p['a']*1.8),400)
    if family=='dcb':P,U,C=dcb(aa,p['E'][0],p['b'],p['h'],p['G'][0]);B=np.zeros_like(aa)
    elif family=='enf':P,U,C=enf(aa,p['E'][0],p['b'],p['h'],p['G'][1],p['L']);B=np.ones_like(aa)
    else:P,U,C,B=mmb(aa,p['E'][0],p['E'][1],p['E'][6],p['b'],p['h'],*p['G'],p['eta'],p['L'],manifest['lever_c_mm'],gc_policy='beam-energy')
    with (folder/'assessment/lefm_equilibrium.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['crack_mm','displacement_mm','load_N','compliance_mm_per_N','beam_GII_over_GT']);writer.writerows(zip(aa,U,P,C,B))
    result['LEFM']=dict(policy='Printed 2010 Appendix A; MMB uses evolving beam energy fraction and is a formal equilibrium path',
                        initial_critical_load_N=float(P[0]),initial_compliance_mm_per_N=float(C[0]),
                        initial_beam_GII_over_GT=float(B[0]),final_beam_GII_over_GT=float(B[-1]),
                        initial_compliance_relative_error=None if metrics['initial_compliance_mm_per_N'] is None else metrics['initial_compliance_mm_per_N']/C[0]-1)
    if family=='mmb':
        pn,un,cn,bn=mmb(aa,p['E'][0],p['E'][1],p['E'][6],p['b'],p['h'],*p['G'],p['eta'],p['L'],manifest['lever_c_mm'],gc_policy='nominal',nominal_B=manifest['mixity_nominal'])
        with (folder/'assessment/lefm_nominal_Gc_equilibrium.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['crack_mm','displacement_mm','load_N','compliance_mm_per_N','beam_GII_over_GT']);writer.writerows(zip(aa,un,pn,cn,bn))
        result['LEFM']['constant_nominal_Gc_initial_critical_load_N']=float(pn[0])
    dump(folder/'assessment/additional_metrics.json',result)
    print(manifest['name'],manifest['status'],'peak eligible',result['peak_eligible'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folders',type=Path,nargs='*');args=parser.parse_args()
    folders=args.folders or sorted((CAMPAIGN/'runs').iterdir())
    for folder in folders:
        manifest=json.loads((folder/'run.json').read_text())
        if 'family' in manifest and manifest['status']!='RUNNING' and (folder/'assessment/force_displacement.csv').exists():reduce(folder)


if __name__=='__main__':main()
