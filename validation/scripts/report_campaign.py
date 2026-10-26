"""Build a traceable result inventory, case matrix and scientific figures.

Reads executed results only. Missing/running jobs cannot become successful rows.
Call reduce_structure.py first to refresh additional structural diagnostics.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from campaign import CAMPAIGN,ROOT
from reduce_structure import rows,reference_for

OUT=CAMPAIGN/'results'
FIG=OUT/'figures'
CASES={
 'element':'element_dt001',
 'turon2010CS_dcb':'dcb_ps03','turon2010CS_enf':'t2010_enf_pc',
 **{'turon2010CS_mmb'+str(m):'t2010_mmb'+str(m)+'_open' for m in [20,50,80]},
 **{'krueger2012CR_mmb'+str(m):'kr_mmb'+str(m)+'_K170k'+('_open' if m==20 else '') for m in [20,50,80]},
 'turon2006MM_exp_dcb':'as4_dcb_ps','turon2006MM_exp_enf':'as4_enf',
 **{'turon2006MM_exp_mmb'+str(m):'as4_mmb'+str(m) for m in [20,50,80]},
 **{'turon2018CS_mmb'+str(m):'t2018_mmb'+str(m)+'_lefmlever' for m in [20,50,70]},
}


def load(path):return json.loads(path.read_text())


def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def inventory():
    records=[]
    for p in sorted((CAMPAIGN/'runs').glob('*/run.json')):
        m=load(p);folder=p.parent
        r=dict(name=m['name'],status=m['status'],family=m.get('family','element_batch'),
               material=m.get('material'),constraint_revision=m.get('constraint_revision',1 if 'family' in m else None),
               input_sha256=m['input_sha256'],uel_sha256=m.get('uel_sha256'),start_utc=m.get('start_utc'),end_utc=m.get('end_utc'),
               max_increment=m.get('max_increment'),rho=m.get('props',[None]*7)[6] if 'props' in m else None,
               K=m.get('parameters',{}).get('K'),mesh_dx_requested=m.get('mesh_dx_requested'),
               stabilization_parameter=m.get('stabilization_energy_fraction'),
               run_path=str(folder.relative_to(ROOT)),source_model=m)
        if (folder/'assessment/metrics.json').exists():r['metrics']=load(folder/'assessment/metrics.json')
        if (folder/'assessment/additional_metrics.json').exists():r['additional']=load(folder/'assessment/additional_metrics.json')
        records.append(r)
    save_json(OUT/'run_inventory.json',records)
    columns=['name','status','family','material','constraint_revision','K','rho','mesh_dx_requested','max_increment','stabilization_parameter',
             'peak_N','maximum_reached_N','last_displacement_mm','work_balance_error','stabilization_over_work','run_path']
    with (OUT/'run_inventory.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for r in records:
            line={k:r.get(k) for k in columns};m=r.get('metrics',{});a=r.get('additional',{})
            if isinstance(m,dict):
                line.update(peak_N=m.get('maximum_reached_force_N') if a.get('peak_eligible') else None,
                            maximum_reached_N=m.get('maximum_reached_force_N'),last_displacement_mm=m.get('last_displacement_mm'),
                            work_balance_error=a.get('work_balance_relative_error'),stabilization_over_work=a.get('stabilization_over_peak_work'))
            writer.writerow(line)
    return records


def case_matrix(records):
    by_name={r['name']:r for r in records};result=[]
    for case,name in CASES.items():
        r=by_name.get(name);classification='NOT RUN';reason='No completed execution record';peak=None;error=None
        evaluated=name
        candidates=(['as4_mmb80_dx01'] if name=='as4_mmb80' else [])+[name+'_adaptive']
        for candidate in candidates:
            alternate=by_name.get(candidate)
            if not alternate:continue
            aa=alternate.get('additional',{})
            if (aa.get('peak_eligible') and aa.get('energy_screen')=='PASS' and aa.get('stabilization_screen')=='PASS'
                    and aa.get('inactive_contact_check',{}).get('status','PASS')=='PASS'):
                if not r or not r.get('additional',{}).get('peak_eligible'):
                    r=alternate;evaluated=alternate['name']
        if r and case=='element':
            classification='FAIL'
            reason='Fixed-basis unit energy checks pass; canonical high-penalty material tangent and elastic changing-area/changing-basis energy checks fail. See separate subchecks.'
        elif r:
            classification='INCONCLUSIVE';a=r.get('additional',{});m=r.get('metrics',{})
            if r['status']=='COMPLETED' and a.get('peak_eligible'):
                peak=m.get('maximum_reached_force_N')
                error=a.get('numerical_reference',{}).get('peak_relative_error')
                reason='Completed prediction; source apparatus equivalence and numerical convergence not established'
                if case=='turon2010CS_dcb':
                    fine=by_name.get('dcb_ps0075',{})
                    reason=('Published numerical curve agrees in this mesh; 0.075mm recovers a full unstabilized path, but intermediate mesh/increment failures and coarse peak sampling prevent a convergence claim'
                            if fine.get('additional',{}).get('peak_eligible') else 'Published numerical curve agrees in this mesh; finer unstabilized mesh/increment paths fail, preventing a convergence claim')
                exp=a.get('experimental_comparisons',[])
                if exp:
                    target=next(x for x in exp if x['source']=='turon2006MM');error=target.get('peak_relative_error')
                    reason='Experimental table comparison only; scatter/raw histories absent, prior toughness fitting and apparatus/numerical uncertainty limit independence'
            else:reason='Loading path incomplete or no resolved postpeak; maximum reached force is not substituted for a verified peak'
        result.append(dict(case=case,primary_run=name,evaluated_run=evaluated,classification=classification,solver_status=r['status'] if r else 'NOT RUN',
                           eligible_peak_N=peak,relative_peak_difference=error,reason=reason))
    save_json(OUT/'verification_validation_matrix.json',result)
    for row in result:
        folder=ROOT/'validation/cases'/row['case']
        if folder.exists():
            current=dict(row,generated_utc=datetime.now(timezone.utc).isoformat(),
                primary_run_manifest='validation/campaign_20261025/runs/'+row['primary_run']+'/run.json',
                evaluated_assessment='validation/campaign_20261025/runs/'+row['evaluated_run']+'/assessment/')
            save_json(folder/'execution_result.json',current)
    with (OUT/'verification_validation_matrix.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    return result


def subcheck_matrix(records):
    checks=[]
    def add(case,check,value,limit,path,eligible=True):
        checks.append(dict(case=case,check=check,metric=value,limit=limit,
            classification='INCONCLUSIVE' if value is None or not eligible else 'PASS' if value<=limit else 'FAIL',
            evidence=path,scope='Diagnostic subcheck; final benchmark verdict requires all relevant evidence and source equivalence'))
    for name in ['element_dt005','element_dt001','element_z8_dt001']:
        path=CAMPAIGN/'runs'/name/'assessment/metrics.json'
        if not path.exists():continue
        for r in load(path):
            add(name+':'+r['label'],'Relative fracture energy error',None if r['fracture_energy_relative_error'] is None else abs(r['fracture_energy_relative_error']),.01,str(path.relative_to(ROOT)))
            add(name+':'+r['label'],'Relative work balance error',r['final_work_balance_relative_error'],.01,str(path.relative_to(ROOT)))
            add(name+':'+r['label'],'Damage decrease beyond rounding',max(0.,-r['minimum_damage_increment']),1e-6,str(path.relative_to(ROOT)))
    for name in ['canonical','zeta1e8','reference_properties_canonical','reference_properties_z8','turon2018_K1e6_canonical','turon2018_K1e6_z8']:
        path=CAMPAIGN/'material'/name/'assessment/metrics.json'
        if path.exists():
            for r in load(path):add('material_'+name+':'+r['label'],'Smooth material tangent error',r['smooth_tangent_max_error'],.01,str(path.relative_to(ROOT)))
        path=CAMPAIGN/'element_tangent'/name/'element_tangent.csv'
        if path.exists():
            for r in rows(path):
                add('assembled_'+name+':'+r['case_id'],'Full assembled residual/tangent error',float(r['tangent_relative_error']),.01,str(path.relative_to(ROOT)))
                add('assembled_'+name+':'+r['case_id'],'Resultant force norm',float(r['resultant_force_norm']),1e-10,str(path.relative_to(ROOT)))
    for r in records:
        a=r.get('additional',{});p=r['run_path']+'/assessment/additional_metrics.json'
        if not a:continue
        loaded=abs(a.get('energies_final_Nmm',{}).get('ALLWK',0.))>1e-8
        add(r['name'],'Relative global work balance error',a.get('work_balance_relative_error'),.01,p,eligible=loaded)
        add(r['name'],'Stabilization energy / peak external work',a.get('stabilization_over_peak_work'),.01,p,eligible=loaded)
        reference=a.get('numerical_reference',{})
        if reference:
            add(r['name'],'Graphical peak discrepancy',None if reference.get('peak_relative_error') is None else abs(reference['peak_relative_error']),.02,p,eligible=a.get('peak_eligible',False))
            prop=reference.get('stable_propagation',{});add(r['name'],'Stable-branch normalized mean absolute error',prop.get('NMAE'),.02,p,eligible=prop.get('status')=='ASSESSED')
        contact=a.get('inactive_contact_check',{})
        if contact:add(r['name'],'Projected initial-crack face penetration [mm]',contact.get('maximum_projected_penetration_mm'),1e-5,p)
        k=a.get('kinematics',{})
        if 'lever_upper_force_magnitude_relative_error' in k:
            add(r['name'],'Lever upper-end force split error',k['lever_upper_force_magnitude_relative_error'],.005,p)
            add(r['name'],'Lever midspan force split error',k['lever_mid_force_magnitude_relative_error'],.005,p)
    for stem in ['area_cycle_checks','basis_cycle_checks']:
        path=OUT/(stem+'.json')
        if path.exists():
            for r in load(path):add(r['name'],'Relative elastic geometry-cycle work balance error',r['maximum_relative_energy_balance_error'],.01,str(path.relative_to(ROOT)))
    path=OUT/'fixed_crack_checks.json'
    if path.exists():
        for r in load(path):
            if 'critical_load_relative_difference' in r:
                add(r['name'],'Independent elastic VCCT critical-load discrepancy',abs(r['critical_load_relative_difference']),.02,str(path.relative_to(ROOT)),eligible=r['status']=='COMPLETED')
    save_json(OUT/'diagnostic_subcheck_matrix.json',checks)
    with (OUT/'diagnostic_subcheck_matrix.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    return checks


def numerical_controls(records):
    """Retain failed paths alongside matched mesh/regularization controls."""
    by_name={r['name']:r for r in records};result=[]
    groups={
        'DCB unstabilized mesh':['dcb_ps06','dcb_ps03','dcb_ps015','dcb_ps0075'],
        'DCB adaptive mesh':['dcb_adaptive_dx0p6','dcb_adaptive_dx0p3','dcb_adaptive_dx0p15'],
        'AS4 MMB80 unstabilized mesh':['as4_mmb80','as4_mmb80_dx01'],
        '2018 MMB20 corrected lever mesh':['t2018_mmb20_lefmlever','t2018_mmb20_lefmlever_dx01'],
        '2018 MMB20 corrected lever penalty':['t2018_mmb20_lefmlever','t2018_mmb20_lefmlever_K1e6'],
        '2018 higher penalty tangent perturbation':['t2018_mmb20_lefmlever_K1e6','t2018_mmb20_lefmlever_K1e6_z8'],
        'DCB fixed viscosity increment':['dcb_ps03_rho1e4_dt02','dcb_ps03_rho1e4_dt001'],
        'DCB fine viscous step stabilization':['dcb_ps03_rho1e4_dt001','dcb_ps03_rho1e4_dt001_adaptive'],
    }
    for group,names in groups.items():
        previous=None
        for name in names:
            if name not in by_name:continue
            r=by_name[name];m=r.get('metrics',{});a=r.get('additional',{})
            peak=m.get('maximum_reached_force_N') if a.get('peak_eligible') else None
            result.append(dict(group=group,run=name,status=r['status'],dx_mm=r.get('mesh_dx_requested'),
                K3_N_per_mm3=r.get('K'),rho=r.get('rho'),max_normalized_increment=r.get('max_increment'),
                stabilization_input=r.get('stabilization_parameter'),resolved_peak_N=peak,
                resolved_peak_displacement_mm=m.get('displacement_at_maximum_reached_force_mm') if peak is not None else None,
                initial_compliance_mm_per_N=m.get('initial_compliance_mm_per_N'),
                maximum_reached_N=m.get('maximum_reached_force_N'),
                peak_relative_change_from_previous_resolved=None if peak is None or previous is None else peak/previous[1]-1,
                previous_resolved_run=previous[0] if previous else None,
                maximum_work_balance_error=a.get('work_balance_relative_error'),
                stabilization_over_peak_work=a.get('stabilization_over_peak_work'),
                energy_screen=a.get('energy_screen'),evidence=r['run_path']+'/assessment/additional_metrics.json'))
            if peak is not None:previous=(name,peak)
    save_json(OUT/'numerical_control_comparisons.json',result)
    with (OUT/'numerical_control_comparisons.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    return result


def execution_audit(records):
    """Expose actual solver endings and audit serial submission intervals."""
    terminations=[];intervals=[]
    for r in records:
        folder=ROOT/r['run_path'];messages=[]
        for suffix in ['.msg','.dat']:
            path=folder/(r['name']+suffix)
            if not path.exists():continue
            with path.open('rb') as handle:
                handle.seek(max(0,path.stat().st_size-131072))
                lines=handle.read().decode(errors='replace').splitlines()
            for i,line in enumerate(lines):
                if '***ERROR' in line.upper():
                    message=' '.join(s.strip() for s in lines[i:i+3] if s.strip())
                    if message not in messages:messages.append(message)
        terminations.append(dict(run=r['name'],status=r['status'],
            error_messages=' | '.join(messages),
            start_utc=r.get('start_utc'),end_utc=r.get('end_utc'),
            evidence=r['run_path']+'/'+r['name']+('.sta' if (folder/(r['name']+'.sta')).exists() else '.dat')))
        if r.get('start_utc') and r.get('end_utc'):
            intervals.append((datetime.fromisoformat(r['start_utc']),datetime.fromisoformat(r['end_utc']),r['name']))
    overlaps=[]
    for i,(start,end,name) in enumerate(sorted(intervals)):
        for later_start,later_end,later_name in sorted(intervals)[i+1:]:
            if later_start>=end:break
            overlaps.append(dict(first=name,second=later_name,overlap_seconds=(min(end,later_end)-later_start).total_seconds()))
    save_json(OUT/'solver_terminations.json',terminations)
    with (OUT/'solver_terminations.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(terminations[0]));writer.writeheader();writer.writerows(terminations)
    by_name={r['name']:r for r in records};queue_errors=[]
    replacements={'t2010_mmb50':'t2010_mmb50_open','kr_mmb20_K170k':'kr_mmb20_K170k_open'}
    for queue in ['queue_open_mmb','queue_diagnostics_final','queue_final_retry','queue_convergence_followup']:
        path=CAMPAIGN/(queue+'.status.json')
        if not path.exists():continue
        for entry in load(path):
            if 'error' not in entry:continue
            original=by_name.get(entry['name'],{});replacement=by_name.get(replacements.get(entry['name']),{})
            folder=CAMPAIGN/'runs'/entry['name'];recovery=original.get('source_model',{}).get('controller_recovery')
            resolved=bool(recovery and (folder/'run_before_controller_recovery.json').exists()
                and original.get('status') in ['FAILED','COMPLETED'] and replacement.get('end_utc')
                and (folder/'assessment/metrics.json').exists())
            queue_errors.append(dict(queue=queue,run=entry['name'],original_error=entry['error'],
                orchestration_resolved=resolved,recovery_record=recovery,replacement_run=replacement.get('name'),
                scope='Resolution concerns stale execution ownership only; original and replacement solver failures remain failed'))
    audit=dict(checked_ended_execution_intervals=len(intervals),overlaps=overlaps,controller_errors=queue_errors,
        unfinished_manifests=[r['name'] for r in records if r['status'] in ['RUNNING','PREPARED']],
        scope='Manifest submission-to-return intervals, including compilation; recovered manifests disclose end-time provenance')
    save_json(OUT/'submission_interval_audit.json',audit)
    return audit


def figure(name,fig,caption,captions):
    for suffix in ['svg','pdf','png']:fig.savefig(FIG/(name+'.'+suffix),dpi=300)
    plt.close(fig);captions.append(dict(name=name,caption=caption))


def curve(name):
    p=CAMPAIGN/'runs'/name/'assessment/force_displacement.csv'
    if not p.exists():return None
    rr=rows(p)
    if not rr:return None
    return np.array([[float(r['displacement_mm']),float(r['total_force_N'])] for r in rr])


def plot_run(ax,name,label=None,color=None):
    c=curve(name)
    if c is None:return
    m=load(CAMPAIGN/'runs'/name/'run.json');failed=m['status']!='COMPLETED'
    line=ax.plot(c[:,0],c[:,1],ls='--' if failed else '-',label=label or name,color=color,lw=1.)[0]
    if failed:ax.plot(*c[-1],marker='x',color=line.get_color(),ms=4,mew=.8)


def reference(ax,name,analytical=True):
    folder=CAMPAIGN/'runs'/name
    if not (folder/'run.json').exists():return
    m=load(folder/'run.json');p,metadata=reference_for(m)
    if p and p.exists():
        rr=rows(p);c=np.array([[float(v) for v in row.values()] for row in rr])
        uncertainty=metadata.get('uncertainty_mm_N') if metadata else None
        if metadata and 'critical equilibrium symbols' in metadata.get('kind',''):
            ax.errorbar(c[:,0],c[:,1],xerr=uncertainty[0] if uncertainty else None,yerr=uncertainty[1] if uncertainty else None,
                color='#555555',ls='none',marker='s',mfc='none',ms=3,capsize=1.5,elinewidth=.5,label='Published critical equilibria')
        else:
            ax.plot(c[:,0],c[:,1],color='#555555',ls=':',lw=.8,label='Published graphical estimate')
            if uncertainty:ax.fill_between(c[:,0],c[:,1]-uncertainty[1],c[:,1]+uncertainty[1],color='#555555',alpha=.10,lw=0)
    path=folder/'assessment/lefm_equilibrium.csv'
    if analytical and path.exists():
        rr=rows(path);c=np.array([[float(r['displacement_mm']),float(r['load_N'])] for r in rr])
        ax.plot(c[:,0],c[:,1],color='#aa6835',ls='-.',lw=.8,label='LEFM equilibrium')
        ax.plot([0,c[0,0]],[0,c[0,1]],color='#aa6835',ls='-.',lw=.8)


def format_ax(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set(xlabel='Applied displacement [mm]',ylabel='Load [N]',xlim=(0,None),ylim=(0,None))


def make_figures(records):
    captions=[];mm=1/25.4
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,
                         'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6})
    fig,axes=plt.subplots(1,2,figsize=(180*mm,70*mm),layout='constrained')
    unit=CAMPAIGN/'runs/element_dt001'
    if (unit/'nodal_response.csv').exists():
        rr=rows(unit/'nodal_response.csv')
        for label in ['B0.0','B0.2','B0.5','B0.8','B1.0','modeIII']:
            r=[v for v in rr if v['group']==label]
            u=np.array([[float(v['u%d_mm'%i]) for i in [1,2,3]] for v in r]);f=np.array([[float(v['rf%d_N'%i]) for i in [1,2,3]] for v in r])
            axes[0].plot(np.linalg.norm(u,axis=1),np.linalg.norm(f,axis=1),lw=.9,label=label.replace('B','B='))
        metrics=load(unit/'assessment/metrics.json');v=[m for m in metrics if m['label'].startswith('B')]
        axes[1].plot([float(m['label'][1:]) for m in v],[m['expected_Gc_N_per_mm'] for m in v],color='#555555',label='BK reference')
        axes[1].plot([float(m['label'][1:]) for m in v],[m['final_dissipation_Nmm'] for m in v],'o',mfc='none',ms=4,color='#267d86',label='Abaqus UEL / unit area')
    axes[0].set(xlabel='Separation norm [mm]',ylabel='Traction norm [MPa]',xlim=(0,.045));axes[1].set(xlabel='Local energy fraction B',ylabel='Dissipated energy [N/mm]')
    for ax in axes:ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    figure('01_element_response',fig,'Actual Abaqus unit-area patches, maximum step increment0.001. Proportional energies are compared with the independently calculated BK reference. Unit patch areas are1mm². This does not establish specimen validation.',captions)
    fig,ax=plt.subplots(figsize=(85*mm,68*mm),layout='constrained')
    for name,label in [('canonical','Canonical perturbation1e-6mm'),('zeta1e8','Test perturbation1e-8mm')]:
        p=CAMPAIGN/'material'/name/'assessment/metrics.json'
        if not p.exists():continue
        r=[r for r in load(p) if r['label'].endswith('_B0.5')]
        ax.loglog([float(r['label'].split('_')[0][1:]) for r in r],[r['smooth_tangent_max_error'] for r in r],'-o',ms=3,label=label)
    ax.axhline(.01,color='#666666',ls=':',lw=.8,label='1% diagnostic threshold');ax.set(xlabel='Normal penalty K [N/mm³]',ylabel='Maximum relative tangent error');ax.legend(frameon=False)
    figure('02_material_tangent',fig,'Actual extracted Fortran constitutive Jacobian against independent central differences on smooth proportional B=.5 branches. The smaller perturbation is an isolated tested copy; the canonical source is unchanged.',captions)
    fig,ax=plt.subplots(figsize=(85*mm,70*mm),layout='constrained');plot_run(ax,'dcb_ps03','UEL, corrected slice',color='#267d86');reference(ax,'dcb_ps03');format_ax(ax);ax.set(xlabel='Full relative opening [mm]',xlim=(0,4.1));ax.legend(frameon=False)
    figure('03_dcb_reference',fig,'Turon2010 DCB, local mesh0.3mm and K=1e6N/mm³. The published30MPa series is a graphical estimate (±0.03mm,±0.8N). Printed Appendix A1–A2 give a different initial compliance; the formal propagation branch and its elastic extension are shown without fitting crack length.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,73*mm),layout='constrained')
    for name,label in [('dcb_ps06','0.6mm'),('dcb_ps03','0.3mm'),('dcb_ps015','0.15mm'),('dcb_ps0075','0.075mm')]:plot_run(axes[0],name,label)
    for name,label in [('dcb_ps03','Unstabilized 0.3mm'),('dcb_adaptive_dx0p3','Adaptive 0.3mm'),('dcb_adaptive_dx0p15','Adaptive 0.15mm'),('dcb_ps03_stab2e5','Constant damping 0.3mm')]:plot_run(axes[1],name,label)
    for ax in axes:format_ax(ax);ax.set(xlabel='Full relative opening [mm]');ax.legend(frameon=False)
    figure('04_dcb_mesh_stabilization',fig,'Corrected plane-strain slice sensitivity. Dashed curves and crosses end at failed jobs; those endpoints are not substituted for peak or propagation results. The constant-damping trace uses about16% stabilization/work and is rejected as a converged prediction. Explicit-adaptive controls use ALLSDTOL=.005; their actual energy fractions are reported separately in the run inventory.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,70*mm),layout='constrained')
    for name,label in [('dcb_ps03_K1e5','K=1e5'),('dcb_ps03','K=1e6'),('dcb_ps03_K1e7','K=1e7')]:plot_run(axes[0],name,label)
    for name,label in [('dcb_ps03','Maximum increment0.02'),('dcb_ps03_inc005','Maximum increment0.005'),('dcb_ps03_gauss','Gauss, increment0.02'),('dcb_ps03_nlgeom','NLGEOM, increment0.02')]:plot_run(axes[1],name,label)
    for ax in axes:format_ax(ax);ax.set(xlabel='Full relative opening [mm]');ax.legend(frameon=False)
    figure('05_dcb_numerical_sensitivity',fig,'One-factor DCB penalty, increment, integration and bulk-kinematics studies. Normal and shear penalty vary together through the UEL stiffness relation. Failed traces are dashed and end with crosses; comparisons do not imply converged propagation.',captions)
    for family,mixes,names,figure_name,caption in [
      ('Turon2010',[20,50,80],['t2010_mmb20_open','t2010_mmb50_open','t2010_mmb80_open'],'06_turon2010_mmb','Turon2010 adjusted-strength MMB comparisons. Graphical targets are conservative lower-envelope estimates with±0.05mm/±3N read-off uncertainty. Printed LEFM uses evolving analytical energy partition. The linear weight-free lever is a documented reconstruction.'),
      ('Krueger2012',[20,50,80],['kr_mmb20_K170k_open','kr_mmb50_K170k','kr_mmb80_K170k'],'07_krueger_mmb','Krueger critical equilibrium symbols, not prescribed-displacement jump trajectories. The diagnostic UEL uses assumed strengths32/61MPa and penalty170000N/mm³, which the VCCT source does not supply. The narrow slice cannot reproduce finite-width fronts.'),
      ('Turon2018',[20,50,70],['t2018_mmb20_lefmlever','t2018_mmb50_lefmlever','t2018_mmb70_lefmlever'],'08_turon2018_mmb','Turon2018 new-law30/80MPa graphical series versus executed UEL traces. The corrected-beam reconstruction enforces the nominal initial analytical mode fraction; the original ideal-lever reconstruction is retained as a separate sensitivity. Neither apparatus is an established author-model reproduction. Material parameters are unchanged.')]:
        fig,axes=plt.subplots(1,3,figsize=(180*mm,68*mm),layout='constrained')
        for ax,mix,name in zip(axes,mixes,names):
            plot_run(ax,name,'Executed UEL',color='#267d86');reference(ax,name);format_ax(ax);ax.set_title('Nominal '+str(mix)+'%',fontsize=8);ax.legend(frameon=False,fontsize=5.6)
            plot_run(ax,name+'_adaptive','Adaptive control',color='#78639a')
            ax.legend(frameon=False,fontsize=5.6)
            if family=='Turon2018':
                plot_run(ax,'t2018_mmb'+str(mix),'Ideal-lever control',color='#9a9a9a')
                if mix==20:plot_run(ax,'t2018_mmb20_lefmlever_dx01','0.1mm mesh',color='#b47b35')
                ax.legend(frameon=False,fontsize=5.8)
        figure(figure_name,fig,caption+' Dashed/crossed traces are incomplete. Named adaptive controls are separate numerical sensitivities and require their own energy checks.',captions)
    fig,ax=plt.subplots(figsize=(180*mm,75*mm),layout='constrained');peak_labels=set()
    for index,case in enumerate(['dcb','mmb20','mmb50','mmb80','enf']):
        rr=rows(ROOT/('validation/cases/turon2006MM_exp_'+case+'/data/experimental_peak_sources.csv'))
        for j,r in enumerate(rr):
            ax.plot(index+(j-.5)*.12,float(r['experimental_Pmax_N']),'s',mfc='#777777' if j==0 else 'none',mec='#777777',ms=4,label=('Turon2006 table' if j==0 else 'Camanho2002 table') if index==0 else None)
        selected={r['case']:r['evaluated_run'] for r in load(OUT/'verification_validation_matrix.json')}
        name=selected['turon2006MM_exp_'+case];c=curve(name)
        if c is not None:
            p=CAMPAIGN/'runs'/name/'assessment/additional_metrics.json';a=load(p) if p.exists() else {}
            eligible=a.get('peak_eligible',False)
            label='UEL peak' if eligible else 'Maximum accepted force; incomplete'
            ax.plot(index+.20,c[:,1].max(),marker='o' if eligible else '^',mfc='#267d86' if eligible else 'none',mec='#267d86',ms=4,label=label if label not in peak_labels else None)
            peak_labels.add(label)
    ax.set_xticks(range(5),['DCB','MMB20%','MMB50%','MMB80%','ENF']);ax.set(ylabel='Load [N]',ylim=(0,None));ax.legend(frameon=False)
    figure('09_experimental_peak_tables',fig,'Separate experimental peak transcriptions are retained. Turon2006 is the primary approved observation table; Camanho2002 is not merged with it. UEL markers use the mechanically screened case-matrix selection. Open triangles denote maximum accepted force from an incomplete prediction, not an eligible peak; energy-failing completed adaptive controls remain visible in figure12. Missing scatter, raw histories and previously fitted toughness prevent an independent validation claim.',captions)
    fig,axes=plt.subplots(2,3,figsize=(180*mm,120*mm),layout='constrained')
    for ax,case in zip(axes.flat,['dcb','mmb20','mmb50','mmb80','enf']):
        name=CASES['turon2006MM_exp_'+case];plot_run(ax,name,'Unstabilized UEL',color='#267d86')
        plot_run(ax,name+'_adaptive','Adaptive control',color='#b47b35')
        if case=='mmb80':plot_run(ax,'as4_mmb80_dx01','0.1mm, unstabilized',color='#6b648d')
        candidates=list((CAMPAIGN/'reference_data/turon2006_experimental_markers').glob('fig*_'+case+'_experimental_markers_graphical.csv'))
        if len(candidates)==1:
            rr=rows(candidates[0]);ax.plot([float(r['displacement_mm']) for r in rr],[float(r['load_N']) for r in rr],'s',mfc='none',mec='#444444',ms=2.7,label='Published experimental markers')
        format_ax(ax);ax.set_title(case.upper(),fontsize=8);ax.legend(frameon=False,fontsize=5.5)
    axes.flat[-1].axis('off')
    figure('12_experimental_marker_comparisons',fig,'Turon2006 Figs9–13 experimental square-marker coordinates, directly digitized from vector paths. No origin or missing measurement is inserted. DCB displacement is full relative opening; MMB displacement is at the load coordinate; ENF is midspan displacement. Dashed/crossed prediction traces are incomplete. Sparse propagation observations, previously fitted toughness/eta and missing raw records/scatter prevent independent validation.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,70*mm),layout='constrained')
    p=CAMPAIGN/'material/extended_diagnostics/assessment/metrics.json'
    if p.exists():
        rr=load(p)
        for rho in ['0.0001','0.001']:
            r=[v for v in rr if v['label'].startswith('visc_rho'+rho+'_')]
            axes[0].loglog([1/int(v['label'].split('_n')[-1]) for v in r],[abs(v['work_balance_error']) for v in r],'-o',ms=3,label='rho='+rho)
    axes[0].axhline(.01,color='#777777',ls=':',lw=.8);axes[0].set(xlabel='Material history increment',ylabel='Relative work/energy error');axes[0].legend(frameon=False)
    labels=[];errors=[]
    for name,label in [('dcb_ps03','rho=0'),('dcb_rho1e4','rho=1e-4'),('dcb_rho1e3','rho=1e-3'),('dcb_local015_rho1e5','rho=1e-5,fine')]:
        p=CAMPAIGN/'runs'/name/'assessment/additional_metrics.json'
        if p.exists():labels.append(label);errors.append(max(load(p)['work_balance_relative_error'],1e-8))
    axes[1].bar(range(len(labels)),errors,color='#267d86',width=.6);axes[1].set_yscale('log');axes[1].set_xticks(range(len(labels)),labels,rotation=20,ha='right');axes[1].axhline(.01,color='#777777',ls=':',lw=.8);axes[1].set(ylabel='Relative work/energy error')
    figure('10_energy_consistency',fig,'Left: endpoint-quadrature energy error from the actual canonical material routines under proportional viscous loading decreases with refinement. Right: actual structural energy errors can remain unacceptable despite successful termination. The viscous structural cases predate the corrected end constraints and are diagnostics; do not compare them as source-equivalent predictions.',captions)
    fig,ax=plt.subplots(figsize=(85*mm,68*mm),layout='constrained')
    for name,label in [('dcb_ps03','0.3mm'),('dcb_ps0075','0.075mm')]:
        folder=CAMPAIGN/'runs'/name;p=folder/'assessment/crack_front.csv'
        if not p.exists():continue
        rr=rows(p);nod=rows(folder/'nodal_response.csv');upper={int(r['increment']):r for r in nod if r['group']=='upper'};lower={int(r['increment']):r for r in nod if r['group']=='lower'}
        valid=[r for r in rr if int(r['increment']) in upper]
        u=[float(upper[int(r['increment'])]['u2_mm'])-float(lower[int(r['increment'])]['u2_mm']) for r in valid]
        for threshold,style in [('0.995','-'),('0.999',':')]:
            ax.plot(u,[float(r['contiguous_all_IP_D_ge_'+threshold+'_mm']) for r in valid],style,marker='o',ms=2,label=label+',D≥'+threshold)
    ax.set(xlabel='Full relative opening [mm]',ylabel='Contiguous separated front x [mm]');ax.legend(frameon=False,fontsize=5.8)
    figure('11_dcb_front_definition',fig,'Completed unstabilized DCB0.3mm and0.075mm controls. Damage-based fronts use undeformed mesh coordinates and require every UEL sampling point across each contiguous mesh interval to exceed the stated threshold. DAT output is rounded and sampled every ten increments plus final output; these sparse histories do not establish temporal crack-front convergence. These fronts are not measured crack lengths.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,70*mm),layout='constrained')
    p=OUT/'fixed_crack_checks.json'
    if p.exists():
        rr=load(p)
        kr=[r for r in rr if r['name'].startswith('fixed_krueger2012_mmb') and '_dx' not in r['name'] and r.get('critical_load_N') is not None]
        if kr:
            kr=sorted(kr,key=lambda r:r['published_annotated_critical_N_mm'][0]);x=[r['published_annotated_critical_N_mm'][0] for r in kr]
            axes[0].plot(x,[r['critical_load_N'] for r in kr],'o',mfc='none',ms=4,label='Independent tied-FE/VCCT')
            axes[0].plot([0,max(x)*1.05],[0,max(x)*1.05],':',color='#777777',label='Equality')
        for family in ['turon2010','krueger2012']:
            values=[]
            for r in rr:
                if not r['name'].startswith('fixed_'+family+'_mmb') or '_dx' in r['name'] or 'B_GII_over_GT' not in r:continue
                manifest=load(CAMPAIGN/'runs'/r['name']/'run.json');values.append([manifest['mixity_nominal'],r['B_GII_over_GT']])
            if values:
                v=np.array(sorted(values));axes[1].plot(v[:,0],v[:,1],'o-',ms=3,label=family)
        axes[1].plot([0,1],[0,1],':',color='#777777',label='Nominal equality')
    axes[0].set(xlabel='Krueger annotated critical load [N]',ylabel='Independent FE/VCCT critical load [N]',xlim=(0,None),ylim=(0,None))
    axes[1].set(xlabel='Nominal GII/GT',ylabel='Independent VCCT GII/GT',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.legend(frameon=False)
    figure('13_fixed_crack_vcct',fig,'Independent tied fixed-crack elastic FE controls: no cohesive law is active. Critical loads are obtained by scaling the measured VCCT energy release to the source BK toughness, using the achieved partition. These checks address elastic loading/energy-release definitions, not UEL propagation or experimental validation. Closure-length sensitivity and apparatus limitations remain separate.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,70*mm),layout='constrained')
    for ax,stem,canonical,test in zip(axes,['area','basis'],['area_cycle_canonical_fine','basis_cycle_canonical'],['area_cycle_reference','basis_cycle_reference']):
        for name,label,color,style in [(canonical,'Canonical: current area','#a44234','-'),(test,'Test: reference area','#267d86','--')]:
            p=CAMPAIGN/'element_tangent'/name/(stem+'_cycle.csv')
            if not p.exists():continue
            d=np.genfromtxt(p,delimiter=',',names=True)
            ax.plot(d['path_time'],d['external_work_Nmm']-d['recoverable_Nmm']-d['dissipation_Nmm'],color=color,label=label,lw=1.,ls=style)
        for x in [.25,.5,.75]:ax.axvline(x,color='#dddddd',lw=.6,ls=':')
        ax.set(xlabel='Closed-cycle path coordinate',ylabel='Work − stored − dissipated [N mm]',xlim=(0,1),title='Area change' if stem=='area' else 'Basis change at constant area')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(frameon=False,fontsize=6.)
    figure('14_elastic_geometry_cycles',fig,'Actual full Fortran UEL with serial runtime stubs, rho=0, elastic0.0001mm maximum gap and restored initial geometry. Left: area1→1.2 at fixed gap; canonical100/1000-interval tests leave -1e-4Nmm net work, whereas the reference-area hypothesis closes this restricted cycle. Right: constant-area midsurface rotation0→45degrees at fixed global gap direction, closure and orientation restoration; both variants leave about+1.3791e-4Nmm net work with zero final stored/dissipated energy. This is not a rigid-body rotation of the entire separated configuration. The reference-area variant is therefore not a complete geometric-energy correction.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,75*mm),layout='constrained')
    controls=[('dcb_ps03','rho=0, dt≤0.02'),('dcb_ps03_rho1e4_dt02','rho=1e-4, dt≤0.02'),
              ('dcb_ps03_rho1e4_dt001','rho=1e-4, dt≤0.001'),
              ('dcb_ps03_rho1e4_dt001_adaptive','rho=1e-4, dt≤0.001, adaptive')]
    for name,label in controls:
        plot_run(axes[0],name,label)
        path=CAMPAIGN/'runs'/name/'energies.json'
        if not path.exists():continue
        energy=load(path)
        # Plot the recorded total-energy ledger on its own time sampling.
        if not all(k in energy for k in ['ALLWK','ETOTAL']):continue
        wk=np.asarray(energy['ALLWK'],dtype=float);total=np.asarray(energy['ETOTAL'],dtype=float)
        scale=np.max(np.abs(wk[:,1]))
        if scale>1e-12:
            drift=np.abs(total[:,1]-total[0,1])/scale
            axes[1].plot(total[:,0],np.where(drift>0,drift,np.nan),label=label,lw=1.)
    format_ax(axes[0]);axes[0].set(xlabel='Full relative opening [mm]');axes[0].legend(frameon=False,fontsize=5.7)
    axes[1].axhline(.01,color='#777777',ls=':',lw=.8);axes[1].set(xlabel='Normalized step time',ylabel='Absolute ETOTAL drift / peak work',yscale='log',ylim=(1e-6,2))
    axes[1].legend(frameon=False,fontsize=5.7)
    figure('15_dcb_viscosity_increment',fig,'Matched corrected-constraint DCB controls at local0.3mm spacing: fixed material/K, rho varied explicitly, then the maximum normalized step increment reduced at fixed rho. Step duration is1; rho is a numerical regularization time scale in the same units. An adaptive control is executed only if the fine-increment viscous baseline fails. Failed traces remain dashed/crossed. The energy screen uses maximum ETOTAL drift divided by peak external work; exact zero drift is omitted on the logarithmic scale and ALLSD is assessed separately. A smaller time increment resolves viscous evolution and does not establish a rate-independent result.',captions)
    fig,axes=plt.subplots(1,2,figsize=(180*mm,73*mm),layout='constrained')
    for name,label,color in [('t2018_mmb20_lefmlever','Source K=1e5','#267d86'),
                             ('t2018_mmb20_lefmlever_K1e6','Test K=1e6','#b47b35'),
                             ('t2018_mmb20_lefmlever_K1e6_z8','K=1e6, smaller perturbation','#78639a')]:
        plot_run(axes[0],name,label,color=color)
    reference(axes[0],'t2018_mmb20_lefmlever',analytical=False);format_ax(axes[0]);axes[0].set(xlabel='Applied displacement [mm]',xlim=(0,11.2));axes[0].legend(frameon=False,fontsize=5.7)
    for names,label,color in [(['reference_properties_canonical','turon2018_K1e6_canonical'],'Canonical perturbation','#b47b35'),
                             (['reference_properties_z8','turon2018_K1e6_z8'],'Smaller perturbation','#78639a')]:
        values=[]
        for K,name in zip([1e5,1e6],names):
            path=CAMPAIGN/'material'/name/'assessment/metrics.json'
            if path.exists():
                row=next(r for r in load(path) if r['label']=='turon2018_B0.5')
                values.append((K,row['smooth_tangent_max_error']))
        if values:
            v=np.array(values);axes[1].loglog(v[:,0],v[:,1],'-o',color=color,ms=3,label=label)
    axes[1].axhline(.01,color='#777777',ls=':',lw=.8,label='1% diagnostic threshold')
    axes[1].set(xlabel='Normal penalty K [N/mm³]',ylabel='Material tangent error at B=0.5');axes[1].legend(frameon=False,fontsize=5.7)
    figure('16_mmb_penalty_tangent',fig,'2018 MMB20 corrected-lever reconstruction, fixed geometry/material/strengths and local0.3mm mesh. The source K3=1e5 remains the baseline; the higher-K control preserves the normal/shear stiffness ratio. Its smaller-perturbation structural retry is conditional on failure of the same-K canonical-perturbation case. Dashed/crossed traces are incomplete and their maxima are not verified peaks. Right: exact property-set standalone material checks isolate tangent accuracy at fixed inputs; all non-tangent responses match between perturbation copies in these histories. These numerical controls do not approve a physical reformulation or a canonical-source replacement.',captions)
    save_json(OUT/'figure_captions.json',captions)
    (OUT/'FIGURE_CAPTIONS.md').write_text('\n\n'.join('**'+v['name']+'**. '+v['caption'] for v in captions)+'\n')
    return captions


def make_overview():
    """Rebuild the review collage from four registered scientific figures."""
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,name in zip(axes.flat,['01_element_response','03_dcb_reference','08_turon2018_mmb','14_elastic_geometry_cycles']):
        ax.imshow(plt.imread(FIG/(name+'.png')));ax.axis('off')
    fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)


def preservation():
    original=load(ROOT/'validation/audit/phase0_2026-10-25/baseline_files.json')
    selected={p:v for p,v in original.items() if p.startswith(('src/','examples/'))}
    changed=[]
    for p,v in selected.items():
        path=ROOT/p
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=v['sha256']:changed.append(p)
    run_mismatches=[];checked_runs=0
    for path in (CAMPAIGN/'runs').glob('*/run.json'):
        m=load(path);folder=path.parent;checked_runs+=1
        deck=folder/(m['name']+'.inp')
        if not deck.exists() or hashlib.sha256(deck.read_bytes()).hexdigest()!=m['input_sha256']:run_mismatches.append(m['name']+': input deck')
        if m.get('uel_sha256'):
            source=folder/'3D_CZM.f'
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=m['uel_sha256']:run_mismatches.append(m['name']+': copied source')
    result=dict(checked_files=len(selected),changed_or_missing=changed,checked_run_decks_and_sources=checked_runs,
                run_hash_mismatches=run_mismatches,canonical_sha256=hashlib.sha256((ROOT/'src/3D_CZM.f').read_bytes()).hexdigest())
    save_json(OUT/'preservation_check.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    records=inventory();matrix=case_matrix(records);checks=subcheck_matrix(records);numerical_controls(records);audit=execution_audit(records);preserved=preservation()
    captions=make_figures(records) if args.figures else load(OUT/'figure_captions.json') if (OUT/'figure_captions.json').exists() else []
    if args.figures:make_overview()
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),solver_status_counts=dict(Counter(r['status'] for r in records)),
                 case_status_counts=dict(Counter(r['classification'] for r in matrix)),figures=len(captions),preservation=preserved)
    summary['diagnostic_subcheck_status_counts']=dict(Counter(r['classification'] for r in checks))
    summary['submission_interval_overlaps']=len(audit['overlaps'])
    summary['unresolved_controller_errors']=sum(not r['orchestration_resolved'] for r in audit['controller_errors'])
    summary['figures_regenerated']=args.figures
    save_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
