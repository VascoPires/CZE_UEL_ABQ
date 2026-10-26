"""Reduce executed campaign outputs; distinguish diagnostic checks and validation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from campaign import CAMPAIGN


def load_rows(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def work_history(u,force):
    return np.r_[0,np.cumsum(np.sum(.5*(force[1:]+force[:-1])*(u[1:]-u[:-1]),axis=1))]


def elements(folder):
    manifest=json.loads((folder/'run.json').read_text())
    rows=load_rows(folder/'nodal_response.csv');sdvs=load_rows(folder/'sdv.csv')
    results=[]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for c in manifest['cases']:
        nodes=[r for r in rows if r['group']==c['label']]
        raw=[r for r in sdvs if int(r['element'])==c['element']]
        lookup={int(r['increment']):r for r in raw}
        nodes=[r for r in nodes if int(r['increment'])==0 or int(r['increment']) in lookup]
        u=np.array([[float(r['u%d_mm'%i]) for i in [1,2,3]] for r in nodes])
        force=np.array([[float(r['rf%d_N'%i]) for i in [1,2,3]] for r in nodes])
        work=work_history(u,force)
        energy=[];d=[];stored=[];t=[]
        for r in nodes:
            inc=int(r['increment']);s=lookup.get(inc)
            t.append(float(r['time']))
            energy.append(sum(float(s['SDV%d'%i]) for i in [14,34,54,74]) if s else 0.)
            stored.append(sum(float(s['SDV%d'%i])*.25 for i in [12,32,52,72]) if s else 0.)
            d.append(float(s['SDV8']) if s else 0.)
        energy=np.asarray(energy);stored=np.asarray(stored);d=np.asarray(d)
        balance=float(abs(work[-1]-energy[-1]-stored[-1])/max(abs(work).max(),1e-12))
        expected=c['expected_Gc'];error=None if expected is None else float((energy[-1]-expected)/expected)
        result=dict(label=c['label'],expected_Gc_N_per_mm=expected,final_dissipation_Nmm=float(energy[-1]),
                    final_recoverable_Nmm=float(stored[-1]),external_work_Nmm=float(work[-1]),
                    final_work_balance_relative_error=balance,fracture_energy_relative_error=error,
                    minimum_damage_increment=float(np.diff(d).min()),minimum_dissipation_increment=float(np.diff(energy).min()),
                    final_damage=float(d[-1]),samples=len(nodes),
                    energy_screen='PASS' if error is not None and abs(error)<=.01 else 'FAIL' if error is not None else 'INCONCLUSIVE',
                    work_screen='PASS' if balance<=.01 else 'FAIL',
                    classification='Diagnostic checks only; complete suite/final physical judgment separate')
        results.append(result)
        if c['label'].startswith('B'):
            axes[0,0].plot(np.linalg.norm(u,axis=1),np.linalg.norm(force,axis=1),label=c['label'])
        if c['label'].startswith('mode_change'):
            axes[0,1].plot(t,work-energy-stored,label=c['label'])
        if c['label'].startswith('unload_reload') or c['label']=='closure':
            axes[1,0].plot(u[:,1],force[:,1],label=c['label'])
        if c['label'].startswith('rho') or c['label']=='B0.5':
            axes[1,1].plot(t,energy,label=c['label'])
    axes[0,0].set(xlabel='Separation norm [mm]',ylabel='Traction norm [MPa]')
    axes[0,1].set(xlabel='Step time',ylabel='Work − stored − dissipation [N mm]')
    axes[1,0].set(xlabel='Opening [mm]',ylabel='Normal traction [MPa]')
    axes[1,1].set(xlabel='Step time',ylabel='Dissipation [N mm]')
    for ax in axes.flat:ax.legend(fontsize=7,frameon=False)
    output=folder/'assessment';output.mkdir(exist_ok=False)
    write_json(output/'metrics.json',results)
    fig.savefig(output/'element_results.svg');fig.savefig(output/'element_results.png',dpi=150);plt.close(fig)
    print('Assessed',folder.name,len(results),'cases')


def material(folder):
    cases=json.loads((folder/'cases.json').read_text())
    data=np.genfromtxt(folder/'response.csv',delimiter=',',names=True)
    result=[]
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for c in cases:
        r=data[data['case_id']==c['id']]
        u=np.column_stack([r['ds1'],r['ds2'],r['dn']]);f=np.column_stack([r['ts1'],r['ts2'],r['tn']])
        work=work_history(u,f)[-1]
        smooth=(r['D']>.02)&(r['D']<.98)&(r['dn']>1e-8)
        maximum=float(r['tangent_error'][smooth].max()) if smooth.any() else None
        expected=c['expected_Gc']
        result.append(dict(label=c['label'],energy_reference=expected,final_dissipation=float(r['diss'][-1]),
                           reaction_work=float(work),final_stored=float(r['psi'][-1]),
                           work_balance_error=float((work-r['diss'][-1]-r['psi'][-1])/max(abs(work),1e-12)),
                           fracture_energy_error=None if expected is None else float(r['diss'][-1]/expected-1),
                           smooth_tangent_max_error=maximum,
                           smooth_tangent_screen='PASS' if maximum is not None and maximum<=.01 else 'FAIL' if maximum is not None else 'INCONCLUSIVE',
                           minimum_damage_increment=float(np.diff(r['D']).min()),
                           min_dissipation_increment=float(np.diff(r['diss']).min())))
        if c['label'].endswith('_B0.5'):
            ax.semilogy(r['D'][smooth],np.maximum(r['tangent_error'][smooth],1e-12),label=c['label'])
    output=folder/'assessment';output.mkdir(exist_ok=False)
    write_json(output/'metrics.json',result)
    ax.axhline(.01,color='gray',ls='--',lw=.8,label='1% diagnostic threshold')
    ax.set(xlabel='Damage D',ylabel='Relative tangent error',title='Actual Fortran tangent vs central differences')
    ax.legend(fontsize=7,frameon=False)
    fig.savefig(output/'tangent_error.svg');fig.savefig(output/'tangent_error.png',dpi=150);plt.close(fig)
    print('Assessed',folder.name,len(result),'material histories')


def specimen(folder):
    manifest=json.loads((folder/'run.json').read_text());rows=load_rows(folder/'nodal_response.csv')
    group={g['label']:[r for r in rows if r['group']==g['label']] for g in manifest['groups']}
    family=manifest['family'];scale=manifest['force_scale']
    if family=='dcb':
        selected=group['upper'];opening=np.array([float(a['u2_mm'])-float(b['u2_mm']) for a,b in zip(selected,group['lower'])])
        force=np.array([float(r['rf2_N']) for r in selected])*scale
    else:
        selected=group['lever' if family=='mmb' else 'mid'];opening=-np.array([float(r['u2_mm']) for r in selected]);force=-np.array([float(r['rf2_N']) for r in selected])*scale
    if not len(force):
        output=folder/'assessment';output.mkdir(exist_ok=False)
        metrics=dict(name=manifest['name'],termination=manifest['status'],target_reached=False,
            target_displacement_mm=manifest['opening_mm'],peak_force_N=None,peak_displacement_mm=None,
            maximum_reached_force_N=None,displacement_at_maximum_reached_force_mm=None,last_displacement_mm=None,
            source_hash=manifest.get('uel_sha256'),assessment_status='INCONCLUSIVE',
            evidence_limit='ODB has no selected response frames; inspect retained input-processor/solver errors. No zero load or energy result is inferred.')
        write_json(output/'metrics.json',metrics)
        (output/'force_displacement.csv').write_text('time,displacement_mm,total_force_N\n')
        print(json.dumps(metrics,indent=2));return metrics
    time=np.array([float(r['time']) for r in selected])
    peak=int(np.argmax(force));mask=(force>.05*force[peak])&(force<.2*force[peak])&(np.arange(len(force))<peak)
    compliance=None;r2=None
    if mask.sum()>=3:
        fit=np.polyfit(force[mask],opening[mask],1);compliance=float(fit[0]);prediction=np.polyval(fit,force[mask])
        r2=float(1-np.sum((prediction-opening[mask])**2)/np.sum((opening[mask]-opening[mask].mean())**2))
    energy=json.loads((folder/'energies.json').read_text())
    balance=None
    raw_balance=None;stabilization_fraction=None;artificial_fraction=None
    if 'ALLWK' in energy and 'ALLIE' in energy:
        w=np.array(energy['ALLWK'])[:,1];ie=np.array(energy['ALLIE'])[:,1]
        denom=max(abs(w).max(),1e-12)
        raw_balance=float(abs(w[-1]-ie[-1])/denom)
        dissipated=sum(energy.get(key,[[0.,0.]])[-1][1] for key in ['ALLSD','ALLVD'])
        balance=float(abs(w[-1]-ie[-1]-dissipated)/denom)
        stabilization_fraction=float(abs(energy.get('ALLSD',[[0.,0.]])[-1][1])/denom)
        artificial_fraction=float(abs(energy.get('ALLAE',[[0.,0.]])[-1][1])/denom)
        if energy.get('ETOTAL'):
            total=np.array(energy['ETOTAL'])[:,1];balance=float(abs(total[-1]-total[0])/denom)
    p=manifest['parameters']
    beam_C=8*p['a']**3/(p['b']*p['h']**3*p['E'][0]) if family=='dcb' else None
    reached=bool(abs(opening[-1]-manifest['opening_mm'])<1e-4*manifest['opening_mm'])
    completed=manifest['status']=='COMPLETED' and reached
    metrics=dict(name=manifest['name'],termination=manifest['status'],peak_force_N=float(force[peak]) if completed else None,peak_displacement_mm=float(opening[peak]) if completed else None,
                 maximum_reached_force_N=float(force[peak]),displacement_at_maximum_reached_force_mm=float(opening[peak]),
                 last_displacement_mm=float(opening[-1]),target_displacement_mm=manifest['opening_mm'],target_reached=bool(abs(opening[-1]-manifest['opening_mm'])<1e-4*manifest['opening_mm']),
                 initial_compliance_mm_per_N=compliance,compliance_fit_R2=r2,source_beam_compliance_mm_per_N=beam_C,
                 relative_beam_compliance_error=None if beam_C is None or compliance is None else compliance/beam_C-1,
                 final_solver_energy_balance_error=balance,source_hash=manifest.get('uel_sha256'),
                 raw_work_minus_internal_energy_relative_error=raw_balance,
                 stabilization_over_peak_work=stabilization_fraction,artificial_energy_over_peak_work=artificial_fraction,
                 evidence_limit='Constrained-slice diagnostic with documented apparatus/element assumptions; no automatic experimental validation claim')
    output=folder/'assessment';output.mkdir(exist_ok=False)
    write_json(output/'metrics.json',metrics)
    with (output/'force_displacement.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['time','displacement_mm','total_force_N']);writer.writerows(zip(time,opening,force))
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    ax.plot(opening,force,label='Executed '+manifest['name'],color='#287d87')
    if family=='dcb':
        from lefm_reference import dcb
        aa=np.linspace(p['a'],max(p['a']*1.8,p['a']+10),500)
        P,u,C=dcb(aa,p['E'][0],p['b'],p['h'],p['G'][0]);ax.plot(u,P,'--',color='#555555',label='Source Appendix A beam reference')
        ax.set_xlim(left=0,right=max(manifest['opening_mm'],opening[-1]))
        reference=CAMPAIGN/'reference_data/turon2010CS_dcb/fig04_tau30_graphical.csv'
        if manifest['material']=='turon2010' and reference.exists():
            r=np.loadtxt(reference,delimiter=',',skiprows=5)
            ax.plot(r[:,0],r[:,1],':',color='#aa4130',label='Published tau30 curve (graphical estimate)')
    ax.set(xlabel='Full opening [mm]' if family=='dcb' else 'Applied displacement [mm]',ylabel='Total load [N]')
    ax.legend(fontsize=7,frameon=False);fig.savefig(output/'force_displacement.svg');fig.savefig(output/'force_displacement.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=['elements','material','specimen']);parser.add_argument('folder',type=Path)
    args=parser.parse_args();globals()[args.kind](args.folder)


if __name__=='__main__':main()
