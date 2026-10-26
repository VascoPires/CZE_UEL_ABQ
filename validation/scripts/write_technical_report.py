"""Write the technical investigation report from current retained assessments."""
import json
import hashlib
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from campaign import ROOT,CAMPAIGN


def load(path):return json.loads(path.read_text())
def number(value,digits=4):return '—' if value is None else ('%.*g'%(digits,value))
def percent(value):return '—' if value is None else ('%+.3f%%'%(100*value))


def reference_area_summary():
    path=CAMPAIGN/'results/reference_area_comparison.json'
    if not path.exists():return []
    data=load(path);lines=['','[Matched area-measure comparisons](campaign_20261025/results/reference_area_comparison.json) retain separate unit and specimen evidence.']
    if data['unit_histories']:
        error=max(abs(r['dissipation_difference_Nmm']) for r in data['unit_histories'])
        lines+=['The '+str(len(data['unit_histories']))+' constant-area unit histories have maximum dissipation difference '+number(error,6)+'Nmm between these copies at retained DAT precision.']
    if data['structural_comparisons']:
        lines+=['','| Area-measure test | Baseline peak [N] | Test peak [N] | Peak change | Baseline work error | Test work error |',
                '|---|---:|---:|---:|---:|---:|']
        for r in data['structural_comparisons']:
            lines.append('| '+r['test']+' | '+number(r['baseline_peak_N'])+' | '+number(r['test_peak_N'])+' | '+percent(r['relative_peak_difference'])+' | '+percent(r['baseline_energy_error'])+' | '+percent(r['test_energy_error'])+' |')
    return lines+['']


def penalty_comparison(by_name):
    before=by_name.get('t2018_mmb20_lefmlever_K1e6',{});after=by_name.get('t2018_mmb20_lefmlever_K1e6_z8',{})
    a=before.get('metrics',{});b=after.get('metrics',{});check=after.get('additional',{})
    if not a or not b:return []
    text='The higher-K canonical-perturbation case '+before['status'].lower()+' at '+number(a.get('last_displacement_mm'),6)+'mm of11mm. '
    text+='The same-K smaller-perturbation case '+after['status'].lower()+' at '+number(b.get('last_displacement_mm'),6)+'mm'
    if check.get('peak_eligible'):
        text+=', with peak '+number(b['maximum_reached_force_N'],6)+'N and maximum energy drift '+percent(check.get('work_balance_relative_error'))
    text+='. Both use zero viscosity and zero step stabilization. This comparison changes the Jacobian approximation while preserving the tested constitutive response; it does not cure the independent geometric work defects.'
    return ['',text,'']


def viscosity_comparison(by_name):
    names=['dcb_ps03_rho1e4_dt02','dcb_ps03_rho1e4_dt001']
    runs=[by_name.get(name,{}) for name in names]
    if not all(r.get('additional',{}).get('peak_eligible') for r in runs):return []
    a,b=[r['metrics'] for r in runs];ea,eb=[r['additional']['work_balance_relative_error'] for r in runs]
    change=b['maximum_reached_force_N']/a['maximum_reached_force_N']-1
    return ['',
        'The matched corrected-constraint rho=1e-4 DCB comparison holds material, geometry, mesh and viscosity fixed. '
        'Reducing the maximum normalized step increment0.02→0.001 lowers maximum energy drift from '+percent(ea)+' to '+percent(eb)+'. '
        'Both runs complete4mm without step stabilization. Their recorded peaks change from '+number(a['maximum_reached_force_N'],6)+' to '+number(b['maximum_reached_force_N'],6)+'N ('+percent(change)+'). '
        'The fine run passes the1% energy screen, but the peak change still exceeds the1% increment-sensitivity screen. '
        'These two levels therefore demonstrate strong increment sensitivity, not established temporal convergence or a zero-viscosity limit. The conditional adaptive retry is skipped because the fine run completes.','']


def main():
    out=CAMPAIGN/'results';inventory=load(out/'run_inventory.json');matrix=load(out/'verification_validation_matrix.json')
    by_name={r['name']:r for r in inventory};counts=Counter(r['status'] for r in inventory)
    running=[r['name'] for r in inventory if r['status'] in ['RUNNING','PREPARED']]
    audit=load(out/'submission_interval_audit.json') if (out/'submission_interval_audit.json').exists() else {}
    unresolved=sum(not r['orchestration_resolved'] for r in audit.get('controller_errors',[]))
    # A controller can be between jobs, so manifests alone do not prove that
    # the planned queue is finished. Include the live queue status too.
    pending=[]
    for name in ['queue_open_mmb','queue_diagnostics_final','queue_final_retry','queue_convergence_followup']:
        tasks=load(CAMPAIGN/(name+'.json'));path=CAMPAIGN/(name+'.status.json')
        state=load(path) if path.exists() else []
        ended={r['name'] for r in state if 'ended_utc' in r}
        pending.extend(t['name'] for t in tasks if t['name'] not in ended)
    lines=['# Abaqus UEL verification and validation investigation — 25 October 2026','',
        'Campaign started25 October2026; assessment generated '+datetime.now(timezone.utc).isoformat()+'.','',
        '**Execution status: '+('IN PROGRESS. Remaining planned tasks: '+str(len(pending))+'. Unresolved controller errors: '+str(unresolved)+'.' if pending or running or unresolved else 'EXECUTION AND ASSESSMENT COMPLETE for the recorded campaign.')+'**',
        '',
        'The campaign executes the updated autonomous [Prompt0.md](../Prompt0.md), under [decisions D6–D17](DECISIONS.md). '
        'It includes actual Abaqus2023/Standard jobs, independent constitutive and assembled-element Fortran checks, numerical sensitivity studies, published graphical comparisons and provisional experimental comparisons. '
        'The canonical UEL is unchanged. An elastic changing-area test demonstrates an energy defect even at rho=0. This report records passed subchecks, failed checks and insufficient evidence separately; it does not declare the implementation fully validated.',
        '',
        'The initial reference-only audit is retained in [REPORT_historical.md](audit/phase0_2026-10-25/REPORT_historical.md). '
        'Its old NOT RUN/checkpoint status is historical. Current source cards, actual model manifests and results have distinct roles.',
        '',
        '## Scope and evidence','',
        '| Evidence | Actual execution / retained record |','|---|---|',
        '| Abaqus jobs | '+', '.join(str(v)+' '+k for k,v in counts.items())+'; [full run inventory](campaign_20261025/results/run_inventory.csv) |',
        '| Unit patches |16 independent patches per batch; both integration schemes, six proportional modes, history changes, closure, K and rho studies |',
        '| Material checks |Canonical and smaller-perturbation copies use the same23 histories; an additional49-history run includes20 tightly bracketed onset probes and six viscous refinements |',
        '| Full UEL assembly |Nine residual/Jacobian states per source variant, including rigid rotations0/15/30degrees; serial runtime stubs, separate from Abaqus solver-interface evidence |',
        '| Structural controls |Native cohesive comparisons, ordinary-plane-strain checks, elastic fixed-crack ties, independent VCCT and compliance derivatives where completed |',
        '| Experiments |Turon2006 Table3 is primary; Camanho2002 Table4 remains separate. Published experimental markers are digitized; original instrument records/repeat scatter remain unavailable |',
        '',
        'All working numerical thresholds are declared in [PLAN.md](PLAN.md). They are diagnostic screens, not previously approved experimental validation tolerances: '
        'proportional energy1%, smooth tangent1%, work balance1%, successive numerical peak changes1%, source-equivalent numerical peak2% and stable-branch mean error2%. '
        'Reference uncertainty, unresolved apparatus equivalence or failed convergence can prevent PASS even when an individual error falls below a threshold. '
        'No force/displacement curve is shifted, and no strength/toughness is fitted to the target observations.',
        '',
        '## Seventeen-case verification and validation matrix','',
        '| Case | Verdict | Evaluated job | Solver | Resolved peak [N] | Peak difference |',
        '|---|---|---|---|---:|---:|']
    for r in matrix:
        run=r.get('evaluated_run',r['primary_run'])
        lines.append('| '+r['case']+' | '+r['classification']+' | ['+run+'](campaign_20261025/runs/'+run+'/run.json) | '+r['solver_status']+' | '+number(r['eligible_peak_N'])+' | '+percent(r['relative_peak_difference'])+' |')
    lines+=['',
        'A resolved peak requires a completed loading path and a recorded postpeak drop. Energy/contact/convergence screens are additional requirements. '
        'Failed-job maxima are retained in the inventory as **maximum reached force**, never substituted for a verified peak. '
        'Experimental differences use the primary2006 table; numerical differences use the registered graphical series. The full machine-readable matrix includes the reason for every verdict.',
        '',
        '[Solver termination register](campaign_20261025/results/solver_terminations.csv) preserves actual error messages, including preprocessing failures. '
        'An ODB without response frames yields an inconclusive assessment with unavailable values; it is not treated as a zero-force or zero-energy result. '
        '[Submission interval audit](campaign_20261025/results/submission_interval_audit.json) checks serial execution from retained timestamps.',
        '',
        'When an unstabilized baseline fails, a named adaptive control or the finer unstabilized AS4MMB80 control is used for the reported resolved peak only if it completes with a postpeak drop, work error≤1%, stabilization/work≤1% and the applicable inactive-contact check passes. The finer unstabilized control takes precedence over an adaptive control when both qualify. This selection does not use agreement with target data; the original baseline remains in the inventory. The overall benchmark verdict still requires convergence and source equivalence. [Diagnostic subcheck matrix](campaign_20261025/results/diagnostic_subcheck_matrix.csv) records individual PASS/FAIL/INCONCLUSIVE checks separately.',
        '',
        '## Constitutive and element verification','']
    unit_path=CAMPAIGN/'runs/element_dt001/assessment/metrics.json'
    if unit_path.exists():
        unit=load(unit_path);proportional=[r for r in unit if r['expected_Gc_N_per_mm'] is not None]
        maximum=max(abs(r['fracture_energy_relative_error']) for r in proportional)
        lines+=['The refined Abaqus unit batch has a maximum proportional/closed-cycle fracture-energy discrepancy of '+number(maximum*100)+'%. '
                'DAT rounding limits the reported precision. The element area is1mm²; reaction work and dissipated energy are normalized consistently. '
                'Closure keeps compressive recoverable energy, so total external work is not equated to fracture energy alone.','',
                '| Unit history | Fracture-energy error | Final work-balance error | Minimum sampled damage increment |',
                '|---|---:|---:|---:|']
        for r in unit:lines.append('| '+r['label']+' | '+percent(r['fracture_energy_relative_error'])+' | '+percent(r['final_work_balance_relative_error'])+' | '+number(r['minimum_damage_increment'])+' |')
    lines+=['',
        'Twenty independent onset probes bracket the predicted initiation separation at0.999 and1.001times its analytical value. '
        'The material checks retain the old state when perturbing the three separation components; they do not differentiate along an already-updated damage history. '
        'Smooth-branch tangent errors exclude onset/closure kinks, which have no unique classical derivative.',
        '',
        '| Normal penalty [N/mm³], B=.5 | Canonical maximum tangent error | Smaller-perturbation test |',
        '|---|---:|---:|']
    paths=[CAMPAIGN/'material'/name/'assessment/metrics.json' for name in ['canonical','zeta1e8']]
    if all(p.exists() for p in paths):
        rr=[[r for r in load(p) if r['label'].endswith('_B0.5')] for p in paths]
        for a,b in zip(*rr):lines.append('| '+a['label'].split('_')[0][1:]+' | '+percent(a['smooth_tangent_max_error'])+' | '+percent(b['smooth_tangent_max_error'])+' |')
    refs=[CAMPAIGN/'material'/name/'assessment/metrics.json' for name in ['reference_properties_canonical','reference_properties_z8']]
    if all(p.exists() for p in refs):
        rr=[[r for r in load(p) if r['label'].endswith('_B0.5')] for p in refs]
        lines+=['','The four actual structural material-property sets are also tested at five proportional fractions each, independently of the generic penalty sweep. Krueger strengths/stiffness remain assumed project inputs.','',
                '| Structural property set, B=.5 | Canonical tangent error | Smaller-perturbation test |','|---|---:|---:|']
        for a,b in zip(*rr):lines.append('| '+a['label'][:-5]+' | '+percent(a['smooth_tangent_max_error'])+' | '+percent(b['smooth_tangent_max_error'])+' |')
    high_k=[CAMPAIGN/'material'/name/'assessment/metrics.json' for name in ['turon2018_K1e6_canonical','turon2018_K1e6_z8']]
    if all(p.exists() for p in high_k):
        rr=[next(r for r in load(p) if r['label'].endswith('_B0.5')) for p in high_k]
        lines+=['','For the separate2018 K3=1e6 sensitivity, five additional proportional histories per source copy are executed. '
            'At B=.5 the canonical tangent error is '+percent(rr[0]['smooth_tangent_max_error'])+' versus '+percent(rr[1]['smooth_tangent_max_error'])+' with the smaller perturbation. '
            'These are standalone material checks, separate from the higher-K Abaqus structural run.']
    lines+=['',
        '**Identified canonical numerical defect:** the fixed1e-6mm forward perturbation loses tangent accuracy as the onset separation becomes smaller at high K. '
        'The isolated1e-8mm source copy improves the tested smooth tangents. [Material response equivalence](campaign_20261025/sources/material_patch_response_equivalence.json) '
        'checks traction/history/energy columns separately from tangent columns. This patch has not replaced the canonical source.',
        '',
        'The full assembled UEL residual-versus-tangent diagnostic has about0.382% normalized error in the tested softening state; reducing the material perturbation barely changes it. '
        'The source updates the local basis and area but omits their full geometric derivatives. The restricted rotated-patch tests do not establish a generally consistent finite-deformation tangent. '
        'Native cohesive DCB controls also fail near the same coarse-mesh opening, so the structural convergence problem is not assigned solely to this UEL tangent.',
        '',
        '## DCB numerical sensitivity and reference comparison','',
        '| Job | Solver | Resolved peak [N] | Maximum accepted [N] | Early-load C [mm/N] | Work-balance error | Stabilization / work |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in inventory:
        if not (r['name'].startswith(('dcb_ps','dcb_adaptive','dcb_native','dcb_rho','dcb_local015_rho'))):continue
        m=r.get('metrics',{});a=r.get('additional',{})
        if not isinstance(m,dict):continue
        lines.append('| '+r['name']+' | '+r['status']+' | '+number(m.get('maximum_reached_force_N') if a.get('peak_eligible') else None)+' | '+number(m.get('maximum_reached_force_N'))+' | '+number(m.get('initial_compliance_mm_per_N'),6)+' | '+percent(a.get('work_balance_relative_error'))+' | '+percent(a.get('stabilization_over_peak_work'))+' |')
    r=by_name.get('dcb_ps03',{});a=r.get('additional',{});m=r.get('metrics',{});ref=a.get('numerical_reference',{})
    if ref:
        lines+=['',
            'The corrected0.3mm DCB baseline gives '+number(m.get('maximum_reached_force_N'),6)+'N versus '+number(ref.get('peak_graphical_N'),6)+'N in the published30MPa graphical series ('+percent(ref.get('peak_relative_error'))+'). '
            'The common stable propagation mean absolute error, normalized by the graphical peak, is '+percent(ref.get('stable_propagation',{}).get('NMAE'))+'. '
            'Graphical uncertainty is±0.03mm/±0.8N, so these error digits are computational descriptors, not reference accuracy. '
            'The0.15mm unstabilized mesh and0.3mm smaller-increment job fail; source-equivalent mesh/increment convergence is therefore not established.']
    fine=by_name.get('dcb_ps0075',{});fm=fine.get('metrics',{});fa=fine.get('additional',{})
    if fa.get('peak_eligible'):
        change=fm['maximum_reached_force_N']/m['maximum_reached_force_N']-1
        lines+=['','The0.075mm longitudinal refinement completes the full4mm path without viscosity or stabilization, whereas the0.15mm run failed. '
            'Its peak is '+number(fm['maximum_reached_force_N'],6)+'N ('+percent(change)+' versus0.3mm), and maximum energy drift is '+percent(fa.get('work_balance_relative_error'))+'. '
            'The recorded peak moves from '+number(m['displacement_at_maximum_reached_force_mm'])+' to '+number(fm['displacement_at_maximum_reached_force_mm'])+'mm; '
            'the maximum displacement increment is0.08mm in both runs, so the peak locations are coarsely sampled. '
            'This demonstrates a beneficial mesh effect but does not establish monotonic solution robustness or temporal convergence.']
    lines+=['',
        'The tabulated early-load compliance is fitted over5–20% of each achieved maximum force where enough samples exist. '
        'This diagnostic interval is not certified undamaged; local cohesive damage can begin before the global peak. '
        'It must not be interpreted as a pure elastic penalty-compliance measurement. Independent tied fixed-crack controls provide separate elastic evidence.',
        '',
        'Printed Appendix A1–A2 give initial DCB compliance0.0383785mm/N, whereas the solid FE controls give about0.047mm/N. '
        'This discrepancy is retained. The published numerical initial slope is closer to the FE result; root rotation/shear are plausible contributors. '
        'The root-correction expression in the paper’s MMB appendix gives a separate diagnostic scale; it is not silently inserted into the printed DCB equation or fitted by changing crack length. '
        'Agreement on the propagation branch alone does not verify initial compliance.',
        '',
        'A stabilization input fraction is not a bound on actual artificial energy. Solver messages show that STABILIZE=2e-4 activated adaptive damping, '
        'whereas2e-5 used constant damping. The latter completed with roughly15–16% stabilization/work and materially higher forces; those runs are rejected as converged predictions. '
        'Later named adaptive controls explicitly hold ALLSDTOL=.005. Their actual energy and force sensitivity remain the deciding evidence.',
        '',
        'The user-directed follow-up adds0.075mm DCB and0.1mm AS4 MMB80 meshes without changing material or regularization, '
        'then holds DCB rho=1e-4 fixed while reducing the maximum normalized step increment0.02→0.001. '
        'Step duration is1. A conditional adaptive retry is a separate intervention. These follow-ups have a declared3000-increment computation cap. '
        'No run stops because peak load was detected. [Matched numerical-control table](campaign_20261025/results/numerical_control_comparisons.csv) '
        'retains failed paths, resolved peaks, successive eligible peak differences and energy screens; missing peak differences are not zero errors.',
        '',
        '## Penalty and strength choices','',
        'Normal/shear strengths and K are source-specific. Krueger’s VCCT benchmark supplies no cohesive strengths or stiffness: '
        '32/61MPa and170000N/mm³ are labelled project assumptions. The sensitivity study preserves that baseline, increases K separately and refines the mesh; '
        'it does not infer strengths from the target critical load. The DCB strength15/30/60MPa tests are named sensitivities, and failed attempts remain in the inventory.',
        '',
        'The simple series-compliance screen in Turon et al.2007, Eqs4–7, gives alpha=K*t/E3 and relative transverse stiffness loss1/(1+alpha). '
        'Using the Krueger arm thickness2.25mm and E3=11380MPa, K=170000 gives about2.89% loss, versus about0.50% at1e6. '
        'This suggests the inherited penalty is not negligible; it does not measure actual MMB compliance. '
        'The thickness scale is explicitly an arm, not an individual ply. [Penalty screen](campaign_20261025/penalty_screen.csv) and the source record retain all tested values. '
        'Increasing K also shrinks onset separation and worsens the canonical fixed-perturbation tangent, so higher K alone is not an established remedy.',
        '',
        'The2018 source explicitly states K3=1e5N/mm³. Its arm-thickness series screen gives6.08% loss; a separate K3=1e6 control gives0.643% in that screen. '
        'This diagnostic changes only the penalty of the completed corrected-lever20% case, retaining the published-K baseline and the constitutive normal/shear stiffness ratio. '
        'It tests compliance and convergence sensitivity without fitting the published force curve. Actual completed/incomplete responses are retained in the matched numerical-control table.',
        'If that higher-K canonical-perturbation run fails, a conditional smaller-perturbation retry holds the same K and solver controls, '
        'separating the directly measured tangent error from the physical penalty sensitivity. Both copies remain isolated from canonical source.',
        *penalty_comparison(by_name),
        '',
        'In the actual Krueger50% runs, the first accepted secant compliance changes from0.003520747mm/N at K170000 to0.003505011mm/N at K1e6, about0.45%. These are single early-load secants, not multi-point undamaged fits. Both paths terminate near1.34mm, so their maxima are not verified peaks. Raising K does not resolve that benchmark discrepancy.',
        '',
        '## MMB loading, dimensionality and independent energy-release checks','',
        'The linear, weight-free equivalent lever enforces u_load=(1+c/L)u_mid−(c/L)u_end and uses the reaction at that load coordinate. '
        'It preserves virtual work for that linear reconstruction; it does not automatically reproduce a finite-rotation rigid apparatus. '
        'The fixed-crack MMB50 control independently checks its force split through bulk NFORC and verifies its displacement identity. '
        'Abaqus2023 exports separate scalar NFORC1..6 fields; the extractor handles those explicitly. '
        'The VCCT sign was calibrated with the pure-I control before comparison with mixed loading.',
        '',
        'Direct hard contact was incompatible with the paired-width equations and caused initial zero pivots. '
        'Explicit penalty contact resolves ENF initial loading and has a separate1e6-versus1e7 stiffness check. '
        'Initially opening MMB contact controls still failed near zero gap. A contact-free fixed-crack MMB control has no face penetration and correct lever statics; '
        'subsequent opening-MMB cases omit inactive initial-face contact and check the gap at every saved frame. '
        'The evolving UEL retains normal compression. A contact-free run that develops overlap cannot establish physical equivalence. '
        'ENF retains contact because its faces close.',
        '',
        'Ordinary-plane-strain nodal constraints enforce width-invariant in-plane motion and zero width displacement. '
        'An initial end-coupling defect allowed small U3 motions because Abaqus ignored extra boundary conditions on coupling nodes; revision2 fixes the coupling and excludes those redundant BCs. '
        'Linear C3D8I controls directly show near-zero out-of-plane strains. Finite-displacement C3D8I reports small internal LE33 residuals despite constrained nodal motion; these values are retained rather than calling the full nonlinear element exactly identical to CPE4I. '
        'The finite-width Krueger control is a separate coarse3D diagnostic, not a width-scaled plane-strain validation.',
        '',
        '| Fixed-crack control | Solver | B=GII/GT | Critical P [N] | Critical displacement [mm] |',
        '|---|---|---:|---:|---:|']
    fp=out/'fixed_crack_checks.json'
    if fp.exists():
        for r in load(fp):lines.append('| '+r['name']+' | '+r['status']+' | '+number(r.get('B_GII_over_GT'))+' | '+number(r.get('critical_load_N'))+' | '+number(r.get('critical_displacement_mm'))+' |')
    cp=out/'dcb_compliance_vs_vcct.json'
    if cp.exists():
        c=load(cp);lines+=['','The independent DCB compliance derivative gives G='+number(c['G_compliance_N_per_mm'],7)+'N/mm, compared with VCCT G='+number(c['G_VCCT_N_per_mm'],7)+'N/mm ('+percent(c['signed_relative_difference'])+'). '
                           'This uses0.5mm virtual closure and a1mm central crack-length interval; it is a diagnostic consistency check, not a crack-front convergence proof.']
    lines+=['',
        'The printed MMB LEFM reference is evaluated twice: with evolving analytical B and with Gc held at nominal B. '
        'Both CSVs are retained per run. Local cohesive B, analytical modal partition and finite-width VCCT partition are distinct quantities. '
        'Krueger symbols describe critical equilibrium, including snap-back, and are not a displacement-controlled jump path. '
        'No curve is interpolated across a detected jump or extended beyond its reference domain.',
        '',
        'The first2018 ideal-lever reconstruction achieves initial corrected-beam fractions below nominal (0.18605 for nominal0.20). A separate reconstruction analytically inverts the corrected beam partition at the source initial crack, giving c100.8067375/42.9266764/31.9981592mm for nominal20/50/70% under the adopted L51mm. This uses geometry, elastic properties and requested mixity, without force/displacement fitting. Both policies remain in the record. The0.1mm control matches the source longitudinal spacing locally in the fracture region; the far field/loading end remains graded, so it is not a uniform author-mesh reproduction.',
        '',
        '## Experimental comparison and independence limits','',
        'Turon2006 Table3 and the digitized black square markers from Figs9–13 are the primary published observations. '
        'Camanho2002 Table4 remains a separate comparison; the later publication date does not establish newer experiments or corrections. '
        'The marker extraction retains only printed markers: no origin, missing measurements, or experimental interpolation is inserted. '
        'Read-off uncertainty is case-specific; ENF has only three markers. The source also notes sparse propagation observations.',
        '',
        '| Baseline | Solver | Resolved peak [N] | 2006 observed peak [N] | Difference | Compared experimental markers |',
        '|---|---|---:|---:|---:|---:|']
    for r in inventory:
        if not r['name'].startswith('as4_'):continue
        a=r.get('additional',{});exp=a.get('experimental_comparisons',[])
        if not exp:continue
        e=next(e for e in exp if e['source']=='turon2006MM')
        lines.append('| '+r['name']+' | '+r['status']+' | '+number(e['predicted_peak_N'])+' | '+number(e['experimental_peak_N'])+' | '+percent(e['peak_relative_error'])+' | '+str(a.get('experimental_marker_comparison',{}).get('common_markers',0))+' |')
    fine_mmb=by_name.get('as4_mmb80_dx01',{});fine_metrics=fine_mmb.get('metrics',{})
    if fine_mmb.get('status')=='FAILED' and fine_metrics.get('last_displacement_mm') is not None:
        lines+=['','The unstabilized AS4 MMB80 local0.1mm refinement fails at '+number(fine_metrics['last_displacement_mm'],6)+'mm of its6mm target. '
            'Its maximum accepted force is '+number(fine_metrics['maximum_reached_force_N'],6)+'N, not a verified full-path peak. '
            'Material, penalty, viscosity and step controls match the0.3mm baseline; mesh refinement alone does not recover this MMB path.']
    lines+=['',
        '**No independent experimental validation claim is made.** Toughness and eta were already fitted to related experiments; raw histories, repeat scatter and full apparatus metadata are missing. '
        'Unresolved numerical convergence or apparatus assumptions further limit interpretation. Peak and common-marker errors are provisional baseline-prediction comparisons, not a new calibration or a held-out validation score.',
        '',
        'The fixed BK fit also differs from individual measured toughnesses: at nominal20%, it gives0.987994N/mm versus1.103N/mm in the2006 table, a10.43% energy difference. Under the same-geometry linear LEFM scaling alone this corresponds to about5.36% lower load. This is an explanatory estimate, not a corrected prediction; it does not account for geometry, achieved partition, finite process zones or numerical errors. [All five nominal toughness comparisons](campaign_20261025/results/experimental_toughness_fit_screen.csv) are retained without refitting eta or endpoint toughnesses.',
        '',
        '## Energy consistency','',
        '**Elastic changing-area failure:** an actual full-UEL standalone cycle opens a unit patch to0.0001mm, stretches area from1 to1.2mm² at fixed gap, closes it, then restores the original geometry. Damage and final stored/dissipated energies are zero, but the canonical residual gives net external work−0.0001Nmm, or20% of peak loading work. The100- and1000-interval tests agree. This failure is distinct from damage quadrature and does not disappear with time refinement. The current-area force/energy weighting has no conjugate in-plane area-change force.',
        '',
        'An isolated [reference-area experiment](campaign_20261025/sources/reference_area_experiment.json) gives net work about−1.26e-19Nmm in the same restricted cycle. It interprets traction and energy per undeformed area; a current-area model with additional variational area/basis forces is another possible formulation. The intended measure requires researcher judgment before adoption. The variant does not replace the canonical source or fix missing basis derivatives, the fixed material perturbation, or viscous quadrature. Constant-area unit and matched specimen tests are retained separately; this is not a general finite-deformation validation claim.',
        '',
        '**Constant-area basis-change failure:** a second elastic full-UEL cycle rotates the midsurface0→45degrees while the small global gap direction stays fixed, then closes and restores the original geometry. This is not a rigid-body rotation of the entire separated configuration. Both canonical and reference-area variants leave about+0.000137909Nmm external work with zero final stored/dissipated energy, about27.58% of peak loading work. The reference-area variant therefore does not cure the missing frame/geometric work terms. Small residual/Jacobian error and correct net force alone do not prove that the residual is the gradient of the reported energy.',
        '',
        'The authoritative additional assessment uses maximum variation of the solver-reported ETOTAL, divided by peak absolute ALLWK; final variation is retained separately. This includes host contact-energy accounting. The subset ledger |ALLWK−ALLIE−ALLSD−ALLVD| is also retained but omits contact elastic energy/discontinuity work. Individual ALLCCE/ALLCCDW histories were not requested in the early decks; ETOTAL captures their net contribution. UEL damage dissipation is already part of ALLIE through ALLPD; '
        'it must not be counted again. These models have no frictional dissipation. Stabilization and artificial strain energy are reported separately. Earlier metrics.json records retain their original final/subset definitions; additional_metrics.json and this report use the complete history criterion. '
        'All listed energies are slice energies unless force/energy width scaling is explicitly applied; relative errors are invariant under that scaling.',
        '',
        'Positive-rho DCB runs can complete with severe energy errors. For example, dcb_rho1e4 reports W=7.076822Nmm and IE=15.057832Nmm. '
        'The canonical viscous dissipation uses endpoint trapezoidal integration of Y dDv. For fixed basis, nonnegative opening and diagonal K, '
        'the independently verified discrete identity is:',
        '',
        '`Delta psi + Q - trapezoidal traction work = .25 DeltaDv (Delta delta)^T K (Delta delta)`.',
        '',
        'Actual extracted Fortran tests with100/1000/10000 uniform increments give energy excess0.00990610/0.0000990610/0.000000990610Nmm for both tested rho values. '
        'The error decreases with refinement and matches the identity; it does not license an unrefined structural energy claim. '
        'Changing basis/area and nonproportional structural paths require their own work checks. No reported energy is altered merely to force a balance.',
        *viscosity_comparison(by_name),
        *reference_area_summary(),
        '',
        '## Reproducibility and retained limitations','',
        '[Exact-run replay and regeneration instructions](campaign_20261025/README.md) retain decks, source copies, command lines, hashes, solver diagnostics, '
        'post-processing failures and derived CSV/JSON data. Original models are never overwritten. The run inventory distinguishes UEL, native and tied elastic controls. '
        'One Abaqus solver is submitted at a time, initially with one CPU. The repository was dirty before the campaign; file hashes, not HEAD alone, identify the baseline.',
        '',
        'Two older controllers had advanced beyond their intended stopping boundary, leaving stale RUNNING manifests after the solver had independently failed. '
        'Original manifests and queue ownership-guard errors are retained. Status was recovered from terminal STA evidence with no live lock; the original outputs were extracted and fresh opening-MMB replacements were executed. '
        'The submission audit records this resolution separately from the still-failed solver results and checks for overlapping execution intervals.',
        '',
        '[Preservation check](campaign_20261025/results/preservation_check.json) compares62 original source/example files, including historical curves. '
        'The first-campaign source SHA256 is `bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c`. '
        'The unproven historical benchmark_turon CSVs are preserved but excluded from targets. Isolated I/O-off and smaller-perturbation copies are separately hashed and tested.',
        '[Final artifact audit](campaign_20261025/results/final_artifact_audit.json) records ended queues, applicable assessment coverage, source/deck preservation, serial execution, local links, Python syntax and figure export checks. These integrity checks do not replace the scientific verdicts.',
        '',
        'The remaining scientific limits are source-equivalent numerical convergence, exact finite-rotation apparatus equivalence, finite-width front convergence, '
        'a general finite-deformation UEL tangent/energy assessment, and independent experimental validation with raw observations/scatter. '
        'A failed or inconclusive benchmark remains a legitimate result; it is not relabelled PASS because a nearby run completes.',
        '',
        '## Figures and source records','',
        'Figures are generated from retained numerical/reference data as SVG/PDF, with PNG review copies, at85 or180mm widths. '
        'Read the [figure captions](campaign_20261025/results/FIGURE_CAPTIONS.md) before thesis use. '
        'Dashed traces/crosses indicate incomplete jobs; open experimental markers are digitized published observations.',
        '',
        '[Figure directory](campaign_20261025/results/figures/) · [Source cards](sources/) · [Digitization register](DIGITIZATION.md) · '
        '[Intervention log](campaign_20261025/INVESTIGATION.md) · [Full run inventory](campaign_20261025/results/run_inventory.json)',
        '',
        'Primary methodological references: [Abaqus coupling constraints](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-coupling.htm), '
        '[static stabilization controls](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-static.htm), '
        '[surface behavior/contact enforcement](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-surfacebehavior.htm), '
        '[nodal element forces](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-std-wholeelementvariables.htm). '
        '[Contact energy accounting](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm) explains why ETOTAL rather than the incomplete element-energy subset is used for the final global check. '
        'The installed solver is2023; online2025 documentation explains the control semantics, and actual2023 messages/decks are the execution evidence. '
        'Published materials/geometry/observations are traced to local PDF pages in the case/source cards. '
        'The additional penalty reference is [Turon et al.2007](https://doi.org/10.1016/j.engfracmech.2006.08.025).',
        '']
    if hashlib.sha256((ROOT/'src/3D_CZM.f').read_bytes()).hexdigest()!='bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c':
        lines[2:2]=['> **Source-version note:** This report concerns the first-campaign v1.3 source and its retained variants. The subsequent optional-input extension is documented and checked [separately](tangent_input_20261026/README.md). Campaign results remain tied to their per-run source hashes.','']
    (ROOT/'validation/REPORT.md').write_text('\n'.join(lines))
    print('Wrote technical report; pending',len(pending),'running/prepared',len(running))


if __name__=='__main__':main()
