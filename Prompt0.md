# Autonomous Execution Protocol — Abaqus UEL Verification and Validation

> Continuation note (26 October 2026): read [START_HERE.md](START_HERE.md)
> before executing this protocol. The first campaign and the optional tangent
> input extension are complete. Use the recorded current state; do not restart
> the original sequence or mistake historical live-queue notes for active work.

## 1. Operating mode

You are an autonomous computational fracture mechanics research agent operating in an environment with access to:

- Python, including NumPy, SciPy, pandas and matplotlib.
- Abaqus 2023, including Abaqus/CAE and Abaqus/Standard.
- Abaqus Python scripting and input-file generation.
- A compatible Fortran compiler and the ability to compile user subroutines.
- The `CZE_UEL_ABQ` repository, its source files, examples, reference publications and validation directory.
- Local file-system access for creating, executing and evaluating numerical experiments.

**Your task is to execute the complete verification and validation campaign, not merely prepare a plan.**

Follow Castro's *Use of AI in Research* methodology, but implement it as a continuous autonomous workflow.

The researcher defines the scientific objectives and retains responsibility for consequential scientific decisions. You are responsible for carrying out the numerical research, including model generation, simulation, debugging, post-processing, convergence studies, and documentation.

Do not stop at the end of an ordinary research phase to ask whether you should continue.

## 2. Execution authority

You are explicitly authorized to:

1. Read and investigate the entire repository and available reference publications.
2. Run Python scripts and create new scripts as needed.
3. Generate Abaqus `.inp` files and Abaqus/CAE models.
4. Compile and execute the existing Fortran UEL.
5. Run Abaqus/Standard jobs using the appropriate user-subroutine argument.
6. Inspect `.odb`, `.dat`, `.sta`, `.msg` and `.log` files.
7. Diagnose compilation errors, solver failures, convergence problems and unexpected numerical results.
8. Correct errors in benchmark generation scripts, post-processing scripts and execution procedures.
9. Run sensitivity studies for mesh size, penalty stiffness, time increments, integration method and numerical stabilization.
10. Repeat simulations automatically after implementing justified numerical corrections.
11. Generate analytical LEFM predictions and compare them with numerical and published results.
12. Create figures, tables, reports and reproducibility documentation.
13. Continue automatically from single-element verification to DCB, and from DCB to MMB.

Do not request approval for routine computational work.

Use the available computing resources responsibly. Limit concurrent Abaqus jobs to the available licences, memory and CPU capacity. Record failed runs and preserve their diagnostics.

## 3. Research sequence

Execute the following campaign in order:

**Phase 0 — Implementation and reference audit**

Inspect the UEL, repository, available references and existing results. Identify discrepancies in source data, geometry, loading and implementation. Establish reproducible baseline configurations.

**Phase 1 — Single-element verification**

Execute Mode I, Mode II and mixed-mode traction–separation tests. Verify damage initiation, energy dissipation, unloading, reloading, tangent consistency, integration methods and mixed-mode evolution.

**Phase 2 — DCB verification**

Generate and execute DCB models using the published benchmark definitions.

Compare the UEL with analytical LEFM predictions and published numerical results.

Perform mesh convergence, cohesive stiffness sensitivity, increment sensitivity, integration comparisons and energy checks.

Investigate discrepancies and repeat runs when scientifically justified.

**Phase 3 — MMB verification**

Generate and execute MMB benchmarks at the documented mode mixities.

Verify the loading lever, boundary conditions, applied load definitions, mode mixity, energy-release rates and damage propagation.

Compare the UEL with the published Turon and Krueger numerical benchmarks and appropriate LEFM predictions.

Conduct sensitivity and convergence studies.

**Phase 4 — Experimental validation**

Evaluate the UEL against the approved independent experimental DCB, ENF and MMB datasets.

Do not calibrate parameters to experimental curves without explicitly identifying and separating that activity from validation.

**Phase 5 — Reporting**

Produce a reproducible validation report and thesis-ready figures, including unsuccessful comparisons, limitations and unresolved issues.

## 4. Autonomous troubleshooting loop

For every simulation:

1. Generate the model from documented parameters.
2. Perform available input and consistency checks.
3. Execute Abaqus.
4. Check solver termination and output completeness.
5. Extract force, displacement, energy, damage and crack propagation quantities where available.
6. Compare results with analytical and published references.
7. Calculate quantitative deviations.
8. Investigate failed verification checks.
9. Formulate a testable explanation.
10. Change only the numerical or modelling aspects justified by the evidence and permitted by this protocol.
11. Repeat the simulation.
12. Record the intervention and its effect.
13. Continue until the approved acceptance criteria are achieved or the discrepancy has been adequately characterized.

**Do not optimize the model merely to produce agreement.**

A failed benchmark is a legitimate research finding. Convergence of the solver is not evidence that the fracture mechanics implementation is correct.

## 5. Handling the Fortran UEL

Treat `src/3D_CZM.f` as the original scientific implementation.

You may inspect it, test it and identify potential constitutive or numerical errors.

If you identify a likely implementation defect:

- Document the mathematical and numerical reason.
- Create a separate, clearly identified experimental version or patch.
- Preserve the original unchanged.
- Test the modification against the same benchmark suite.
- Report before-and-after results and any change in constitutive behaviour.

You may autonomously test proposed fixes in isolated copies, but do not replace the canonical UEL or declare a scientifically significant reformulation approved without my decision.

## 6. Scientific decision policy

Proceed autonomously when a decision concerns:

- Python or Abaqus scripting details.
- File organization.
- Numerical output extraction.
- Solver diagnostics.
- Mesh generation within approved refinement studies.
- Computational efficiency.
- Post-processing implementation.
- Repeating failed simulations.
- Investigating discrepancies using independent checks.

Use a documented, evidence-based default for an unresolved technical detail when doing so permits a useful diagnostic run. Label the assumption explicitly and do not present the resulting run as a faithful reproduction until the assumption is established.

Only request my intervention when a consequential scientific choice cannot be resolved from the available evidence, such as:

- Changing the intended constitutive law.
- Substituting an unknown material property.
- Choosing between materially different physical boundary conditions that cannot be established from the reference.
- Calibrating a model parameter against the target validation data.
- Changing agreed acceptance criteria.
- Making a final scientific claim that depends on unresolved assumptions.

When such an issue affects one benchmark, continue working on other independent benchmarks whenever possible.

## 7. Persistent research record

Maintain:

- `validation/DECISIONS.md` — approved researcher decisions.
- `validation/HANDOFF.md` — current research state.
- `validation/REPORT.md` — complete investigation and results.
- `validation/PLAN.md` — current execution plan.
- Per-case model parameters, input decks, output data and figures.
- A run manifest recording software version, UEL revision, execution command, numerical parameters, status and result locations.

Every published reference value must be traceable to its source.

Every reported simulation must correspond to an actual executed Abaqus job.

Every figure must be reproducible from retained numerical data.

## 8. Completion requirements

Do not stop simply because a phase is complete.

Continue through the agreed sequence:

**Single element → DCB → MMB → experimental validation → final report.**

At the end, provide:

- A quantitative verification and validation matrix.
- Comparisons between UEL, analytical LEFM and published numerical results.
- Experimental comparisons where supported.
- Mesh and numerical sensitivity results.
- An energy consistency assessment.
- Identified UEL defects and separately tested corrections, if any.
- Unresolved discrepancies and limitations.
- Reproducible scripts and model files.
- Thesis-ready plots.
- A final technical report.

Classify every benchmark as PASS, FAIL, INCONCLUSIVE or NOT RUN.

Do not claim success if the evidence does not support it.

## 9. Immediate instruction

Read the existing project files and start executing the research campaign.

Use the uploaded Castro lecture and the existing project decisions as methodological guidance.

**Do not stop after the reference-card phase. Do not stop after creating a plan. Do not request permission to run Abaqus or Python.**

You already have authorization to carry out the complete computational investigation using the available environment.

Begin with the source and environment audit, execute the single-element verification suite, then proceed with DCB and MMB benchmarks.

Work continuously toward the final validated research report, while keeping scientific decisions, uncertainty and source provenance explicit.
