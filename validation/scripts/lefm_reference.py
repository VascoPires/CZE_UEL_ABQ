"""Turon 2010 Appendix A reference equations, N-mm-MPa. No FE solver.

DCB needs no support-span assumption. ENF/MMB functions require L explicitly;
the source's ambiguous L notation is resolved by decision D8 for execution. The MMB caller
must also choose how Gc is evaluated; no implicit nominal/variable-B policy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def dcb(a, E11, b, h, GIc):
    """A1-A2: P is the force on one arm; u is full relative opening."""
    a = np.asarray(a)
    load = np.sqrt(GIc*b*b*h**3*E11/(12*a*a))
    compliance = 8*a**3/(b*h**3*E11)
    return load, load*compliance, compliance


def enf(a, E11, b, h, GIIc, L):
    """A3-A4, formal equilibrium; does not select a stable loading path."""
    a = np.asarray(a)
    load = np.sqrt(GIIc*16*b*b*h**3*E11/(9*a*a))
    compliance = (3*a**3+2*L**3)/(8*b*h**3*E11)
    return load, load*compliance, compliance


def mmb(a, E11, E22, G13, b, h, GIc, GIIc, eta, L, c, *, gc_policy, nominal_B=None):
    """A5-A7. Return P,u,C,B; small-rotation, corrected beam reference.

    B is derived from modal terms of (1/2b)dC/da. This is an analytical
    partition, distinct from UEL SDV10 and from VCCT at a 3D crack front.
    """
    a = np.asarray(a)
    gamma = 1.18*np.sqrt(E11*E22)/G13
    chi = np.sqrt(E11/(11*G13)*(3-2*(gamma/(1+gamma))**2))
    EI = E11*b*h**3/12
    mode_I = 4*(3*c-L)**2*(a+chi*h)**2
    mode_II = 3*(c+L)**2*(a+0.42*chi*h)**2
    B = mode_II/(mode_I+mode_II)
    if gc_policy == "beam-energy":
        gc = GIc+(GIIc-GIc)*B**eta
    elif gc_policy == "nominal" and nominal_B is not None and 0 <= nominal_B <= 1:
        gc = GIc+(GIIc-GIc)*nominal_B**eta
    else:
        raise ValueError("Choose beam-energy or supply an explicit nominal_B.")
    compliance = (4*(3*c-L)**2*(a+chi*h)**3
                  +(c+L)**2*(3*(a+0.42*chi*h)**3+2*L**3))/(96*L*L*EI)
    load = np.sqrt(gc*64*b*L*L*EI/(mode_I+mode_II))
    return load, load*compliance, compliance, B


def dcb_sanity(out):
    """Compare A1-A2 with the printed analytical curve; not a UEL test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    validation = Path(__file__).resolve().parents[1]
    source = validation/"cases/turon2010CS_dcb/data/fig04_lefm_graphical.csv"
    points = np.loadtxt(source, delimiter=",", comments="#", skiprows=6)
    # Figure 4 draws an equilibrium branch including a < a0; retain that
    # graphical domain for transcription checking only, not specimen loading.
    E, b, h, G, a0 = 120000., 20., 1.55, 0.260, 35.
    A = np.sqrt(G*b*b*h**3*E/12)
    a_graph = np.sqrt(points[:, 0]*b*h**3*E/(8*A))
    expected, _, _ = dcb(a_graph, E, b, h, G)
    difference = points[:, 1]-expected
    out.mkdir(parents=True, exist_ok=False)
    with (out/"analytical_reference.csv").open("w", newline="") as handle:
        handle.write("# Turon2010CS Appendix A1-A2, p.1863; Fig.4 graphical-domain comparison; no FE results\n")
        writer = csv.writer(handle)
        writer.writerow(["opening_mm", "graphical_P_N", "analytical_P_N", "difference_N", "derived_a_mm"])
        writer.writerows(zip(points[:, 0],points[:, 1],expected,difference,a_graph))
    metrics = {"kind": "reference-equation transcription check; not UEL verification",
               "source": "Turon2010CS p.1859; Table1 p.1860; Appendix A1-A2 p.1863; Fig4 p.1860",
               "inputs": {"E11_MPa": E,"b_mm": b,"h_mm": h,"GIc_N_per_mm": G,"a0_mm": a0},
               "samples": len(points), "max_abs_error_N": float(max(abs(difference))),
               "max_abs_relative_error": float(max(abs(difference/expected))),
               "graphical_uncertainty_mm_N": [0.01, 0.3], "UEL_status": "NOT RUN",
               "initial_compliance_mm_per_N": float(dcb(a0,E,b,h,G)[2]),
               "acceptance_tolerance": "NOT APPROVED"}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(5,3.5),layout="constrained")
    ax.plot(points[:,0],expected,label="Appendix A1-A2",color="#287d87")
    ax.scatter(points[::4,0],points[::4,1],s=12,facecolors="none",edgecolors="#555555",label="Fig. 4 graphical reference")
    ax.set(xlabel="Full opening [mm]",ylabel="Force on one arm [N]",title="DCB reference transcription check")
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/"reference_check.svg")
    plt.close(fig)
    print(json.dumps(metrics,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory.")
    dcb_sanity(args.out)


if __name__ == "__main__":
    main()
