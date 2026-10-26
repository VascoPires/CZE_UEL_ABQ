"""Plot the extracted graphical data, preserving source order and raw zeros."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VALIDATION = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new directory; plots are preserved.")
    specs = json.loads((VALIDATION/"reference_extraction.json").read_text())["curves"]
    fig, axes = plt.subplots(4,2,figsize=(10,12),layout="constrained")
    for ax, spec in zip(axes.flat,specs):
        data = np.loadtxt(VALIDATION/"cases"/spec["case"]/"data"/spec["filename"],delimiter=",",skiprows=6)
        # Scatter avoids inventing lines across obscured data or jump segments.
        ax.scatter(data[:,0],data[:,1],s=9,color="#287d87")
        ax.set(title=spec["case"]+" / "+spec["figure"],xlabel="Source displacement [mm]",ylabel="Source total load [N]")
        ax.text(.03,.03,"Graphical reference; no UEL results",transform=ax.transAxes,fontsize=8,color="#555555")
    axes.flat[-1].axis("off")
    args.out.mkdir(parents=True)
    for suffix in ["svg","png"]:
        fig.savefig(args.out/("reference_extracts."+suffix),dpi=160)
    plt.close(fig)


if __name__=="__main__":
    main()
