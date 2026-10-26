"""Turon2010 Fig5,60MPa shear-strength drawn curve; no offsets or fitting."""
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    pdf=ROOT/'validation/references/turon2010CS.pdf'
    out=ROOT/'validation/campaign_20261025/reference_data/turon2010CS_enf';out.mkdir(parents=True,exist_ok=False)
    scale=4000/793.701
    with tempfile.TemporaryDirectory(prefix='cze_enf_graph_') as temporary:
        prefix=Path(temporary)/'page4';subprocess.run(['pdftoppm','-f','4','-l','4','-scale-to','4000','-png','-singlefile',str(pdf),str(prefix)],check=True)
        page=Image.open(str(prefix)+'.png').convert('RGB');left,top=int(69*scale),int((793.701-380)*scale)
        crop=page.crop((left,top,int(272*scale),int((793.701-227)*scale)));pixels=np.asarray(crop)
    yy,xx=np.indices(pixels.shape[:2]);px=(xx+left)/scale;py=793.701-(yy+top)/scale
    x0,y0,x1,y1=69.927,228.460,271.108,378.120
    u=(px-x0)/(x1-x0)*9;force=(py-y0)/(y1-y0)*600;colour=np.array([.690,.129,.051])
    mask=(np.linalg.norm(pixels.astype(float)-colour*255,axis=2)<25)&(u>0)&(u<9)&(force>=0)&(force<=600)
    mask &= ~((px<155)&(py>330))
    points=[]
    for a in np.linspace(0,9,401)[:-1]:
        selected=mask&(u>=a)&(u<a+9/400)
        if np.count_nonzero(selected)>=2:points.append([float(np.median(u[selected])),float(np.median(force[selected]))])
    a=np.array(points)
    with (out/'fig05_tau60_graphical.csv').open('w',newline='') as handle:
        handle.write('# Turon2010CS Fig5,p1860,PDFpage4;60MPa shear strength,dark-red filled triangle series\n')
        handle.write('# Drawn-curve graphical estimate,not raw solver samples; no offset or missing-point insertion\n')
        handle.write('# Estimated uncertainty +/-0.04mm,+/-5N; jump neighbourhood is ambiguous\n')
        writer=csv.writer(handle);writer.writerow(['displacement_mm','load_N']);writer.writerows(points)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained');axes[0].imshow(pixels,extent=[px.min(),px.max(),py.min(),py.max()],aspect='auto')
    axes[0].scatter(x0+a[:,0]/9*(x1-x0),y0+a[:,1]/600*(y1-y0),s=2,color='black');axes[0].set(xlabel='PDF x[pt]',ylabel='PDF y[pt]')
    axes[1].plot(a[:,0],a[:,1],color=colour);axes[1].set(xlabel='Midspan displacement[mm]',ylabel='Load[N]')
    fig.savefig(out/'digitization_review.png',dpi=170);fig.savefig(out/'digitization_review.svg');plt.close(fig)
    (out/'digitization.json').write_text(json.dumps(dict(source_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdf_page=4,printed_page=1860,figure=5,
        shear_strength_MPa=60,axis_pdf_points=[x0,y0,x1,y1],axis_max=[9,600],rgb_target=colour.tolist(),rgb_distance_max=25,bin_width_mm=9/400,
        legend_exclusion='pdfx<155 and pdfy>330',uncertainty_mm_N=[.04,5.],samples=len(points),limitation='Drawn curve; finite displacement jump is not a stable equilibrium interpolation domain'),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
