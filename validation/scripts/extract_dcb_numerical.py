"""Graphical Turon 2010 Fig.4 tau3=30MPa series; no shifts or fitting.

Raster colour selection of the dark-red numerical curve, with axis metadata.
The result estimates the drawn curve, not original experimental/solver points.
"""
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'validation/references/turon2010CS.pdf'
    out=ROOT/'validation/campaign_20261025/reference_data/turon2010CS_dcb'
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='cze_dcb_digitization_') as temporary:
        prefix=Path(temporary)/'page4'
        subprocess.run(['pdftoppm','-f','4','-l','4','-scale-to','4000','-png','-singlefile',str(source),str(prefix)],check=True)
        pixels=np.asarray(Image.open(str(prefix)+'.png').convert('RGB'))
    height,width=pixels.shape[:2];scale=4000/793.701
    yy,xx=np.indices((height,width));px=xx/scale;py=793.701-yy/scale
    x0,y0,x1,y1=68.593,462.412,269.119,611.603
    displacement=(px-x0)/(x1-x0)*4
    force=(py-y0)/(y1-y0)*60
    colour=np.array([.694,.133,.051])*255
    distance=np.linalg.norm(pixels.astype(float)-colour,axis=2)
    mask=(distance<22)&(displacement>0)&(displacement<4)&(force>=0)&(force<=60)
    # Legend is below the numerical propagation branch at these abscissae.
    mask &= ~((px>200)&(py<535))
    rows=[]
    for a in np.linspace(0,4,301)[:-1]:
        take=mask&(displacement>=a)&(displacement<a+4/300)
        if np.count_nonzero(take)<2:continue
        rows.append([float(np.median(displacement[take])),float(np.median(force[take]))])
    with (out/'fig04_tau30_graphical.csv').open('w',newline='') as handle:
        handle.write('# source: Turon2010CS Fig.4,p1860,PDFpage4; tau3=30MPa, dark-red filled-circle series\n')
        handle.write('# graphical curve estimate, not original raw solver points; no offset or origin insertion\n')
        handle.write('# method: 4000px Poppler raster, RGB distance<22 from(.694,.133,.051), median per4/300mm bin\n')
        handle.write('# conservative estimated read-off uncertainty: displacement +/-0.03mm; force +/-0.8N\n')
        writer=csv.writer(handle);writer.writerow(['opening_mm','load_N']);writer.writerows(rows)
    (out/'digitization.json').write_text(json.dumps(dict(pdf=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pdf_page=4,printed_page=1860,figure=4,normal_strength_MPa=30,
        axis_pdf_points=[x0,y0,x1,y1],axis_max=[4,60],rgb_target=colour.tolist(),rgb_distance_max=22,
        legend_mask='pdfx>200 and pdfy<535',rows=len(rows),uncertainty_mm_N=[.03,.8],
        limitation='Drawn-curve graphical estimates; markers/overlap affect resolution; not raw samples'),indent=2)+'\n')
    print('Extracted',len(rows),'drawn-curve samples',out)


if __name__=='__main__':main()
