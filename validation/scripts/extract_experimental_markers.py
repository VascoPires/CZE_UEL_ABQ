"""Digitize the black experimental square markers in Turon2006 Figs9–13.

Centres of registered filled vector rectangles. No origin is inserted and no
curve is interpolated across missing measurements. These are republished
graphical observations, not original instrument records or independent repeats.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pdf_paths import paths

ROOT=Path(__file__).resolve().parents[2]
SPECS=[
 dict(case='dcb',page=12,figure=9,ids=list(range(39,62,2)),axis=[159.017,370.068,409.262,553.201],maximum=[8,170],uncertainty=[.02,1.5]),
 dict(case='mmb20',page=12,figure=10,ids=list(range(202,219,2)),axis=[152.731,104.334,416.576,298.947],maximum=[13,120],uncertainty=[.02,1.5]),
 dict(case='mmb50',page=13,figure=11,ids=list(range(86,101,2)),axis=[159.257,480.570,422.015,672.895],maximum=[8,300],uncertainty=[.02,3.]),
 dict(case='mmb80',page=13,figure=12,ids=list(range(192,209,2)),axis=[159.259,209.047,422.027,401.380],maximum=[7,600],uncertainty=[.02,5.]),
 dict(case='enf',page=14,figure=13,ids=[220,222,224],axis=[154.917,482.784,417.648,675.054],maximum=[5,850],uncertainty=[.02,5.]),
]


def main():
    pdf=ROOT/'validation/references/turon2006MM.pdf';digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    out=ROOT/'validation/campaign_20261025/reference_data/turon2006_experimental_markers'
    out.mkdir(parents=True,exist_ok=False);cache={}
    fig,axes=plt.subplots(2,3,figsize=(9,5),layout='constrained')
    for spec,ax in zip(SPECS,axes.flat):
        page=spec['page']
        if page not in cache:cache[page]=paths(pdf,page)
        objects=cache[page];points=[]
        for i in spec['ids']:
            p=objects[i];x,y,xx,yy=p['bbox']
            if p['op']!='f' or len(p['points'])!=5 or len(p['parts'])!=1 or abs((xx-x)-(yy-y))>.01 or not 3.3<xx-x<3.8:
                raise ValueError('Experimental marker geometry changed: page%d path%d'%(page,i))
            points.append([(x+xx)/2,(y+yy)/2])
        a=np.array(points);x0,y0,x1,y1=spec['axis'];mx,my=spec['maximum']
        a[:,0]=(a[:,0]-x0)/(x1-x0)*mx;a[:,1]=(a[:,1]-y0)/(y1-y0)*my
        filename='fig%02d_%s_experimental_markers_graphical.csv'%(spec['figure'],spec['case'])
        with (out/filename).open('w',newline='') as handle:
            handle.write('# Turon2006MM Fig%d,printedp%d,PDFpage%d; black experimental square markers\n'%(spec['figure'],1071+page,page))
            handle.write('# Graphical coordinates, not raw instrument data; no origin insertion or interpolation; original marker order retained\n')
            handle.write('# Source PDF SHA256: '+digest+'; axes and pathIDs: digitization.json\n')
            handle.write('# Read-off uncertainty mm,N: '+str(spec['uncertainty'])+'\n')
            writer=csv.writer(handle);writer.writerow(['displacement_mm','load_N']);writer.writerows(a)
        spec.update(filename=filename,samples=len(a),peak_graphical_N=float(a[:,1].max()),peak_graphical_displacement_mm=float(a[a[:,1].argmax(),0]))
        ax.plot(a[:,0],a[:,1],'s',mfc='none',mec='#444444',ms=3);ax.set(title=spec['case'],xlabel='Applied displacement [mm]',ylabel='Load [N]',xlim=(0,None),ylim=(0,None))
    axes.flat[-1].axis('off');fig.savefig(out/'marker_review.png',dpi=170);fig.savefig(out/'marker_review.svg');plt.close(fig)
    (out/'digitization.json').write_text(json.dumps(dict(source_pdf_sha256=digest,source='Turon2006MM Figs9–13',
        method='pypdf vector path centres of registered black filled squares',series=SPECS,
        limitation='Republished observations; missing raw records, repeat/specimen scatter and detailed acquisition metadata. Toughness/eta were already fitted to related experimental data; no independent held-out validation claim.'),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
