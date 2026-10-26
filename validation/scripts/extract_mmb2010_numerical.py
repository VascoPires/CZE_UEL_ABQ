"""Conservative drawn-curve estimates for Turon2010 Fig12; never raw solver data.

All coloured strokes are shared by analytical and numerical series. Select the
lower vertical cluster at each displacement, which follows the numerical series
where the analytical equilibrium hook lies above it. Review the retained overlay;
this extraction is a graphical envelope estimate, not a marker-only transcription.
"""
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
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[2]


def main():
    pdf=ROOT/'validation/references/turon2010CS.pdf'
    out=ROOT/'validation/campaign_20261025/reference_data/turon2010CS_mmb_v2'
    out.mkdir(parents=True,exist_ok=False)
    scale=4000/793.701
    with tempfile.TemporaryDirectory(prefix='cze_mmb_graph_') as temporary:
        prefix=Path(temporary)/'page6'
        subprocess.run(['pdftoppm','-f','6','-l','6','-scale-to','4000','-png','-singlefile',str(pdf),str(prefix)],check=True)
        page=Image.open(str(prefix)+'.png').convert('RGB')
        left,top=int(340*scale),int((793.701-238)*scale)
        crop=page.crop((left,top,int(543*scale),int((793.701-85)*scale)))
        pixels=np.asarray(crop)
    crop.save(out/'fig12_source_crop.png')
    yy,xx=np.indices(pixels.shape[:2]);px=(xx+left)/scale;py=793.701-(yy+top)/scale
    x0,y0,x9,y240=340.654,86.98,530.7555,236.29
    u=(px-x0)/(x9-x0)*9;force=(py-y0)/(y240-y0)*240
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].imshow(pixels,extent=[px.min(),px.max(),py.min(),py.max()],aspect='auto')
    metadata=dict(source_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdf_page=6,printed_page=1862,
                  figure=12,normal_shear_strengths_MPa=[30,58.9],axis_pdf_points=[x0,y0,x9,y240],axis_values=[0,0,9,240],
                  uncertainty_mm_N=[.05,3.],method='4000px Poppler colour selection with white-antialias projection; retain 8-connected components of at least100pixels; lowest cluster separated by >4N, median per9.5/400mm bin',
                  limitation='Analytical and numerical colours overlap. This is a lower drawn-curve envelope estimate; peak/jump neighbourhood ambiguous. No raw solver or experimental points.',series=[])
    for mix,colour in [(20,[1.,.012,.012]),(50,[.012,.012,1.]),(80,[.012,.506,.012])]:
        target=255-np.array(colour)*255;ink=255-pixels.astype(float)
        opacity=ink@target/(target@target)
        residual=np.linalg.norm(ink-opacity[:,:,None]*target,axis=2)
        mask=(residual<8)&(opacity>.15)&(u>0)&(u<9.5)&(force>=0)&(force<=240)
        mask &= ~((px<431.6)&(py>173.5))
        labels,count=label(mask,structure=np.ones((3,3)))
        sizes=np.bincount(labels.ravel());sizes[0]=0
        mask &= sizes[labels]>=100
        result=[]
        for a in np.linspace(0,9.5,401)[:-1]:
            take=mask&(u>=a)&(u<a+9.5/400)
            if np.count_nonzero(take)<2:continue
            fy=np.sort(np.unique(force[take]));gaps=np.flatnonzero(np.diff(fy)>4.)
            cutoff=fy[gaps[0]]+.01 if len(gaps) else np.inf
            take &= force<=cutoff
            if np.count_nonzero(take)<2:continue
            result.append([float(np.median(u[take])),float(np.median(force[take]))])
        result=np.array(result)
        filename='fig12_B%d_tau30_58p9_graphical.csv'%mix
        with (out/filename).open('w',newline='') as handle:
            handle.write('# Turon2010CS Fig12, p1862, PDFpage6; adjusted strengths30/58.9MPa\n')
            handle.write('# Graphical lower-envelope estimate; colour shared with analytical curve; no shifts or fitting\n')
            handle.write('# Uncertainty +/-0.05mm,+/-3N; jump neighbourhood ambiguous; not original raw solver samples\n')
            writer=csv.writer(handle);writer.writerow(['displacement_mm','load_N']);writer.writerows(result)
        axes[0].scatter(x0+result[:,0]/9*(x9-x0),y0+result[:,1]/240*(y240-y0),s=2,color='black')
        axes[1].plot(result[:,0],result[:,1],color=colour,label=str(mix)+'%')
        metadata['series'].append(dict(mixity_percent=mix,filename=filename,samples=len(result),rgb_target=colour))
    axes[0].set(xlabel='PDF x [pt]',ylabel='PDF y [pt]',title='Source crop with extracted sample locations')
    axes[1].set(xlabel='Lever displacement [mm]',ylabel='Applied load [N]',title='Graphical envelope estimates')
    axes[1].legend(frameon=False)
    fig.savefig(out/'digitization_review.png',dpi=160);fig.savefig(out/'digitization_review.svg');plt.close(fig)
    (out/'digitization.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
