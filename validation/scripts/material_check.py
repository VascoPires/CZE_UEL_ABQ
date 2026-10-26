"""Compile actual canonical constitutive routines with Intel Fortran; independent checks.

Only a minimal ABA_PARAM implicit-double include is substituted outside Abaqus.
This is a material check, not a substitute for the full UEL solver tests.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from campaign import ROOT, CAMPAIGN, win


PROGRAM='''program material_driver
implicit none
integer id,idx,oldid,ios,j,k
real*8 gi,gii,tn,ts,eta,kn,ks,rho,dt,du(3),dold,dvold,d,dv,lam,l0,lf,b
real*8 t(3,1),c(3,3),tp(3,1),tm(3,1),dp,dvp,lp,l0p,lfp,bp,ref(3,3),pert(3)
real*8 yold,psi,ddiss,y,diss,err,hh,zeta
zeta=TANGENT_PERTURBATION
open(10,file='input.txt',status='old')
open(11,file='response.csv',status='new')
write(11,'(A)') 'case_id,index,ds1,ds2,dn,ts1,ts2,tn,D,Dv,lambda,lambda0,lambdaf,B,psi,diss,Y,tangent_error'
oldid=-1
do
 read(10,*,iostat=ios) id,idx,gi,gii,tn,ts,eta,kn,rho,dt,du
 if(ios/=0) exit
 if(id/=oldid) then
  dold=0d0;dvold=0d0;yold=0d0;diss=0d0
 endif
 ks=kn*gi/gii*(ts/tn)**2
 call k_cohesive_law(t,c,gi,gii,tn,ts,eta,kn,ks,du,3,1,dold,dvold,d,dv,lam,l0,lf,b,dt,rho,zeta)
 call k_cohesive_energy(gi,gii,eta,kn,ks,du,dold,d,dvold,dv,l0,lf,b,rho,yold,psi,ddiss,y)
 diss=diss+ddiss
 hh=1d-9
 do j=1,3
  pert=du;pert(j)=pert(j)+hh
  call k_cohesive_response(tp,gi,gii,tn,ts,eta,kn,ks,pert,3,1,dold,dvold,dp,dvp,lp,l0p,lfp,bp,dt,rho)
  pert=du;pert(j)=pert(j)-hh
  call k_cohesive_response(tm,gi,gii,tn,ts,eta,kn,ks,pert,3,1,dold,dvold,dp,dvp,lp,l0p,lfp,bp,dt,rho)
  ref(:,j)=(tp(:,1)-tm(:,1))/(2d0*hh)
 enddo
 err=sqrt(sum((c-ref)**2))/max(1d0,sqrt(sum(ref**2)))
 write(11,'(I0,A,I0,16(A,ES24.16))') id,',',idx,(',',du(j),j=1,3), &
   (',',t(j,1),j=1,3),',',d,',',dv,',',lam,',',l0,',',lf,',',b,',',psi,',',diss,',',y,',',err
 dold=d;dvold=dv;yold=y;oldid=id
enddo
close(10);close(11)
end program
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--zeta',type=float)
    parser.add_argument('--extended',action='store_true',help='Add closely bracketed onset probes and viscous refinement histories')
    parser.add_argument('--reference-materials',action='store_true',help='Test the four actual structural property sets instead of the generic K sweep')
    parser.add_argument('--material',choices=['turon2010','krueger2012','turon2006_dcb','turon2018'],help='Test one actual property set')
    parser.add_argument('--K',type=float,help='Penalty override for the explicitly selected material')
    parser.add_argument('--out',type=Path,help='Fresh output folder; default: campaign material directory')
    args=parser.parse_args()
    if args.zeta is not None and (not math.isfinite(args.zeta) or args.zeta<=0):parser.error('--zeta must be finite and positive')
    if args.K is not None and not args.material:parser.error('--K requires --material')
    out=args.out or CAMPAIGN/'material'/args.name;out.mkdir(parents=True,exist_ok=False)
    original=(ROOT/'src/3D_CZM.f').read_text()
    chunks=[]
    for name in ['k_cohesive_law','k_cohesive_response','k_cohesive_energy','k_matrix_zero']:
        match=re.search(r'^      subroutine '+name+r'\b.*?^      end\s*$',original,re.M|re.S|re.I)
        if match is None:raise ValueError(name)
        chunks.append(match.group())
    kernels='\n'.join(chunks)+'\n'
    (out/'kernels.f').write_text(kernels)
    (out/'ABA_PARAM.INC').write_text('      implicit real*8 (a-h,o-z)\n')
    constant=('%.12e'%(args.zeta if args.zeta is not None else 1e-6)).replace('e','d')
    (out/'driver.f90').write_text(PROGRAM.replace('TANGENT_PERTURBATION',constant))
    cases=[];lines=[]
    def add(label,props,vectors,expected=None):
        id=len(cases)+1
        cases.append(dict(id=id,label=label,props=props,expected_Gc=expected,n=len(vectors)))
        for index,v in enumerate(vectors):
            lines.append(' '.join(map(str,[id,index]+props+[1./max(1,len(vectors)-1)]+list(v))))
    properties=[('K%g'%K,.187,.781,38.,52.,1.22,K) for K in [1e4,1e5,1e6,1e7]]
    if args.reference_materials or args.material:
        from structured_specimen import MATERIALS
        properties=[]
        for family in ([args.material] if args.material else ['turon2010','krueger2012','turon2006_dcb','turon2018']):
            p=MATERIALS[family];properties.append((family,*p['G'],p['tn'],p['ts'],p['eta'],args.K if args.K is not None else p['K']))
    for label,gi,gii,tn,ts,eta,K in properties:
        for B in [0.,.2,.5,.8,1.]:
            ks=K*gi/gii*(ts/tn)**2;kb=K*(1-B)+ks*B
            G=gi+(gii-gi)*B**eta
            l0=math.sqrt((tn*tn/K+(ts*ts/ks-tn*tn/K)*B**eta)/kb)
            lf=2*G/(kb*l0)
            vec=np.array([math.sqrt(B/ks),0,math.sqrt((1-B)/K)])
            effective=(ks*vec[0]**2+K*vec[2]**2)/math.sqrt(ks*ks*vec[0]**2+K*K*vec[2]**2)
            lam=np.concatenate([np.linspace(0,.95*l0,80),np.linspace(1.05*l0,.995*lf,1800),np.linspace(1.005*lf,1.2*lf,40)])
            add(label+'_B%g'%B,[gi,gii,tn,ts,eta,K,0.],lam[:,None]*vec/effective,G)
            if args.extended:
                probe=np.array([0,.999*l0,1.001*l0,1.01*l0])
                add('onset_'+label+'_B%g'%B,[gi,gii,tn,ts,eta,K,0.],probe[:,None]*vec/effective)
    for count in ([] if args.reference_materials or args.material else [200,1000,5000]):
        times=np.array([0,.25,.5,.7,1.]);vertices=np.array([[0,0,0],[.003,0,.002],[-.002,.002,.003],[.003,-.003,.001],[.06,.04,.04]])
        t=np.linspace(0,1,count+1)
        v=np.column_stack([np.interp(t,times,vertices[:,i]) for i in range(3)])
        add('changing_%d'%count,[.187,.781,38.,52.,1.22,1e5,0.],v)
    if args.extended and not (args.reference_materials or args.material):
        for rho in [1e-4,1e-3]:
            for count in [100,1000,10000]:
                kn=1e5;ks=kn*.187/.781*(52./38.)**2
                direction=np.array([math.sqrt(kn/ks),0.,1.]);direction/=np.linalg.norm(direction)
                add('visc_rho%g_n%d'%(rho,count),[.187,.781,38.,52.,1.22,kn,rho],np.linspace(0,.08,count+1)[:,None]*direction)
    (out/'input.txt').write_text('\n'.join(lines)+'\n')
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    batch='@echo off\r\ncd /d "'+win(out)+'"\r\n'
    batch+='call "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.0.0\\env\\vars.bat" intel64 vs2022\r\n'
    batch+='ifort /nologo /O2 /extend-source:132 driver.f90 kernels.f /exe:material_driver.exe > compiler.txt 2>&1\r\n'
    batch+='if errorlevel 1 exit /b 1\r\nmaterial_driver.exe > driver_stdout.txt 2>&1\r\nexit /b %errorlevel%\r\n'
    (out/'compile_run.cmd').write_text(batch)
    result=subprocess.run(['cmd.exe','/c',win(out/'compile_run.cmd')],cwd=out)
    manifest=dict(canonical_sha256=hashlib.sha256(original.encode()).hexdigest(),kernels_sha256=hashlib.sha256(kernels.encode()).hexdigest(),
                  zeta=args.zeta if args.zeta is not None else 1e-6,returncode=result.returncode,
                  reference_materials=args.reference_materials,
                  selected_material=args.material,penalty_override=args.K,
                  statement='Constitutive routines extracted from canonical source; ABA_PARAM implicit-double stub; not a full UEL assembly test.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
