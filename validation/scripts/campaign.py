"""Isolated, serial Abaqus campaign. Preserve all decks, source copies and logs."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT/'validation/campaign_20261025'


def win(path):
    return subprocess.check_output(['wslpath','-w',str(Path(path).resolve())],text=True).strip()


def create_run(name, deck, parameters, source=None):
    folder = CAMPAIGN/'runs'/name
    folder.mkdir(parents=True,exist_ok=False)
    (folder/(name+'.inp')).write_text(deck)
    source = source or ROOT/'src/3D_CZM.f'
    if '*USER ELEMENT' in deck.upper():
        shutil.copy2(source,folder/'3D_CZM.f')
        parameters['uel_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    parameters.update(name=name,created_utc=datetime.now(timezone.utc).isoformat(),status='PREPARED',
                      input_sha256=hashlib.sha256(deck.encode()).hexdigest())
    (folder/'run.json').write_text(json.dumps(parameters,indent=2)+'\n')
    return folder


def execute(folder, datacheck=False):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'run.json').read_text())
    name = manifest['name']
    if (folder/(name+'.odb')).exists() or (folder/'execution_stdout.txt').exists():
        raise FileExistsError('Existing execution artifacts; choose a new run folder.')
    args = ['job='+name,'input='+name+'.inp','cpus=1','interactive']
    if (folder/'3D_CZM.f').exists():
        args.append('user=3D_CZM.f')
    if datacheck:
        args.append('datacheck')
    command = 'call "C:\\SIMULIA\\Commands\\abaqus.bat" '+' '.join(args)
    batch = '@echo off\r\ncd /d "'+win(folder)+'"\r\n'+command+' > execution_stdout.txt 2>&1\r\nexit /b %errorlevel%\r\n'
    (folder/'execute.cmd').write_text(batch)
    manifest.update(command=command,start_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING',datacheck=datacheck)
    (folder/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result = subprocess.run(['cmd.exe','/c',win(folder/'execute.cmd')],cwd=folder)
    sta = folder/(name+'.sta')
    completed = sta.exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
    manifest.update(returncode=result.returncode,end_utc=datetime.now(timezone.utc).isoformat(),
                    status='COMPLETED' if completed else 'DATACHECK_FINISHED' if datacheck and result.returncode==0 else 'FAILED')
    (folder/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(name,manifest['status'],flush=True)
    return manifest


def extract(folder):
    folder=Path(folder).resolve()
    if (folder/'nodal_response.csv').exists():
        if (folder/'energies.json').exists():
            raise FileExistsError('Postprocessed response already exists.')
        attempt=len(list(folder.glob('nodal_response_incomplete_*.csv')))+1
        (folder/'nodal_response.csv').rename(folder/('nodal_response_incomplete_%02d.csv'%attempt))
    command='call "C:\\SIMULIA\\Commands\\abaqus.bat" python "'+win(ROOT/'validation/scripts/extract_odb.py')+'"'
    attempt=len(list(folder.glob('extraction_attempt_*.txt')))+1
    log='extraction_attempt_%02d.txt'%attempt
    batch='@echo off\r\ncd /d "'+win(folder)+'"\r\n'+command+' > '+log+' 2>&1\r\nexit /b %errorlevel%\r\n'
    (folder/'extract.cmd').write_text(batch)
    result=subprocess.run(['cmd.exe','/c',win(folder/'extract.cmd')],cwd=folder)
    if result.returncode:
        raise RuntimeError((folder/log).read_text(errors='replace'))
    print('Extracted',folder,flush=True)


def proportional_vector(B, props, amplitude=.08):
    GI,GII,tn,ts,eta,K,rho,scheme = props
    Ks=K*GI/GII*(ts/tn)**2
    if B==1:
        return [amplitude,0.,0.]
    ratio=math.sqrt(B*K/((1-B)*Ks))
    normal=amplitude/math.sqrt(1+ratio*ratio)
    return [normal*ratio,normal,0.]


def element_cases():
    p=[.187,.781,38.,52.,1.22,1e5,0.,2]
    cases=[]
    for B in [0.,.2,.5,.8,1.]:
        v=proportional_vector(B,p)
        cases.append(dict(label='B'+str(B),props=p[:],points=[[0,0,0,0],[1,*v]],expected_Gc=p[0]+(p[1]-p[0])*B**p[4]))
    cases.append(dict(label='modeIII',props=p[:],points=[[0,0,0,0],[1,0,0,.08]],expected_Gc=p[1]))
    for scheme in [1,2]:
        pp=p[:];pp[-1]=scheme
        cases.append(dict(label='unload_reload_s'+str(scheme),props=pp,points=[[0,0,0,0],[.2,0,.004,0],[.4,0,.001,0],[.6,0,.004,0],[1,0,.04,0]],expected_Gc=p[0]))
    cases.append(dict(label='closure',props=p[:],points=[[0,0,0,0],[.65,0,.04,0],[1,0,-.001,0]],expected_Gc=p[0]))
    for scheme in [1,2]:
        pp=p[:];pp[-1]=scheme
        cases.append(dict(label='mode_change_s'+str(scheme),props=pp,points=[[0,0,0,0],[.25,.003,.002,0],[.5,-.002,.003,.002],[.7,.003,.001,-.003],[1,.06,.04,.04]],expected_Gc=None))
    for K in [1e4,1e6,1e7]:
        pp=p[:];pp[5]=K
        v=proportional_vector(.5,pp)
        cases.append(dict(label='K'+str(K),props=pp,points=[[0,0,0,0],[1,*v]],expected_Gc=p[0]+(p[1]-p[0])*.5**p[4]))
    for rho in [.0001,.001]:
        pp=p[:];pp[6]=rho
        v=proportional_vector(.5,pp)
        cases.append(dict(label='rho'+str(rho),props=pp,points=[[0,0,0,0],[1,*v]],expected_Gc=None))
    return cases


def build_elements(name,max_inc,source=None,tangent_perturbation=None):
    if tangent_perturbation is not None and (not math.isfinite(tangent_perturbation) or tangent_perturbation<=0):raise ValueError('Tangent perturbation must be finite and positive')
    cases=element_cases()
    lines=['*HEADING','** Fully prescribed, independent unit-area U8 tests; source and run parameters retained.','*NODE']
    for e,case in enumerate(cases,1):
        n=(e-1)*8
        coords=[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]*2
        for j,(x,y,z) in enumerate(coords,1):lines.append('%d, %.12g, %.12g, %.12g'%(n+j,x+2*e,y,z))
    lines+=['*USER ELEMENT, TYPE=U8, NODES=8, COORDINATES=3, PROPERTIES=%d, IPROPERTIES=1, VARIABLES=200, UNSYMM'%(7 if tangent_perturbation is None else 8),'1,2,3']
    for e,case in enumerate(cases,1):
        n=(e-1)*8
        case.update(element=e,top_nodes=[n+5,n+6,n+7,n+8])
        values=case['props'][:7]+([] if tangent_perturbation is None else [tangent_perturbation])+case['props'][7:]
        lines+=['*ELEMENT, TYPE=U8, ELSET=E%d'%e,','.join(map(str,[e,n+1,n+4,n+3,n+2,n+5,n+8,n+7,n+6])),
                '*UEL PROPERTY, ELSET=E%d'%e]+[','.join(map(str,values[j:j+8])) for j in range(0,len(values),8)]+[
                '*NSET, NSET=TOP%d'%e,','.join(map(str,case['top_nodes'])),
                '*NSET, NSET=BOT%d'%e,','.join(map(str,[n+1,n+2,n+3,n+4]))]
        for dof in range(1,4):
            lines+=['*AMPLITUDE, NAME=A%d_%d'%(e,dof)]
            pairs=[]
            for point in case['points']:pairs.extend([point[0],point[dof]])
            for j in range(0,len(pairs),8):lines.append(','.join('%.14g'%x for x in pairs[j:j+8]))
    lines+=['*ELSET, ELSET=ALL_E',','.join(str(e) for e in range(1,len(cases)+1)),'*BOUNDARY']
    for e in range(1,len(cases)+1):lines+=['BOT%d,1,3,0'%e,'TOP%d,1,3,0'%e]
    lines+=['*STEP, NAME=TEST, INC=100000, NLGEOM','*STATIC','%.12g,1.,1e-12,%.12g'%(max_inc,max_inc)]
    for e in range(1,len(cases)+1):
        for dof in range(1,4):lines+=['*BOUNDARY, AMPLITUDE=A%d_%d'%(e,dof),'TOP%d,%d,%d,1.'%(e,dof,dof)]
    lines+=['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U,RF','*OUTPUT,HISTORY,FREQUENCY=1','*ENERGY OUTPUT','ALLSE,ALLPD,ALLVD,ALLIE,ALLWK','*EL PRINT,ELSET=ALL_E,FREQUENCY=1']
    for i in range(1,81,8):lines.append(','.join('SDV%d'%j for j in range(i,i+8)))
    lines+=['*END STEP']
    return create_run(name,'\n'.join(lines)+'\n',dict(family='element',cases=cases,max_increment=max_inc,cpus=1,tangent_perturbation=tangent_perturbation),source=source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    b=sub.add_parser('elements');b.add_argument('--name',required=True);b.add_argument('--max-inc',type=float,default=.005);b.add_argument('--run',action='store_true')
    b.add_argument('--source',type=Path)
    b.add_argument('--tangent-perturbation',type=float)
    r=sub.add_parser('run');r.add_argument('folder',type=Path);r.add_argument('--datacheck',action='store_true')
    e=sub.add_parser('extract');e.add_argument('folder',type=Path)
    args=parser.parse_args()
    if args.action=='elements':
        folder=build_elements(args.name,args.max_inc,args.source,args.tangent_perturbation)
        print('Prepared',folder,flush=True)
        if args.run:execute(folder)
    elif args.action=='run':execute(args.folder,args.datacheck)
    elif args.action=='extract':extract(args.folder)


if __name__=='__main__':main()
