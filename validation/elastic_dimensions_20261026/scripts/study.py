"""Matched elastic DCB study. Preparation is nonprocessing; execution is explicit."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'validation/scripts'))
from structured_specimen import MATERIALS, x_grid
from campaign import execute, win


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def protected_files():
    # Hash inputs, source and assessment data; record sizes/times for large
    # binary solver artifacts without rereading the entire 14 GB campaign.
    paths = list((ROOT / 'validation/campaign_20261025').rglob('*'))
    paths += [ROOT / 'src/3D_CZM.f', ROOT / 'validation/REPORT.md']
    result = {}
    for p in paths:
        if not p.is_file(): continue
        stat = p.stat()
        entry = dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if p.suffix.lower() in ['.inp','.f','.json','.csv','.md','.py']:
            entry['sha256'] = digest(p)
        result[str(p.relative_to(ROOT))] = entry
    return result


def build(element, dx, ny, a, nz=1, finite=False):
    p = MATERIALS['turon2010']
    is2d = element.startswith('CP')
    width = p['b'] if finite else 1.
    xs = x_grid(p['length'], 35., p['L'], dx)
    # Keep exactly the same stations for all three crack lengths.
    assert a in xs
    nodes, xyz, elems = {}, {}, []
    for arm in range(2):
        for k in range(1 if is2d else nz + 1):
            for j in range(ny + 1):
                for i, x in enumerate(xs):
                    n = len(xyz) + 1
                    nodes[arm, i, j, k] = n
                    xyz[n] = (x, (arm - 1) * p['h'] + j * p['h'] / ny,
                              0. if is2d else width * k / nz)
        for k in range(1 if is2d else nz):
            for j in range(ny):
                for i in range(len(xs) - 1):
                    face = [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]
                    conn = [nodes[arm, ii, jj, k] for ii,jj in face]
                    if not is2d:
                        conn += [nodes[arm, ii, jj, k+1] for ii,jj in face]
                    elems.append([len(elems)+1] + conn)
    bulk_count = len(xyz)
    up, down = bulk_count+1, bulk_count+2
    xyz[up], xyz[down] = (0.,p['h']/2,0.), (0.,-p['h']/2,0.)
    out = ['*HEADING', '** Matched fixed-crack elastic DCB; N, mm, MPa.', '*NODE']
    out += [','.join(map(str, [n]+list(v[:2] if is2d else v))) for n,v in xyz.items()]
    out += ['*ELEMENT,TYPE='+element+',ELSET=BULK']
    out += [','.join(map(str,e)) for e in elems]
    out += ['*MATERIAL,NAME=PLY', '*ELASTIC,TYPE=ENGINEERING CONSTANTS',
            ','.join(map(str,p['E'][:8])), str(p['E'][8]),
            '*ORIENTATION,NAME=AXES', '1,0,0,0,1,0',
            '*SOLID SECTION,ELSET=BULK,MATERIAL=PLY,ORIENTATION=AXES',
            '1.' if is2d else ',']
    sets = {}
    def nset(name, labels):
        sets[name] = labels
        out.append('*NSET,NSET='+name)
        out.extend(','.join(map(str,labels[i:i+16])) for i in range(0,len(labels),16))
    def equation(terms):
        out.extend(['*EQUATION', str(len(terms)), ','.join(str(v) for term in terms for v in term)])
    nset('LOADUP',[up]); nset('LOADDOWN',[down])
    nset('ALL_BULK_NODES',list(range(1,bulk_count+1)))
    coupled = set()
    for arm, tag, rp in [(1,'UP',up),(0,'DOWN',down)]:
        labels = [nodes[arm,0,j,k] for k in range(1 if is2d or not finite else nz+1) for j in range(ny+1)]
        nset('END'+tag,labels); coupled.update(labels)
        out += ['*SURFACE,TYPE=NODE,NAME=SURF'+tag,'END'+tag+',1.',
                '*COUPLING,CONSTRAINT NAME=COUP'+tag+',REF NODE='+str(rp)+',SURFACE=SURF'+tag,
                '*KINEMATIC', '1,2' if is2d or finite else '1,3']
    if not is2d and not finite:
        for arm in range(2):
            for j in range(ny+1):
                for i in range(len(xs)):
                    for d in [1,2]:
                        equation([(nodes[arm,i,j,1],d,1.),(nodes[arm,i,j,0],d,-1.)])
    tip = xs.index(a)
    for tag, arm, j, i in [('TIPUP',1,0,tip),('TIPLOW',0,ny,tip),
                           ('BEHINDUP',1,0,tip-1),('BEHINDLOW',0,ny,tip-1)]:
        nset(tag,[nodes[arm,i,j,k] for k in range(1 if is2d else nz+1)])
    for k in range(1 if is2d or not finite else nz+1):
        for i in range(tip,len(xs)):
            for d in ([1,2] if is2d or not finite else [1,2,3]):
                equation([(nodes[1,i,0,k],d,1.),(nodes[0,i,ny,k],d,-1.)])
    if not is2d:
        if finite:
            nset('WIDTH_ANCHOR',[nodes[0,0,0,0]])
            out += ['*BOUNDARY','WIDTH_ANCHOR,3,3,0.']
        else:
            nset('PLANE_STRAIN_NODES',[n for n in range(1,bulk_count+1) if n not in coupled])
            out += ['*BOUNDARY','PLANE_STRAIN_NODES,3,3,0.']
        out += ['*BOUNDARY',str(up)+',3,5,0.',str(down)+',3,5,0.']
    out += ['*BOUNDARY',str(up)+',1,1,0.',str(down)+',1,1,0.',
            '*STEP,NAME=LOAD,NLGEOM=NO,INC=10','*STATIC','0.25,1.,1e-10,0.25',
            '*BOUNDARY',str(up)+',2,2,0.1',str(down)+',2,2,-0.1',
            '*OUTPUT,FIELD,FREQUENCY=1','*NODE OUTPUT','U,RF',
            '*ELEMENT OUTPUT,ELSET=BULK','S,E,NFORC',
            '*OUTPUT,HISTORY,FREQUENCY=1','*ENERGY OUTPUT','ALLSE,ALLWK,ALLAE,ETOTAL','*END STEP']
    meta = dict(element=element,dx_mm=dx,ny_per_arm=ny,a_mm=a,nz=0 if is2d else nz,
                representation='plane stress' if element.startswith('CPS') else 'plane strain' if is2d else
                'finite width' if finite else 'constrained plane-strain slice',
                width_mm=width,reference_width_mm=p['b'],force_scale=p['b']/width,
                opening_mm=.2,closure_length_mm=xs[tip]-xs[tip-1],parameters=p,
                sets=sets,node_count=bulk_count,element_count=len(elems),
                inplane_grid_sha256=hashlib.sha256(json.dumps([xs,ny,p['h']]).encode()).hexdigest(),
                active_uel=False,nlgeom=False,stabilization=0.,viscosity=0.)
    return '\n'.join(out)+'\n',meta


def prepare():
    if (HERE/'plan.json').exists():
        raise FileExistsError('Study already prepared. Preserve it; choose a fresh study folder.')
    (HERE/'preservation_before.json').write_text(json.dumps(protected_files(),indent=2)+'\n')
    jobs = []
    for dx,ny in [(.5,5),(.25,10)]:
        for element in ['CPS4','CPS4I','CPE4','CPE4I','C3D8','C3D8I']:
            for a in [34.5,35.,35.5]:
                jobs.append((element,dx,ny,a,1,False))
    for nz in [3,7,15,25]:
        for a in [34.5,35.,35.5]:
            jobs.append(('C3D8I',.5,5,a,nz,True))
    names = []
    for element,dx,ny,a,nz,finite in jobs:
        name = ('fw%d_'%nz if finite else '')+element+'_dx'+str(dx)+'_a'+str(a)
        name = name.replace('.','p')
        folder = HERE/'runs'/name
        folder.mkdir(parents=True,exist_ok=False)
        deck,meta = build(element,dx,ny,a,nz,finite)
        assert '*USER ELEMENT' not in deck and '*STATIC\n' in deck
        (folder/(name+'.inp')).write_text(deck)
        meta.update(name=name,status='PREPARED',created_utc=stamp(),input_sha256=digest(folder/(name+'.inp')))
        (folder/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
        names.append(name)
    sources = [Path(__file__), ROOT/'validation/scripts/structured_specimen.py',
               ROOT/'validation/scripts/campaign.py', HERE/'scripts/extract.py', HERE/'scripts/assess.py']
    (HERE/'plan.json').write_text(json.dumps(dict(created_utc=stamp(),jobs=names,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
        status='PREPARED',job_count=len(names)),indent=2)+'\n')
    print('Prepared',len(names),'jobs; no solver submitted.')


def active_solvers():
    command = "Get-CimInstance Win32_Process -Filter \"name='standard.exe' or name='explicit.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('Cannot verify solver availability: '+result.stderr)
    raw = result.stdout.strip()
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data,list) else [data]


def run():
    plan = json.loads((HERE/'plan.json').read_text())
    for path,expected in plan['source_hashes'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Prepared source changed: '+path)
    # A process snapshot can fall in a gap between an unrelated controller's
    # jobs. Once an external queue is identified, require its explicit release.
    hold = HERE/'external_queue_hold.json'
    for name in plan['jobs']:
        folder = HERE/'runs'/name
        meta = json.loads((folder/'run.json').read_text())
        if meta['status'] == 'COMPLETED':
            valid = False
            if (folder/'response.json').exists():
                try:
                    response = json.loads((folder/'response.json').read_text())
                    valid = abs(response['last']['time']-1.)<1e-7
                except (ValueError,KeyError):
                    pass
            if not valid:
                extract(folder)
            continue
        if meta['status'] != 'PREPARED':
            raise RuntimeError('Preserve nonfresh attempt '+name+'; assess it before preparing a retry.')
        if digest(folder/(name+'.inp')) != meta['input_sha256']:
            raise RuntimeError('Input changed: '+name)
        active = active_solvers()
        if active or hold.exists():
            state = dict(status='DEFERRED_EXTERNAL_SOLVER',checked_utc=stamp(),next_job=name,active=active)
            if hold.exists(): state['external_queue_hold'] = json.loads(hold.read_text())
            (HERE/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            return
        execute(folder)
        meta = json.loads((folder/'run.json').read_text())
        if meta['status'] != 'COMPLETED':
            raise RuntimeError('Analysis failed; preserved '+name)
        extract(folder)
    (HERE/'queue_status.json').write_text(json.dumps(dict(status='COMPLETED',end_utc=stamp()),indent=2)+'\n')


def extract(folder):
    command = 'call "C:\\SIMULIA\\Commands\\abaqus.bat" python "'+win(HERE/'scripts/extract.py')+'"'
    attempt = len(list(folder.glob('extraction_attempt_*.txt')))+1
    log = 'extraction_attempt_%02d.txt'%attempt
    batch = '@echo off\r\ncd /d "'+win(folder)+'"\r\n'+command+' > '+log+' 2>&1\r\nexit /b %errorlevel%\r\n'
    (folder/'extract.cmd').write_text(batch)
    result = subprocess.run(['cmd.exe','/c',win(folder/'extract.cmd')],cwd=folder)
    if result.returncode:
        raise RuntimeError((folder/log).read_text(errors='replace'))


def check():
    plan = json.loads((HERE/'plan.json').read_text())
    for name in plan['jobs']:
        folder = HERE/'runs'/name
        meta = json.loads((folder/'run.json').read_text())
        assert digest(folder/(name+'.inp')) == meta['input_sha256']
    before = json.loads((HERE/'preservation_before.json').read_text())
    changed = []
    for path,expected in before.items():
        p = ROOT/path
        if not p.exists():
            changed.append(path); continue
        stat = p.stat()
        if stat.st_size!=expected['size'] or stat.st_mtime_ns!=expected['mtime_ns'] or (
                'sha256' in expected and digest(p)!=expected['sha256']):
            changed.append(path)
    result = dict(checked_utc=stamp(),prepared_jobs=len(plan['jobs']),protected_files=len(before),changed=changed,
                  status='PASS' if not changed else 'FAIL',scope='Input hashes and preservation only; no solver validation')
    (HERE/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert not changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--check',action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.run: run()
    else: check()
