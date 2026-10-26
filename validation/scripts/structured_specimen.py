"""Documented structured C3D8I/U8 slice; independent of the legacy CAE builder.

3D geometry, ordinary plane-strain constraints, zero-thickness interface,
explicit width scaling. MMB uses a linear, weight-free equivalent lever:
w_load=(1+c/L)*w_mid-(c/L)*w_end. This is a labelled reconstruction.
"""
import argparse
import math
from pathlib import Path

from campaign import create_run, execute


MATERIALS={
 'turon2010':dict(E=[120000,10500,10500,.3,.3,.5,5250,5250,3480],length=150.,b=20.,h=1.55,a=35.,L=75.,G=[.260,1.002],eta=2.,tn=30.,ts=58.9,K=1e6),
 'krueger2012':dict(E=[161000,11380,11380,.32,.32,.45,5200,5200,3900],length=152.4,b=25.4,h=2.25,a=25.4,L=50.4,G=[.212,.774],eta=2.1,tn=32.,ts=61.,K=170000.),
 'turon2006_dcb':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=32.9,L=51.,G=[.969,1.719],eta=2.284,tn=80.,ts=100.,K=1e6),
 'turon2006_mmb20':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=33.7,L=51.,G=[.969,1.719],eta=2.284,tn=80.,ts=100.,K=1e6),
 'turon2006_mmb50':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=34.1,L=51.,G=[.969,1.719],eta=2.284,tn=80.,ts=100.,K=1e6),
 'turon2006_mmb80':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=31.4,L=51.,G=[.969,1.719],eta=2.284,tn=80.,ts=100.,K=1e6),
 'turon2006_enf':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=39.2,L=51.,G=[.969,1.719],eta=2.284,tn=80.,ts=100.,K=1e6),
 'turon2018':dict(E=[122700,10100,10100,.25,.25,.45,5500,5500,3700],length=102.,b=25.4,h=1.56,a=32.64,L=51.,G=[.969,1.719],eta=2.,tn=30.,ts=80.,K=1e5),
}
MATERIALS['turon2010_enf']=dict(MATERIALS['turon2010'],ts=60.)


def x_grid(length,a,L,dx,refine_to=None):
    knots=sorted(set([0.,a,L,2*L,length]))
    knots=[x for x in knots if 0<=x<=length]
    if refine_to is not None:knots=sorted(set(knots+[max(0.,a-8),min(length,refine_to)]))
    result=[0.]
    for left,right in zip(knots[:-1],knots[1:]):
        local_dx=dx
        if refine_to is not None:
            middle=(left+right)/2
            local_dx=1.2 if middle<a-8 else dx if middle<refine_to else 3.
        n=int(math.ceil((right-left)/local_dx))
        result.extend(left+(right-left)*i/n for i in range(1,n+1))
    return result


def build(name,material,kind='dcb',mix=.5,dx=.6,ny=5,K=None,scheme=2,rho=0.,max_inc=.005,
          opening=4.,nlgeom=False,strength=None,nz=1,finite_width=False,source=None,line_search=0,native=False,refine_to=None,
          bulk_type='C3D8I',stabilize=None,bonded=False,a0=None,stabilization_tolerance=.005,contact_penalty=1e6,precrack_contact=True,
          contact_discretization='surface',lever_policy='source-or-ideal',max_increments=100000,tangent_perturbation=None):
    if tangent_perturbation is not None:
        if not math.isfinite(tangent_perturbation) or tangent_perturbation<=0:raise ValueError('Tangent perturbation must be finite and positive')
        if native or bonded:raise ValueError('Tangent perturbation applies only to active UEL models')
    p=dict(MATERIALS[material]);p['E']=p['E'][:]
    if a0 is not None:p['a']=a0
    if bonded and native:raise ValueError('Choose either tied fixed-crack or native cohesive control')
    if K is not None:p['K']=K
    if strength is not None:
        p['ts']*=strength/p['tn'];p['tn']=strength
    if kind=='dcb':c=None
    elif kind=='enf':c=None
    elif material=='turon2010':c={.2:149.05,.5:63.18,.8:41.06}[mix]
    elif material=='krueger2012':c={.2:92.9,.5:41.3,.8:27.3}[mix]
    elif material.startswith('turon2006'):c={.2:109.4,.5:44.4,.8:28.4}[mix]
    else:
        q=math.sqrt(3*(1-mix)/mix)
        c=p['L']*(1+q/2)/(3-q/2)  # Camanho Eq.70, derived ideal lever.
    if kind=='mmb' and lever_policy=='corrected-beam-initial-mode-fraction':
        gamma=1.18*math.sqrt(p['E'][0]*p['E'][1])/p['E'][6]
        chi=math.sqrt(p['E'][0]/(11*p['E'][6])*(3-2*(gamma/(1+gamma))**2))
        q=math.sqrt(3*(1-mix)/mix)*(p['a']+.42*chi*p['h'])/(p['a']+chi*p['h'])
        c=p['L']*(1+q/2)/(3-q/2)
    elif lever_policy!='source-or-ideal':raise ValueError(lever_policy)
    width=p['b'] if finite_width else 1.
    xs=x_grid(p['length'],p['a'],p['L'],dx,refine_to)
    nx=len(xs)-1
    nodes={};coords={};elements=[];uel=[];label=0;eid=0;surfaces=[[],[]]
    for arm in range(2):
        for k in range(nz+1):
            for j in range(ny+1):
                y=(-p['h']+p['h']*j/ny) if arm==0 else p['h']*j/ny
                for i,x in enumerate(xs):
                    label+=1;nodes[arm,i,j,k]=label;coords[label]=(x,y,width*k/nz)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    eid+=1
                    conn=[nodes[arm,i,j,k],nodes[arm,i+1,j,k],nodes[arm,i+1,j+1,k],nodes[arm,i,j+1,k],
                          nodes[arm,i,j,k+1],nodes[arm,i+1,j,k+1],nodes[arm,i+1,j+1,k+1],nodes[arm,i,j+1,k+1]]
                    elements.append([eid]+conn)
                    if xs[i+1]<=p['a']+1e-8 and ((arm==0 and j==ny-1) or (arm==1 and j==0)):
                        surfaces[arm].append((eid,'S5' if arm==0 else 'S3'))
    for k in range(nz):
        for i in range(nx):
            if xs[i]<p['a']-1e-8:continue
            eid+=1
            low=[nodes[0,i,ny,k],nodes[0,i,ny,k+1],nodes[0,i+1,ny,k+1],nodes[0,i+1,ny,k]]
            up=[nodes[1,i,0,k],nodes[1,i,0,k+1],nodes[1,i+1,0,k+1],nodes[1,i+1,0,k]]
            uel.append([eid]+low+up)
    up=label+1;down=label+2;load=label+3
    active_rps=([up,down] if kind=='dcb' else [up,load] if kind=='mmb' else [])
    if up in active_rps:coords[up]=(0,p['h']/2,0)
    if down in active_rps:coords[down]=(0,-p['h']/2,0)
    if load in active_rps:coords[load]=(p['L']+(c or 0),p['h'],0)
    out=['*HEADING','** Source-traceable structured slice. See run.json for assumptions.','*NODE']
    out+=['%d,%.12g,%.12g,%.12g'%((n,)+xyz) for n,xyz in coords.items()]
    if bulk_type not in ['C3D8I','C3D8','C3D8R']:raise ValueError(bulk_type)
    out+=['*ELEMENT,TYPE='+bulk_type+',ELSET=BULK']+[','.join(map(str,e)) for e in elements]
    declaration='*USER ELEMENT,TYPE=U8,NODES=8,COORDINATES=3,PROPERTIES=%d,IPROPERTIES=1,VARIABLES=200,UNSYMM'%(7 if tangent_perturbation is None else 8)
    out+=[declaration,'1,2,3','*ELEMENT,TYPE=U8,ELSET=INTERFACE']
    out += [','.join(map(str,e)) for e in uel]
    props=p['G']+[p['tn'],p['ts'],p['eta'],p['K'],rho,scheme]
    values=props[:7]+([] if tangent_perturbation is None else [tangent_perturbation])+props[7:]
    out+=['*UEL PROPERTY,ELSET=INTERFACE']+[','.join(map(str,values[i:i+8])) for i in range(0,len(values),8)]+[
          '*MATERIAL,NAME=PLY','*ELASTIC,TYPE=ENGINEERING CONSTANTS',','.join(map(str,p['E'][:8])),str(p['E'][8]),
          '*ORIENTATION,NAME=AXES','1,0,0,0,1,0','*SOLID SECTION,ELSET=BULK,MATERIAL=PLY,ORIENTATION=AXES',',']
    def nset(name,labels):
        out.append('*NSET,NSET='+name)
        for i in range(0,len(labels),16):out.append(','.join(map(str,labels[i:i+16])))
    if up in active_rps:nset('LOADUP',[up])
    if down in active_rps:nset('LOADDOWN',[down])
    if load in active_rps:nset('LEVER',[load])
    nset('ALL_BULK_NODES',list(range(1,label+1)))
    nset('ALL_END_UP',[nodes[1,0,j,k] for k in range(nz+1) for j in range(ny+1)])
    coupled_nodes=set()
    for arm,tag,rp in [(1,'UP',up),(0,'DOWN',down)]:
        if kind=='enf' or (kind=='mmb' and arm==0):
            continue
        end=[nodes[arm,0,j,k] for k in (range(nz+1) if finite_width else [0]) for j in range(ny+1)]
        coupled_nodes.update(end)
        nset('END'+tag,end)
        out+=['*SURFACE,TYPE=NODE,NAME=SURF'+tag,'END'+tag+',1.',
              '*COUPLING,CONSTRAINT NAME=COUP'+tag+',REF NODE=%d,SURFACE=SURF'%rp+tag,'*KINEMATIC','1,2' if finite_width else '1,3']
    if not finite_width:
        for arm in range(2):
            for j in range(ny+1):
                for i in range(nx+1):
                    for dof in [1,2]:out+=['*EQUATION','2','%d,%d,1.,%d,%d,-1.'%(nodes[arm,i,j,1],dof,nodes[arm,i,j,0],dof)]
    mid_i=min(range(nx+1),key=lambda i:abs(xs[i]-p['L']))
    right_i=min(range(nx+1),key=lambda i:abs(xs[i]-2*p['L']))
    mid=nodes[1,mid_i,ny,0]
    nset('MID',[mid])
    nset('MIDLINE',[nodes[1,mid_i,ny,k] for k in range(nz+1)])
    tip_i=min(range(nx+1),key=lambda i:abs(xs[i]-p['a']))
    nset('TIPUP',[nodes[1,tip_i,0,k] for k in range(nz+1)])
    nset('TIPLOW',[nodes[0,tip_i,ny,k] for k in range(nz+1)])
    nset('BEHINDUP',[nodes[1,tip_i-1,0,k] for k in range(nz+1)])
    nset('BEHINDLOW',[nodes[0,tip_i-1,ny,k] for k in range(nz+1)])
    if kind=='mmb':
        if finite_width:
            for k in range(1,nz+1):
                out+=['*EQUATION','2','%d,2,1.,%d,2,-1.'%(nodes[1,mid_i,ny,k],mid)]
        # The first variable is eliminated by Abaqus. Keep the prescribed
        # lever displacement independent; eliminate the unprescribed midspan.
        out+=['*EQUATION','3','%d,2,1.,%d,2,%.14g,%d,2,%.14g'%(mid,load,-p['L']/(p['L']+c),up,-c/(p['L']+c))]
    if kind in ['mmb','enf'] and precrack_contact:
        out+=['*SURFACE,TYPE=ELEMENT,NAME=CRACKLOW']+['%d,%s'%e for e in surfaces[0]]
        if contact_discretization=='surface':
            out+=['*SURFACE,TYPE=ELEMENT,NAME=CRACKHIGH']+['%d,%s'%e for e in surfaces[1]]
        elif contact_discretization!='nodes':raise ValueError(contact_discretization)
        else:out+=['*SURFACE,TYPE=NODE,NAME=CRACKHIGH']
        # A constrained slice has only one independent normal displacement
        # per x station. Contact both width faces would duplicate constraints.
        # Give the independent line the full tributary width; 3D nodes use
        # their actual tributary width. Exclude an exactly tied crack tip.
        for k in ((range(nz+1) if finite_width else [0]) if contact_discretization=='nodes' else []):
            dz=width/nz*(.5 if k in [0,nz] else 1.) if finite_width else width
            for i,x in enumerate(xs):
                if x>p['a']+1e-8 or (bonded and x>=p['a']-1e-8):continue
                left=xs[i-1] if i else x
                right=min(xs[i+1],p['a']) if i<nx else x
                area=.5*(right-left)*dz
                if area>0:out+=['%d,%.12g'%(nodes[1,i,0,k],area)]
        out+=['*SURFACE INTERACTION,NAME=FRICTIONLESS','*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=HARD,PENALTY=LINEAR',str(contact_penalty),
              '*CONTACT PAIR,INTERACTION=FRICTIONLESS,TYPE='+('NODE TO SURFACE' if contact_discretization=='nodes' else 'SURFACE TO SURFACE')+',SMALL SLIDING','CRACKHIGH,CRACKLOW']
    out+=['*BOUNDARY']
    if not finite_width:
        # Abaqus eliminates all translation variables on coupling nodes and
        # ignores a separate U3 boundary there. Couple U3 to the fixed RP U3,
        # and apply the zero-width BC only to the remaining bulk nodes.
        nset('PLANE_STRAIN_NODES',[n for n in range(1,label+1) if n not in coupled_nodes])
        out+=['*BOUNDARY','PLANE_STRAIN_NODES,3,3,0.']
    else:
        # Suppress width rigid translation at the symmetry plane, without locking contraction.
        nset('WIDTH_ANCHOR',[nodes[0,0,0,0]])
        out+=['*BOUNDARY','WIDTH_ANCHOR,3,3,0.']
    for rp in active_rps:out+=['%d,3,5,0.'%rp]
    if kind=='dcb':out+=['%d,1,1,0.'%up,'%d,1,1,0.'%down]
    else:
        for k in (range(nz+1) if finite_width else [0]):
            out+=['%d,1,2,0.'%nodes[0,0,0,k],'%d,2,2,0.'%nodes[0,right_i,0,k]]
        if kind=='mmb':out+=['%d,1,1,0.'%load,'%d,6,6,0.'%load]
    out+=['*STEP,NAME=LOAD,INC=%d,UNSYMM=YES'%max_increments+(',NLGEOM=YES' if nlgeom else ',NLGEOM=NO'),
          '*STATIC'+(',STABILIZE=%.12g,ALLSDTOL=%.12g'%(stabilize,stabilization_tolerance) if stabilize is not None else ''),'%.12g,1.,1e-10,%.12g'%(max_inc,max_inc)]
    if line_search:out+=['*CONTROLS,PARAMETERS=LINE SEARCH',str(line_search)]
    out+=['*BOUNDARY']
    if kind=='dcb':out+=['%d,2,2,%.12g'%(up,opening/2),'%d,2,2,%.12g'%(down,-opening/2)]
    elif kind=='mmb':out+=['%d,2,2,%.12g'%(load,-opening)]
    else:out+=['%d,2,2,%.12g'%(mid,-opening)]
    out+=['*OUTPUT,FIELD,FREQUENCY=1','*NODE OUTPUT','U,RF','*ELEMENT OUTPUT,ELSET=BULK','S,LE,NFORC',
          '*OUTPUT,HISTORY,FREQUENCY=1','*ENERGY OUTPUT','ALLSE,ALLPD,ALLVD,ALLIE,ALLWK,ALLAE,ALLSD,ALLDMD,ETOTAL',
          '*EL PRINT,ELSET=INTERFACE,FREQUENCY=10']
    sdvs=[j+i for i in [0,20,40,60] for j in range(1,16)]
    for i in range(0,len(sdvs),8):out.append(','.join('SDV%d'%j for j in sdvs[i:i+8]))
    out+=['*END STEP']
    if kind in ['mmb','enf'] and precrack_contact:
        where=out.index('*OUTPUT,HISTORY,FREQUENCY=1')
        out[where:where]=['*CONTACT OUTPUT','CSTRESS,CDISP']
    params=dict(family=kind,material=material,parameters=p,props=props,mixity_nominal=mix,lever_c_mm=c,
                mesh_dx_requested=dx,mesh_dx_actual_max=max(b-a for a,b in zip(xs[:-1],xs[1:])),nx=nx,ny_per_arm=ny,nz=nz,
                refine_to_mm=refine_to,x_grid_mm=xs,
                slice_width_mm=width,reference_width_mm=p['b'],force_scale=p['b']/width,nlgeom=nlgeom,line_search=line_search,
                opening_mm=opening,max_increment=max_inc,bulk_type=bulk_type,integration_scheme=scheme,stabilization_energy_fraction=stabilize,
                stabilization_adaptive_tolerance=stabilization_tolerance if stabilize is not None else None,
                contact_enforcement=contact_discretization+' linear penalty hard contact' if kind in ['mmb','enf'] and precrack_contact else None,
                contact_penalty_N_per_mm3=contact_penalty if kind in ['mmb','enf'] else None,
                precrack_contact_active=bool(kind in ['mmb','enf'] and precrack_contact),
                generator_revision=7,tangent_perturbation=tangent_perturbation,
                lever_policy=lever_policy,
                maximum_increment_count=max_increments,
                representation='finite-width 3D' if finite_width else 'ordinary plane-strain constrained 3D slice',constraint_revision=2,
                bonded_fixed_crack=bonded,vcct_closure_length_mm=xs[tip_i]-xs[tip_i-1],
                groups=([{'label':'upper','set':'LOADUP'},{'label':'lower','set':'LOADDOWN'}] if kind=='dcb' else [{'label':'lever','set':'LEVER'},{'label':'upper','set':'LOADUP'}] if kind=='mmb' else [{'label':'mid','set':'MID'}]),
                interface_elements=[{'label':e[0],'x0':coords[e[1]][0],'x1':coords[e[3]][0],'z0':coords[e[1]][2]} for e in uel],
                assumptions=['MMB linear weight-free lever reconstruction' if kind=='mmb' else 'end-section kinematic load distribution with free in-plane rotation',
                             'No hidden strength or toughness fitting','Work and forces scale by reference_width/slice_width'])
    if bonded:
        start=out.index(declaration)
        end=out.index('*MATERIAL,NAME=PLY')
        ties=[]
        for k in (range(nz+1) if finite_width else [0]):
            for i in range(tip_i,nx+1):
                for dof in ([1,2,3] if finite_width else [1,2]):
                    ties+=['*EQUATION','2','%d,%d,1.,%d,%d,-1.'%(nodes[1,i,0,k],dof,nodes[0,i,ny,k],dof)]
        out[start:end]=ties
        start=next(i for i,line in enumerate(out) if line.startswith('*EL PRINT'))
        out=out[:start]+['*END STEP']
        params.update(law='Elastic fixed-crack control with coincident nodal ties; no active cohesive law')
    elif native:
        start=out.index(declaration)
        del out[start:start+2]
        out[out.index('*ELEMENT,TYPE=U8,ELSET=INTERFACE')]='*ELEMENT,TYPE=COH3D8,ELSET=INTERFACE'
        start=out.index('*UEL PROPERTY,ELSET=INTERFACE')
        ks=p['K']*p['G'][0]/p['G'][1]*(p['ts']/p['tn'])**2
        out[start:start+2]=['*MATERIAL,NAME=COH','*ELASTIC,TYPE=TRACTION','%.14g,%.14g,%.14g'%(p['K'],ks,ks),
                          '*DAMAGE INITIATION,CRITERION=QUADS','%.14g,%.14g,%.14g'%(p['tn'],p['ts'],p['ts']),
                          '*DAMAGE EVOLUTION,TYPE=ENERGY,MIXED MODE BEHAVIOR=BK,POWER=%.14g'%p['eta'],
                          '%.14g,%.14g,%.14g'%(p['G'][0],p['G'][1],p['G'][1]),
                          '*COHESIVE SECTION,ELSET=INTERFACE,MATERIAL=COH,RESPONSE=TRACTION SEPARATION,THICKNESS=SPECIFIED','1.']
        start=next(i for i,line in enumerate(out) if line.startswith('*EL PRINT'))
        out=out[:start]+['*END STEP']
        params.update(native=True,law='Native QUADS/BK comparison; different mixed-mode initiation criterion; pure-I control')
    return create_run(name,'\n'.join(out)+'\n',params,source=source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--material',choices=MATERIALS,default='turon2010')
    parser.add_argument('--kind',choices=['dcb','mmb','enf'],default='dcb');parser.add_argument('--mix',type=float,default=.5)
    parser.add_argument('--dx',type=float,default=.6);parser.add_argument('--ny',type=int,default=5)
    parser.add_argument('--K',type=float);parser.add_argument('--scheme',type=int,choices=[1,2],default=2)
    parser.add_argument('--rho',type=float,default=0.);parser.add_argument('--max-inc',type=float,default=.005)
    parser.add_argument('--opening',type=float,default=4.);parser.add_argument('--nlgeom',action='store_true')
    parser.add_argument('--strength',type=float);parser.add_argument('--source',type=Path)
    parser.add_argument('--line-search',type=int,default=0);parser.add_argument('--native',action='store_true')
    parser.add_argument('--refine-to',type=float)
    parser.add_argument('--bulk-type',choices=['C3D8I','C3D8','C3D8R'],default='C3D8I');parser.add_argument('--stabilize',type=float)
    parser.add_argument('--bonded',action='store_true');parser.add_argument('--a0',type=float)
    parser.add_argument('--contact-penalty',type=float,default=1e6)
    parser.add_argument('--no-precrack-contact',dest='precrack_contact',action='store_false',default=True)
    parser.add_argument('--contact-discretization',choices=['surface','nodes'],default='surface')
    parser.add_argument('--lever-policy',choices=['source-or-ideal','corrected-beam-initial-mode-fraction'],default='source-or-ideal')
    parser.add_argument('--max-increments',type=int,default=100000)
    parser.add_argument('--tangent-perturbation',type=float)
    parser.add_argument('--run',action='store_true')
    args=vars(parser.parse_args());run=args.pop('run');folder=build(**args)
    print('Prepared',folder,flush=True)
    if run:execute(folder)


if __name__=='__main__':main()
