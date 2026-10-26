"""Run with Abaqus Python in one retained study run folder (read-only ODB)."""
from __future__ import print_function
import json
import os
from odbAccess import openOdb


def main():
    meta = json.load(open('run.json'))
    odb = openOdb(str(meta['name'])+'.odb',readOnly=True)
    def region(tag):
        if tag in odb.rootAssembly.nodeSets:
            return odb.rootAssembly.nodeSets[str(tag)]
        found = [inst.nodeSets[str(tag)] for inst in odb.rootAssembly.instances.values() if tag in inst.nodeSets]
        if len(found)!=1:
            raise RuntimeError('Expected unique set '+tag)
        return found[0]
    def values(frame,field,tag):
        return frame.fieldOutputs[str(field)].getSubset(region=region(str(tag))).values
    def avg(frame,field,tag,d):
        data = values(frame,field,tag)
        return sum(v.data[d] for v in data)/len(data)
    records = []
    try:
        step = odb.steps['LOAD']
        for frame in step.frames:
            if frame.frameValue==0:
                continue
            opening = avg(frame,'U','LOADUP',1)-avg(frame,'U','LOADDOWN',1)
            upper = sum(v.data[1] for v in values(frame,'RF','LOADUP'))
            lower = sum(v.data[1] for v in values(frame,'RF','LOADDOWN'))
            force = (upper-lower)/2*meta['force_scale']
            per_tag = {}
            for tag in ['TIPUP','TIPLOW','BEHINDUP','BEHINDLOW']:
                per_tag[tag] = [v.data for v in sorted(values(frame,'U',tag),key=lambda v:v.nodeLabel)]
            tip = list(sorted(values(frame,'U','TIPUP'),key=lambda v:v.nodeLabel))
            by_node = {v.nodeLabel:[0.,0.] for v in tip}
            if 'NFORC' in frame.fieldOutputs:
                for v in values(frame,'NFORC','TIPUP'):
                    for j in [0,1]: by_node[v.nodeLabel][j] += v.data[j]
            else:
                for j in [0,1]:
                    for v in values(frame,'NFORC'+str(j+1),'TIPUP'):
                        by_node[v.nodeLabel][j] += float(v.data)
            closure = [0.,0.]
            front = []
            for i,v in enumerate(tip):
                delta = [per_tag['BEHINDUP'][i][j]-per_tag['BEHINDLOW'][i][j] for j in [0,1]]
                work = [by_node[v.nodeLabel][j]*delta[j]/2 for j in [0,1]]
                for j in [0,1]: closure[j] += work[j]
                dz = meta['width_mm']/meta['nz'] if meta['nz'] else meta['width_mm']
                if meta['nz'] and i in [0,len(tip)-1]: dz *= .5
                front.append(dict(node=v.nodeLabel,tributary_width_mm=dz,
                                  GI_N_per_mm=work[1]/(dz*meta['closure_length_mm']),
                                  GII_N_per_mm=work[0]/(dz*meta['closure_length_mm'])))
            area = meta['width_mm']*meta['closure_length_mm']
            strain = frame.fieldOutputs['E' if 'E' in frame.fieldOutputs else 'LE']
            stress = frame.fieldOutputs['S']
            checks = {}
            for field in [strain,stress]:
                for j,tag in enumerate(field.componentLabels):
                    checks['max_abs_'+tag] = max(abs(v.data[j]) for v in field.values)
            jump = max(abs(per_tag['TIPUP'][i][j]-per_tag['TIPLOW'][i][j])
                       for i in range(len(tip)) for j in [0,1])
            record = dict(time=frame.frameValue,opening_mm=opening,force_N=force,
                          compliance_mm_per_N=opening/force,
                          reaction_imbalance_fraction=abs(upper+lower)/max(abs(upper),abs(lower)),
                          GI_N_per_mm=closure[1]/area,GII_N_per_mm=closure[0]/area,
                          tip_tie_jump_mm=jump,front=front,strain_stress_checks=checks)
            energies = {}
            for name,r in step.historyRegions.items():
                if name.upper().startswith('ASSEMBLY'):
                    for tag,h in r.historyOutputs.items():
                        energies[tag] = min(h.data,key=lambda tv:abs(tv[0]-frame.frameValue))[1]
            record['energies_slice_Nmm'] = energies
            work = .5*force/meta['force_scale']*opening
            record['reaction_work_slice_Nmm'] = work
            record['strain_energy_relative_error'] = abs(energies['ALLSE']-work)/abs(work)
            records.append(record)
        if not records or abs(records[-1]['time']-1.)>1e-7:
            raise RuntimeError('Incomplete loading path')
        result = dict(name=meta['name'],frames=records,last=records[-1],
                      compliance_spread_fraction=(max(r['compliance_mm_per_N'] for r in records)-
                      min(r['compliance_mm_per_N'] for r in records))/records[-1]['compliance_mm_per_N'])
        encoded = json.dumps(result,indent=2,default=float)
        if os.path.exists('response.json'):
            i = 1
            while os.path.exists('response_previous_%02d.json'%i): i += 1
            os.rename('response.json','response_previous_%02d.json'%i)
        with open('response.json','w') as handle:
            handle.write(encoded+'\n')
    finally:
        odb.close()


if __name__=='__main__': main()
