"""Abaqus Python 2/3: export converged nodal group response and global energies."""
from __future__ import print_function
import csv
import json
import os
import sys
import bisect
from odbAccess import openOdb


def nforc_component(frame,region,index):
    if 'NFORC' in frame.fieldOutputs:
        return sum(v.data[index] for v in frame.fieldOutputs['NFORC'].getSubset(region=region).values)
    key='NFORC%d'%(index+1)
    return sum(float(v.data) for v in frame.fieldOutputs[key].getSubset(region=region).values)


def main():
    manifest=json.load(open('run.json'))
    odb=openOdb(str(manifest['name'])+'.odb',readOnly=True)
    groups=manifest.get('groups') or [{'label':c['label'],'set':'TOP%d'%c['element']} for c in manifest['cases']]
    handle=open('nodal_response.csv','wb' if sys.version_info[0]<3 else 'w')
    writer=csv.writer(handle)
    writer.writerow(['step','increment','time','group','u1_mm','u2_mm','u3_mm','rf1_N','rf2_N','rf3_N'])
    energies={}
    checks=[]
    try:
        elapsed=0.
        for step_name in odb.steps.keys():
            step=odb.steps[step_name]
            for frame in step.frames:
                for group in groups:
                    set_name=str(group['set']).upper()
                    if set_name in odb.rootAssembly.nodeSets:
                        region=odb.rootAssembly.nodeSets[set_name]
                    else:
                        matches=[inst.nodeSets[set_name] for inst in odb.rootAssembly.instances.values() if set_name in inst.nodeSets]
                        if len(matches)!=1:
                            raise RuntimeError('Expected one node set '+set_name)
                        region=matches[0]
                    us=[v.data for v in frame.fieldOutputs['U'].getSubset(region=region).values]
                    fs=[v.data for v in frame.fieldOutputs['RF'].getSubset(region=region).values]
                    avg=[sum(v[i] for v in us)/len(us) for i in range(3)]
                    force=[sum(v[i] for v in fs) for i in range(3)]
                    writer.writerow([step_name,frame.incrementNumber,elapsed+frame.frameValue,group['label']]+avg+force)
                if 'family' in manifest:
                    check=dict(step=str(step_name),increment=frame.incrementNumber,time=elapsed+frame.frameValue)
                    displacements={}
                    pairs={}
                    faces={0:{},1:{}}
                    for instance in odb.rootAssembly.instances.values():
                        if 'ALL_BULK_NODES' not in instance.nodeSets:continue
                        bulk=instance.nodeSets['ALL_BULK_NODES']
                        us=frame.fieldOutputs['U'].getSubset(region=bulk).values
                        displacements={v.nodeLabel:v.data for v in us}
                        for node in bulk.nodes:
                            per_arm=(manifest['nx']+1)*(manifest['ny_per_arm']+1)*(manifest['nz']+1)
                            key=((node.label-1)//per_arm,round(node.coordinates[0],7),round(node.coordinates[1],7))
                            pairs.setdefault(key,[]).append(displacements[node.label])
                            if abs(node.coordinates[1])<1e-7 and node.coordinates[0]<manifest['parameters']['a']-1e-7:
                                faces[key[0]][(round(node.coordinates[0],7),round(node.coordinates[2],7))]=displacements[node.label]
                    if displacements:
                        check['max_abs_U3_mm']=max(abs(v[2]) for v in displacements.values())
                        check['max_inplane_width_pair_difference_mm']=max(max(v[j] for v in vs)-min(v[j] for v in vs) for vs in pairs.values() for j in [0,1])
                        keys=set(faces[0])&set(faces[1])
                        if keys:check['max_precrack_global_Y_penetration_mm']=max(0.,-min(faces[1][key][1]-faces[0][key][1] for key in keys))
                        gaps=[]
                        for z in set(key[1] for key in keys):
                            line=sorted(key for key in keys if key[1]==z)
                            lower=sorted((key[0]+faces[0][key][0],faces[0][key][1]) for key in line)
                            xx=[v[0] for v in lower]
                            for key in line:
                                ux=key[0]+faces[1][key][0];uy=faces[1][key][1]
                                if len(xx)<2 or ux<xx[0] or ux>xx[-1]:continue
                                j=max(1,min(len(xx)-1,bisect.bisect_right(xx,ux)))
                                dx=xx[j]-xx[j-1]
                                if dx<=0:continue
                                slope=(lower[j][1]-lower[j-1][1])/dx
                                ly=lower[j-1][1]+slope*(ux-xx[j-1])
                                gaps.append((uy-ly)/(1+slope*slope)**.5)
                        if gaps:check['max_precrack_projected_normal_penetration_mm']=max(0.,-min(gaps))
                    contact_openings=[float(v.data) for key,field in frame.fieldOutputs.items() if key.startswith('COPEN') for v in field.values]
                    if contact_openings:check['max_contact_penetration_mm']=max(0.,-min(contact_openings))
                    strain_name='LE' if 'LE' in frame.fieldOutputs else 'E'
                    if strain_name in frame.fieldOutputs:
                        le=frame.fieldOutputs[strain_name]
                        values=[v.data for v in le.values if len(v.data)==6]
                        if values:
                            check['max_abs_'+strain_name]=max(abs(x) for v in values for x in v)
                            for component,index in [(strain_name+'33',2),(strain_name+'13',4),(strain_name+'23',5)]:
                                check['max_abs_'+component]=max(abs(v[index]) for v in values)
                    has_nforc='NFORC' in frame.fieldOutputs or 'NFORC1' in frame.fieldOutputs
                    if manifest.get('bonded_fixed_crack') and has_nforc:
                        regions={}
                        for tag in ['TIPUP','TIPLOW','BEHINDUP','BEHINDLOW']:
                            matches=[inst.nodeSets[tag] for inst in odb.rootAssembly.instances.values() if tag in inst.nodeSets]
                            if tag in odb.rootAssembly.nodeSets:matches=[odb.rootAssembly.nodeSets[tag]]
                            if len(matches)==1:regions[tag]=matches[0]
                        if len(regions)==4:
                            mean={}
                            for tag,region in regions.items():
                                values=frame.fieldOutputs['U'].getSubset(region=region).values
                                mean[tag]=[sum(v.data[j] for v in values)/len(values) for j in range(3)]
                            tip_force=[nforc_component(frame,regions['TIPUP'],j) for j in range(3)]
                            relative=[mean['BEHINDUP'][j]-mean['BEHINDLOW'][j] for j in range(3)]
                            area=manifest['slice_width_mm']*manifest['vcct_closure_length_mm']
                            # Standard NFORC is the element action on the rest
                            # of the model. Pure-I fixed-crack calibration gives
                            # positive upper-tip NFORC2 with positive opening.
                            check['vcct_raw_normal_closure_work_N_per_mm']=tip_force[1]*relative[1]/(2*area)
                            check['vcct_GI_N_per_mm']=tip_force[1]*max(relative[1],0.)/(2*area)
                            check['vcct_GII_N_per_mm']=tip_force[0]*relative[0]/(2*area)
                            check['vcct_tip_tie_jump_norm_mm']=sum((mean['TIPUP'][j]-mean['TIPLOW'][j])**2 for j in range(3))**.5
                            check['vcct_tip_force_upper_x_N']=tip_force[0]
                            check['vcct_tip_force_upper_y_N']=tip_force[1]
                            check['vcct_behind_relative_U1_mm']=relative[0]
                            check['vcct_behind_relative_U2_mm']=relative[1]
                    if manifest['family']=='mmb':
                        means={}
                        for tag in ['MID','LEVER','LOADUP']:
                            matches=[inst.nodeSets[tag] for inst in odb.rootAssembly.instances.values() if tag in inst.nodeSets]
                            if tag in odb.rootAssembly.nodeSets:matches=[odb.rootAssembly.nodeSets[tag]]
                            if len(matches)==1:
                                values=frame.fieldOutputs['U'].getSubset(region=matches[0]).values
                                means[tag]=sum(v.data[1] for v in values)/len(values)
                        if len(means)==3:
                            ratio=manifest['lever_c_mm']/manifest['parameters']['L']
                            check['lever_equation_residual_mm']=means['LEVER']-(1+ratio)*means['MID']+ratio*means['LOADUP']
                        if has_nforc:
                            forces={}
                            for tag in ['ALL_END_UP','MIDLINE','LEVER']:
                                matches=[inst.nodeSets[tag] for inst in odb.rootAssembly.instances.values() if tag in inst.nodeSets]
                                if tag in odb.rootAssembly.nodeSets:matches=[odb.rootAssembly.nodeSets[tag]]
                                if len(matches)!=1:continue
                                if tag=='LEVER':forces[tag]=sum(v.data[1] for v in frame.fieldOutputs['RF'].getSubset(region=matches[0]).values)
                                else:forces[tag]=nforc_component(frame,matches[0],1)
                            if len(forces)==3 and abs(forces['LEVER'])>1e-6:
                                check['upper_end_bulk_NFORC2_N']=forces['ALL_END_UP']
                                check['midline_bulk_NFORC2_N']=forces['MIDLINE']
                                check['lever_RF2_N']=forces['LEVER']
                                check['lever_upper_force_magnitude_relative_error']=abs(abs(forces['ALL_END_UP']/forces['LEVER'])-ratio)/ratio
                                check['lever_mid_force_magnitude_relative_error']=abs(abs(forces['MIDLINE']/forces['LEVER'])-(1+ratio))/(1+ratio)
                    checks.append(check)
            for key,region in step.historyRegions.items():
                if key.upper().startswith('ASSEMBLY'):
                    for name,h in region.historyOutputs.items():
                        if name.startswith('ALL') or name in ['ETOTAL']:
                            energies.setdefault(name,[]).extend([[elapsed+t,v] for t,v in h.data])
            elapsed+=step.timePeriod
        json.dump(energies,open('energies.json','w'),indent=2)
        if checks:json.dump(checks,open('kinematic_checks.json','w'),indent=2,default=float)
    finally:
        handle.close();odb.close()


if __name__=='__main__':main()
