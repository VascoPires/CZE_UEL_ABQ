from odbAccess import openOdb
o=openOdb("dcb_local015_rho1e5.odb",readOnly=True)
i=o.rootAssembly.instances.values()[0]
f=o.steps.values()[0].frames[-1]
print("FIELDS",f.fieldOutputs.keys())
v=sorted(f.fieldOutputs["U"].values,key=lambda v:abs(v.data[2]),reverse=True)
for u in v[:15]:
 print(u.nodeLabel,tuple(u.data),tuple(i.getNodeFromLabel(u.nodeLabel).coordinates))
o.close()
