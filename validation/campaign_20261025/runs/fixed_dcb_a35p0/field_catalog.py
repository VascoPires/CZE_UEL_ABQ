from odbAccess import openOdb
import json
o=openOdb("fixed_dcb_a35p0.odb",readOnly=True)
f=o.steps.values()[0].frames[-1]
d={str(k):dict(components=list(v.componentLabels),count=len(v.values),first=str(v.values[0].data) if len(v.values) else None) for k,v in f.fieldOutputs.items()}
json.dump(d,open("field_catalog.json","w"),indent=2)
o.close()
