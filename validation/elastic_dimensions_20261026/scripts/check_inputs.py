"""Structural input checks; these do not certify Abaqus keyword acceptance."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    checked = []
    grids = {}
    for name in plan['jobs']:
        folder = HERE/'runs'/name
        meta = json.loads((folder/'run.json').read_text())
        deck = (folder/(name+'.inp')).read_text()
        assert '*USER ELEMENT' not in deck and 'STABILIZE' not in deck
        nodes, elements, equations = {}, [], []
        state, pending_count = '', None
        for line in deck.splitlines():
            if line.startswith('**'): continue
            if line.startswith('*'):
                state = line.split(',')[0].upper()
                continue
            data = [v.strip() for v in line.split(',') if v.strip()]
            if state=='*NODE':
                nodes[int(data[0])] = tuple(float(v) for v in data[1:])
            elif state=='*ELEMENT':
                elements.append([int(v) for v in data[1:]])
            elif state=='*EQUATION':
                if pending_count is None:
                    pending_count = int(data[0])
                else:
                    assert len(data)==3*pending_count
                    terms = [(int(data[i]),int(data[i+1]),float(data[i+2])) for i in range(0,len(data),3)]
                    equations.append(terms); pending_count = None
        assert len(elements)==meta['element_count']
        assert len(nodes)==meta['node_count']+2
        is2d = meta['element'].startswith('CP')
        for conn in elements:
            assert len(conn)==(4 if is2d else 8)
            assert len(set(conn))==len(conn) and all(n in nodes for n in conn)
            v0,v1,v2,v3 = [nodes[n] for n in conn[:4]]
            assert v1[0]>v0[0] and v3[1]>v0[1]
            assert v0[1]==v1[1] and v3[1]==v2[1] and v0[0]==v3[0] and v1[0]==v2[0]
            if not is2d: assert nodes[conn[4]][2]>v0[2]
        dependent = []
        for terms in equations:
            for n,d,c in terms:
                assert n in nodes and d in ([1,2] if is2d else [1,2,3]) and abs(c)==1.
            assert len(terms)==2 and terms[0][2]+terms[1][2]==0
            dependent.append(terms[0][:2])
        assert len(dependent)==len(set(dependent)), 'Duplicate eliminated DOF'
        for tag in ['TIPUP','TIPLOW']:
            assert all(abs(nodes[n][0]-meta['a_mm'])<1e-10 and nodes[n][1]==0. for n in meta['sets'][tag])
        for tag in ['BEHINDUP','BEHINDLOW']:
            assert all(abs(nodes[n][0]-(meta['a_mm']-meta['closure_length_mm']))<1e-10 for n in meta['sets'][tag])
        key = (meta['dx_mm'],meta['ny_per_arm'])
        grids.setdefault(key,set()).add(meta['inplane_grid_sha256'])
        checked.append(dict(name=name,nodes=len(nodes),elements=len(elements),equations=len(equations)))
    assert all(len(values)==1 for values in grids.values())
    (HERE/'input_checks.json').write_text(json.dumps(dict(status='PASS',checked=checked,
        scope='Connectivity, positive rectangular geometry, crack stations, unique dependent equation DOFs and matched grids; not solver acceptance'),indent=2)+'\n')
    print('PASS:',len(checked),'prepared input structures; not solver acceptance.')


if __name__=='__main__': main()
