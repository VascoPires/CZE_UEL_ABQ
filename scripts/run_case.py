"""Prepare an isolated Abaqus job; submit only with --submit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def validate_deck(text):
    if re.search(r'^\*INCLUDE\b', text, re.I | re.M):
        raise ValueError('This runner requires a self-contained deck; expand includes first.')
    users = re.findall(r'^\*USER ELEMENT[^\n]*', text, re.I | re.M)
    property_counts = set()
    for line in users:
        if not re.search(r'\bUNSYMM\b', line, re.I):
            raise ValueError('Declare U8 with UNSYMM to retain the mixed-mode tangent.')
        props = re.search(r'(?<!I)\bPROPERTIES\s*=\s*(\d+)', line, re.I)
        iprops = re.search(r'\bIPROPERTIES\s*=\s*(\d+)', line, re.I)
        nvars = re.search(r'\bVARIABLES\s*=\s*(\d+)', line, re.I)
        if not props or int(props[1]) not in (7,8) or not iprops or int(iprops[1]) != 1:
            raise ValueError('Current U8 requires PROPERTIES=7 or 8, IPROPERTIES=1.')
        property_counts.add(int(props[1]))
        if not nvars or int(nvars[1]) < 108:
            raise ValueError('Allocate at least 108 state variables (examples use 200).')
    if len(property_counts)>1:
        raise ValueError('Use one real-property count throughout the deck; use eight properties with explicit zeta for mixed settings.')
    expected_count=next(iter(property_counts),0)+1
    for m in re.finditer(r'^\*UEL PROPERTY[^\n]*\n((?:(?:\*\*[^\n]*|[^*\n][^\n]*|)\n)*)', text+'\n', re.I | re.M):
        values = []
        for line in m[1].splitlines():
            if not line.strip() or line.lstrip().startswith('**'):
                continue
            fields = line.split(',')
            if not fields[-1].strip():fields.pop()
            if len(fields)>8 or any(not x.strip() for x in fields):
                raise ValueError('UEL property data requires at most eight values per line, without empty fields.')
            values.extend(float(x.strip().replace('D','E').replace('d','e')) for x in fields)
            # Validate the declared values. Legacy supplied MMB decks contain
            # an unused subsequent line; do not change their preparation.
            if len(values)>=expected_count:break
        real_count = len(values)-1
        if (real_count not in property_counts or any(not math.isfinite(v) for v in values)
                or any(v <= 0 for v in values[:6]) or values[6] < 0 or values[-1] not in (1,2)):
            raise ValueError('Expected declared real properties GIc, GIIc, strengths, eta, K3, rho, optional zeta, then integer scheme.')
        if real_count == 8 and values[7] <= 0:
            raise ValueError('Tangent perturbation zeta must be finite and strictly positive.')
    if users and not re.search(r'^\*UEL PROPERTY', text, re.I | re.M):
        raise ValueError('UEL properties missing.')
    return bool(users)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--name', help='Job name, default: input stem')
    p.add_argument('--out', type=Path, help='New run folder; default: runs/<name>')
    p.add_argument('--source', type=Path, default=ROOT/'src/3D_CZM.f')
    p.add_argument('--abaqus', default='abaqus', help='Executable or full path to abaqus.bat')
    p.add_argument('--cpus', type=int, default=1)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    name = args.name or args.input.stem
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name) or args.cpus < 1:
        p.error('Use a job name starting with a letter, and at least one CPU.')
    inp = args.input.resolve()
    text = inp.read_text()
    uses_uel = validate_deck(text)
    source_bytes = args.source.read_bytes() if uses_uel else None
    out = (args.out or ROOT/'runs'/name).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(inp, out/(name+'.inp'))
    cmd = [args.abaqus, 'job='+name, 'input='+name+'.inp', 'cpus='+str(args.cpus), 'interactive']
    if uses_uel:
        (out/'3D_CZM.f').write_bytes(source_bytes)
        cmd.append('user=3D_CZM.f')
    provenance = {'input':str(inp), 'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
                  'uel_sha256':hashlib.sha256(source_bytes).hexdigest() if uses_uel else None,
                  'command':cmd, 'submitted':args.submit}
    (out/'run.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Prepared:',out)
    print(subprocess.list2cmdline(cmd))
    if args.submit:
        if os.name == 'nt':
            cmd = ['cmd', '/c'] + cmd
        subprocess.run(cmd, cwd=out, check=True)


if __name__ == '__main__':
    main()
