"""Convert one COH3D8 interface to U8 using explicit material parameters.

Importing this module never submits a job. The constitutive source is kept
independent of paths; output goes to the solver working directory.
"""
import argparse
import json
import math
import re
from pathlib import Path


def _parse_manual_uel_values(cfg):
    evolution = cfg['evolution']
    values = list(evolution['table'][:2]) + list(cfg['initiation'][:2])
    values += [evolution['power'], cfg['elastic'], cfg.get('viscosity', 0.0)]
    if len(values) != 7 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values[:6]):
        raise ValueError('Seven properties required; first six must be positive.')
    scheme = float(cfg.get('integration_scheme', 1))
    if scheme not in (1, 2) or not math.isfinite(float(values[6])) or float(values[6]) < 0:
        raise ValueError('Integration scheme must be 1 or 2; viscosity must be nonnegative.')
    if 'tangent_perturbation' in cfg:
        zeta = float(cfg['tangent_perturbation'])
        if not math.isfinite(zeta) or zeta <= 0:
            raise ValueError('tangent_perturbation must be finite and strictly positive (length units).')
        values.append(zeta)
    return values + [int(scheme)]


def format_uel_values(values):
    """Abaqus accepts eight real/integer property values per data line."""
    return '\n'.join(', '.join('{:g}'.format(v) for v in values[i:i+8])
                     for i in range(0, len(values), 8))


def convert_to_uel(text, manual_cfg):
    """Return a U8 deck without modifying the original input."""
    if re.search(r'^\*INCLUDE\b', text, re.I | re.M):
        raise ValueError('Expand include files before conversion.')
    sections = list(re.finditer(r'^\*COHESIVE SECTION[^\n]*\n(?:[^*\n][^\n]*\n|\n)*', text, re.I | re.M))
    if len(sections) != 1:
        raise ValueError('Converter supports exactly one cohesive section.')
    elset = re.search(r'ELSET\s*=\s*([^,\n]+)', sections[0][0], re.I)
    if not elset:
        raise ValueError('Cohesive section requires ELSET.')
    headers = list(re.finditer(r'^\*ELEMENT\s*,[^\n]*TYPE\s*=\s*COH3D8\b[^\n]*',text,re.I|re.M))
    all_coh = re.findall(r'^\*ELEMENT[^\n]*TYPE\s*=\s*COH\w+', text, re.I|re.M)
    if len(headers) != 1 or len(all_coh) != 1:
        raise ValueError('Converter supports exactly one COH3D8 element block.')
    first = re.search(r'\n\s*(\d+)\s*,', text[headers[0].end():])
    if not first:
        raise ValueError('Missing cohesive connectivity.')
    values = _parse_manual_uel_values(manual_cfg)
    # Retain the original element set and all node/element labels.
    header = re.sub(r'COH3D8\b', 'U8', headers[0][0], flags=re.I)
    declaration = '*USER ELEMENT, TYPE=U8, NODES=8, COORDINATES=3, PROPERTIES={}, IPROPERTIES=1, VARIABLES=200, UNSYMM\n1, 2, 3\n'.format(len(values)-1)
    text = text[:headers[0].start()] + declaration + header + text[headers[0].end():]
    replacement = '*UEL PROPERTY, ELSET=' + elset[1].strip() + '\n'
    replacement += format_uel_values(values) + '\n'
    text = re.sub(r'^\*COHESIVE SECTION[^\n]*\n(?:[^*\n][^\n]*\n|\n)*',lambda m: replacement,text,flags=re.I|re.M)
    # Unused material and section-control definitions are retained intentionally.
    return text


def _patch_inp_for_uel(inp_path, manual_cfg):
    """Convert an existing file in place after validating the complete input."""
    path = Path(inp_path)
    text = convert_to_uel(path.read_text(), manual_cfg)
    first = re.search(r'^\*ELEMENT[^\n]*TYPE\s*=\s*U8\b[^\n]*\n\s*(\d+)', text, re.I | re.M)
    path.write_text(text)
    return int(first[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--config', required=True, type=Path, help='JSON with cohesive_UEL or direct properties')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')
    cfg = json.loads(args.config.read_text())
    _parse_manual_uel_values(cfg.get('cohesive_UEL', cfg))
    text = convert_to_uel(args.input.read_text(), cfg.get('cohesive_UEL', cfg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == '__main__':
    main()
