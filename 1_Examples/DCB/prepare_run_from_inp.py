import os
import shutil
import subprocess


name = "try"
working_folder = r"C:\Users\p2321038\Documents\GitHub\PythonLearning\subroutine\002_dev_DCB\try_script"
input_file_name = "try.inp"

# Abaqus runner (set run_abaqus=False to only prepare files)
run_abaqus = True
abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.bat"
job_name = ""

# Optional manual UEL properties (set use_manual=True to override parsing).
cohesive_uel = {
    "use_manual": True,
    "elset": "Cohesive_Layer",
    "elastic": 1.0e6,
    "initiation": [38.0, 52.0],
    "evolution": {
        "power": 1.23,
        "table": [0.187, 0.786]
    },
    "viscosity": 1.0e-6,
    "integration_scheme": 1
}

uel_file_name = "3D_CZM.f"


def _ensure_trailing_slash(path_value):
    path_value = path_value.replace('\\', '/')
    if not path_value.endswith('/'):
        path_value += '/'
    return path_value


def _build_dirname_lines(run_folder):
    path_value = _ensure_trailing_slash(run_folder)
    max_len = 60
    if len(path_value) <= max_len:
        return ['      dirname="{0}"\n'.format(path_value)]

    split_pos = path_value.rfind('/', 0, max_len)
    if split_pos <= 0:
        split_pos = max_len
    part1 = path_value[:split_pos + 1]
    part2 = path_value[split_pos + 1:]
    return [
        '      dirname="{0}"//\n'.format(part1),
        '     & "{0}"\n'.format(part2)
    ]


def _insert_uel_metadata_fixed(f_path, run_folder, first_label, label_line=65, dirname_line=66):
    with open(f_path, 'r') as f_handle:
        lines = f_handle.readlines()

    insert_lines = _build_dirname_lines(run_folder)
    label_idx = label_line - 1
    dirname_idx = dirname_line - 1
    required_len = dirname_idx + len(insert_lines)
    while len(lines) < required_len:
        lines.append('\n')

    lines[label_idx] = '      first_el_label = {0}\n'.format(first_label)
    lines[dirname_idx] = insert_lines[0]
    if len(insert_lines) > 1:
        lines[dirname_idx + 1] = insert_lines[1]

    with open(f_path, 'w') as f_handle:
        f_handle.writelines(lines)


def _remove_cohesive_material_keyword_blocks(lines, cohesive_material):
    """
    Removes ONLY the following keyword blocks (and their data lines)
    inside *MATERIAL, NAME=<cohesive_material>:

      *DAMAGE INITIATION
      *DAMAGE EVOLUTION
      *DENSITY
      *ELASTIC, TYPE=TRACTION
    """
    target = cohesive_material.strip().upper()
    in_target_material = False
    out = []
    i = 0

    def _is_comment_or_blank(s):
        s2 = s.strip()
        return (not s2) or s2.startswith('**')

    while i < len(lines):
        line = lines[i]
        u = line.strip().upper()

        if u.startswith('*MATERIAL') and 'NAME=' in u:
            name = u.split('NAME=')[1].split(',')[0].strip()
            in_target_material = (name == target)
            out.append(line)
            i += 1
            continue

        if in_target_material and u.startswith('*') and not u.startswith('**'):
            if u.startswith('*DAMAGE INITIATION') or \
               u.startswith('*DAMAGE EVOLUTION') or \
               u.startswith('*DENSITY') or \
               (u.startswith('*ELASTIC') and 'TYPE=TRACTION' in u):

                i += 1
                while i < len(lines) and _is_comment_or_blank(lines[i]):
                    i += 1
                if i < len(lines):
                    i += 1
                continue

        out.append(line)
        i += 1

    return out


def _extract_first_element_label(lines, type_token):
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('**'):
            continue
        if stripped.upper().startswith('*ELEMENT') and type_token in stripped.upper():
            in_block = True
            continue
        if in_block:
            if stripped.startswith('*'):
                break
            parts = stripped.replace(',', ' ').split()
            if parts:
                return int(parts[0])
    return None


def _parse_manual_uel_values(cfg):
    evolution = cfg.get('evolution', {})
    table = evolution.get('table', [])
    power = evolution.get('power', None)
    initiation = cfg.get('initiation', [])
    elastic = cfg.get('elastic', None)
    viscosity = cfg.get('viscosity', 0.0)
    integration_scheme = cfg.get('integration_scheme', 1)

    if len(table) < 2 or len(initiation) < 2 or power is None or elastic is None:
        raise ValueError('Manual cohesive_uel is missing required fields.')

    return [
        table[0],
        table[1],
        initiation[0],
        initiation[1],
        power,
        elastic,
        viscosity,
        integration_scheme
    ]


def _parse_material_values(lines, material_name):
    raise ValueError('Material parsing not implemented. Set cohesive_uel["use_manual"] = True.')


def _patch_inp_for_uel(inp_path, manual_cfg):
    with open(inp_path, 'r') as inp_handle:
        lines = inp_handle.readlines()

    # Find cohesive section info (elset + material)
    cohesive_elset = None
    cohesive_material = None
    for line in lines:
        uline = line.strip().upper()
        if uline.startswith('*COHESIVE SECTION'):
            if 'ELSET=' in uline:
                cohesive_elset = uline.split('ELSET=')[1].split(',')[0].strip()
            if 'MATERIAL=' in uline:
                cohesive_material = uline.split('MATERIAL=')[1].split(',')[0].strip()
            break

    if not cohesive_elset and not manual_cfg.get('elset'):
        raise ValueError('Could not find cohesive elset from *COHESIVE SECTION and no manual elset provided.')

    # Remove ONLY specific keyword blocks inside the cohesive material (keep *Material header and anything else)
    if cohesive_material:
        lines = _remove_cohesive_material_keyword_blocks(lines, cohesive_material)

    # After removals, locate first cohesive element label (for JELEM guard patching)
    first_label = _extract_first_element_label(lines, 'TYPE=COH3D')
    if first_label is None:
        raise ValueError('Could not find cohesive element labels (TYPE=COH3D).')

    # Build UEL property values
    if manual_cfg.get('use_manual'):
        cohesive_elset = manual_cfg.get('elset', cohesive_elset)
        uel_values = _parse_manual_uel_values(manual_cfg)
    else:
        # Keep original behavior if you later set use_manual=False
        # (this parser expects the original traction/damage keywords to still exist)
        if not cohesive_material:
            raise ValueError('Could not find cohesive material from *COHESIVE SECTION.')
        material_vals = _parse_material_values(lines, cohesive_material)
        integration_scheme = manual_cfg.get('integration_scheme', 1)
        uel_values = [
            material_vals['evolution'][0],
            material_vals['evolution'][1],
            material_vals['initiation'][0],
            material_vals['initiation'][1],
            material_vals['power'],
            material_vals['elastic'],
            material_vals['viscosity'],
            integration_scheme
        ]

    uel_header = '*UEL PROPERTY, ELSET={0}\n'.format(cohesive_elset)
    uel_values_line = ', '.join('{0:g}'.format(v) for v in uel_values) + '\n'
    user_element_line = '*USER ELEMENT, TYPE=U8, NODES=8, COORDINATES=3, IPROPERTIES=1, PROPERTIES=8, VARIABLES=200\n'
    dof_line = '1, 2, 3\n'

    new_lines = []
    inserted_user = False
    inserted_uel = False

    skip_section = False
    skip_controls = False

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        uline = line.strip().upper()

        # --- FIX: skip blocks WITHOUT swallowing the next keyword line ---
        if skip_controls:
            if uline.startswith('*') and not uline.startswith('**'):
                skip_controls = False
                continue  # reprocess this keyword line
            idx += 1
            continue

        if skip_section:
            if uline.startswith('*') and not uline.startswith('**'):
                skip_section = False
                continue  # reprocess this keyword line (keeps *END PART, etc.)
            idx += 1
            continue
        # ---------------------------------------------------------------

        # Optional: remove "Section controls" block if present (same as your original script)
        if uline.startswith('**') and 'ELEMENT CONTROLS' in uline:
            idx += 1
            continue
        if uline.startswith('*SECTION CONTROLS'):
            skip_controls = True
            idx += 1
            continue

        # Optional: remove the comment line announcing the cohesive section (same as your original script)
        if uline.startswith('**') and 'SECTION:' in uline and 'COH_SECTION' in uline:
            idx += 1
            continue

        # Replace *COHESIVE SECTION block with *UEL PROPERTY block (skip original cohesive section lines)
        if uline.startswith('*COHESIVE SECTION'):
            new_lines.append(uel_header)
            new_lines.append(uel_values_line)
            inserted_uel = True
            skip_section = True
            idx += 1
            continue

        # Replace cohesive element definition block header (COH3D -> U8)
        if uline.startswith('*ELEMENT') and 'TYPE=COH3D' in uline:
            if not inserted_user:
                new_lines.append(user_element_line)
                new_lines.append(dof_line)
                inserted_user = True
            new_lines.append('*ELEMENT, TYPE=U8, ELSET={0}\n'.format(cohesive_elset))
            idx += 1
            continue

        # Default: keep line
        new_lines.append(line)
        idx += 1

    # If no cohesive section existed, append UEL property at end (fallback)
    if not inserted_uel:
        new_lines.append('\n')
        new_lines.append(uel_header)
        new_lines.append(uel_values_line)

    with open(inp_path, 'w') as inp_handle:
        inp_handle.writelines(new_lines)

    return first_label



def _patch_uel_guard(f_path, first_label):
    with open(f_path, 'r') as f:
        lines = f.readlines()

    label_line = '      first_el_label = {0}\n'.format(first_label)
    target_guard = 'jelem .EQ. first_el_label'
    new_lines = []
    inserted = False
    replaced = False
    skip_next = False

    for idx, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if (not inserted) and 'first_el_label' in line and line.lstrip().startswith('!'):
            new_lines.append(line)
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ''
            if 'first_el_label' in next_line and '=' in next_line:
                new_lines.append(label_line)
                inserted = True
                skip_next = True
                continue
            new_lines.append(label_line)
            inserted = True
            continue

        if 'jelem' in line and '.EQ.' in line:
            if 'jelem' in line and '1' in line:
                new_lines.append(
                    line.replace('jelem .EQ. 1', target_guard)
                        .replace('jelem.eq.1', target_guard)
                        .replace('jelem.eq. 1', target_guard)
                )
                replaced = True
                continue
            if 'jelem' in line and 'first_el_label' in line:
                replaced = True

        new_lines.append(line)

    if not inserted:
        raise ValueError('first_el_label comment not found in UEL file')
    if not replaced:
        raise ValueError('jelem guard not found in UEL file')

    with open(f_path, 'w') as f:
        f.writelines(new_lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(script_dir, working_folder))

    src_inp = os.path.join(working_dir, input_file_name)
    src_uel = os.path.join(working_dir, uel_file_name)

    dest_folder = os.path.join(working_dir, name)
    os.makedirs(dest_folder, exist_ok=True)

    dst_inp = os.path.join(dest_folder, os.path.basename(src_inp))
    dst_uel = os.path.join(dest_folder, uel_file_name)

    shutil.copy(src_inp, dst_inp)
    shutil.copy(src_uel, dst_uel)

    first_label = _patch_inp_for_uel(dst_inp, cohesive_uel)
    _insert_uel_metadata_fixed(dst_uel, os.path.abspath(dest_folder), first_label)

    print("Prepared run folder: {0}".format(dest_folder))

    if run_abaqus:
        abaqus_job = job_name.strip() if job_name else os.path.splitext(os.path.basename(dst_inp))[0]
        subprocess.run([
            abaqus_cmd,
            'job={0}'.format(abaqus_job),
            'input={0}'.format(os.path.basename(dst_inp)),
            'user={0}'.format(uel_file_name)
        ], cwd=dest_folder)


if __name__ == "__main__":
    main()
