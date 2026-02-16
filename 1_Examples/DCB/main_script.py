import json
import os
import shutil
import subprocess
import sys


try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from prepare_run_from_inp import _insert_uel_metadata_fixed
except Exception:
    _insert_uel_metadata_fixed = None


def _load_config(config_path):
    if not os.path.isfile(config_path):
        raise IOError('Config file not found: {0}'.format(config_path))
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)



def _find_first_uel_label(inp_path):
    if not os.path.isfile(inp_path):
        return None
    with open(inp_path, 'r') as inp_handle:
        lines = inp_handle.readlines()

    in_uel_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('**'):
            continue
        if stripped.upper().startswith('*ELEMENT') and 'TYPE=U8' in stripped.upper():
            in_uel_block = True
            continue
        if in_uel_block:
            if stripped.startswith('*'):
                break
            parts = stripped.replace(',', ' ').split()
            if parts:
                try:
                    return int(parts[0])
                except Exception:
                    return None
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'dcb_config.json')
    config = _load_config(config_path)

    run_name = config.get('run_name', None)
    if not run_name:
        raise ValueError('run_name must be defined in dcb_config.json')

    dest_folder = os.path.join(script_dir, run_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    src_dcb = os.path.join(script_dir, 'dcb_script.py')
    src_cfg = os.path.join(script_dir, 'dcb_config.json')
    src_uel = os.path.join(script_dir, '3D_CZM.f')
    src_prep = os.path.join(script_dir, 'prepare_run_from_inp.py')

    dst_dcb = os.path.join(dest_folder, 'dcb_script.py')
    dst_cfg = os.path.join(dest_folder, 'dcb_config.json')
    dst_uel = os.path.join(dest_folder, '3D_CZM.f')
    dst_prep = os.path.join(dest_folder, 'prepare_run_from_inp.py')

    shutil.copy(src_dcb, dst_dcb)
    shutil.copy(src_cfg, dst_cfg)
    shutil.copy(src_uel, dst_uel)
    if os.path.isfile(src_prep):
        shutil.copy(src_prep, dst_prep)

    if _insert_uel_metadata_fixed is None:
        raise ImportError('prepare_run_from_inp._insert_uel_metadata_fixed could not be imported.')

    abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.bat"
    subprocess.run([abaqus_cmd, 'cae', 'noGUI=dcb_script.py'], cwd=dest_folder)

    inp_path = os.path.join(dest_folder, run_name + '_CZM.inp')
    first_label = _find_first_uel_label(inp_path)
    if first_label is None:
        raise ValueError('Could not find first U8 element label in {0}'.format(inp_path))

    _insert_uel_metadata_fixed(dst_uel, os.path.abspath(dest_folder), first_label)
    subprocess.run([abaqus_cmd, 'cae', 'noGUI=dcb_script.py'], cwd=dest_folder)


if __name__ == '__main__':
    main()
