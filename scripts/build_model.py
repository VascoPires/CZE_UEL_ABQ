"""Generate a native cohesive model with CAE, then write a separate UEL deck."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from run_case import validate_deck

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('uel_input', ROOT/'scripts/convert_input.py')
uel_input = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uel_input)


def abaqus_call(executable, arguments, cwd):
    command = [executable] + list(arguments)
    if os.name == 'nt':
        command = ['cmd', '/c'] + command
    subprocess.run(command, cwd=cwd, check=True)


def validate_config(config, native=False):
    name = config.get('run_name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
        raise ValueError('Set run_name to a name starting with a letter; use letters, digits, _ or -.')
    if config.get('mode', 'CZM').upper() != 'CZM' or config.get('cohesive_tie', False):
        raise ValueError('This generator supports the CZM solid/cohesive-element workflow.')
    material = config['materials'][config['material_key']]
    index = config.get('specimen_index', 0)
    if not isinstance(index, int) or not 0 <= index < len(material['specimens']):
        raise ValueError('specimen_index is outside the selected material specimens.')
    use_uel = not native and config.get('Run Subroutine', True)
    if use_uel:
        uel_input._parse_manual_uel_values(config['cohesive_UEL'])
    return name, use_uel


def build_model(family, config_path, out, abaqus='abaqus', native=False, submit=False, cpus=1):
    family = family.upper()
    if family not in ('DCB', 'MMB'):
        raise ValueError('Choose DCB or MMB.')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    name, use_uel = validate_config(config, native=native)
    if cpus < 1:
        raise ValueError('cpus must be positive.')
    backend = ROOT/'examples'/family.lower()/(family.lower()+'_script.py')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_copy = out/(family.lower()+'_config.json')
    config['Run Subroutine'] = use_uel
    config_copy.write_text(json.dumps(config, indent=2)+'\n')
    print('Generating {} model in {}'.format(family, out), flush=True)
    abaqus_call(abaqus, ['cae', 'noGUI='+str(backend), '--', str(config_copy)], cwd=out)
    native_input = out/(name+'_CZM.inp')
    # CAE noGUI can report errors without a nonzero process return code.
    if not native_input.is_file():
        raise RuntimeError('CAE did not create {}. Check the CAE output above.'.format(native_input.name))
    if validate_deck(native_input.read_text()):
        raise ValueError('CAE must produce a native cohesive deck before conversion.')
    selected = native_input
    job_name = native_input.stem
    if use_uel:
        uel_path = out/(name+'_UEL.inp')
        # The native deck remains unchanged for direct inspection/comparison.
        text = uel_input.convert_to_uel(native_input.read_text(), config['cohesive_UEL'])
        validate_deck(text)
        uel_path.write_text(text)
        shutil.copy2(ROOT/'src/3D_CZM.f', out/'3D_CZM.f')
        selected = uel_path
        job_name = uel_path.stem
    command = ['job='+job_name, 'input='+selected.name, 'cpus='+str(cpus), 'interactive']
    if use_uel:
        command.append('user=3D_CZM.f')
    print('Native input:', native_input, flush=True)
    if use_uel:
        print('UEL input:', selected, flush=True)
    print('Solve from this folder:', subprocess.list2cmdline([abaqus]+command), flush=True)
    if submit:
        abaqus_call(abaqus, command, cwd=out)
    return selected


def main(family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if family is None:
        parser.add_argument('family', choices=['DCB', 'MMB', 'dcb', 'mmb'])
    parser.add_argument('--config', type=Path, help='Defaults to the selected example configuration')
    parser.add_argument('--out', type=Path, required=True, help='New output folder')
    parser.add_argument('--abaqus', default='abaqus', help='Abaqus executable or full path to abaqus.bat')
    parser.add_argument('--native', action='store_true', help='Generate only the native cohesive input')
    parser.add_argument('--submit', action='store_true', help='Solve the generated input after building')
    parser.add_argument('--cpus', type=int, default=1, help='Solver CPUs when --submit is used')
    args = parser.parse_args()
    family = (family or args.family).upper()
    config = args.config or ROOT/'examples'/family.lower()/(family.lower()+'_config.json')
    try:
        build_model(family, config, args.out, args.abaqus, args.native, args.submit, args.cpus)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, 'Build failed: {}\n'.format(exc))


if __name__ == '__main__':
    main()
