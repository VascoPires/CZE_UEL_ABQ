"""Check damage energy in a completed single-interface Abaqus ODB.

Run with Abaqus Python. --expected is total energy (N mm in these examples),
not toughness per area; multiply a reference toughness by interface area.
"""
from __future__ import print_function
import argparse
import json
import math
from odbAccess import openOdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--expected', type=float, help='Expected final total damage dissipation')
    parser.add_argument('--expected-elastic', type=float, help='Expected final recoverable energy')
    parser.add_argument('--rtol', type=float, default=1e-6)
    args = parser.parse_args()
    odb = openOdb(path=args.odb, readOnly=True)
    try:
        final = {}
        dissipation = []
        peak_elastic = 0.0
        for name in odb.steps.keys():
            step = odb.steps[name]
            regions = [r for key,r in step.historyRegions.items()
                       if key.upper().startswith('ASSEMBLY') and 'ALLPD' in r.historyOutputs]
            if len(regions) != 1:
                raise RuntimeError('Expected one assembly energy history in step '+str(name))
            region = regions[0]
            for key in ['ALLSE','ALLPD','ALLVD','ALLIE','ALLWK']:
                values = region.historyOutputs[key].data
                if not values:
                    raise RuntimeError('Empty energy history: '+key)
                if not all(not math.isnan(v) and not math.isinf(v) for t,v in values):
                    raise RuntimeError('Nonfinite energy history: '+key)
                final[key] = values[-1][1]
            dissipation.extend(v for t,v in region.historyOutputs['ALLPD'].data)
            peak_elastic = max(peak_elastic, max(v for t,v in region.historyOutputs['ALLSE'].data))
        tol = args.rtol*max(1.0,abs(final['ALLPD']))
        if any(v < -tol for v in dissipation):
            raise AssertionError('Negative accumulated damage dissipation')
        if any(b < a-tol for a,b in zip(dissipation,dissipation[1:])):
            raise AssertionError('Accumulated damage dissipation decreases')
        if args.expected is not None and abs(final['ALLPD']-args.expected) > args.rtol*max(abs(args.expected),1e-12):
            raise AssertionError('ALLPD {} differs from expected {}'.format(final['ALLPD'],args.expected))
        if args.expected_elastic is not None and abs(final['ALLSE']-args.expected_elastic) > args.rtol*max(abs(args.expected_elastic),1e-12):
            raise AssertionError('ALLSE differs from expected recoverable energy')
        internal = final['ALLSE']+final['ALLPD']+final['ALLVD']
        if abs(final['ALLIE']-internal) > args.rtol*max(abs(internal),1e-12):
            raise AssertionError('Internal energy components do not sum to ALLIE')
        final['relative_work_balance_error'] = abs(final['ALLIE']-final['ALLWK'])/max(abs(final['ALLWK']),1e-12)
        if peak_elastic <= 0:
            raise AssertionError('No recoverable interface energy was recorded')
        final['peak_ALLSE'] = peak_elastic
        final['energy_samples'] = len(dissipation)
        print(json.dumps(final, indent=2, sort_keys=True))
        print('Energy checks passed.')
    finally:
        odb.close()


if __name__ == '__main__':
    main()
