"""Check or replay an exact retained input deck in a fresh folder.

The original generator changed during troubleshooting. Exact deck replay does
not depend on reconstructing its earlier generator revision. Never overwrite an
existing run. --check performs no submission or processing.
"""
import argparse
import hashlib
import json
from pathlib import Path

from campaign import CAMPAIGN,create_run,execute


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('original',help='Existing run name')
    parser.add_argument('--name');parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();folder=CAMPAIGN/'runs'/args.original
    manifest=json.loads((folder/'run.json').read_text());deck=folder/(manifest['name']+'.inp')
    digest=hashlib.sha256(deck.read_bytes()).hexdigest()
    if digest!=manifest['input_sha256']:raise ValueError('Retained deck hash differs from the execution manifest')
    source=folder/'3D_CZM.f'
    if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['uel_sha256']:raise ValueError('Retained UEL hash differs from manifest')
    if args.check:
        print(json.dumps(dict(name=manifest['name'],deck_sha256=digest,uel_sha256=manifest.get('uel_sha256'),status='CHECKED; no solver submission'),indent=2));return
    if not args.name:parser.error('--name is required for a fresh replay')
    params=dict(manifest)
    for key in ['command','start_utc','end_utc','returncode','datacheck','name','created_utc','status']:params.pop(key,None)
    params.update(replayed_from=args.original,reproduction_method='Exact retained input/source files; no regeneration of historical model assumptions')
    target=create_run(args.name,deck.read_text(),params,source if source.exists() else None)
    print('Prepared',target)
    if args.run:execute(target)


if __name__=='__main__':main()
