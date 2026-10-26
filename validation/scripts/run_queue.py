"""Execute a documented serial queue; preserve failed attempts and continue.

One process owns solver submission. Existing run folders are never rebuilt.
"""
import argparse
import json
import subprocess
import sys
import traceback
import time
from datetime import datetime,timezone
from pathlib import Path

from campaign import ROOT,CAMPAIGN,execute,extract,build_elements
from structured_specimen import build
from analyze_campaign import specimen,elements


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('queue',type=Path);parser.add_argument('--wait-for',type=Path);args=parser.parse_args()
    if args.wait_for:
        required=len(json.loads(args.wait_for.read_text()))
        previous_status=args.wait_for.with_suffix('.status.json')
        while True:
            try:
                previous=json.loads(previous_status.read_text())
                if len(previous)==required and all('ended_utc' in item for item in previous):break
            except (OSError,ValueError):pass
            time.sleep(2)
    tasks=json.loads(args.queue.read_text());state=[]
    status=args.queue.with_suffix('.status.json')
    for task in tasks:
        if args.queue.with_suffix('.stop').exists():
            marker=dict(status='STOPPED_BETWEEN_JOBS',next_unstarted=task['name'],stopped_utc=datetime.now(timezone.utc).isoformat(),
                        reason='Cooperative stop sentinel; current job/extraction completed before stopping')
            args.queue.with_suffix('.controller_stopped.json').write_text(json.dumps(marker,indent=2)+'\n')
            print('STOPPED between jobs before',task['name'],flush=True);break
        name=task['name'];folder=CAMPAIGN/'runs'/name
        entry={'name':name,'started_utc':datetime.now(timezone.utc).isoformat()};state.append(entry)
        status.write_text(json.dumps(state,indent=2)+'\n')
        print('START',name,flush=True)
        try:
            condition=task.get('when_failed')
            if condition:
                previous=json.loads((CAMPAIGN/'runs'/condition/'run.json').read_text())
                if previous['status']!='FAILED':
                    entry.update(solver_status='SKIPPED',reason='Baseline '+condition+' did not fail',ended_utc=datetime.now(timezone.utc).isoformat())
                    status.write_text(json.dumps(state,indent=2)+'\n');print('SKIP',name,entry['reason'],flush=True);continue
            if not folder.exists():
                cfg=dict(task);cfg.pop('phase',None);cfg.pop('when_failed',None);kind=cfg.pop('type','specimen')
                if cfg.get('source'):cfg['source']=ROOT/cfg['source']
                folder=build_elements(**cfg) if kind=='elements' else build(**cfg)
            manifest=json.loads((folder/'run.json').read_text())
            if manifest['status']=='PREPARED':manifest=execute(folder)
            if manifest['status']=='RUNNING':raise RuntimeError('Another live execution owns '+name)
            entry['solver_status']=manifest['status']
            if (folder/(name+'.odb')).exists():
                if not (folder/'energies.json').exists():extract(folder)
                if not task.get('native') and not task.get('bonded') and not (folder/'sdv.csv').exists():
                    subprocess.run([sys.executable,str(ROOT/'examples/element/parse_elprint_sdv_by_increment.py'),
                                    '--datFile',str(folder/(name+'.dat')),'--outCsv',str(folder/'sdv.csv'),'--sdvCount','80'],check=True)
                if not (folder/'assessment').exists():
                    (elements if task.get('type')=='elements' else specimen)(folder)
                entry['assessment']=str(folder/'assessment/metrics.json')
        except Exception as exc:
            entry['error']=str(exc)
            (CAMPAIGN/(name+'_queue_error.txt')).write_text(traceback.format_exc())
            print('ERROR',name,str(exc),flush=True)
        entry['ended_utc']=datetime.now(timezone.utc).isoformat()
        status.write_text(json.dumps(state,indent=2)+'\n')
        print('END',name,entry.get('solver_status'),flush=True)


if __name__=='__main__':main()
