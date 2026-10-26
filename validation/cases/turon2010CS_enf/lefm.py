"""Source equation available; span/policy unresolved at Checkpoint1."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from lefm_reference import enf
if __name__=='__main__':
    raise SystemExit('NOT RUN: resolve Q1.3; import enf only with explicit source/approved inputs. No defaults for missing span or mixity policy.')
