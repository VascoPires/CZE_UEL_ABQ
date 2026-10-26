"""Run the source-equation/graphical DCB check; no Abaqus."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from lefm_reference import main
if __name__=='__main__':main()
