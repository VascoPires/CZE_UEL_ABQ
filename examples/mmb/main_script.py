"""Generate a native cohesive input and convert it to U8."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from build_model import main

if __name__ == '__main__':
    main('MMB')
