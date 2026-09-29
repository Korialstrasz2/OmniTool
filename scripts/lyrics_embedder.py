"""Migration adapter: read-only scan/sidecar preview; reviewed writes use the workspace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitool_core.lyrics_workbench import main

if __name__ == '__main__':
    main()
