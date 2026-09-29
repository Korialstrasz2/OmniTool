#!/usr/bin/env python3
"""Read-only folder comparison. Use the workspace for paginated reports or --help."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitool_core.file_workbench import compare_main

if __name__ == '__main__':
    compare_main()
