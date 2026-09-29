#!/usr/bin/env python3
"""Explicit preview/apply/undo CLI; opening the script never renames files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitool_core.rename import main

if __name__ == '__main__':
    main()
