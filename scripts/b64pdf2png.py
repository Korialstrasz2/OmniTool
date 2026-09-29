#!/usr/bin/env python3
"""Compatibility CLI: strict input, bounded rendering, NEW output directory only.
Use --inspect for a read-only preview. Existing output directories are rejected.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitool_core.conversion import main

if __name__ == '__main__':
    main()
