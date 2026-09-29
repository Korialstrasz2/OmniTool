#!/usr/bin/env python3
"""Compatibility entrypoint. The immediate-on-selection Tkinter renamer is removed.
Open Dual File Renamer in OmniTool for the staged UI, or use --help for the CLI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitool_core.file_workbench import dual_main

if __name__ == '__main__':
    dual_main()
