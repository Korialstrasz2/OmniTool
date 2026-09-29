"""Run the complete suite with a native watchdog, including shutdown.

Disable pytest's faulthandler plugin: its exception hooks cancel external
faulthandler timers. The native timer also works when a thread holds the GIL.
"""
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    faulthandler.enable(file=sys.__stderr__)
    faulthandler.dump_traceback_later(120, file=sys.__stderr__, exit=True)
    import pytest
    # Keep the watchdog alive through interpreter shutdown as well as execution.
    return int(pytest.main(['-vv', '-p', 'no:faulthandler', 'tests']))


if __name__ == '__main__':
    raise SystemExit(main())
