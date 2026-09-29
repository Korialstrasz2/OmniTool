"""Run the complete suite with an independent watchdog, including shutdown.

Pytest's exception hooks can cancel faulthandler timers. A separate daemon
thread preserves the CI deadline even after a test failure or during atexit.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
from pathlib import Path


def expired() -> None:
    try:
        print('CI watchdog: the suite or interpreter shutdown exceeded 120 seconds.', file=sys.__stderr__, flush=True)
        faulthandler.dump_traceback(file=sys.__stderr__, all_threads=True)
    finally:
        os._exit(124)  # Do not deadlock again in worker/atexit cleanup.


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    timer = threading.Timer(120, expired)
    timer.daemon = True
    timer.start()
    import pytest
    # Keep the watchdog alive through interpreter shutdown as well as test execution.
    return int(pytest.main(['-vv', 'tests']))


if __name__ == '__main__':
    raise SystemExit(main())
