#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 or newer required"'
if ! .venv/bin/python -c 'import flask, cryptography, waitress' 2>/dev/null; then
    .venv/bin/python -m pip install -r requirements-core.txt
fi
exec .venv/bin/python app.py
