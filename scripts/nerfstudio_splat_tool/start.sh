#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${TOOL_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment..."
  python -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${TOOL_DIR}/requirements.txt"

python "${TOOL_DIR}/app.py"
