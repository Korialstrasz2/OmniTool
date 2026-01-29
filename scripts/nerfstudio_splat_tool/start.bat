@echo off
setlocal

set TOOL_DIR=%~dp0
set VENV_DIR=%TOOL_DIR%.venv

if not exist "%VENV_DIR%" (
  echo Creating virtual environment...
  python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r "%TOOL_DIR%requirements.txt"

python "%TOOL_DIR%app.py"
endlocal
