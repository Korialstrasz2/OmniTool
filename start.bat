@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PIP_USER=0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" py -3 -m venv .venv
if not exist "%PYTHON_EXE%" python -m venv .venv
if not exist "%PYTHON_EXE%" (
  echo Unable to create a virtual environment. Install Python 3.11 or newer.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or newer is required'"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -c "import flask, cryptography, waitress" >nul 2>&1
if errorlevel 1 (
  "%PYTHON_EXE%" -m pip install -r requirements-core.txt
  if errorlevel 1 exit /b 1
)
echo Starting OmniTool. Use the local access code shown below.
echo Optional tool dependencies are in requirements.txt; they are not installed automatically.
start "" http://127.0.0.1:5000
"%PYTHON_EXE%" app.py
endlocal
