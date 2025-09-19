@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment...
    py -3 -m venv "%VENV_DIR%" >nul 2>&1
)

if not exist "%PYTHON_EXE%" (
    python -m venv "%VENV_DIR%"
)

if not exist "%PYTHON_EXE%" (
    echo Failed to create virtual environment. Ensure that Python 3 is installed and available in PATH.
    exit /b 1
)

echo Updating pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt

start "OmniTool" "%PYTHON_EXE%" app.py
timeout /t 3 > nul
start "" http://localhost:5000

endlocal
