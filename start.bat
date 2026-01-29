@echo off
setlocal

rem Ensure Python ignores user site-packages so installations stay isolated to the
rem virtual environment and avoid permission issues with global packages.
set PYTHONNOUSERSITE=1

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

rem Ensure installations always happen inside the virtual environment.
set "PYTHONNOUSERSITE=1"
set "PIP_USER=0"

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
if errorlevel 1 (
    echo Failed to update pip. See the messages above for details.
    exit /b 1
)

echo Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. See the messages above for details.
    exit /b 1
)

start "Gaussian Splat Scene Builder" "%PYTHON_EXE%" app.py
timeout /t 3 > nul
start "" http://localhost:7860

endlocal
