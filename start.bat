@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto failed
set "PYTHONNOUSERSITE=1"
set "PIP_USER=0"
set "PYTHONUTF8=1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto environment_ready
if /i "%~1"=="--diagnose" goto no_environment

echo Creating the local virtual environment. Python 3.13 x64 is preferred.
py -3.13 -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,11) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3.13 -m venv .venv
  goto environment_ready
)
py -3 -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,11) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3 -m venv .venv
  goto environment_ready
)
python -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,11) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 (
  python -m venv .venv
  goto environment_ready
)
echo Install 64-bit Python 3.11 or newer, then run start.bat again.
goto failed

:environment_ready
if not exist "%PYTHON_EXE%" goto no_environment
"%PYTHON_EXE%" -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,11) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
  echo The existing .venv is invalid or uses unsupported Python. No files were deleted.
  echo Rename .venv and rerun with 64-bit Python 3.11 or newer installed.
  goto failed
)
if /i "%~1"=="--diagnose" (
  "%PYTHON_EXE%" -m omnitool_core.bootstrap --diagnose
  if errorlevel 1 goto failed
  goto done
)
"%PYTHON_EXE%" -m omnitool_core.bootstrap --check-core
if errorlevel 1 (
  echo Installing or repairing the required core packages only...
  "%PYTHON_EXE%" -m pip install -r requirements-core.txt
  if errorlevel 1 goto failed
  "%PYTHON_EXE%" -m omnitool_core.bootstrap --check-core
  if errorlevel 1 goto failed
)
echo Starting OmniTool. The browser opens only after the login page responds.
echo Use start.bat --diagnose for local dependency checks. Optional tool packages are not installed automatically.
if /i "%~1"=="--no-browser" (
  "%PYTHON_EXE%" app.py --no-browser
) else (
  "%PYTHON_EXE%" app.py --open-browser
)
if errorlevel 1 goto failed
goto done

:no_environment
echo No usable local .venv was found. Run start.bat normally to create it.
:failed
echo.
echo OmniTool did not start successfully. Read the error above.
if not defined OMNITOOL_NO_PAUSE pause
exit /b 1

:done
endlocal
exit /b 0
