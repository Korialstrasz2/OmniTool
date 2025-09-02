@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Auto-run b64pdf2png.py on 'input.txt' in the same folder as this BAT.
set "SCRIPT_DIR=%~dp0"
set "INPUT=%SCRIPT_DIR%input.txt"
set "OUTDIR=%SCRIPT_DIR%out_png"
set "DPI=220"

if not exist "%INPUT%" (
  echo [error] Expected file not found: "%INPUT%"
  echo Create "input.txt" next to this .bat and paste your Base64 there.
  pause
  exit /b 1
)

set "VENV_DIR=%SCRIPT_DIR%.venv"

REM Ensure Python launcher exists
where py >nul 2>nul
if errorlevel 1 (
  echo [error] Python launcher not found. Install Python 3 from python.org, then retry.
  pause
  exit /b 1
)

if not exist "%VENV_DIR%" (
  py -3 -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --disable-pip-version-check --quiet --upgrade pip
pip install --quiet --upgrade pymupdf pillow

python "%SCRIPT_DIR%b64pdf2png.py" "%INPUT%" --out "%OUTDIR%" --dpi %DPI%
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
  echo [error] Conversion failed. Exit code: %ERR%
) else (
  echo [ok] Done. PNGs in: "%OUTDIR%"
)
pause
exit /b %ERR%
