@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Start OmniTool with start.bat first to create its virtual environment.
  exit /b 1
)
if "%~1"=="" (
  echo Open Base64 / PDF to PNG in OmniTool for preview and confirmation.
  echo CLI: b64pdf2png.bat INPUT --out NEW_FOLDER --dpi 220 --max-pages 100
  echo Read-only preview: b64pdf2png.bat INPUT --inspect
  echo Install optional dependencies explicitly from requirements-conversion.txt.
  exit /b 2
)
"%PYTHON_EXE%" scripts\b64pdf2png.py %*
exit /b %ERRORLEVEL%
