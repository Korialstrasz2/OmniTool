@echo off
REM Launch OmniTool
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt
start "OmniTool" python app.py
timeout /t 3 > nul
start "" http://localhost:5000
