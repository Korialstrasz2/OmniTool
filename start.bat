@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PYTHONNOUSERSITE=1"
set "PIP_USER=0"

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "HELPER=%SCRIPT_DIR%start_bat_helper.py"
set "SELF_BAT=%~f0"
set "APP_PORT=5000"

if not exist "%HELPER%" (
    echo Missing helper script: "%HELPER%"
    exit /b 1
)

set "CANDIDATES=;"

rem Load persisted candidates from this batch file.
for /f "tokens=2 delims==" %%P in ('findstr /b /c:"rem AUTO_CANDIDATE_PYTHON=" "%SELF_BAT%" 2^>nul') do (
    call :add_candidate "%%~P"
)

rem Preferred local venv first.
call :add_candidate "%SCRIPT_DIR%.venv\Scripts\python.exe"

rem Add interpreters discovered through py launcher.
call :add_py_launcher_candidate ""
call :add_py_launcher_candidate "-3"
call :add_py_launcher_candidate "-3.13"
call :add_py_launcher_candidate "-3.12"
call :add_py_launcher_candidate "-3.11"
call :add_py_launcher_candidate "-3.10"
call :add_py_launcher_candidate "-3.9"

rem Add PATH python executables.
for /f "usebackq delims=" %%P in (`where python 2^>nul`) do (
    call :add_candidate "%%~fP"
)

call :try_candidates
if not errorlevel 1 goto :opened

echo.
echo Initial candidates failed. Scanning "%SCRIPT_DIR%" and subfolders for python.exe...
for /r "%SCRIPT_DIR%" %%F in (python.exe) do (
    call :add_candidate "%%~fF"
    call :remember_candidate "%%~fF"
)

call :try_candidates
if not errorlevel 1 goto :opened

echo.
echo Failed to start OmniTool with every discovered Python environment.
exit /b 1

:opened
echo Opening browser...
start "" "http://localhost:%APP_PORT%"
exit /b 0

:add_py_launcher_candidate
set "PYFLAG=%~1"
set "PY_DISCOVERED="
for /f "usebackq delims=" %%P in (`py %PYFLAG% -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "PY_DISCOVERED=%%~fP"
)
if defined PY_DISCOVERED call :add_candidate "!PY_DISCOVERED!"
exit /b 0

:add_candidate
set "CANDIDATE=%~1"
if not defined CANDIDATE exit /b 0
if not exist "%CANDIDATE%" exit /b 0
set "CANON=%~f1"
set "NEEDLE=;!CANON!;"
if not "!CANDIDATES:%NEEDLE%=!"=="!CANDIDATES!" exit /b 0
set "CANDIDATES=!CANDIDATES!!CANON!;"
exit /b 0

:remember_candidate
set "CANDIDATE=%~f1"
if not exist "%CANDIDATE%" exit /b 0
for /f "usebackq delims=" %%M in (`"%CANDIDATE%" "%HELPER%" --remember-candidate "%CANDIDATE%" "%SELF_BAT%" 2^>nul`) do (
    rem no-op; helper prints status if needed.
)
exit /b 0

:try_candidates
set "WORKING_PYTHON="
set "TO_TRY=!CANDIDATES:;= !"
for %%P in (!TO_TRY!) do (
    if exist "%%~fP" (
        echo.
        echo Trying Python interpreter: %%~fP
        "%%~fP" "%HELPER%" --try-start --candidate "%%~fP" --script-dir "%SCRIPT_DIR%" --port %APP_PORT%
        if not errorlevel 1 (
            set "WORKING_PYTHON=%%~fP"
            echo OmniTool started with: %%~fP
            exit /b 0
        )
    )
)
exit /b 1

rem AUTO_CANDIDATE_PYTHON=C:\Path\To\Python\python.exe
