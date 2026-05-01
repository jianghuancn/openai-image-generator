@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

cd /d "%PROJECT_DIR%"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=py -3"
)

%PYTHON% "%PROJECT_DIR%\generate_image.py" %*
