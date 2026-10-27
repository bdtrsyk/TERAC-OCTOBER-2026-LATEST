@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM -- Locate Python --
set "PYTHON_CMD=python"
where python >nul 2>&1
if not errorlevel 1 goto :install_deps
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :install_deps
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :install_deps
)
if exist "%PROGRAMFILES%\Python312\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python312\python.exe"
    goto :install_deps
)
if exist "%PROGRAMFILES%\Python311\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python311\python.exe"
    goto :install_deps
)

echo [ERROR] Python was not found in PATH or standard installation paths.
echo Please install Python 3.10+ from https://www.python.org/
pause
exit /b 1

:install_deps
echo [1/2] Installing dependencies from requirements.txt...
"%PYTHON_CMD%" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    "%PYTHON_CMD%" -m pip install -r requirements.txt --user --quiet --disable-pip-version-check
)

echo [2/2] Launching PyQt6 GUI...
"%PYTHON_CMD%" gui.py %*

exit /b %ERRORLEVEL%
