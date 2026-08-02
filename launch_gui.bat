@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Optional: set RVC_ROOT before launch, or configure it in Settings.
REM set RVC_ROOT=D:\path\to\your\RVC_install

set "HERE=%cd%"
set "PYW="
set "PYC="

if defined RVC_ROOT (
  if exist "%RVC_ROOT%\runtime\pythonw.exe" set "PYW=%RVC_ROOT%\runtime\pythonw.exe"
  if exist "%RVC_ROOT%\runtime\python.exe" set "PYC=%RVC_ROOT%\runtime\python.exe"
)

if not defined PYC (
  for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYC=%%i"
    goto :found_pyc
  )
)
:found_pyc

if not defined PYW (
  for /f "delims=" %%i in ('where pythonw 2^>nul') do (
    set "PYW=%%i"
    goto :found_pyw
  )
)
:found_pyw

if not defined PYC if defined PYW set "PYC=%PYW%"
if not defined PYW if defined PYC set "PYW=%PYC%"

if not defined PYC (
  echo [ERROR] No Python found.
  echo Set RVC_ROOT to your RVC install ^(folder with runtime\python.exe^),
  echo or ensure python is on PATH.
  pause
  exit /b 1
)

echo Using: %PYC%
echo Preflight check...
"%PYC%" "%HERE%\preflight.py"
if errorlevel 1 (
  echo.
  echo [ERROR] GUI failed to import. See traceback above.
  pause
  exit /b 1
)

echo Starting GUI...
start "" "%PYW%" "%HERE%\gui.py"
exit /b 0
