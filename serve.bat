@echo off
rem Runs ScriptInsight with the waitress production server on http://127.0.0.1:5001
rem (the Flask development server used by run.bat is for local testing only).
cd /d "%~dp0"
set PY=
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
if not defined PY if exist .venv313tf\Scripts\python.exe set PY=.venv313tf\Scripts\python.exe
if not defined PY if exist .venv313\Scripts\python.exe set PY=.venv313\Scripts\python.exe
if not defined PY (
  echo No environment found. Run setup.bat first.
  exit /b 1
)
%PY% -c "import waitress" 2>nul
if errorlevel 1 (
  echo Installing waitress...
  %PY% -m pip install waitress==3.0.2
)
echo Using %PY%
%PY% -m waitress --host=127.0.0.1 --port=5001 --call app:create_app
