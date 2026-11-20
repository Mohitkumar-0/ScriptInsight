@echo off
rem Starts ScriptInsight on http://127.0.0.1:5001
cd /d "%~dp0"
set PY=
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
if not defined PY if exist .venv313tf\Scripts\python.exe set PY=.venv313tf\Scripts\python.exe
if not defined PY if exist .venv313\Scripts\python.exe set PY=.venv313\Scripts\python.exe
if not defined PY (
  echo No environment found. Run setup.bat first.
  exit /b 1
)
echo Using %PY%
%PY% app.py
