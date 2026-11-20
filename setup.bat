@echo off
rem Creates a Windows virtual environment (Python 3.10) and installs everything.
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo The Python launcher "py" was not found. Install Python 3.10 from python.org, then run this again.
  exit /b 1
)
py -3.10 --version
if errorlevel 1 (
  echo Python 3.10 is not installed. TensorFlow 2.13 needs Python 3.8 to 3.11, and this project is tested on 3.10.
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3.10 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
if errorlevel 1 (
  echo Installation failed. Send me the last 40 lines of the output above.
  exit /b 1
)
echo.
echo Setup finished. Start the app with run.bat
