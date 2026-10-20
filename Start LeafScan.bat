@echo off
rem Double-click to start LeafScan (no terminal needed).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The Python environment .venv was not found in this folder.
  echo Install it once with:  python -m venv .venv  then  .venv\Scripts\python -m pip install -r requirements.txt  then  .venv\Scripts\python -m pip install -e . --no-deps
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m app
