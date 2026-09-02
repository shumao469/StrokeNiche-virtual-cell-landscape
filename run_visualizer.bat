@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app\perturbation_explorer.py
if errorlevel 1 (
  echo.
  echo The visualizer dependencies may be missing.
  echo Install them once with: python -m pip install -e ".[visualizer]"
  pause
)
