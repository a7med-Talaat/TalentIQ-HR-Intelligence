@echo off
REM Standalone test script for Deliverable 3: Trained Model (.pkl)

if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" test_model_loading.py
) else if exist ".\venv\Scripts\python.exe" (
    ".\venv\Scripts\python.exe" test_model_loading.py
) else (
    python test_model_loading.py
)
if "%1" neq "nopause" pause
