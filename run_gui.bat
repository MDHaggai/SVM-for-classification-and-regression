@echo off
echo =======================================
echo  Heart Disease Prediction GUI - Fixed
echo =======================================
echo.
echo Starting the application...
echo.

cd /d "%~dp0"
python heart_disease_gui_fixed.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error occurred. Press any key to exit...
    pause >nul
) else (
    echo.
    echo ✅ Application closed successfully.
    timeout 3 >nul
)
