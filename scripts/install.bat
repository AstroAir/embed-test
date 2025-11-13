@echo off
REM VectorFlow Installation Script for Windows

setlocal enabledelayedexpansion

echo VectorFlow Installation Script
echo ==============================
echo.

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check if pip is installed
echo [INFO] Checking pip installation...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Please install pip
    pause
    exit /b 1
)
echo [SUCCESS] pip found

REM Install UV if possible
echo [INFO] Checking for UV package manager...
uv --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] UV found, using for faster installation
    set USE_UV=true
) else (
    echo [INFO] UV not found, installing...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    uv --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] UV installed successfully
        set USE_UV=true
    ) else (
        echo [WARNING] UV installation failed, falling back to pip
        set USE_UV=false
    )
)

REM Install VectorFlow
echo [INFO] Installing VectorFlow...
if "!USE_UV!"=="true" (
    uv pip install vectorflow
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed with UV, trying pip...
        pip install vectorflow
        if %errorlevel% neq 0 (
            echo [ERROR] Installation failed
            pause
            exit /b 1
        )
    )
) else (
    pip install vectorflow
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed
        pause
        exit /b 1
    )
)
echo [SUCCESS] VectorFlow installed successfully

REM Verify installation
echo [INFO] Verifying installation...
vectorflow --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] CLI command not found
    pause
    exit /b 1
)
echo [SUCCESS] CLI command available

python -c "import vectorflow; print(f'Version: {vectorflow.__version__}')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python package import failed
    pause
    exit /b 1
)
echo [SUCCESS] Python package import successful

echo.
echo [SUCCESS] Installation completed successfully!
echo.
echo Next steps:
echo   1. Run 'vectorflow --help' to see available commands
echo   2. Check the documentation at: https://your-username.github.io/vectorflow/
echo   3. Try the quick start guide: https://your-username.github.io/vectorflow/getting-started/quickstart/
echo.
pause
