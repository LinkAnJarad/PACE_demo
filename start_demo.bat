@echo off
setlocal EnableExtensions

rem One-click launcher for the Predictive Processing Live Demo (Windows)

set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo Initializing virtual environment with uv...
    uv sync
)

if not exist "%PYTHON%" (
    echo.
    echo Missing virtual environment. Please run "uv sync" from "%PROJECT_ROOT%".
    pause
    exit /b 1
)

echo.
echo =========================================================================
echo   Starting Predictive Processing Live Demo
echo =========================================================================

rem Auto-detect installed Ollama model if available
where ollama >nul 2>&1
if not errorlevel 1 (
    for /f "skip=1 tokens=1" %%m in ('ollama list 2^>nul') do (
        if not defined OPENAI_MODEL_NAME (
            set "OPENAI_MODEL_NAME=%%m"
            echo   [Model Agnostic] Detected Ollama model: %%m
        )
    )
)

echo   Interactive Studio : http://127.0.0.1:8000
echo =========================================================================
echo.

"%PYTHON%" download_weights.py

echo Opening Web UI in default browser...
start http://127.0.0.1:8000

echo Starting Unified Demo Server on http://127.0.0.1:8000...
"%PYTHON%" -m uvicorn icub_server:app --host 127.0.0.1 --port 8000
pause
exit /b
