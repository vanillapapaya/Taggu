@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found: %CD%\.venv
    echo.
    echo Create it first, then re-run this launcher:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON%" -c "import importlib.util as u, sys; missing=[m for m in ['fastapi','uvicorn','torch','open_clip','transformers','qwen_vl_utils','PIL','numpy','onnxruntime'] if u.find_spec(m) is None]; sys.exit(0 if not missing else (print('MISSING:',','.join(missing)) or 1))"
if errorlevel 1 (
    echo.
    echo [ERROR] Missing dependencies. Install with:
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM HTTPS 인증서가 있으면 https://, 없으면 http:// 로 열기
set "SCHEME=http"
if exist "192.168.0.75+2.pem" if exist "192.168.0.75+2-key.pem" set "SCHEME=https"

REM Open browser once port 8000 is listening (only on first launch).
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i=0; $i -lt 240; $i++) { try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',8000); $c.Close(); break } catch { Start-Sleep -Milliseconds 500 } }; Start-Process '%SCHEME%://localhost:8000'"

echo ============================================================
echo   Yoink server launcher
echo   - Browser opens automatically once the server is ready.
echo   - Web UI [Restart] button restarts the server (exit 42).
echo   - To fully stop: press Ctrl+C and answer Y, or close window.
echo ============================================================
echo.

:loop
"%PYTHON%" app.py
set "EC=%ERRORLEVEL%"
if "%EC%"=="42" (
    echo.
    echo [Restart] Server requested restart, relaunching in 1s...
    timeout /t 1 /nobreak >nul
    goto loop
)

echo.
echo Server stopped (exit code %EC%).
pause
