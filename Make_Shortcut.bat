@echo off
setlocal
cd /d "%~dp0"

set "TARGET=%CD%\.venv\Scripts\pythonw.exe"
set "SCRIPT=%CD%\desktop.py"
set "LNK=%CD%\Yoink.lnk"
set "ICON=%CD%\icon.ico"

if not exist "%TARGET%" (
    echo [ERROR] Not found: %TARGET%
    echo Create the venv first.
    pause
    exit /b 1
)

if not exist "%ICON%" (
    echo Generating icon.ico...
    "%TARGET%\..\python.exe" make_icon.py
)

set "ICON_LOC=%ICON%,0"
if not exist "%ICON%" set "ICON_LOC=%TARGET%,0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$s = New-Object -ComObject WScript.Shell;" ^
    "$l = $s.CreateShortcut('%LNK%');" ^
    "$l.TargetPath = '%TARGET%';" ^
    "$l.Arguments = '\"%SCRIPT%\"';" ^
    "$l.WorkingDirectory = '%CD%';" ^
    "$l.WindowStyle = 7;" ^
    "$l.IconLocation = '%ICON_LOC%';" ^
    "$l.Description = 'Yoink desktop app';" ^
    "$l.Save();" ^
    "Write-Host 'Created:' '%LNK%'"

if errorlevel 1 (
    echo [ERROR] Shortcut creation failed.
    pause
    exit /b 1
)

echo.
echo Done. You can now:
echo   1. Double-click Yoink.lnk to launch (no console).
echo   2. Move/copy it to your Desktop or pin to taskbar.
echo   3. Right-click - Properties - Change Icon... if you want a custom icon.
echo.
pause
