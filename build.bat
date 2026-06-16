@echo off
REM Taggu PyInstaller 빌드 스크립트
REM 사전 요건: .venv-build (CPU torch + pyinstaller + 모든 런타임 의존성)
REM 결과: dist\Taggu\Taggu.exe + 사이드 파일

setlocal
cd /d "%~dp0"

if not exist ".venv-build\Scripts\pyinstaller.exe" (
    echo [ERROR] .venv-build가 없습니다. 먼저 빌드 환경을 만드세요:
    echo   uv venv --python 3.13 .venv-build
    echo   VIRTUAL_ENV=.venv-build uv pip install --index-url https://download.pytorch.org/whl/cpu torch
    echo   VIRTUAL_ENV=.venv-build uv pip install -r requirements.txt pyinstaller
    pause
    exit /b 1
)

echo === 이전 빌드 정리 중 ===
if exist "build" rmdir /s /q build
if exist "dist\Taggu" rmdir /s /q dist\Taggu

echo === PyInstaller 실행 중 (10~20분 소요) ===
".venv-build\Scripts\pyinstaller.exe" taggu.spec --noconfirm

if errorlevel 1 (
    echo [ERROR] 빌드 실패
    pause
    exit /b 1
)

echo.
echo === 빌드 완료 ===
echo 결과: dist\Taggu\Taggu.exe
echo 배포: dist\Taggu 폴더 전체를 zip으로 묶어 배포
echo.

REM 빌드 크기 표시
for /f "tokens=3" %%a in ('dir /s /-c dist\Taggu ^| findstr "File(s)"') do set SIZE=%%a
echo 총 크기: %SIZE% 바이트

pause
