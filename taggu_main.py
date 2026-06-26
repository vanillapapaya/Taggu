"""PyInstaller 엔트리 — 단일 프로세스로 서버 + 브라우저 띄움.

데스크톱 dev 환경의 desktop.py는 subprocess로 Python을 띄우지만,
PyInstaller bundle에선 별도 Python 인터프리터가 없으므로 모든 걸 한 프로세스에서 처리.

흐름:
  1. 단일 인스턴스 lock (포트 50815)
  2. 백그라운드 스레드에서 uvicorn 서버 실행 (인-프로세스)
  3. 메인 스레드는 서버 listen 대기 후 Edge/Chrome --app 띄움
  4. 브라우저 닫히면 종료
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional


HOST = "127.0.0.1"
PORT = int(os.environ.get("TAGGU_PORT", "8000"))
SINGLE_INSTANCE_PORT = int(os.environ.get("TAGGU_INSTANCE_PORT", "50815"))
CREATE_NO_WINDOW = 0x08000000

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]


def _exe_dir() -> Path:
    """패키지된 EXE의 실행 디렉토리 (or dev 모드면 스크립트 디렉토리)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def find_browser() -> Optional[Path]:
    for p in EDGE_PATHS + CHROME_PATHS:
        if p.exists():
            return p
    return None


def acquire_single_instance() -> Optional[socket.socket]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
        s.listen(1)
        return s
    except OSError:
        try: s.close()
        except Exception: pass
        return None


def show_message(title: str, text: str, icon: int = 0x40):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, icon | 0x1000)
    except Exception:
        pass


def wait_for_port(timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_server(use_https: bool, certfile: str, keyfile: str, db_path: str):
    """uvicorn을 in-process로 실행. 데몬 스레드에서 호출됨."""
    import uvicorn
    from app import create_app
    app = create_app(db_path, None)
    if use_https:
        uvicorn.run(app, host="0.0.0.0", port=PORT,
                    ssl_certfile=certfile, ssl_keyfile=keyfile, log_level="warning")
    else:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def main():
    # EXE 옆 디렉토리를 작업 디렉토리로 (DB, settings.json, templates 등이 여기에 위치)
    os.chdir(_exe_dir())

    # windowed(console=False) 빌드에선 sys.stdout/stderr가 None이라,
    # 서버 스레드의 print()/로그가 예외를 일으켜 부팅이 통째로 멈춘다.
    # 로그 파일로 우회 (열기 실패 시 devnull). 향후 디버깅에도 유용.
    if sys.stdout is None or sys.stderr is None:
        try:
            _log = open(_exe_dir() / "server_log.txt", "a", encoding="utf-8", buffering=1)
        except Exception:
            _log = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = _log
        if sys.stderr is None:
            sys.stderr = _log

    lock = acquire_single_instance()
    if lock is None:
        show_message("Taggu", "Taggu가 이미 실행 중입니다.\n작업 표시줄에서 윈도우를 확인하세요.")
        sys.exit(0)

    browser = find_browser()
    if browser is None:
        show_message(
            "Taggu - 오류",
            "Microsoft Edge 또는 Google Chrome을 찾을 수 없습니다.\n둘 중 하나를 설치하세요.",
            icon=0x10,
        )
        sys.exit(1)

    cert = _exe_dir() / "192.168.0.75+2.pem"
    key = _exe_dir() / "192.168.0.75+2-key.pem"
    use_https = cert.exists() and key.exists()
    db_path = str(_exe_dir() / "images.db")

    server_thread = threading.Thread(
        target=run_server,
        args=(use_https, str(cert), str(key), db_path),
        daemon=True,
    )
    server_thread.start()

    if not wait_for_port():
        show_message("Taggu - 오류", "서버가 60초 내에 응답하지 않았습니다.", icon=0x10)
        sys.exit(1)

    scheme = "https" if use_https else "http"
    url = f"{scheme}://{HOST}:{PORT}"
    browser_data = _exe_dir() / ".browser_data"
    browser_data.mkdir(exist_ok=True)

    browser_proc = subprocess.Popen(
        [
            str(browser),
            f"--app={url}",
            f"--user-data-dir={browser_data}",
            "--window-size=1400,900",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=Translate",
        ],
        creationflags=CREATE_NO_WINDOW,
    )

    try:
        browser_proc.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
