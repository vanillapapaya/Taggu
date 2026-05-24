"""데스크톱 launcher — 시스템 브라우저(Edge/Chrome)의 --app 모드.

PyWebView 같은 webview 래퍼 없이 시스템에 이미 설치된 Edge(또는 Chrome)를
별도 user-data-dir + --app 모드로 띄움. 결과: 진짜 데스크톱 앱처럼 보이는
독립 윈도우 (URL 막대 없음, 탭 없음, 브라우저 즐겨찾기와도 격리).

흐름:
  1. 단일 인스턴스 lock (포트 50815)
  2. 백그라운드 스레드에서 uvicorn 서버를 subprocess로 실행 (콘솔 없이)
     - 서버가 exit 42로 종료하면 (헤더 [↻] 재시작) 자동 재실행
     - 그 외 exit 코드면 loop 종료
  3. 메인 스레드는 서버 listen 대기 후 Edge/Chrome --app 띄움
  4. 브라우저 닫히면 서버 thread 종료 신호 + cleanup
"""

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Optional

ROOT = Path(__file__).parent.resolve()
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
APP = ROOT / "app.py"
LOG_PATH = ROOT / "server_log.txt"
BROWSER_DATA = ROOT / ".browser_data"
HOST = "127.0.0.1"
PORT = 8000
# app.py 가 인증서 페어를 발견하면 자동으로 HTTPS 로 뜨므로 URL 스킴도 맞춰준다.
_CERT = ROOT / "192.168.0.75+2.pem"
_KEY = ROOT / "192.168.0.75+2-key.pem"
_SCHEME = "https" if _CERT.exists() and _KEY.exists() else "http"
URL = f"{_SCHEME}://{HOST}:{PORT}"
SINGLE_INSTANCE_PORT = 50815

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

state: dict = {"proc": None, "stop": threading.Event()}


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
        try:
            s.close()
        except Exception:
            pass
        return None


def show_message(title: str, text: str, icon: int = 0x40):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, icon | 0x1000)
    except Exception:
        pass


def wait_for_port(timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def server_loop(log: IO):
    """서버 subprocess를 실행하고 exit code 42(헤더 [↻] 재시작)이면 자동 재실행."""
    while not state["stop"].is_set():
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} server spawn ===\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(APP)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
        )
        state["proc"] = proc
        rc = proc.wait()
        log.write(f"=== server exited rc={rc} ===\n")
        if state["stop"].is_set():
            break
        if rc == 42:
            # 재시작 요청
            continue
        # 의도치 않은 종료 — 한 번만 재시도 후 멈춤
        log.write("=== unexpected exit, breaking loop ===\n")
        break


def main():
    lock = acquire_single_instance()
    if lock is None:
        show_message(
            "Yoink",
            "Yoink가 이미 실행 중입니다.\n작업 표시줄에서 윈도우를 확인하세요.",
        )
        sys.exit(0)

    browser = find_browser()
    if browser is None:
        show_message(
            "Yoink - 오류",
            "Microsoft Edge 또는 Google Chrome을 찾을 수 없습니다.\n둘 중 하나를 설치하세요.",
            icon=0x10,
        )
        sys.exit(1)

    BROWSER_DATA.mkdir(exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} desktop start (browser={browser.name}) ===\n")

    server_thread = threading.Thread(target=server_loop, args=(log,), daemon=True)
    server_thread.start()

    if not wait_for_port():
        state["stop"].set()
        log.write("Server failed to start within 120s\n")
        log.close()
        proc = state.get("proc")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        show_message(
            "Yoink - 오류",
            f"서버가 120초 내에 응답하지 않았습니다.\n로그: {LOG_PATH}",
            icon=0x10,
        )
        sys.exit(1)

    browser_proc = subprocess.Popen(
        [
            str(browser),
            f"--app={URL}",
            f"--user-data-dir={BROWSER_DATA}",
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
    finally:
        state["stop"].set()
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} browser closed, stopping server ===\n")
        proc = state.get("proc")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        log.close()


if __name__ == "__main__":
    main()
