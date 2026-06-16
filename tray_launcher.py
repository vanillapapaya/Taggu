"""시스템 트레이 아이콘으로 Taggu 서버 관리.

흐름:
  1. python app.py를 subprocess로 실행 (콘솔 창 없음)
  2. 시스템 트레이 아이콘 표시 — 우클릭 메뉴: [브라우저 열기] / [서버 재시작] / [종료]
  3. 서버가 exit 42(웹 [재시작] 버튼)로 종료되면 자동 재시작
  4. 다른 exit code면 트레이만 유지하고 재시작은 안 함 (메뉴에서 [재시작]으로 수동)
  5. 종료 메뉴 클릭 시: 서버 종료 + 트레이 아이콘 제거

서버 stdout/stderr는 server_log.txt에 누적 기록 (디버깅용).

VBS launcher (Taggu.vbs)에서 pythonw.exe로 호출하면 콘솔 창이 안 뜸.
"""

import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.resolve()
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
APP = ROOT / "app.py"
LOG_PATH = ROOT / "server_log.txt"
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

CREATE_NO_WINDOW = 0x08000000  # Windows: subprocess 콘솔 창 없이 실행

state: dict = {
    "proc": None,
    "log_file": None,
    "stop": False,
    "icon": None,
}


def make_icon_image() -> Image.Image:
    """64x64 트레이 아이콘 이미지 생성 (워밍 다크 + 오렌지 T)."""
    img = Image.new("RGB", (64, 64), "#322e2a")
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill="#1f1d1a", outline="#d97757", width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), "T", font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text(((64 - w) / 2 - bbox[0], (64 - h) / 2 - bbox[1] - 2), "T", fill="#d97757", font=font)
    return img


def _open_log():
    return open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def _spawn_server() -> subprocess.Popen:
    log = _open_log()
    state["log_file"] = log
    log.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} server start ===\n")
    proc = subprocess.Popen(
        [str(PYTHON), str(APP)],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
    )
    return proc


def server_loop():
    """서버 프로세스를 띄우고, exit 42면 자동 재시작."""
    while not state["stop"]:
        proc = _spawn_server()
        state["proc"] = proc
        rc = proc.wait()
        try:
            if state["log_file"]:
                state["log_file"].write(f"=== exit {rc} ===\n")
                state["log_file"].close()
        except Exception:
            pass

        if state["stop"]:
            break

        if rc == 42:
            time.sleep(1)
            continue
        else:
            # 비정상 종료: 트레이는 유지하고 재시작은 메뉴에서 수동
            _notify("서버 종료", f"exit code {rc}. 트레이 [서버 재시작]으로 다시 시작 가능.")
            # idle 상태로 대기 — proc는 None으로 두고 wait
            state["proc"] = None
            while not state["stop"] and state["proc"] is None:
                time.sleep(0.5)


def _notify(title: str, message: str):
    icon = state.get("icon")
    if icon is None:
        return
    try:
        icon.notify(message, title)
    except Exception:
        pass


def wait_for_port_then_open():
    """서버 listen 잡힐 때까지 대기 후 브라우저 열기 (최대 60초)."""
    for _ in range(120):
        try:
            with socket.create_connection((HOST, PORT), timeout=0.3):
                webbrowser.open(URL)
                return
        except OSError:
            time.sleep(0.5)


def open_browser(icon, item):
    webbrowser.open(URL)


def restart_server(icon, item):
    proc = state.get("proc")
    if proc is None:
        # idle 상태였다면 새로 띄우도록 트리거
        threading.Thread(target=_kick_restart, daemon=True).start()
        return
    if proc.poll() is not None:
        # 이미 죽음
        threading.Thread(target=_kick_restart, daemon=True).start()
        return
    # 서버에 graceful restart 요청 (exit 42)
    try:
        req = urllib.request.Request(URL + "/api/restart", method="POST")
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        # 응답 없거나 실패 → 강제 종료. server_loop이 다시 띄움.
        try:
            proc.terminate()
        except Exception:
            pass


def _kick_restart():
    """idle 상태에서 서버 다시 띄우기 (server_loop 깨우기)."""
    proc = _spawn_server()
    state["proc"] = proc


def open_log(icon, item):
    try:
        import os
        os.startfile(str(LOG_PATH))
    except Exception:
        pass


def quit_app(icon, item):
    state["stop"] = True
    proc = state.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            req = urllib.request.Request(URL + "/api/restart", method="POST")
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    icon.stop()


def main():
    threading.Thread(target=server_loop, daemon=True).start()
    threading.Thread(target=wait_for_port_then_open, daemon=True).start()

    icon = pystray.Icon(
        "taggu",
        make_icon_image(),
        "Taggu",
        menu=pystray.Menu(
            pystray.MenuItem("브라우저 열기", open_browser, default=True),
            pystray.MenuItem("서버 재시작", restart_server),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("로그 파일 열기", open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("종료", quit_app),
        ),
    )
    state["icon"] = icon
    icon.run()


if __name__ == "__main__":
    main()
