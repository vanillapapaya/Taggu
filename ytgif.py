"""
ytgif — 유튜브 구간 GIF / 한 컷 JPEG 추출.

두 가지로 쓰인다.
  1. CLI:    python ytgif.py "주소" 10 15        (구간 → GIF)
             python ytgif.py "주소" 11           (한 컷 → JPEG)
  2. 모듈:   app.py가 generate()/tools_status()를 import 해서 사용.

외부 의존: yt-dlp(영상 구간 다운로드) + ffmpeg(인코딩). 둘 다 선택적 —
없으면 tools_status()가 알려주고, generate()는 YtgifError를 던진다.
서버에서 잡아 사용자에게 설치 안내를 띄울 수 있게 sys.exit는 쓰지 않는다.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class YtgifError(Exception):
    """짤 생성 중 발생한, 사용자에게 보여줄 만한 오류."""


# ---------------------------------------------------------------------------
# 외부 도구 탐지
# ---------------------------------------------------------------------------

def find_exe(name: str):
    """실행파일 절대경로를 찾는다. PATH → uv tool 기본 위치(~/.local/bin) 순.

    Taggu가 EXE/런처로 뜨면 서버 프로세스 PATH가 셸과 달라서 PATH만으론
    uv tool로 깐 yt-dlp를 못 찾을 수 있다. 그래서 알려진 위치도 같이 본다.
    """
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / name,
        Path.home() / ".local" / "bin" / f"{name}.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # ffmpeg는 imageio-ffmpeg(pip)가 venv 안에 동봉한 바이너리로 폴백
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).exists():
                return exe
        except Exception:
            pass
    return None


def tools_status() -> dict:
    """{'yt_dlp': bool, 'ffmpeg': bool} — UI 버튼 활성/비활성 판단용."""
    return {
        "yt_dlp": find_exe("yt-dlp") is not None,
        "ffmpeg": find_exe("ffmpeg") is not None,
    }


def _require(name: str) -> str:
    exe = find_exe(name)
    if exe is None:
        hint = (
            "yt-dlp 설치: uv tool install yt-dlp (또는 pip install -U yt-dlp)"
            if name == "yt-dlp"
            else "ffmpeg 설치: winget install ffmpeg"
        )
        raise YtgifError(f"'{name}' 를 찾을 수 없습니다. {hint}")
    return exe


# ---------------------------------------------------------------------------
# 시간 파싱
# ---------------------------------------------------------------------------

def parse_time(s: str) -> float:
    """'75', '1:15', '1:02:03' -> 초(float)."""
    parts = str(s).strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise YtgifError(f"시간 형식이 이상합니다: {s!r}  (예: 75, 1:15, 1:02:03)")
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    if sec < 0:
        raise YtgifError(f"시간이 음수입니다: {s!r}")
    return sec


# ---------------------------------------------------------------------------
# 내부 실행 헬퍼
# ---------------------------------------------------------------------------

def _run(cmd: list, what: str):
    """조용히 실행하고, 실패하면 마지막 로그와 함께 YtgifError."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=0x08000000 if os.name == "nt" else 0,  # CREATE_NO_WINDOW
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
        raise YtgifError(f"{what} 실패\n{tail}")
    return proc.stdout or ""


def _download_section(url: str, start: float, end: float, tmpdir: str) -> str:
    """필요한 구간만 받아서 임시 파일 경로를 돌려준다."""
    yt = _require("yt-dlp")
    out_tmpl = os.path.join(tmpdir, "seg.%(ext)s")
    cmd = [
        yt,
        "-f", "bv*[ext=mp4]/bv*/b",          # 영상만 (오디오 불필요)
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",          # 구간 경계를 정확히 맞춤
        "-o", out_tmpl,
        "--no-playlist",
        url,
    ]
    _run(cmd, "유튜브 구간 다운로드")
    files = glob.glob(os.path.join(tmpdir, "seg.*"))
    if not files:
        raise YtgifError("다운로드된 임시 파일을 찾지 못했습니다. 주소/시간을 확인하세요.")
    return files[0]


def _make_gif(seg: str, out: str, width: int, fps: int):
    """팔레트 2-pass 로 깔끔한 GIF 생성."""
    ff = _require("ffmpeg")
    tmpdir = os.path.dirname(seg)
    palette = os.path.join(tmpdir, "palette.png")
    vf = f"fps={fps},scale={width}:-2:flags=lanczos"
    _run([ff, "-y", "-i", seg, "-vf", f"{vf},palettegen", palette], "팔레트 생성")
    _run(
        [ff, "-y", "-i", seg, "-i", palette,
         "-lavfi", f"{vf}[x];[x][1:v]paletteuse", out],
        "GIF 인코딩",
    )


def _make_jpeg(seg: str, out: str, width: int):
    """구간 첫 프레임(=찍은 시점)을 JPEG 한 장으로 저장."""
    ff = _require("ffmpeg")
    _run(
        [ff, "-y", "-i", seg, "-frames:v", "1",
         "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "2", out],
        "JPEG 캡처",
    )


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def generate(
    url: str,
    start,
    end=None,
    out_path: str = None,
    width: int = 480,
    fps: int = 15,
) -> str:
    """유튜브에서 GIF 또는 JPEG 한 장을 만들어 out_path에 저장하고 경로 반환.

    end가 주어지면 [start, end] 구간 GIF, 없으면 start 시점 한 컷 JPEG.
    실패 시 YtgifError.
    """
    if not url or not str(url).strip():
        raise YtgifError("유튜브 주소가 비어 있습니다.")

    start_s = parse_time(start)
    if end is not None and str(end).strip() != "":
        end_s = parse_time(end)
        if end_s <= start_s:
            raise YtgifError("끝 시간이 시작 시간보다 빨라요.")
        mode = "gif"
    else:
        # 한 컷이라도 키프레임 정렬을 위해 아주 짧은 구간을 받는다.
        end_s = start_s + 0.5
        mode = "jpg"

    if out_path is None:
        out_path = "clip.gif" if mode == "gif" else "shot.jpg"
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        seg = _download_section(url, start_s, end_s, tmpdir)
        if mode == "gif":
            _make_gif(seg, out_path, width, fps)
        else:
            _make_jpeg(seg, out_path, width)

    if not os.path.exists(out_path):
        raise YtgifError("출력 파일이 생성되지 않았습니다.")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="유튜브 구간 GIF / 한 컷 JPEG 추출기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            '  python ytgif.py "https://youtu.be/xxxx" 10 15        # GIF\n'
            '  python ytgif.py "https://youtu.be/xxxx" 1:05 1:12 -o 짤.gif\n'
            '  python ytgif.py "https://youtu.be/xxxx" 32           # JPEG 한 컷\n'
            '  python ytgif.py "https://youtu.be/xxxx" 1:02:03 -o 캡처.jpg\n'
        ),
    )
    ap.add_argument("url", help="유튜브 주소")
    ap.add_argument("times", nargs="+", help="시간. 두 개(시작 끝)면 GIF, 한 개면 JPEG 한 컷")
    ap.add_argument("-o", "--out", help="저장 이름 (기본: GIF=clip.gif, JPEG=shot.jpg)")
    ap.add_argument("--width", type=int, default=480, help="가로 픽셀 (기본 480)")
    ap.add_argument("--fps", type=int, default=15, help="GIF 초당 프레임 (기본 15)")
    args = ap.parse_args()

    if len(args.times) > 2:
        print("[오류] 시간은 한 개(JPEG) 또는 두 개(GIF)만 받습니다.")
        sys.exit(1)

    start = args.times[0]
    end = args.times[1] if len(args.times) == 2 else None
    mode = "GIF" if end is not None else "JPG"
    out = args.out or ("clip.gif" if end is not None else "shot.jpg")

    try:
        print(f"[1/2] 유튜브에서 구간 받는 중... ({mode} 모드)")
        result = generate(args.url, start, end, out, args.width, args.fps)
    except YtgifError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    size = os.path.getsize(result) / 1024
    unit = f"{size:.0f} KB" if size < 1024 else f"{size/1024:.1f} MB"
    print(f"완료 -> {os.path.abspath(result)}  ({unit})")


if __name__ == "__main__":
    main()
