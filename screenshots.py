"""Taggu 웹 UI 자동 스크린샷 도구.

실행 중인 Taggu 서버(기본 https://localhost:8000)에 헤드리스 Edge(Playwright)로 붙어
주요 기능 화면을 순서대로 캡처한다. 각 컷에 한국어 제목/캡션을 붙이고,
docs/screenshots/ 에 PNG로 저장한 뒤 README.md의 스크린샷 섹션을 자동 갱신한다.

사전 준비:
    python -m pip install playwright     # Chromium 다운로드 불필요 (시스템 Edge 사용)
    # Taggu 서버가 떠 있어야 함 (Taggu.bat 또는 python app.py)

사용:
    python screenshots.py                       # 전체 캡처 + README 갱신
    python screenshots.py --url https://localhost:8000
    python screenshots.py --no-readme           # 캡처만, README 안 건드림

재실행하면 같은 파일을 덮어쓰고 README 섹션을 다시 만든다(마커 사이 idempotent).
"""

import argparse
import json
import sqlite3
import ssl
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

# Windows 콘솔(cp949)에서도 한국어/em-dash 로그가 깨지지 않도록
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent.resolve()
SHOT_DIR = ROOT / "docs" / "screenshots"
README = ROOT / "README.md"
MARK_START = "<!-- SCREENSHOTS:START (screenshots.py가 자동 생성 — 수동 편집 금지) -->"
MARK_END = "<!-- SCREENSHOTS:END -->"

VIEWPORT = {"width": 1440, "height": 900}


def _api(url, path):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url + path, timeout=15, context=ctx) as r:
        return json.load(r)


def _hide_excluded(substrs):
    """제외 대상 폴더(path에 substr 포함) 이미지를 임시 숨김(hidden=1). 원래 hidden=0이던 id만 반환(정확 복구용)."""
    if not substrs:
        return []
    db = ROOT / "images.db"
    con = sqlite3.connect(str(db), timeout=15)
    con.row_factory = sqlite3.Row
    fids = [r["id"] for r in con.execute("SELECT id, path FROM folders")
            if any(s in r["path"] for s in substrs)]
    ids = []
    for fid in fids:
        ids += [r["id"] for r in con.execute(
            "SELECT id FROM images WHERE folder_id=? AND hidden=0", (fid,))]
    for i in ids:
        con.execute("UPDATE images SET hidden=1 WHERE id=?", (i,))
    con.commit()
    con.close()
    return ids


def _restore_hidden(ids):
    if not ids:
        return
    con = sqlite3.connect(str(ROOT / "images.db"), timeout=15)
    con.executemany("UPDATE images SET hidden=0 WHERE id=?", [(i,) for i in ids])
    con.commit()
    con.close()


def _test_folder(url):
    """스코프 데모용 폴더 (경로에 '그림 모음' 포함) id/path 반환. 없으면 첫 폴더."""
    try:
        data = _api(url, "/api/folders")
        folders = data.get("folders", [])
        if not folders:
            return None, None
        for f in folders:
            if "그림 모음" in f["path"]:
                return f["id"], f["path"]
        return folders[0]["id"], folders[0]["path"]
    except Exception:
        return None, None


def build_shots(url, exclude=None):
    """각 컷: (파일명, 제목, 캡션, 준비 JS, 정착 대기 ms, 이미지 로딩 대기 여부)."""
    fid, fpath = _test_folder(url)
    fpath_js = (fpath or "").replace("\\", "\\\\").replace("'", "\\'")

    # 폴더 목록 컷: 제외 대상 폴더 행을 클라이언트에서 제거(이름 노출 방지)
    panel_action = "toggleFolderPanel();"
    if exclude:
        arr = json.dumps(exclude, ensure_ascii=False)
        panel_action += (
            f"setTimeout(()=>{{document.querySelectorAll('#folderList .folder-item')"
            f".forEach(el=>{{if({arr}.some(s=>el.textContent.includes(s))) el.remove();}});}},150);")

    shots = [
        ("01-gallery", "갤러리 — 전체 보기",
         "인덱싱한 이미지를 정사각 썸네일 그리드로. 세로/가로 긴 이미지도 cover 크롭으로 균일하게 표시.",
         None, 700, False),

        ("02-search", "한국어 검색",
         "검색창에 한국어를 입력하면 WD14 태그·AI 설명·캐릭터·내 태그 텍스트에 매치되는 이미지를 찾는다.",
         "document.getElementById('searchInput').value='소녀'; doSearch();", 1500, False),

        ("03-detail-modal", "이미지 상세 — 캐릭터·태그·한국어 설명",
         "카드를 열면 캐릭터(성+이름), AI 태그, Qwen이 생성한 한국어 설명, 내 태그를 한 화면에서 보고 편집한다.",
         "const i=cardData.findIndex(c=>(c.wd_chars_ko||'').trim()); if(i>=0) openModalAt(i); i;", 900, True),

        ("04-similar", "유사 이미지 검색 (CLIP)",
         "CLIP 시각 임베딩 코사인 유사도로 '이것과 비슷한 이미지'를 찾는다.",
         "const i=cardData.findIndex(c=>(c.wd_chars_ko||'').trim()); if(i>=0) findSimilar(cardData[i].id);", 1600, False),

        ("05-folder-scope", "폴더별 보기",
         "특정 폴더로 좁혀 그 폴더의 이미지만 표시. 상단 배너로 현재 스코프를 항상 표시하고 한 번에 전체로 복귀.",
         (f"setScope({fid}, '{fpath_js}');" if fid else None), 1200, False),

        ("06-filters", "필터 — 조건별 좁히기",
         "캐릭터/내 태그/AI 태그가 비어 있는 이미지만 골라내 후속 작업(태깅·정리) 대상을 빠르게 찾는다.",
         "if(document.getElementById('filterBar').classList.contains('collapsed')) toggleFilterBar(); toggleFilter('no_char');", 1200, False),

        ("07-folder-panel", "등록된 폴더 목록",
         "인덱싱한 폴더 목록. 폴더별 이미지 수·마지막 추가 시각을 보고, 경로 클릭으로 그 폴더만 보기.",
         panel_action, 700, False),

        ("08-toolbar", "사진 추가 · 관리 도구",
         "폴더 경로를 넣어 새 이미지 추가(인덱싱), 유튜브 짤 생성, 유지보수 도구 등에 접근.",
         "toggleToolbar();", 700, False),

        ("09-settings", "AI 백엔드 설정",
         "로컬 GPU(Qwen2.5-VL) 또는 OpenAI/Anthropic/Gemini API 중 선택하고 키를 검증·저장.",
         "openSettings();", 1600, False),

        ("10-youtube", "유튜브 짤 생성",
         "유튜브 URL과 구간을 넣어 GIF/JPEG 짤을 만들고 바로 인덱싱까지 연결.",
         "openYoutubeModal();", 1000, False),

        ("11-guide", "사용 안내",
         "처음 사용자를 위한 단계별 가이드.",
         "openGuide();", 900, False),
    ]
    return shots


def capture(url, headless=True, exclude=None):
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    shots = build_shots(url, exclude)
    hidden_ids = _hide_excluded(exclude)
    if hidden_ids:
        print(f"  제외 폴더 이미지 {len(hidden_ids)}장 임시 숨김")
    done = []
    try:
        _run_capture(url, headless, shots, done)
    finally:
        _restore_hidden(hidden_ids)
        if hidden_ids:
            print(f"  숨김 복구 완료 ({len(hidden_ids)}장)")
    return done


def _run_capture(url, headless, shots, done):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=headless,
                                     args=["--ignore-certificate-errors"])
        page = browser.new_page(ignore_https_errors=True, viewport=VIEWPORT,
                                device_scale_factor=2)
        # 첫 방문 가이드 모달 자동 오픈 억제 (매 로드마다 뜨는 걸 막음).
        # 가이드 컷은 openGuide()로 직접 띄우므로 영향 없음.
        page.add_init_script("try{localStorage.setItem('taggu.guideSeen','true')}catch(e){}")
        for file, title, caption, action, settle, wait_img in shots:
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_selector("#grid .card", timeout=20000)
                if action:
                    page.evaluate(action)
                if wait_img:
                    # 모달 원본 이미지 로딩 완료까지 대기
                    page.wait_for_function(
                        "() => {const m=document.getElementById('modalImg');"
                        "return m && m.complete && m.naturalWidth>0;}", timeout=15000)
                page.wait_for_timeout(settle)
                out = SHOT_DIR / f"{file}.png"
                page.screenshot(path=str(out))
                done.append((file, title, caption))
                print(f"  OK  {file}.png  — {title}")
            except Exception as e:
                print(f"  실패 {file}: {e}")
        browser.close()
    return done


def update_readme(done):
    if not README.exists():
        print("README.md 없음 — 삽입 건너뜀")
        return
    rel = "docs/screenshots"
    lines = [MARK_START, "", "## 스크린샷", ""]
    for file, title, caption in done:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel}/{file}.png)")
        lines.append("")
        lines.append(caption)
        lines.append("")
    lines.append(MARK_END)
    block = "\n".join(lines)

    text = README.read_text(encoding="utf-8")
    if MARK_START in text and MARK_END in text:
        pre = text.split(MARK_START)[0]
        post = text.split(MARK_END)[1]
        new = pre.rstrip() + "\n\n" + block + "\n" + post
    else:
        # '## 라이선스' 앞에 삽입, 없으면 문서 끝
        anchor = "\n## 라이선스"
        if anchor in text:
            i = text.index(anchor)
            new = text[:i].rstrip() + "\n\n" + block + "\n" + text[i:]
        else:
            new = text.rstrip() + "\n\n" + block + "\n"
    README.write_text(new, encoding="utf-8")
    print(f"README.md 갱신 — {len(done)}컷 삽입")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://localhost:8000")
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--show", action="store_true", help="브라우저 창 표시(디버그)")
    ap.add_argument("--exclude-folder", action="append", default=[], metavar="SUBSTR",
                    help="경로에 이 문자열이 들어간 폴더를 캡처에서 숨김(이미지 임시 hidden + 폴더 목록 행 제거). 반복 가능")
    args = ap.parse_args()

    print(f"Taggu 스크린샷 — {args.url}")
    done = capture(args.url, headless=not args.show, exclude=args.exclude_folder)
    if not done:
        print("캡처된 컷 없음 — 서버가 떠 있는지 확인")
        sys.exit(1)
    if not args.no_readme:
        update_readme(done)
    print(f"\n완료: {len(done)}컷 → {SHOT_DIR}")


if __name__ == "__main__":
    main()
