"""
MemeTracker 웹 서버: 검색 + 폴더 인덱싱 + 이미지 서빙

사용법:
    python app.py
    python app.py --db custom.db --port 8000
    python app.py --images /path/to/images  # (선택) 시작 시 폴더 자동 등록
"""

import argparse
import sqlite3
import threading
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import wd14_tagger
from index import (
    backfill_vlm,
    backfill_wd14,
    index_folder,
    init_db,
    load_vlm,
    relocalize,
)
import make_icon

DB_DEFAULT = "images.db"
ALIASES_PATH = "character_aliases.json"
ICON_PATH = Path(__file__).parent / "icon.ico"


class IndexRequest(BaseModel):
    path: str
    reindex: bool = False
    with_ai: bool = False  # AI 한국어 분석(VLM)을 함께 돌릴지. 기본은 빠른 모드 (False)


class StateRequest(BaseModel):
    hidden: Optional[bool] = None
    favorite: Optional[bool] = None


class TagsRequest(BaseModel):
    tags: list[str]


MAX_USER_TAG_LEN = 50
MAX_USER_TAGS = 30


def _normalize_user_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip().replace(",", " ")[:MAX_USER_TAG_LEN]
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= MAX_USER_TAGS:
            break
    return out


def _is_relative(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def create_app(db_path: str, initial_folder: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="MemeTracker")
    templates = Jinja2Templates(directory="templates")

    conn0 = init_db(db_path)
    if initial_folder:
        resolved = str(Path(initial_folder).resolve())
        conn0.execute("INSERT OR IGNORE INTO folders (path) VALUES (?)", (resolved,))
        conn0.commit()
    conn0.close()

    try:
        make_icon.ensure_icon()
    except Exception as e:
        print(f"아이콘 생성 실패 (무시): {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    print("CLIP 로딩 중...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print("CLIP 로딩 완료")

    aliases_state = {"data": wd14_tagger.load_aliases(ALIASES_PATH)}
    print(f"이름 사전 로딩 완료 (캐릭터 매핑 {len(aliases_state['data'].get('characters', {}))}개)")

    # WD14는 lazy load (첫 캐릭터 인식 시 ~15-20초 추가 대기, 그 후엔 메모리 유지)
    wd14_state = {"model": None}
    vlm_state = {"model": None, "processor": None}

    def ensure_wd14():
        if wd14_state["model"] is None:
            index_state["message"] = "캐릭터 인식 모델 준비 중... (~15-20초, 첫 1회만)"
            wd14_state["model"] = wd14_tagger.load_wd14()
        return wd14_state["model"]

    index_lock = threading.Lock()
    index_state: dict = {
        "status": "idle",
        "folder": "",
        "total": 0,
        "current": 0,
        "skipped": 0,
        "success": 0,
        "errors": 0,
        "current_file": "",
        "last_error": "",
        "message": "",
        "started_at": None,
        "finished_at": None,
    }

    def get_db() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def text_to_embedding(text: str) -> np.ndarray:
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32).flatten()

    def get_allowed_roots() -> list[Path]:
        conn = get_db()
        rows = conn.execute("SELECT path FROM folders").fetchall()
        conn.close()
        return [Path(r["path"]) for r in rows]

    def _reset_progress(folder_label: str, message: str):
        index_state.update({
            "status": "loading_model" if message else "running",
            "folder": folder_label,
            "started_at": time.time(),
            "finished_at": None,
            "total": 0, "current": 0,
            "skipped": 0, "success": 0, "errors": 0,
            "current_file": "", "last_error": "",
            "message": message,
        })

    def run_indexing(folder_path: str, reindex: bool, with_ai: bool):
        if not index_lock.acquire(blocking=False):
            return
        try:
            need_wd14 = wd14_state["model"] is None
            need_vlm = with_ai and vlm_state["model"] is None
            msg_parts = []
            if need_wd14:
                msg_parts.append("캐릭터 인식 모델 준비 중...")
            if need_vlm:
                msg_parts.append("AI 분석 모델 준비 중...")
            _reset_progress(folder_path, " · ".join(msg_parts))

            wd14 = ensure_wd14()

            if need_vlm:
                model, processor = load_vlm(device)
                vlm_state["model"] = model
                vlm_state["processor"] = processor

            index_state["status"] = "running"
            index_state["message"] = ""

            index_folder(
                folder_path,
                db_path,
                vlm_state["model"] if with_ai else None,
                vlm_state["processor"] if with_ai else None,
                clip_model,
                clip_preprocess,
                device,
                progress=index_state,
                reindex=reindex,
                wd14=wd14,
                aliases=aliases_state["data"],
                use_vlm=with_ai,
            )

            index_state["status"] = "done"
            index_state["finished_at"] = time.time()
            index_state["message"] = (
                f"완료 (처리 {index_state['success']}, 오류 {index_state['errors']}, "
                f"스킵 {index_state['skipped']})"
            )
        except Exception as e:
            traceback.print_exc()
            index_state["status"] = "error"
            index_state["message"] = str(e)
            index_state["finished_at"] = time.time()
        finally:
            index_lock.release()

    def run_backfill():
        if not index_lock.acquire(blocking=False):
            return
        try:
            need_wd14 = wd14_state["model"] is None
            _reset_progress(
                "(캐릭터 인식)",
                "캐릭터 인식 모델 준비 중..." if need_wd14 else "캐릭터 인식 중...",
            )
            wd14 = ensure_wd14()
            index_state["status"] = "running"
            backfill_wd14(db_path, wd14, aliases_state["data"], progress=index_state)
            index_state["status"] = "done"
            index_state["finished_at"] = time.time()
            index_state["message"] = (
                f"백필 완료 (성공 {index_state['success']}, 오류 {index_state['errors']})"
            )
        except Exception as e:
            traceback.print_exc()
            index_state["status"] = "error"
            index_state["message"] = str(e)
            index_state["finished_at"] = time.time()
        finally:
            index_lock.release()

    @app.get("/", response_class=HTMLResponse)
    async def search_page(request: Request):
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        wd_missing = conn.execute("SELECT COUNT(*) FROM images WHERE wd_chars IS NULL").fetchone()[0]
        conn.close()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "total": total, "wd_missing": wd_missing},
        )

    @app.get("/api/random")
    async def random_images(n: int = Query(5, ge=1, le=50), view: str = Query("all")):
        conn = get_db()
        where = _view_filter(view)
        rows = conn.execute(
            f"SELECT id, path, filename, tags, description, wd_chars_ko, wd_chars, hidden, favorite, user_tags "
            f"FROM images WHERE {where} ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        conn.close()
        results = [
            {
                "id": row["id"],
                "filename": row["filename"],
                "tags": row["tags"],
                "description": row["description"],
                "wd_chars_ko": row["wd_chars_ko"],
                "wd_chars": row["wd_chars"],
                "user_tags": row["user_tags"] or "",
                "hidden": bool(row["hidden"]),
                "favorite": bool(row["favorite"]),
                "image_url": f"/images/{urllib.parse.quote(row['path'])}",
            }
            for row in rows
        ]
        return {"results": results, "total": len(results)}

    @app.post("/api/image/{image_id}/copy")
    def copy_image_to_os_clipboard(image_id: int):
        """이미지 파일 자체를 OS 클립보드에 복사 (탐색기에서 파일 복사한 것과 동일).

        Discord/Slack/카톡 등에 paste 시 파일 그대로 업로드 → GIF 애니메이션 유지.
        Windows 전용 (PowerShell Set-Clipboard).
        """
        import subprocess as _sp
        conn = get_db()
        row = conn.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        conn.close()
        if not row:
            return JSONResponse({"error": "이미지 없음"}, status_code=404)
        file_path = row["path"]
        if not Path(file_path).exists():
            return JSONResponse({"error": "파일 없음"}, status_code=404)
        try:
            # PowerShell single-quoted string에서 ' 는 '' 로 escape
            safe = file_path.replace("'", "''")
            ps_cmd = f"Set-Clipboard -LiteralPath '{safe}'"
            result = _sp.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                creationflags=0x08000000,
                capture_output=True,
                timeout=10,
                text=True,
            )
            if result.returncode != 0:
                return JSONResponse(
                    {"error": f"clipboard 복사 실패: {result.stderr.strip() or 'unknown'}"},
                    status_code=500,
                )
            return {"copied": file_path}
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/open_log")
    async def open_log_file():
        log_path = (Path(__file__).parent / "server_log.txt").resolve()
        if not log_path.exists():
            return JSONResponse({"error": "로그 파일이 아직 없습니다"}, status_code=404)
        try:
            import os as _os
            _os.startfile(str(log_path))
            return {"opened": str(log_path)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/info")
    async def get_info():
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        folders = conn.execute("SELECT COUNT(*) FROM folders").fetchone()[0]
        favs = conn.execute("SELECT COUNT(*) FROM images WHERE favorite=1").fetchone()[0]
        hidden = conn.execute("SELECT COUNT(*) FROM images WHERE hidden=1").fetchone()[0]
        wd_missing = conn.execute("SELECT COUNT(*) FROM images WHERE wd_chars IS NULL").fetchone()[0]
        vlm_missing = conn.execute(
            "SELECT COUNT(*) FROM images WHERE (tags IS NULL OR tags = '') AND (description IS NULL OR description = '')"
        ).fetchone()[0]
        user_tagged = conn.execute("SELECT COUNT(*) FROM images WHERE user_tags IS NOT NULL AND user_tags != ''").fetchone()[0]
        conn.close()
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        return {
            "images": total,
            "folders": folders,
            "favorites": favs,
            "hidden": hidden,
            "user_tagged": user_tagged,
            "wd_missing": wd_missing,
            "vlm_missing": vlm_missing,
            "alias_chars": len(aliases_state["data"].get("characters", {})),
            "alias_works": len(aliases_state["data"].get("_works", {})),
            "device": cuda_name,
            "torch_version": torch.__version__,
        }

    @app.get("/favicon.ico")
    async def favicon():
        if ICON_PATH.exists():
            return FileResponse(str(ICON_PATH), media_type="image/x-icon")
        return JSONResponse({"error": "no icon"}, status_code=404)

    def _view_filter(view: str) -> str:
        """view 모드 → SQL WHERE 절 (선행 공백 + AND 또는 빈 문자열)."""
        if view == "favorite":
            return "favorite=1 AND hidden=0"
        if view == "hidden":
            return "hidden=1"
        return "hidden=0"

    @app.get("/api/search")
    async def search(
        q: str = Query(..., min_length=1, description="검색어"),
        limit: int = Query(40, ge=1, le=200, description="결과 수"),
        view: str = Query("all", description="all | favorite | hidden"),
    ):
        try:
            query_emb = text_to_embedding(q)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": f"텍스트 임베딩 실패: {e}"}, status_code=500)

        try:
            conn = get_db()
            rows = conn.execute(
                f"SELECT id, path, filename, tags, description, clip_embedding, "
                f"wd_chars, wd_chars_ko, wd_general, hidden, favorite, user_tags "
                f"FROM images WHERE {_view_filter(view)}"
            ).fetchall()
            conn.close()
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": f"DB 조회 실패: {e}"}, status_code=500)

        if not rows:
            return {"results": [], "total": 0, "query": q}

        q_lower = q.lower().strip()
        q_terms = [t for t in q_lower.split() if t]

        results = []
        for row in rows:
            try:
                emb_blob = row["clip_embedding"]
                if not emb_blob:
                    continue
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                score = float(np.dot(query_emb, emb))
            except Exception:
                continue

            user_tags_str = row["user_tags"] or ""
            user_haystack = user_tags_str.lower()
            other_haystack = " ".join(filter(None, [
                row["wd_chars_ko"], row["tags"], row["description"],
                row["wd_chars"], row["wd_general"], row["filename"],
            ])).lower()

            boost = 0.0
            if q_lower and q_lower in user_haystack:
                boost = 0.7
            elif q_lower and q_lower in other_haystack:
                boost = 0.5
            elif len(q_terms) > 1 and all((t in user_haystack) or (t in other_haystack) for t in q_terms):
                boost = 0.3

            if boost == 0:
                continue

            results.append({
                "id": row["id"],
                "filename": row["filename"],
                "tags": row["tags"],
                "description": row["description"],
                "wd_chars_ko": row["wd_chars_ko"],
                "wd_chars": row["wd_chars"],
                "user_tags": user_tags_str,
                "hidden": bool(row["hidden"]),
                "favorite": bool(row["favorite"]),
                "score": round(score + boost, 4),
                "matched": True,
                "image_url": f"/images/{urllib.parse.quote(row['path'])}",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]
        return {"results": results, "total": len(results), "scanned": len(rows), "query": q}

    @app.get("/api/browse")
    async def browse(
        offset: int = Query(0, ge=0),
        limit: int = Query(40, ge=1, le=200),
        view: str = Query("all", description="all | favorite | hidden"),
    ):
        conn = get_db()
        where = _view_filter(view)
        rows = conn.execute(
            f"SELECT id, path, filename, tags, description, wd_chars_ko, wd_chars, hidden, favorite, user_tags "
            f"FROM images WHERE {where} ORDER BY favorite DESC, indexed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) FROM images WHERE {where}").fetchone()[0]
        conn.close()

        results = [
            {
                "id": row["id"],
                "filename": row["filename"],
                "tags": row["tags"],
                "description": row["description"],
                "wd_chars_ko": row["wd_chars_ko"],
                "wd_chars": row["wd_chars"],
                "user_tags": row["user_tags"] or "",
                "hidden": bool(row["hidden"]),
                "favorite": bool(row["favorite"]),
                "image_url": f"/images/{urllib.parse.quote(row['path'])}",
            }
            for row in rows
        ]
        return {"results": results, "total": total, "offset": offset}

    @app.post("/api/image/{image_id}/state")
    async def set_image_state(image_id: int, req: StateRequest):
        if req.hidden is None and req.favorite is None:
            return JSONResponse({"error": "hidden 또는 favorite 필요"}, status_code=400)
        sets = []
        params = []
        if req.hidden is not None:
            sets.append("hidden=?")
            params.append(1 if req.hidden else 0)
        if req.favorite is not None:
            sets.append("favorite=?")
            params.append(1 if req.favorite else 0)
        params.append(image_id)
        conn = get_db()
        cur = conn.execute(f"UPDATE images SET {','.join(sets)} WHERE id=?", params)
        conn.commit()
        if cur.rowcount == 0:
            conn.close()
            return JSONResponse({"error": "이미지 없음"}, status_code=404)
        row = conn.execute(
            "SELECT id, hidden, favorite FROM images WHERE id=?", (image_id,)
        ).fetchone()
        conn.close()
        return {"id": row["id"], "hidden": bool(row["hidden"]), "favorite": bool(row["favorite"])}

    @app.post("/api/image/{image_id}/tags")
    async def set_image_tags(image_id: int, req: TagsRequest):
        cleaned = _normalize_user_tags(req.tags)
        joined = ",".join(cleaned)
        conn = get_db()
        cur = conn.execute("UPDATE images SET user_tags=? WHERE id=?", (joined, image_id))
        conn.commit()
        if cur.rowcount == 0:
            conn.close()
            return JSONResponse({"error": "이미지 없음"}, status_code=404)
        conn.close()
        return {"id": image_id, "user_tags": joined, "tags": cleaned}

    @app.post("/api/relocalize")
    async def do_relocalize():
        if index_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        try:
            aliases_state["data"] = wd14_tagger.load_aliases(ALIASES_PATH)
            result = relocalize(db_path, aliases_state["data"], skip_unmapped=True)
            return {
                "status": "done",
                "updated": result["updated"],
                "alias_count": len(aliases_state["data"].get("characters", {})),
            }
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/similar/{image_id}")
    async def find_similar(
        image_id: int,
        limit: int = Query(40, ge=1, le=200),
        view: str = Query("all"),
    ):
        conn = get_db()
        target = conn.execute(
            "SELECT clip_embedding FROM images WHERE id=?", (image_id,)
        ).fetchone()
        if not target or not target["clip_embedding"]:
            conn.close()
            return JSONResponse({"error": "이미지 또는 임베딩 없음"}, status_code=404)
        target_emb = np.frombuffer(target["clip_embedding"], dtype=np.float32)

        rows = conn.execute(
            f"SELECT id, path, filename, tags, description, clip_embedding, "
            f"wd_chars, wd_chars_ko, hidden, favorite, user_tags "
            f"FROM images WHERE {_view_filter(view)} AND id != ? AND clip_embedding IS NOT NULL",
            (image_id,),
        ).fetchall()
        conn.close()

        results = []
        for row in rows:
            try:
                emb = np.frombuffer(row["clip_embedding"], dtype=np.float32)
                score = float(np.dot(target_emb, emb))
            except Exception:
                continue
            results.append({
                "id": row["id"],
                "filename": row["filename"],
                "tags": row["tags"],
                "description": row["description"],
                "wd_chars_ko": row["wd_chars_ko"],
                "wd_chars": row["wd_chars"],
                "user_tags": row["user_tags"] or "",
                "hidden": bool(row["hidden"]),
                "favorite": bool(row["favorite"]),
                "score": round(score, 4),
                "matched": False,
                "image_url": f"/images/{urllib.parse.quote(row['path'])}",
            })

        results.sort(key=lambda x: -x["score"])
        results = results[:limit]
        return {"results": results, "total": len(results), "scanned": len(rows), "query": f"#{image_id}"}

    pick_folder_lock = threading.Lock()

    @app.post("/api/pick_folder")
    def pick_folder():
        """tkinter native 폴더 선택 다이얼로그를 띄움. (sync route → threadpool에서 실행)"""
        if not pick_folder_lock.acquire(blocking=False):
            return JSONResponse({"error": "이미 폴더 선택 다이얼로그가 떠 있습니다"}, status_code=409)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                path = filedialog.askdirectory(title="MemeTracker - 인덱싱할 폴더 선택")
            finally:
                root.destroy()
            return {"path": path or None}
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": f"폴더 선택 실패: {e}"}, status_code=500)
        finally:
            pick_folder_lock.release()

    @app.post("/api/restart")
    async def restart_server():
        """서버를 exit code 42로 종료. .bat 런처가 이를 감지하면 자동으로 재시작."""
        import os as _os

        def _exit_soon():
            time.sleep(0.15)
            _os._exit(42)

        threading.Thread(target=_exit_soon, daemon=True).start()
        return {"status": "restarting"}

    @app.get("/api/counts")
    async def view_counts():
        conn = get_db()
        all_n = conn.execute("SELECT COUNT(*) FROM images WHERE hidden=0").fetchone()[0]
        fav_n = conn.execute("SELECT COUNT(*) FROM images WHERE favorite=1 AND hidden=0").fetchone()[0]
        hid_n = conn.execute("SELECT COUNT(*) FROM images WHERE hidden=1").fetchone()[0]
        conn.close()
        return {"all": all_n, "favorite": fav_n, "hidden": hid_n}

    @app.post("/api/index")
    async def start_indexing(req: IndexRequest):
        path = req.path.strip()
        if not path:
            return JSONResponse({"error": "경로가 비어있습니다"}, status_code=400)
        p = Path(path)
        if not p.exists():
            return JSONResponse({"error": f"경로가 존재하지 않습니다: {path}"}, status_code=400)
        if not p.is_dir():
            return JSONResponse({"error": "폴더가 아닙니다"}, status_code=400)
        if index_lock.locked():
            return JSONResponse({"error": "이미 다른 작업이 진행 중입니다"}, status_code=409)

        thread = threading.Thread(
            target=run_indexing,
            args=(str(p.resolve()), req.reindex, req.with_ai),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "folder": str(p.resolve())}

    def run_backfill_vlm():
        if not index_lock.acquire(blocking=False):
            return
        try:
            need_vlm = vlm_state["model"] is None
            _reset_progress("(AI 분석)", "AI 모델 준비 중..." if need_vlm else "")
            if need_vlm:
                model, processor = load_vlm(device)
                vlm_state["model"] = model
                vlm_state["processor"] = processor
            index_state["status"] = "running"
            index_state["message"] = ""
            backfill_vlm(db_path, vlm_state["model"], vlm_state["processor"], progress=index_state)
            index_state["status"] = "done"
            index_state["finished_at"] = time.time()
            index_state["message"] = (
                f"AI 분석 완료 (성공 {index_state['success']}, 오류 {index_state['errors']})"
            )
        except Exception as e:
            traceback.print_exc()
            index_state["status"] = "error"
            index_state["message"] = str(e)
            index_state["finished_at"] = time.time()
        finally:
            index_lock.release()

    @app.post("/api/backfill_vlm")
    async def start_backfill_vlm():
        if index_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        thread = threading.Thread(target=run_backfill_vlm, daemon=True)
        thread.start()
        return {"status": "started"}

    @app.post("/api/backfill_wd14")
    async def start_backfill():
        if index_lock.locked():
            return JSONResponse({"error": "이미 다른 작업이 진행 중입니다"}, status_code=409)
        thread = threading.Thread(target=run_backfill, daemon=True)
        thread.start()
        return {"status": "started"}

    @app.get("/api/index/status")
    async def get_index_status():
        return index_state

    @app.get("/api/folders")
    async def list_folders():
        conn = get_db()
        rows = conn.execute("""
            SELECT f.id, f.path, f.added_at, f.last_indexed_at,
                   COUNT(i.id) AS image_count,
                   SUM(CASE WHEN i.wd_chars IS NULL THEN 1 ELSE 0 END) AS wd_missing
            FROM folders f
            LEFT JOIN images i ON i.folder_id = f.id
            GROUP BY f.id, f.path, f.added_at, f.last_indexed_at
            ORDER BY f.added_at DESC
        """).fetchall()
        wd_total_missing = conn.execute(
            "SELECT COUNT(*) FROM images WHERE wd_chars IS NULL"
        ).fetchone()[0]
        conn.close()
        return {
            "folders": [
                {
                    "id": r["id"],
                    "path": r["path"],
                    "added_at": r["added_at"],
                    "last_indexed_at": r["last_indexed_at"],
                    "image_count": r["image_count"],
                    "wd_missing": r["wd_missing"] or 0,
                }
                for r in rows
            ],
            "wd_total_missing": wd_total_missing,
        }

    @app.get("/images/{file_path:path}")
    async def serve_image(file_path: str):
        decoded = urllib.parse.unquote(file_path)
        try:
            resolved = Path(decoded).resolve()
        except Exception:
            return JSONResponse({"error": "잘못된 경로"}, status_code=400)

        allowed = get_allowed_roots()
        if not any(_is_relative(resolved, root) for root in allowed):
            return JSONResponse({"error": "접근 불가"}, status_code=403)

        if not resolved.exists():
            return JSONResponse({"error": "파일 없음"}, status_code=404)

        suffix = resolved.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(str(resolved), media_type=media_type)

    return app


def main():
    parser = argparse.ArgumentParser(description="MemeTracker 웹 서버")
    parser.add_argument("--images", help="(선택) 시작 시 자동 등록할 이미지 폴더 경로")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite DB 경로 (기본값: {DB_DEFAULT})")
    parser.add_argument("--host", default="0.0.0.0", help="호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="포트 (기본값: 8000)")
    args = parser.parse_args()

    app = create_app(args.db, args.images)
    print(f"\n서버 시작: http://{args.host}:{args.port}")
    print(f"DB: {args.db}")
    if args.images:
        print(f"초기 이미지 폴더: {args.images}")
    print("브라우저에서 폴더 경로를 입력하고 [인덱싱] 버튼을 누르세요.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
