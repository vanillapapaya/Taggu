"""
Yoink 웹 서버: 검색 + 폴더 인덱싱 + 이미지 서빙

사용법:
    python app.py
    python app.py --db custom.db --port 8000
    python app.py --images /path/to/images  # (선택) 시작 시 폴더 자동 등록
"""

import argparse
import socket
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
    dedupe_all_tags,
    index_folder,
    init_db,
    relocalize,
)
import make_icon
import providers

def _bundle_resource(rel: str) -> Path:
    """번들된 리소스(templates/icon/aliases) 경로. PyInstaller 시 _MEIPASS, 일반 dev 시 스크립트 디렉토리."""
    base = Path(getattr(__import__("sys"), "_MEIPASS", Path(__file__).parent))
    return base / rel


DB_DEFAULT = "images.db"
ALIASES_PATH = str(_bundle_resource("character_aliases.json"))
ICON_PATH = _bundle_resource("icon.ico")


class IndexRequest(BaseModel):
    path: str
    reindex: bool = False
    with_ai: bool = False  # AI 한국어 분석(VLM)을 함께 돌릴지. 기본은 빠른 모드 (False)


class StateRequest(BaseModel):
    hidden: Optional[bool] = None
    favorite: Optional[bool] = None


class TagsRequest(BaseModel):
    tags: list[str]


class FieldTagsRequest(BaseModel):
    field: str  # "user_tags" | "wd_chars_ko" | "tags" | "wd_general"
    tags: list[str]


class MoveTagRequest(BaseModel):
    tag: str
    source: str  # "user" | "char" | "ai"
    target: str  # "user" | "char" | "ai"


class SettingsRequest(BaseModel):
    settings: dict


class SettingsTestRequest(BaseModel):
    settings: dict


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


def _collect_local_ips() -> set[str]:
    """현재 머신이 가지고 있는 모든 IP 주소를 수집.

    같은 머신에서 LAN IP(https://192.168.x.x)로 접속해도 로컬로 인식하기 위해 사용.
    """
    ips: set[str] = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0])
    except Exception:
        pass
    return ips


_LOCAL_IPS: set[str] = _collect_local_ips()


def _is_local_request(request: Request) -> bool:
    """요청이 서버와 같은 머신에서 왔는지 판정 (URL과 무관)."""
    client = request.client
    if not client:
        return False
    return client.host in _LOCAL_IPS


def _backup_db(db_path: str, reason: str) -> Optional[str]:
    """위험한 일괄 변경 직전 DB 스냅샷. 최근 5개만 유지하여 디스크 절약.

    반환: 백업 경로 (성공) / None (실패).
    """
    try:
        src = Path(db_path)
        if not src.exists():
            return None
        backup_dir = src.parent / "db_backups"
        backup_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = backup_dir / f"{src.stem}_{ts}_{reason}.db"
        import shutil
        shutil.copy2(src, dst)
        # 오래된 백업 정리 — 최근 5개만
        existing = sorted(backup_dir.glob(f"{src.stem}_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in existing[5:]:
            try: old.unlink()
            except Exception: pass
        return str(dst)
    except Exception:
        traceback.print_exc()
        return None


def create_app(db_path: str, initial_folder: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Yoink")
    templates = Jinja2Templates(directory=str(_bundle_resource("templates")))

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

    # VLM provider — 설정에 따라 로컬 / OpenAI / Anthropic / Gemini 중 하나
    settings_path = Path(__file__).parent / "settings.json"
    vlm_state: dict = {"provider": None, "settings": providers.load_settings(settings_path)}

    def ensure_wd14():
        if wd14_state["model"] is None:
            index_state["message"] = "캐릭터 인식 모델 준비 중... (~15-20초, 첫 1회만)"
            wd14_state["model"] = wd14_tagger.load_wd14()
        return wd14_state["model"]

    def get_vlm_provider():
        """현재 설정으로 provider 인스턴스 보장. 설정 변경 시 invalidate_vlm_provider 호출."""
        if vlm_state["provider"] is None:
            vlm_state["provider"] = providers.make_provider(vlm_state["settings"])
        return vlm_state["provider"]

    def invalidate_vlm_provider():
        """GPU 모델 / API client 해제. 설정 변경 또는 모드 전환 시 호출.

        주의: 인덱싱 중에는 backfill_vlm이 로컬 변수로 provider를 들고 있어서
        실제 해제는 인덱싱 완료 후 GC 시점에 일어남. 이 함수는 vlm_state만 비움.
        """
        old = vlm_state["provider"]
        vlm_state["provider"] = None
        if old is None:
            return

        # GPU 텐서를 CPU로 옮긴 뒤 None 할당 — 빠른 해제
        if hasattr(old, "_model") and old._model is not None:
            try:
                old._model.cpu()
            except Exception:
                pass
            old._model = None
        if hasattr(old, "_processor"):
            old._processor = None
        if hasattr(old, "_client"):
            old._client = None

        try:
            import gc
            del old
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            traceback.print_exc()

    def _row_to_item(row, **extras) -> dict:
        """이미지 row → 클라이언트용 dict. 4개 검색/브라우즈 엔드포인트가 공유."""
        item = {
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
            "thumb_url": f"/thumbs/{row['id']}",
        }
        item.update(extras)
        return item

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
            need_vlm = with_ai and vlm_state["provider"] is None
            msg_parts = []
            if need_wd14:
                msg_parts.append("캐릭터 인식 모델 준비 중...")
            if need_vlm:
                msg_parts.append("AI 분석 모델 준비 중...")
            _reset_progress(folder_path, " · ".join(msg_parts))

            wd14 = ensure_wd14()

            provider = get_vlm_provider() if with_ai else None
            if provider is not None:
                provider.ensure_ready()

            index_state["status"] = "running"
            index_state["message"] = ""

            index_folder(
                folder_path,
                db_path,
                provider,
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
            request,
            "index.html",
            {
                "total": total,
                "wd_missing": wd_missing,
                "is_local": _is_local_request(request),
            },
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
            _row_to_item(row) for row in rows
        ]
        return {"results": results, "total": len(results)}

    @app.post("/api/image/{image_id}/copy")
    def copy_image_to_os_clipboard(image_id: int, request: Request):
        """이미지 파일 자체를 OS 클립보드에 복사 (탐색기에서 파일 복사한 것과 동일).

        Discord/Slack/카톡 등에 paste 시 파일 그대로 업로드 → GIF 애니메이션 유지.
        Windows 전용 (PowerShell Set-Clipboard). 같은 머신에서 온 요청만 허용.
        """
        if not _is_local_request(request):
            return JSONResponse(
                {"error": "서버 클립보드는 같은 머신에서만 사용 가능"},
                status_code=403,
            )
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

            results.append(_row_to_item(row, score=round(score + boost, 4), matched=True))

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

        results = [_row_to_item(row) for row in rows]
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

    @app.post("/api/image/{image_id}/field_tags")
    async def set_field_tags(image_id: int, req: FieldTagsRequest):
        """임의의 태그 필드(user_tags/wd_chars_ko/tags/wd_general) 갱신 + dedupe."""
        allowed = {"user_tags", "wd_chars_ko", "tags", "wd_general"}
        if req.field not in allowed:
            return JSONResponse({"error": "허용되지 않는 필드"}, status_code=400)
        seen: set[str] = set()
        cleaned: list[str] = []
        for t in req.tags:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                cleaned.append(t)
        if req.field == "user_tags":
            cleaned = _normalize_user_tags(cleaned)
            joined = ",".join(cleaned)
        else:
            joined = ", ".join(cleaned)
        conn = get_db()
        cur = conn.execute(
            f"UPDATE images SET {req.field}=? WHERE id=?", (joined, image_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            conn.close()
            return JSONResponse({"error": "이미지 없음"}, status_code=404)
        conn.close()
        return {"id": image_id, "field": req.field, "tags": cleaned}

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

    @app.post("/api/image/{image_id}/move_tag")
    async def move_tag(image_id: int, req: MoveTagRequest):
        """user/char/ai 3필드 간 단일 태그 이동 (atomic, 6방향).

        source/target ∈ {"user", "char", "ai"}.
            user → user_tags ("," 구분), char → wd_chars_ko (", "), ai → tags (", ").
        """
        field_map = {"user": "user_tags", "char": "wd_chars_ko", "ai": "tags"}
        if req.source not in field_map or req.target not in field_map:
            return JSONResponse({"error": "source/target는 user|char|ai 중 하나"}, status_code=400)
        if req.source == req.target:
            return JSONResponse({"error": "source와 target이 같음"}, status_code=400)
        tag = req.tag.strip()
        if not tag:
            return JSONResponse({"error": "빈 태그"}, status_code=400)

        src_field = field_map[req.source]
        dst_field = field_map[req.target]

        conn = get_db()
        row = conn.execute(
            "SELECT user_tags, wd_chars_ko, tags FROM images WHERE id=?", (image_id,)
        ).fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": "이미지 없음"}, status_code=404)

        fields = {
            "user_tags": [t.strip() for t in (row["user_tags"] or "").split(",") if t.strip()],
            "wd_chars_ko": [t.strip() for t in (row["wd_chars_ko"] or "").split(",") if t.strip()],
            "tags": [t.strip() for t in (row["tags"] or "").split(",") if t.strip()],
        }
        fields[src_field] = [t for t in fields[src_field] if t != tag]
        if tag not in fields[dst_field]:
            fields[dst_field].append(tag)
        if dst_field == "user_tags":
            fields["user_tags"] = _normalize_user_tags(fields["user_tags"])

        def _join(field: str) -> str:
            sep = "," if field == "user_tags" else ", "
            return sep.join(fields[field])

        conn.execute(
            f"UPDATE images SET {src_field}=?, {dst_field}=? WHERE id=?",
            (_join(src_field), _join(dst_field), image_id),
        )
        conn.commit()
        conn.close()
        return {
            "id": image_id,
            "user_tags": fields["user_tags"],
            "wd_chars_ko": fields["wd_chars_ko"],
            "tags": fields["tags"],
        }

    @app.post("/api/cleanup_library_dupes")
    async def cleanup_library_dupes():
        """경로 components에 *.library / *.eagle 등이 포함된 row 일괄 삭제.

        Eagle, Aseprite 등 이미지 관리 앱의 라이브러리 폴더는 원본 사본을 보관하므로
        기존 인덱싱에서 같은 파일이 2~3번 들어간 경우 정리. 미래 인덱싱은 scan_images에서 사전 제외.
        """
        if index_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        _backup_db(db_path, "cleanup_library")
        suffixes = (".library", ".eagle", ".aseprite-cache", ".thumbs")
        conn = get_db()
        rows = conn.execute("SELECT id, path FROM images").fetchall()
        to_delete = []
        for row in rows:
            parts = [seg.lower() for seg in Path(row["path"]).parts]
            if any(seg.endswith(suffixes) for seg in parts):
                to_delete.append(row["id"])
        if to_delete:
            conn.executemany("DELETE FROM images WHERE id=?", [(i,) for i in to_delete])
            conn.commit()
        conn.close()
        return {"status": "done", "deleted": len(to_delete), "checked": len(rows)}

    @app.post("/api/dedupe_tags")
    async def do_dedupe_tags():
        """모든 이미지의 tags / wd_chars(_ko) / wd_general / user_tags 중복 정리."""
        if index_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        _backup_db(db_path, "dedupe_tags")
        try:
            result = dedupe_all_tags(db_path)
            return {"status": "done", **result}
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    auto_tx_lock = threading.Lock()

    def run_auto_translate_chars():
        if not auto_tx_lock.acquire(blocking=False):
            return
        try:
            # 1. DB에서 모든 wd_chars 영어명 수집
            conn = get_db()
            rows = conn.execute(
                "SELECT wd_chars FROM images WHERE wd_chars IS NOT NULL AND wd_chars != ''"
            ).fetchall()
            conn.close()
            all_chars: set[str] = set()
            for row in rows:
                for t in (row["wd_chars"] or "").split(","):
                    t = t.strip().lower()
                    if t:
                        all_chars.add(t)

            # 2. 이미 매핑된 것 제외
            aliases = aliases_state["data"]
            char_map = aliases.get("characters", {})
            work_map = aliases.get("_works", {})
            # _strip_work_suffix용
            import re as _re
            suffix_re = _re.compile(r"_\([^()]+\)$")
            unmapped: list[str] = []
            for tag in all_chars:
                clean = suffix_re.sub("", tag)
                if clean in char_map or tag in char_map or tag in work_map or clean in work_map:
                    continue
                unmapped.append(tag)

            total = len(unmapped)
            _reset_progress("(자동 캐릭터 번역)", f"{total}개 미매핑 캐릭터 번역 준비 중...")
            index_state["status"] = "running"
            index_state["total"] = total

            if total == 0:
                index_state["status"] = "done"
                index_state["finished_at"] = time.time()
                index_state["message"] = "번역할 새 캐릭터 없음"
                return

            # 3. 현재 provider로 번역 (analyze가 아닌 텍스트 호출 경로)
            provider = get_vlm_provider()
            provider.ensure_ready()

            translations: dict[str, str] = {}
            batch = 40
            for i in range(0, total, batch):
                chunk = unmapped[i:i + batch]
                index_state["current"] = min(i + batch, total)
                index_state["current_file"] = f"{chunk[0]} … 외 {len(chunk)-1}개"
                try:
                    result = providers.translate_chars_to_ko(provider, chunk, batch_size=batch)
                    translations.update(result)
                except Exception as e:
                    index_state["last_error"] = str(e)
                    traceback.print_exc()

            # 4. character_aliases.json에 머지 저장 (clean key 기준)
            with open(ALIASES_PATH, "r", encoding="utf-8") as f:
                raw = __import__("json").load(f)
            chars_section = raw.get("characters", {})
            added = 0
            for en_tag, ko in translations.items():
                if not ko or ko == "null":
                    continue
                clean = suffix_re.sub("", en_tag)
                if clean not in chars_section:
                    chars_section[clean] = ko
                    added += 1
            raw["characters"] = chars_section
            with open(ALIASES_PATH, "w", encoding="utf-8") as f:
                __import__("json").dump(raw, f, ensure_ascii=False, indent=2)

            # 5. aliases 메모리 갱신 + DB 재매핑 (사용자 편집은 보존되는 새 relocalize 사용)
            aliases_state["data"] = wd14_tagger.load_aliases(ALIASES_PATH)
            _backup_db(db_path, "auto_translate")
            relocalize(db_path, aliases_state["data"], skip_unmapped=True)

            index_state["status"] = "done"
            index_state["finished_at"] = time.time()
            index_state["message"] = f"자동 번역 완료 — {added}개 캐릭터 추가, DB 재매핑 완료"
        except Exception as e:
            traceback.print_exc()
            index_state["status"] = "error"
            index_state["message"] = str(e)
            index_state["finished_at"] = time.time()
        finally:
            auto_tx_lock.release()

    @app.post("/api/auto_translate_chars")
    async def start_auto_translate():
        if index_lock.locked() or auto_tx_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        thread = threading.Thread(target=run_auto_translate_chars, daemon=True)
        thread.start()
        return {"status": "started"}

    @app.post("/api/relocalize")
    async def do_relocalize():
        if index_lock.locked():
            return JSONResponse({"error": "다른 작업이 진행 중입니다"}, status_code=409)
        _backup_db(db_path, "relocalize")
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
            results.append(_row_to_item(row, score=round(score, 4), matched=False))

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
                path = filedialog.askdirectory(title="Yoink - 인덱싱할 폴더 선택")
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
            need_vlm = vlm_state["provider"] is None
            provider = get_vlm_provider()
            label = f"(AI 분석 · {provider.name})"
            _reset_progress(label, f"{provider.name} 준비 중..." if need_vlm else "")
            provider.ensure_ready()
            index_state["status"] = "running"
            index_state["message"] = ""
            backfill_vlm(db_path, provider, progress=index_state)
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

    @app.get("/api/settings")
    async def get_settings_endpoint():
        """현재 설정 (API 키 마스킹) + 모델 카탈로그 + 감지된 VRAM."""
        return {
            "settings": providers.settings_for_display(vlm_state["settings"]),
            "local_models": providers.LOCAL_MODELS,
            "openai_models": providers.OPENAI_MODELS,
            "anthropic_models": providers.ANTHROPIC_MODELS,
            "gemini_models": providers.GEMINI_MODELS,
            "vram_gb": providers.detect_vram_gb(),
            "active": vlm_state["provider"].name if vlm_state["provider"] is not None else None,
        }

    @app.post("/api/settings")
    async def save_settings_endpoint(req: SettingsRequest):
        """설정 저장 + 즉시 적용. API 키 빈 값이면 기존 키 유지.

        인덱싱/백필이 돌고 있으면 거부 — 진행 중 모델 교체는 위험.
        """
        if index_lock.locked():
            return JSONResponse(
                {"error": "분석 작업이 진행 중입니다. 끝난 뒤 설정을 변경하세요"},
                status_code=409,
            )
        try:
            new = _merge_settings(vlm_state["settings"], req.settings)
            providers.make_provider(new)  # 유효성 검증 (생성 가능 여부만)
            providers.save_settings(settings_path, new)
            vlm_state["settings"] = new
            invalidate_vlm_provider()
            return {"status": "ok", "settings": providers.settings_for_display(new)}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/settings/test")
    async def test_settings_endpoint(req: SettingsTestRequest):
        """저장 없이 제안된 설정으로 1x1 더미 이미지 1회 호출 테스트."""
        try:
            merged = _merge_settings(vlm_state["settings"], req.settings)
            provider = providers.make_provider(merged)
            # 1x1 흰색 PNG 임시 파일
            import tempfile
            from PIL import Image as PILImage
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                PILImage.new("RGB", (32, 32), (255, 255, 255)).save(tf, "PNG")
                tmp_path = Path(tf.name)
            try:
                provider.ensure_ready()
                tags, desc = provider.analyze(tmp_path)
            finally:
                try: tmp_path.unlink()
                except Exception: pass
            return {"status": "ok", "tags": tags, "description": desc, "name": provider.name}
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=400)

    def _merge_settings(current: dict, incoming: dict) -> dict:
        """incoming을 current에 머지. API 키가 빈 문자열이면 기존 키 보존."""
        import copy
        merged = copy.deepcopy(current)
        merged["provider"] = incoming.get("provider", merged.get("provider", "local"))
        for key in ("local", "openai", "anthropic", "gemini"):
            if key in incoming:
                if key not in merged:
                    merged[key] = {}
                for k, v in incoming[key].items():
                    if k == "api_key" and not v:
                        continue  # 빈 키는 무시 (기존 보존)
                    if k.endswith("_mask"):
                        continue  # 표시용 필드는 저장 X
                    merged[key][k] = v
        return merged

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

    THUMB_DIR = Path("thumbnails")
    THUMB_SIZE = 480  # 카드 그리드는 240px 표시, retina 대비 2x

    @app.get("/thumbs/{image_id}")
    async def serve_thumbnail(image_id: int):
        conn = get_db()
        row = conn.execute("SELECT path, mtime FROM images WHERE id=?", (image_id,)).fetchone()
        conn.close()
        if not row:
            return JSONResponse({"error": "이미지 없음"}, status_code=404)

        src = Path(row["path"])
        if not src.exists():
            return JSONResponse({"error": "원본 파일 없음"}, status_code=404)

        THUMB_DIR.mkdir(exist_ok=True)
        mtime_int = int(row["mtime"] or src.stat().st_mtime)
        cache = THUMB_DIR / f"{image_id}_{mtime_int}_{THUMB_SIZE}.webp"

        if not cache.exists():
            try:
                from PIL import Image as PILImage
                img = PILImage.open(src)
                # GIF 등 애니메이션은 첫 프레임만 (정적 썸네일)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE), PILImage.LANCZOS)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(cache, "WEBP", quality=80, method=4)
            except Exception as e:
                # 썸네일 생성 실패 시 원본 fallback
                return FileResponse(str(src))

        return FileResponse(str(cache), media_type="image/webp")

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
    here = Path(__file__).parent
    default_cert = here / "192.168.0.75+2.pem"
    default_key = here / "192.168.0.75+2-key.pem"

    parser = argparse.ArgumentParser(description="Yoink 웹 서버")
    parser.add_argument("--images", help="(선택) 시작 시 자동 등록할 이미지 폴더 경로")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite DB 경로 (기본값: {DB_DEFAULT})")
    parser.add_argument("--host", default="0.0.0.0", help="호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="포트 (기본값: 8000)")
    parser.add_argument("--certfile", default=str(default_cert) if default_cert.exists() else "",
                        help="HTTPS 인증서 (PEM). 비우면 HTTP")
    parser.add_argument("--keyfile", default=str(default_key) if default_key.exists() else "",
                        help="HTTPS 개인키 (PEM). 비우면 HTTP")
    args = parser.parse_args()

    app = create_app(args.db, args.images)

    use_https = bool(args.certfile) and bool(args.keyfile) \
        and Path(args.certfile).exists() and Path(args.keyfile).exists()
    scheme = "https" if use_https else "http"
    print(f"\n서버 시작: {scheme}://{args.host}:{args.port}")
    print(f"DB: {args.db}")
    if args.images:
        print(f"초기 이미지 폴더: {args.images}")
    print("브라우저에서 폴더 경로를 입력하고 [인덱싱] 버튼을 누르세요.")

    if use_https:
        uvicorn.run(app, host=args.host, port=args.port,
                    ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
