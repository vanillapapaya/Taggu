"""
이미지 인덱서: 폴더 스캔 → VLM 태깅 → CLIP 임베딩 → WD14 캐릭터 태깅 → SQLite 저장

CLI 사용법:
    python index.py /path/to/images
    python index.py /path/to/images --db custom.db
    python index.py /path/to/images --reindex  # 기존 인덱스 무시하고 재인덱싱

함수 인터페이스 (app.py에서 import):
    init_db(db_path) -> Connection
    load_clip(device) -> (model, preprocess)
    index_folder(folder, db_path, vlm_provider, clip_model, clip_preprocess,
                 device, progress=None, reindex=False, on_item=None, wd14=None, aliases=None) -> dict
    backfill_vlm(db_path, vlm_provider, progress=None, on_item=None) -> dict
    backfill_wd14(db_path, wd14, aliases, progress=None, on_item=None) -> dict

VLM 분석은 providers.py의 VLMProvider 추상화 사용 — 로컬 Qwen / OpenAI / Anthropic / Gemini 지원.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open_clip
import torch
from PIL import Image

import wd14_tagger
import ccip_tagger
import providers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DB_DEFAULT = "images.db"

WD14_GENERAL_TOPK = 30


_MIGRATIONS: list[tuple[str, str]] = [
    # (column_name, column_type) — 신규 컬럼은 여기에 append만 하면 됨
    ("mtime", "REAL"),
    ("folder_id", "INTEGER"),
    ("wd_chars", "TEXT"),
    ("wd_chars_ko", "TEXT"),
    ("wd_general", "TEXT"),
    ("hidden", "INTEGER DEFAULT 0"),
    ("favorite", "INTEGER DEFAULT 0"),
    ("user_tags", "TEXT"),
    ("ccip_embedding", "BLOB"),  # 캐릭터 동일성 임베딩 (CCIP 768d float32)
]


def init_db(db_path: str) -> sqlite3.Connection:
    """SQLite DB 초기화 + 마이그레이션. WAL 모드, 마이그레이션은 단일 트랜잭션."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # PRAGMA는 트랜잭션 밖에서
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                filename TEXT,
                tags TEXT,
                description TEXT,
                clip_embedding BLOB,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_indexed_at TIMESTAMP
            )
        """)
        # 캐릭터별 prototype (centroid) — 사용자 라벨 누적으로 자동 학습
        conn.execute("""
            CREATE TABLE IF NOT EXISTS char_prototypes (
                name TEXT PRIMARY KEY,
                centroid BLOB NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
        for col, col_type in _MIGRATIONS:
            if col not in cols:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} {col_type}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return conn


def get_indexed_paths(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT path FROM images").fetchall()
    return {row[0] for row in rows}


# 이미지 관리 앱이 원본 사본을 보관하는 라이브러리 폴더 — 스캔 제외
# (path component 끝이 이 중 하나로 끝나면 그 안쪽은 모두 스킵)
EXCLUDED_DIR_SUFFIXES = (".library", ".eagle", ".aseprite-cache", ".thumbs")


def is_library_path(p: Path) -> bool:
    """경로 components 중 하나라도 라이브러리 디렉토리이면 True."""
    parts = [seg.lower() for seg in p.parts]
    return any(seg.endswith(EXCLUDED_DIR_SUFFIXES) for seg in parts)


def scan_images(root: str) -> list[Path]:
    """폴더 재귀 스캔하여 이미지 파일 목록 반환. 라이브러리 폴더(*.library 등)는 제외."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {root}")

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(root_path.rglob(f"*{ext}"))
        images.extend(root_path.rglob(f"*{ext.upper()}"))
    seen = set()
    unique = []
    for p in images:
        if is_library_path(p):
            continue
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return sorted(unique)


def load_clip(device: str):
    """CLIP ViT-B/32 로드."""
    print("CLIP 로딩 중: ViT-B-32")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    print("CLIP 로딩 완료")
    return model, preprocess


def _dedupe_csv(s: str) -> str:
    """Comma-separated 태그 문자열 중복 제거 (순서 유지)."""
    if not s:
        return ""
    seen = set()
    out = []
    for t in s.split(","):
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def dedupe_all_tags(db_path: str) -> dict:
    """모든 이미지의 tags / wd_chars / wd_chars_ko / wd_general / user_tags에서 중복 제거."""
    columns = ["tags", "wd_chars", "wd_chars_ko", "wd_general", "user_tags"]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT id, {', '.join(columns)} FROM images").fetchall()
    changed = 0
    for row in rows:
        updates = {}
        for col in columns:
            original = row[col] or ""
            cleaned = _dedupe_csv(original)
            # user_tags 형식은 콤마-공백 없이 콤마만이라 별도 처리
            if col == "user_tags":
                cleaned = ",".join([t.strip() for t in cleaned.split(",") if t.strip()])
            if cleaned != original:
                updates[col] = cleaned
        if updates:
            cols_sql = ", ".join(f"{c}=?" for c in updates)
            conn.execute(
                f"UPDATE images SET {cols_sql} WHERE id=?",
                (*updates.values(), row["id"]),
            )
            changed += 1
    conn.commit()
    conn.close()
    return {"checked": len(rows), "changed": changed}


def generate_embedding(
    image_path: Path,
    clip_model,
    clip_preprocess,
    device: str,
    pil: Optional[Image.Image] = None,
) -> np.ndarray:
    """CLIP 이미지 임베딩 생성 (512d float32). pil 주어지면 디스크 디코드 스킵."""
    if pil is not None:
        image = pil if pil.mode == "RGB" else pil.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype(np.float32).flatten()


def _wd14_for_image(
    image_path: Path,
    wd14: Optional[dict],
    aliases: Optional[dict],
    pil: Optional[Image.Image] = None,
) -> tuple[str, str, str]:
    """이미지 한 장에 대한 WD14 결과 (chars_en_str, chars_ko_str, general_en_str)."""
    if wd14 is None:
        return "", "", ""
    chars, works, general = wd14_tagger.tag_image(image_path, wd14, pil=pil)
    all_chars = chars + works
    chars_en_str = ",".join(all_chars)
    if aliases is not None:
        # skip_unmapped=False: 별칭 있으면 한국어, 없으면 romaji로라도 캐릭터 태그에 표시
        # (인식된 캐릭터가 별칭 없다고 사라지지 않게 — 일관성)
        ko = wd14_tagger.localize_tags(all_chars, aliases, skip_unmapped=False)
    else:
        ko = []
    chars_ko_str = ",".join(ko)
    general_en_str = ",".join(general[:WD14_GENERAL_TOPK])
    return chars_en_str, chars_ko_str, general_en_str


def backfill_vlm(
    db_path: str,
    provider,
    progress: Optional[dict] = None,
    on_item: Optional[Callable[[int, int, str, str, str], None]] = None,
) -> dict:
    """AI 한국어 태그/설명이 비어있는 모든 이미지에 VLM 분석만 추가 (CLIP/WD14는 건드리지 않음).

    provider는 providers.VLMProvider — analyze(path) → (tags_csv, description) 반환.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, path, filename FROM images "
        "WHERE (tags IS NULL OR tags = '') AND (description IS NULL OR description = '')"
    ).fetchall()

    if progress is not None:
        progress["total"] = len(rows)
        progress["current"] = 0
        progress["success"] = 0
        progress["errors"] = 0
        progress["skipped"] = 0
        progress["last_error"] = ""
        progress["current_file"] = ""

    for i, row in enumerate(rows, 1):
        path = Path(row["path"])
        if progress is not None:
            progress["current"] = i
            progress["current_file"] = row["filename"] or path.name

        if not path.exists():
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{path.name}: 파일 없음"
            if on_item is not None:
                on_item(i, len(rows), path.name, "err", "파일 없음")
            continue

        try:
            tags, description = providers.with_retry(lambda: provider.analyze(path))
            conn.execute(
                "UPDATE images SET tags=?, description=? WHERE id=?",
                (tags, description, row["id"]),
            )
            conn.commit()
            if progress is not None:
                progress["success"] += 1
            if on_item is not None:
                on_item(i, len(rows), path.name, "ok", tags)
        except Exception as e:
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{path.name}: {e}"
            if on_item is not None:
                on_item(i, len(rows), path.name, "err", str(e))

    conn.close()
    return {
        "processed": len(rows),
        "success": progress["success"] if progress is not None else 0,
        "errors": progress["errors"] if progress is not None else 0,
    }


def relocalize(
    db_path: str,
    aliases: dict,
    skip_unmapped: bool = True,
) -> dict:
    """wd_chars(영어) → wd_chars_ko(한국어) 재생성. 이미지 재처리 없이 빠름.

    중요: 사용자가 수동으로 추가/편집한 wd_chars_ko 항목은 보존.
    새 alias 매핑 결과에 없는 기존 한국어 항목은 사용자가 직접 단 것으로 간주 → 합쳐서 저장.
    매핑 사전(character_aliases.json) 변경 후 호출.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, wd_chars, wd_chars_ko FROM images WHERE wd_chars IS NOT NULL OR wd_chars_ko IS NOT NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        chars_en = [t.strip() for t in (row["wd_chars"] or "").split(",") if t.strip()]
        new_from_alias = wd14_tagger.localize_tags(chars_en, aliases, skip_unmapped=skip_unmapped) if chars_en else []
        # alias로 만들어지지 않는 기존 한국어 항목 = 사용자 수동 편집 → 보존
        old_ko = [t.strip() for t in (row["wd_chars_ko"] or "").split(",") if t.strip()]
        preserved = [k for k in old_ko if k not in new_from_alias]
        # 새 매핑 + 사용자 편집 보존 (순서 유지 + 중복 제거)
        merged = list(dict.fromkeys(new_from_alias + preserved))
        conn.execute(
            "UPDATE images SET wd_chars_ko=? WHERE id=?",
            (",".join(merged), row["id"]),
        )
        updated += 1
    conn.commit()
    conn.close()
    return {"updated": updated}


def _register_folder(conn: sqlite3.Connection, folder_path: str) -> tuple[int, str]:
    """folders 테이블에 등록하고 (id, resolved_path) 반환."""
    resolved = str(Path(folder_path).resolve())
    conn.execute("INSERT OR IGNORE INTO folders (path) VALUES (?)", (resolved,))
    conn.commit()
    row = conn.execute("SELECT id FROM folders WHERE path = ?", (resolved,)).fetchone()
    return row["id"], resolved


def index_folder(
    folder_path: str,
    db_path: str,
    vlm_provider,
    clip_model,
    clip_preprocess,
    device: str,
    progress: Optional[dict] = None,
    reindex: bool = False,
    on_item: Optional[Callable[[int, int, str, str, str], None]] = None,
    wd14: Optional[dict] = None,
    aliases: Optional[dict] = None,
    use_vlm: bool = True,
    ccip: Optional[dict] = None,
) -> dict:
    """폴더 incremental 인덱싱.

    - mtime 비교로 변경/추가된 파일만 처리
    - reindex=True면 전부 다시 처리
    - wd14가 주어지면 캐릭터/작품 태그도 함께 저장 (한국어 매핑 포함)
    - use_vlm=False면 AI 한국어 태그/설명 생성 건너뜀 (빠른 모드, 이미지당 ~0.5초)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    folder_id, resolved_folder = _register_folder(conn, folder_path)

    images = scan_images(folder_path)

    todo = []
    skipped_count = 0
    for p in images:
        resolved = str(p.resolve())
        try:
            current_mtime = p.stat().st_mtime
        except OSError:
            continue

        if reindex:
            todo.append((p, resolved, current_mtime))
            continue

        row = conn.execute("SELECT mtime FROM images WHERE path = ?", (resolved,)).fetchone()
        if row is None:
            todo.append((p, resolved, current_mtime))
        elif row["mtime"] is None:
            skipped_count += 1
        elif abs(current_mtime - row["mtime"]) > 0.01:
            todo.append((p, resolved, current_mtime))
        else:
            skipped_count += 1

    if progress is not None:
        progress["total"] = len(todo)
        progress["current"] = 0
        progress["skipped"] = skipped_count
        progress["success"] = 0
        progress["errors"] = 0
        progress["last_error"] = ""
        progress["current_file"] = ""

    for i, (image_path, resolved, mtime) in enumerate(todo, 1):
        if progress is not None:
            progress["current"] = i
            progress["current_file"] = image_path.name
        try:
            if use_vlm and vlm_provider is not None:
                tags, description = providers.with_retry(lambda: vlm_provider.analyze(image_path))
            else:
                tags, description = "", ""
            # 디스크 디코드 1회 — 3개 모델(CLIP/WD14/CCIP)에 공유
            try:
                pil_shared = Image.open(image_path)
                pil_shared.load()  # 실제 디코드 강제 (lazy 디코드 회피)
            except Exception:
                pil_shared = None
            embedding = generate_embedding(image_path, clip_model, clip_preprocess, device, pil=pil_shared)
            wd_chars_str, wd_chars_ko_str, wd_general_str = _wd14_for_image(image_path, wd14, aliases, pil=pil_shared)
            ccip_blob = None
            if ccip is not None:
                try:
                    ccip_blob = ccip_tagger.embed(image_path, ccip, pil=pil_shared).tobytes()
                except Exception:
                    ccip_blob = None
            conn.execute(
                """
                INSERT INTO images (path, filename, tags, description, clip_embedding, mtime, folder_id,
                                    wd_chars, wd_chars_ko, wd_general, ccip_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,
                    tags=excluded.tags,
                    description=excluded.description,
                    clip_embedding=excluded.clip_embedding,
                    mtime=excluded.mtime,
                    folder_id=excluded.folder_id,
                    wd_chars=excluded.wd_chars,
                    wd_chars_ko=excluded.wd_chars_ko,
                    wd_general=excluded.wd_general,
                    ccip_embedding=excluded.ccip_embedding,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (resolved, image_path.name, tags, description, embedding.tobytes(), mtime, folder_id,
                 wd_chars_str, wd_chars_ko_str, wd_general_str, ccip_blob),
            )
            conn.commit()
            if progress is not None:
                progress["success"] += 1
            if on_item is not None:
                msg = wd_chars_ko_str or tags
                on_item(i, len(todo), image_path.name, "ok", msg)
        except Exception as e:
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{image_path.name}: {e}"
            if on_item is not None:
                on_item(i, len(todo), image_path.name, "err", str(e))

    conn.execute(
        "UPDATE folders SET last_indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (folder_id,),
    )
    conn.commit()
    conn.close()

    return {
        "folder": resolved_folder,
        "processed": len(todo),
        "skipped": skipped_count,
        "success": progress["success"] if progress is not None else 0,
        "errors": progress["errors"] if progress is not None else 0,
    }


def backfill_ccip(
    db_path: str,
    ccip: dict,
    progress: Optional[dict] = None,
) -> dict:
    """ccip_embedding이 NULL인 모든 이미지에 CCIP 임베딩만 추가."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, path, filename FROM images WHERE ccip_embedding IS NULL"
    ).fetchall()

    if progress is not None:
        progress["total"] = len(rows)
        progress["current"] = 0
        progress["success"] = 0
        progress["errors"] = 0
        progress["skipped"] = 0
        progress["last_error"] = ""
        progress["current_file"] = ""

    for i, row in enumerate(rows, 1):
        path = Path(row["path"])
        if progress is not None:
            progress["current"] = i
            progress["current_file"] = row["filename"] or path.name
        if not path.exists():
            if progress is not None:
                progress["errors"] += 1
            continue
        try:
            emb = ccip_tagger.embed(path, ccip)
            conn.execute(
                "UPDATE images SET ccip_embedding=? WHERE id=?",
                (emb.tobytes(), row["id"]),
            )
            conn.commit()
            if progress is not None:
                progress["success"] += 1
        except Exception as e:
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{path.name}: {e}"

    conn.close()
    return {"processed": len(rows)}


def backfill_wd14(
    db_path: str,
    wd14: dict,
    aliases: dict,
    progress: Optional[dict] = None,
    on_item: Optional[Callable[[int, int, str, str, str], None]] = None,
) -> dict:
    """wd_chars가 NULL인 모든 이미지에 WD14 태깅만 추가 (VLM/CLIP 재실행 X)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, path, filename FROM images WHERE wd_chars IS NULL").fetchall()

    if progress is not None:
        progress["total"] = len(rows)
        progress["current"] = 0
        progress["success"] = 0
        progress["errors"] = 0
        progress["skipped"] = 0
        progress["last_error"] = ""
        progress["current_file"] = ""

    for i, row in enumerate(rows, 1):
        path = Path(row["path"])
        if progress is not None:
            progress["current"] = i
            progress["current_file"] = row["filename"] or path.name

        if not path.exists():
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{path.name}: 파일 없음"
            if on_item is not None:
                on_item(i, len(rows), path.name, "err", "파일 없음")
            continue

        try:
            wd_chars_str, wd_chars_ko_str, wd_general_str = _wd14_for_image(path, wd14, aliases)
            conn.execute(
                "UPDATE images SET wd_chars=?, wd_chars_ko=?, wd_general=? WHERE id=?",
                (wd_chars_str, wd_chars_ko_str, wd_general_str, row["id"]),
            )
            conn.commit()
            if progress is not None:
                progress["success"] += 1
            if on_item is not None:
                on_item(i, len(rows), path.name, "ok", wd_chars_ko_str)
        except Exception as e:
            if progress is not None:
                progress["errors"] += 1
                progress["last_error"] = f"{path.name}: {e}"
            if on_item is not None:
                on_item(i, len(rows), path.name, "err", str(e))

    conn.close()
    return {
        "processed": len(rows),
        "success": progress["success"] if progress is not None else 0,
        "errors": progress["errors"] if progress is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="이미지 인덱서")
    parser.add_argument("image_dir", help="이미지 폴더 경로")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite DB 경로 (기본값: {DB_DEFAULT})")
    parser.add_argument("--reindex", action="store_true", help="기존 인덱스 무시하고 재인덱싱")
    parser.add_argument("--no-wd14", action="store_true", help="WD14 캐릭터 태깅 비활성화")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")

    conn = init_db(args.db)
    conn.close()

    if not Path(args.image_dir).exists():
        print(f"오류: 경로가 존재하지 않습니다: {args.image_dir}")
        sys.exit(1)

    from providers import LocalQwenProvider
    vlm_provider = LocalQwenProvider(model_key="qwen2.5-vl-7b", device=device)
    vlm_provider.ensure_ready()
    clip_model, clip_preprocess = load_clip(device)
    wd14 = None
    aliases = None
    if not args.no_wd14:
        wd14 = wd14_tagger.load_wd14()
        aliases = wd14_tagger.load_aliases()

    def cli_progress(i: int, total: int, name: str, status: str, msg: str):
        if status == "ok":
            short = (msg or "")[:50] + ("..." if msg and len(msg) > 50 else "")
            print(f"[{i}/{total}] {name} OK ({short})")
        else:
            print(f"[{i}/{total}] {name} ERROR: {msg}")

    if wd14 is not None:
        print("\n기존 이미지 WD14 백필 중...")
        backfill_wd14(args.db, wd14, aliases, progress={}, on_item=cli_progress)

    progress = {}
    start = time.time()
    result = index_folder(
        args.image_dir,
        args.db,
        vlm_provider,
        clip_model,
        clip_preprocess,
        device,
        progress=progress,
        reindex=args.reindex,
        on_item=cli_progress,
        wd14=wd14,
        aliases=aliases,
    )
    elapsed = time.time() - start
    print(
        f"\n완료! 처리: {result['success']}, 오류: {result['errors']}, 스킵: {result['skipped']}, 소요: {elapsed:.1f}s"
    )
    print(f"DB 저장: {args.db}")


if __name__ == "__main__":
    main()
