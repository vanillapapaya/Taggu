# 03 — API 명세

## 베이스

- 로컬: `http://127.0.0.1:8000` 또는 `https://127.0.0.1:8000` (인증서 있을 시 자동)
- LAN: 동일 머신 IP, 포트 8000
- 인증 없음 (단일 사용자 가정)
- 모든 응답 JSON (`/thumbs/*`, `/images/*` 제외)

## 인덱싱 / 백필

| Method | Path | Body / Query | 비고 |
|---|---|---|---|
| POST | `/api/index` | `{path, reindex, with_ai}` | 폴더 인덱싱 시작 (백그라운드) |
| POST | `/api/backfill_wd14` | — | WD14 누락분만 |
| POST | `/api/backfill_ccip` | — | CCIP 임베딩 누락분만 (학습 활성화 전 1회) |
| POST | `/api/backfill_vlm` | — | AI 한국어 태그/설명 누락분만 |
| POST | `/api/auto_translate_chars` | — | 미매핑 영어 캐릭터 → AI 한국어 일괄 번역 |
| POST | `/api/relocalize` | — | alias 갱신 후 한국어 매핑 재적용 (사용자 편집 보존) |
| POST | `/api/dedupe_tags` | — | 모든 이미지 중복 태그 정리 |
| POST | `/api/cleanup_library_dupes` | — | `*.library` 등 사본 row 삭제 |
| GET | `/api/index/status` | — | 진행률 + 현재 상태 폴링 |

### `/api/index` 요청 예시
```json
{ "path": "D:\\Download\\그림\\씹덕", "reindex": false, "with_ai": false }
```

### `/api/index/status` 응답
```json
{
  "status": "running" | "idle" | "done" | "loading_model" | "error",
  "folder": "D:\\...",
  "total": 207, "current": 56, "skipped": 0,
  "success": 55, "errors": 1,
  "current_file": "abc.jpg",
  "last_error": "...",
  "message": "AI 분석 중...",
  "started_at": 1716800000.0,
  "finished_at": null
}
```

## 검색 / 브라우징

| Method | Path | Query | 비고 |
|---|---|---|---|
| GET | `/api/search` | `q, limit, view, no_char, no_user, no_ai` | 텍스트 검색, 띄어쓰기 AND |
| GET | `/api/random` | `n, view, no_char, no_user, no_ai` | 랜덤 |
| GET | `/api/browse` | `offset, limit, view, no_char, no_user, no_ai` | 페이지네이션 |
| GET | `/api/similar/{id}` | `limit, view, no_char, no_user, no_ai` | CLIP cosine 유사 이미지 |
| GET | `/api/counts` | — | `{all, favorite, hidden, no_char, no_user, no_ai}` |
| GET | `/api/info` | — | 전체 통계 (popover용) |

### `view` 값
- `all` (기본, hidden=0)
- `favorite` (favorite=1 AND hidden=0)
- `hidden` (hidden=1)

### 필터 플래그 (0 or 1)
- `no_char`: wd_chars_ko 비어있음
- `no_user`: user_tags 비어있음
- `no_ai`: tags 비어있음

여러 개 동시에 AND 조합됨.

### 결과 row 형식
```json
{
  "id": 123,
  "filename": "abc.jpg",
  "tags": "고양이, 밈, 웃긴",
  "description": "놀란 표정의 고양이",
  "wd_chars_ko": "꽉",
  "wd_chars": "kyaru_(princess_connect!)",
  "user_tags": "라프텔,웃긴",
  "hidden": false,
  "favorite": true,
  "image_url": "/images/D%3A%5C...",
  "thumb_url": "/thumbs/123",
  "score": 0.847,        // search/similar에서만
  "matched": true        // search에서만
}
```

## 이미지 상태 / 태그

| Method | Path | Body | 비고 |
|---|---|---|---|
| POST | `/api/image/{id}/state` | `{favorite?, hidden?}` | 즐겨찾기/숨김 토글 |
| POST | `/api/image/{id}/tags` | `{tags: list[str]}` | user_tags 전체 갱신 |
| POST | `/api/image/{id}/field_tags` | `{field, tags}` | 임의 필드 갱신 + 학습 hook (wd_chars_ko일 때) |
| POST | `/api/image/{id}/move_tag` | `{tag, source, target}` | 6방향 atomic 이동 + 학습 hook |
| GET | `/api/image/{id}/suggest_chars` | `min_score, limit` | CCIP centroid 추천 |
| POST | `/api/image/{id}/copy` | — | (로컬 only) OS 클립보드에 이미지 복사 |

### `move_tag` source/target
- `user` ↔ `char` ↔ `ai` 6 방향 (자기 자신 제외)
- 필드 매핑: `user → user_tags`, `char → wd_chars_ko`, `ai → tags`

### `suggest_chars` 응답
```json
{
  "candidates": [
    {"name": "히마리", "score": 0.873, "count": 8},
    {"name": "사야",   "score": 0.712, "count": 3}
  ],
  "already": ["꽉"],
  "threshold": 0.70
}
```

## 설정 (AI Provider)

| Method | Path | Body / Query | 비고 |
|---|---|---|---|
| GET | `/api/settings` | — | 설정 + 모델 카탈로그 + VRAM (API 키 마스킹) |
| POST | `/api/settings` | `{settings}` | 저장 + 즉시 적용. 빈 키는 기존 유지 |
| POST | `/api/settings/test` | `{settings}` | 32×32 더미 이미지로 1회 검증 호출 |

### settings 구조
```json
{
  "provider": "local" | "openai" | "anthropic" | "gemini",
  "local":     {"model_key": "qwen2.5-vl-7b-bnb4", "device": "cuda"},
  "openai":    {"api_key": "sk-...", "model": "gpt-4o-mini"},
  "anthropic": {"api_key": "sk-ant-...", "model": "claude-haiku-4-5"},
  "gemini":    {"api_key": "AIza...", "model": "gemini-2.5-flash"}
}
```

### local 모델 옵션
- `qwen2.5-vl-7b` — FP16, ~16GB VRAM
- `qwen2.5-vl-7b-bnb4` — bnb 4bit, ~5.5GB VRAM (권장)
- `qwen2.5-vl-3b` — FP16, ~7GB VRAM

## 시스템

| Method | Path | 비고 |
|---|---|---|
| POST | `/api/restart` | exit 42 → 데스크톱 런처가 자동 재실행 |
| POST | `/api/open_log` | 로그 파일을 OS 기본 앱으로 열기 (로컬 only) |
| POST | `/api/pick_folder` | 폴더 선택 다이얼로그 (로컬 only) |
| GET | `/api/folders` | 등록된 폴더 목록 + 각 통계 |
| GET | `/images/{path:path}` | 이미지 서빙 (등록 폴더 안만 허용 — path traversal 방지) |
| GET | `/thumbs/{id}` | 480px WebP 썸네일 (애니메이션 보존, 캐시) |
| GET | `/favicon.ico` | 아이콘 |
| GET | `/` | 메인 HTML (Jinja2 템플릿) |

## 에러 처리

- 4xx: `{"error": "한국어 메시지"}` 형식
- 5xx: 동일 + `traceback.print_exc()` 서버 로그에
- 409: 다른 작업 진행 중 (`index_lock` 또는 `pick_folder_lock` 점유)

## 관련

- [[02-데이터-파이프라인]] — 각 엔드포인트가 호출되는 흐름
- [[04-프론트엔드-명세]] — UI가 어떻게 이 API를 호출하는가
