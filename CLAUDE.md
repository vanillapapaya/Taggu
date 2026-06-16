# Taggu v2

로컬 이미지 자동 태깅 + 검색 시스템. GPU 서버에서 VLM 태깅 + CLIP 임베딩, 브라우저로 검색.

## Architecture

- **index.py**: 이미지 폴더 스캔 → Qwen2.5-VL-7B 한국어 태깅 → CLIP ViT-B/32 임베딩 → SQLite 저장
- **app.py**: FastAPI 웹 서버 (검색 API + 이미지 서빙)
- **templates/index.html**: 검색 UI (단일 HTML + vanilla JS)

## Tech Stack

| 항목 | 선택 |
|------|------|
| VLM | Qwen2.5-VL-7B-Instruct (한국어 태그/설명 생성) |
| 임베딩 | CLIP ViT-B/32 (open_clip, 512d) |
| DB | SQLite (images.db) |
| 웹 | FastAPI + Jinja2 + uvicorn |
| 프론트 | 단일 HTML + vanilla JS |

## Project Structure

```
index.py              # 이미지 인덱서 (GPU 서버에서 실행)
app.py                # 웹 서버 (GPU 서버에서 실행)
templates/
  index.html          # 검색 UI
requirements.txt      # Python 의존성
images.db             # 생성되는 SQLite DB (gitignore)
docs/                 # 설계 문서 (한국어)
```

## Usage

```bash
# GPU 서버 (Windows, RTX 5080)
pip install -r requirements.txt

# 1. 인덱싱
python index.py /path/to/images

# 2. 웹 서버 실행
python app.py --images /path/to/images

# Mac 브라우저에서 접속
# http://192.168.0.75:8000
```

## Conventions

- Documentation language: Korean
- Code comments: Korean or English
- Python type hints 사용
