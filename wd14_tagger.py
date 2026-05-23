"""
WD14 캐릭터/일반 태거 (SmilingWolf/wd-eva02-large-tagger-v3, ONNX).

- HuggingFace Hub에서 모델/태그 csv 다운로드 (자동 캐시)
- onnxruntime CPU 추론
- 한국어 별칭 매핑 (character_aliases.json)
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

WD14_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
ALIASES_DEFAULT = "character_aliases.json"

# Danbooru 태그 카테고리: 0=general, 1=artist, 3=copyright, 4=character, 5=meta
CATEGORY_GENERAL = 0
CATEGORY_COPYRIGHT = 3
CATEGORY_CHARACTER = 4

GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
COPYRIGHT_THRESHOLD = 0.50


def load_wd14(providers: Optional[list[str]] = None):
    """WD14 ONNX 모델 + 태그 목록 로드. 첫 호출 시 모델 다운로드 (~1.5GB)."""
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    print(f"WD14 로딩 중: {WD14_REPO}")
    model_path = hf_hub_download(WD14_REPO, "model.onnx")
    tags_path = hf_hub_download(WD14_REPO, "selected_tags.csv")

    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)

    tags: list[dict] = []
    with open(tags_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append({
                "name": row["name"],
                "category": int(row["category"]),
            })

    input_shape = sess.get_inputs()[0].shape
    target_size = input_shape[1] if isinstance(input_shape[1], int) else 448
    print(f"WD14 로딩 완료 (input={target_size}x{target_size}, tags={len(tags)})")
    return {"session": sess, "tags": tags, "target_size": target_size}


def load_aliases(path: str = ALIASES_DEFAULT) -> dict:
    """character_aliases.json 로드."""
    p = Path(path)
    if not p.exists():
        return {"_works": {}, "characters": {}}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "_works": data.get("_works", {}),
        "characters": data.get("characters", {}),
    }


def _preprocess(image_path: Path, target_size: int) -> np.ndarray:
    """WD14 입력 형식: target_size x target_size, BGR float32, 흰 배경 패딩."""
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    w, h = img.size
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (255, 255, 255))
    padded.paste(img, ((s - w) // 2, (s - h) // 2))
    padded = padded.resize((target_size, target_size), Image.BICUBIC)

    arr = np.array(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    arr = np.expand_dims(arr, axis=0)
    return arr


def tag_image(
    image_path: Path,
    wd14: dict,
    general_threshold: float = GENERAL_THRESHOLD,
    character_threshold: float = CHARACTER_THRESHOLD,
    copyright_threshold: float = COPYRIGHT_THRESHOLD,
) -> tuple[list[str], list[str], list[str]]:
    """이미지를 태깅하여 (캐릭터, 작품, 일반) 영어 태그 리스트 반환."""
    sess = wd14["session"]
    tags = wd14["tags"]
    target_size = wd14["target_size"]

    arr = _preprocess(image_path, target_size)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    probs = sess.run([output_name], {input_name: arr})[0][0]

    characters: list[tuple[str, float]] = []
    works: list[tuple[str, float]] = []
    general: list[tuple[str, float]] = []

    for i, t in enumerate(tags):
        p = float(probs[i])
        cat = t["category"]
        name = t["name"]
        if cat == CATEGORY_CHARACTER and p >= character_threshold:
            characters.append((name, p))
        elif cat == CATEGORY_COPYRIGHT and p >= copyright_threshold:
            works.append((name, p))
        elif cat == CATEGORY_GENERAL and p >= general_threshold:
            general.append((name, p))

    characters.sort(key=lambda x: -x[1])
    works.sort(key=lambda x: -x[1])
    general.sort(key=lambda x: -x[1])

    return (
        [n for n, _ in characters],
        [n for n, _ in works],
        [n for n, _ in general],
    )


_WORK_SUFFIX_RE = re.compile(r"_\([^()]+\)$")


def _strip_work_suffix(tag: str) -> str:
    """Danbooru 캐릭터 태그의 '_(작품명)' 접미사 제거."""
    return _WORK_SUFFIX_RE.sub("", tag)


def localize_tags(
    tags_en: list[str],
    aliases: dict,
    skip_unmapped: bool = True,
) -> list[str]:
    """영어 태그 리스트 → 한국어 별칭. 캐릭터 + 작품 매핑 모두 시도.

    skip_unmapped=True: 매핑 못 찾은 영어 태그는 결과에서 제외 (UI 깔끔)
    skip_unmapped=False: 언더스코어를 공백으로 바꿔 그대로 유지
    """
    char_map = aliases.get("characters", {})
    work_map = aliases.get("_works", {})

    out: list[str] = []
    seen: set[str] = set()

    for tag in tags_en:
        if not tag:
            continue
        lower = tag.lower()
        clean = _strip_work_suffix(lower)
        ko = char_map.get(clean) or char_map.get(lower) or work_map.get(lower) or work_map.get(clean)
        if ko is None:
            if skip_unmapped:
                continue
            ko = clean.replace("_", " ")
        if ko not in seen:
            out.append(ko)
            seen.add(ko)

    return out


def localize_characters(chars_en: list[str], works_en: list[str], aliases: dict) -> list[str]:
    """[Deprecated] 호환용. localize_tags 사용 권장."""
    return localize_tags((chars_en or []) + (works_en or []), aliases, skip_unmapped=False)
