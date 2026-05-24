"""Yoink 아이콘 생성기.

PIL로 다크 + 오렌지 Y 아이콘을 멀티 사이즈 .ico (16~256px)로 저장.

- icon.ico: Windows .lnk / favicon 용
- icon.png: 256px 미리보기 / OG 이미지 등

자기만의 디자인을 쓰려면 이 파일을 무시하고 같은 경로(icon.ico)에 직접 .ico를 두면 됨.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
ICO_PATH = ROOT / "icon.ico"
PNG_PATH = ROOT / "icon.png"

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
SOURCE_SIZE = 256

BG_COLOR = "#322e2a"
BORDER_COLOR = "#d97757"
TEXT_COLOR = "#d97757"


def _try_font(size: int):
    for name in ("seguibl.ttf", "arialbd.ttf", "segoeuib.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_source_image(size: int = SOURCE_SIZE) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = max(2, size // 24)
    border_w = max(2, size // 32)
    radius = size // 5
    d.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=radius,
        fill=BG_COLOR,
        outline=BORDER_COLOR,
        width=border_w,
    )

    font = _try_font(int(size * 0.58))
    bbox = d.textbbox((0, 0), "Y", font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1] - size * 0.04
    d.text((x, y), "Y", fill=TEXT_COLOR, font=font)
    return img


def ensure_icon(force: bool = False) -> Path:
    """icon.ico가 없거나 force=True면 생성. 항상 경로 반환."""
    if ICO_PATH.exists() and not force:
        return ICO_PATH
    src = make_source_image(SOURCE_SIZE)
    src.save(ICO_PATH, format="ICO", sizes=ICO_SIZES)
    src.save(PNG_PATH, format="PNG")
    return ICO_PATH


if __name__ == "__main__":
    p = ensure_icon(force=True)
    print(f"Created: {p}")
    print(f"         {PNG_PATH}")
