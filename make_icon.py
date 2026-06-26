"""Taggu 아이콘 생성기 — 'Moe T' 마스코트.

PIL로 마스코트 아이콘(다크 라운드 사각 + 분홍 T + 청록 눈)을 멀티 사이즈 .ico (16~256px)로 저장.
디자인 출처: claude.ai/design 프로젝트 "Taggu 프로젝트 아이콘", TagguIcon variant=mascot.

- icon.ico: Windows .lnk / favicon / EXE 아이콘 용
- icon.png: 256px 미리보기 / OG 이미지 등

자기만의 디자인을 쓰려면 이 파일을 무시하고 같은 경로(icon.ico)에 직접 .ico를 두면 됨.
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
ICO_PATH = ROOT / "icon.ico"
PNG_PATH = ROOT / "icon.png"

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
SOURCE_SIZE = 256

# 팔레트
DARK = (26, 21, 48, 255)      # #1a1530 배경
PINK = (255, 92, 138, 255)    # #ff5c8a T / 볼터치
TEAL = (79, 227, 211, 255)    # #4fe3d3 눈
CREAM = (255, 244, 234, 255)  # #fff4ea 하이라이트


def make_source_image(size: int = SOURCE_SIZE) -> Image.Image:
    """마스코트 아이콘 렌더. 256 viewBox 좌표를 4x 슈퍼샘플로 그린 뒤 size로 다운스케일."""
    ss = 4
    n = size * ss
    k = n / 256.0  # 256 좌표계 → n 픽셀

    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def rr(x, y, w, h, r, fill):
        d.rounded_rectangle([x * k, y * k, (x + w) * k, (y + h) * k], radius=r * k, fill=fill)

    def cc(cx, cy, r, fill):
        d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], fill=fill)

    rr(0, 0, 256, 256, 58, DARK)            # 둥근 사각 배경 (코너 바깥 투명)
    rr(56, 82, 144, 34, 17, PINK)           # T 가로바
    rr(110, 82, 36, 112, 16, PINK)          # T 세로기둥
    cc(92, 90, 20, TEAL); cc(164, 90, 20, TEAL)     # 눈
    cc(95, 94, 9, DARK);  cc(161, 94, 9, DARK)      # 동공
    cc(90, 85, 3.6, CREAM); cc(159, 85, 3.6, CREAM)  # 하이라이트

    # 볼터치 (분홍 50% 투명) — 별도 레이어 합성
    overlay = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for cx in (70, 186):
        od.ellipse([(cx - 11) * k, (150 - 11) * k, (cx + 11) * k, (150 + 11) * k],
                   fill=(255, 92, 138, 128))
    img = Image.alpha_composite(img, overlay)

    return img.resize((size, size), Image.LANCZOS)


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
