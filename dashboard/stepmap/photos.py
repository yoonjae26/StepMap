"""현장 사진 사본 (대시보드·신고서 공통).

events.csv의 image는 events.csv 기준 상대 경로다 (예: photos/step_curb_gap.jpg, img/ev_0001.jpg).
HTML을 만들 때 원본을 찾아 짧은 변 PHOTO_SHORT_SIDE px 이하로 줄인 사본을 <HTML 폴더>/photos/에 저장하고
HTML 기준 상대 경로로 연결한다. 원본은 건드리지 않는다.
- EXIF 회전을 반영하고, 사본에는 메타데이터(촬영 위치 등)를 넣지 않는다
- 원본이 없거나 열 수 없으면 그 지점은 사진 없음으로 처리한다
- stamps를 주면 사본 아래쪽에 촬영 일시·위치 띠를 찍는다 (사진만 따로 보거나 내려받아도 언제·어디인지 남게)
"""

from __future__ import annotations

import html
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .events import parse_image_mark

PHOTO_SHORT_SIDE = 720
PHOTO_DIR = "photos"
LEVEL_COLOR = {"danger": "#EF6461", "warn": "#F7B558", "ok": "#A5D48C"}  # 대시보드·신고서 등급 색 (위험·주의·정상)
LEG_NAMES = {"FR": "앞 오른발", "FL": "앞 왼발", "RR": "뒤 오른발", "RL": "뒤 왼발"}
# 사진 아래 설명 (데이터에 넣지 않고 종류에 따라 자동으로 붙인다)
PHOTO_CAPTIONS = {"loose": "외관 정상, 발 접촉으로 확인"}
# 대시보드 상세 패널 썸네일 상자: CSS .ph --maxh 300px에서 테두리 1px씩을 빼면 사진은 298px.
# 여유를 두어 296px로 계산한다 (패널 안쪽 폭도 300px 이상)
THUMB_BOX_PX = 296
THUMB_MIN_FONT_PX = 11
KST = timezone(timedelta(hours=9))
# 사진 띠 글꼴: 한글이 되는 글꼴을 차례로 찾는다 (Windows 맑은 고딕, 로봇·Linux Noto CJK). 없으면 영문·숫자만 쓴다
STAMP_FONTS = ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
               "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
               str(Path.home() / "terrain_dataset" / "fonts" / "NotoSansCJKkr-Bold.otf"))   # 로봇 (Jetson)


def kst_text(t: float) -> str:
    """시각(유닉스 초) → 'YYYY-MM-DD HH:MM:SS' (한국 시간)."""
    return datetime.fromtimestamp(t, KST).strftime("%Y-%m-%d %H:%M:%S")


def stamp_text(e: dict, route_m: float | None = None) -> tuple[str, str]:
    """사진 띠 문구 (한글, 영문 대체). 촬영 일시 + 출발점에서 걸은 거리."""
    when = kst_text(e["t"])
    if route_m is None:
        return f"촬영 {when} KST", f"{when} KST"
    return f"촬영 {when} KST · 출발점에서 {round(route_m)} m", f"{when} KST  start +{round(route_m)} m"


def _stamp(im, text: tuple[str, str]):
    """사진 아래쪽에 반투명 검은 띠 + 흰 글씨 (사진 크기는 그대로)."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = im.size
    size = max(12, round(min(w, h) * 0.04))
    font, label = None, text[1]
    for f in STAMP_FONTS:
        try:
            font, label = ImageFont.truetype(f, size), text[0]
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    band = round(size * 1.8)
    over = Image.new("RGBA", (w, band), (0, 0, 0, 150))
    ImageDraw.Draw(over).text((round(size * 0.6), round((band - size) / 2) - 1), label, fill=(255, 255, 255, 255), font=font)
    im = im.convert("RGBA")
    im.alpha_composite(over, (0, h - band))
    return im.convert("RGB")


def _copy_name(event_id: str, src: Path) -> str:
    stem = re.sub(r"[^\w.-]", "_", src.stem)[:40]
    return f"{event_id}_{stem}.jpg"


def prepare_photos(events: list[dict], events_dir: str | Path, out_dir: str | Path,
                   stamps: dict[str, tuple[str, str]] | None = None) -> dict[str, dict]:
    """{id: {"src": "photos/<파일>", "w": 폭, "h": 높이, "mark": [u, v, r] | None}}. 사진 없는 지점은 빠진다.
    stamps: {id: stamp_text(...)} 이면 사본 아래쪽에 그 문구 띠를 찍는다."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    events_dir, out_dir = Path(events_dir), Path(out_dir)
    photos: dict[str, dict] = {}
    for e in events:
        if not e.get("image"):
            continue
        src = Path(e["image"])
        if not src.is_absolute():
            src = events_dir / src
        if not src.is_file():
            continue
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                k = PHOTO_SHORT_SIDE / min(w, h)
                if k < 1:  # 줄이기만 한다
                    # Image.Resampling은 Pillow 9.1+. 로봇(Jetson, Ubuntu 20.04)은 Pillow 7이라 예전 이름으로 대신한다
                    lanczos = getattr(Image, "Resampling", Image).LANCZOS
                    im = im.resize((round(w * k), round(h * k)), lanczos)
                if stamps and e["id"] in stamps:
                    im = _stamp(im, stamps[e["id"]])
                name = _copy_name(e["id"], src)
                (out_dir / PHOTO_DIR).mkdir(parents=True, exist_ok=True)
                im.save(out_dir / PHOTO_DIR / name, "JPEG", quality=85, optimize=True)
                w, h = im.size
        except (OSError, UnidentifiedImageError):
            continue
        try:
            mark = parse_image_mark(e.get("image_mark") or "")
        except ValueError:
            mark = None
        photo = {"src": f"{PHOTO_DIR}/{name}", "w": w, "h": h, "mark": list(mark) if mark else None,
                 "caption": photo_caption(e["kind"])}
        photo["overlay"] = overlay_svg(photo, e["kind"], e["level"], e.get("leg", ""))  # 크게 보기·신고서
        photo["overlay_thumb"] = overlay_svg(photo, e["kind"], e["level"], e.get("leg", ""),  # 썸네일
                                             min_font_px=THUMB_MIN_FONT_PX, scale=thumb_scale(w, h))
        photos[e["id"]] = photo
    return photos


def photo_caption(kind: str) -> str:
    return PHOTO_CAPTIONS.get(kind, "")


def thumb_scale(w: int, h: int) -> float:
    """썸네일에서 사진 1px이 화면 몇 px인지 (상자 THUMB_BOX_PX 안에 비율 유지)."""
    return min(THUMB_BOX_PX / w, THUMB_BOX_PX / h)


# 동그라미는 겉보기에 멀쩡한 흔들림 사진에만 (단차·균열은 문제가 사진에 보이므로 사진만)
MARK_KINDS = ("loose",)


def contact_label(leg: str, short: bool = False) -> str:
    """흔들림 사진 위 라벨: 발 접촉 위치 (추정)."""
    if short:
        return f"발 접촉 (추정) · {leg}" if leg else "발 접촉 (추정)"
    return f"발 접촉 위치 (추정) · {LEG_NAMES.get(leg, leg)}({leg})" if leg else "발 접촉 위치 (추정)"


def _text_width(text: str, fs: float) -> float:
    """라벨 폭 어림값 (한글 0.95em, 공백 0.3em, 그 밖 0.6em)."""
    return sum(0.95 if ord(c) >= 0x1100 else 0.3 if c == " " else 0.6 for c in text) * fs


def _label_box(label: str, fs: float, w: float, h: float, cx: float, cy: float, R: float):
    """라벨 상자 (lx, ly, bw, bh, 넘침 없음 여부). 동그라미 위 → 아래 → (자리 없으면) 가운데."""
    pad, accent = 0.45 * fs, 0.25 * fs
    bw, bh = accent + 2 * pad + _text_width(label, fs), 1.55 * fs
    gap = 0.3 * fs
    lx = min(max(cx - bw / 2, 4), w - bw - 4)
    fits = bw <= w - 8
    ly = cy - R - gap - bh  # 동그라미 위
    if ly < 4:
        ly = cy + R + gap  # 위에 자리가 없으면 아래
    if ly + bh > h - 4:
        ly = max(4, cy - bh / 2)  # 둘 다 없으면 가운데 (동그라미를 가린다)
        fits = False
    return lx, ly, bw, bh, fits


def overlay_svg(photo: dict, kind: str, level: str, leg: str = "",
                min_font_px: float | None = None, scale: float | None = None) -> str:
    """흔들림 사진 위 발 접촉 위치 표시 (원본 사진은 그대로). 사진 픽셀 좌표의 SVG.

    kind가 흔들림(loose)이고 image_mark가 있을 때만 그린다. 다른 종류는 image_mark가 있어도 빈 문자열
    (단차·균열은 문제가 사진에 보이므로 사진만 보여 준다).
    점선 동그라미 + "발 접촉 위치 (추정) · 다리". 동그라미는 판정 색, 사진 밝기와 상관없이 보이게 바깥에 흰 테두리.
    min_font_px·scale(사진 1px당 화면 px)을 주면 라벨 글씨를 화면에서 최소 min_font_px로 키우고,
    그래도 사진 밖으로 넘치거나 동그라미를 가리면 짧은 라벨("발 접촉 (추정) · FL")을 쓴다 (썸네일용).
    """
    if kind not in MARK_KINDS or not photo.get("mark"):
        return ""
    w, h = photo["w"], photo["h"]
    u, v, r = photo["mark"]
    cx, cy, R = u * w, v * h, r * min(w, h)
    color = LEVEL_COLOR[level]
    dash = ' stroke-dasharray="9 6"'
    ns = ' vector-effect="non-scaling-stroke"'

    fs = 0.045 * min(w, h)
    if min_font_px and scale:
        fs = max(fs, math.ceil(min_font_px / scale * 10) / 10)  # SVG에 0.1 단위로 쓰므로 올림
    label = contact_label(leg)
    lx, ly, bw, bh, fits = _label_box(label, fs, w, h, cx, cy, R)
    if min_font_px and not fits:
        label = contact_label(leg, short=True)
        lx, ly, bw, bh, _ = _label_box(label, fs, w, h, cx, cy, R)
    pad, accent = 0.45 * fs, 0.25 * fs

    return (
        f'<svg class="ph-mark" viewBox="0 0 {w} {h}" aria-hidden="true" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R:.1f}" fill="none" stroke="#ffffff" stroke-width="7"{ns}/>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R:.1f}" fill="none" stroke="{color}" stroke-width="3.5"{dash}{ns}/>'
        f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="{0.25 * fs:.1f}" '
        f'fill="#111111" fill-opacity="0.82" stroke="#ffffff" stroke-width="1.5"{ns}/>'
        f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{accent:.1f}" height="{bh:.1f}" fill="{color}"/>'
        f'<text x="{lx + accent + pad:.1f}" y="{ly + bh * 0.7:.1f}" font-size="{fs:.1f}" font-weight="700" fill="#ffffff" '
        f'font-family="Malgun Gothic, Apple SD Gothic Neo, system-ui, sans-serif">{html.escape(label)}</text>'
        f'</svg>'
    )
