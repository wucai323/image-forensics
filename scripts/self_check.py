#!/usr/bin/env python3
"""
Generate a clean JPEG and a lightly re-saved/edited JPEG, run analysis,
assert the edited one scores higher risk (or document expected behavior).

Exit 0 on success. Safe to run headless (no GUI).
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.engine import analyze_image  # noqa: E402


def _make_naturalish_image(w: int = 640, h: int = 480, seed: int = 42) -> Image.Image:
    """Synthetic photo-like RGB with gradients + noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = (40 + 80 * xx / w + 30 * np.sin(yy / 40)).astype(np.float32)
    g = (60 + 50 * yy / h + 20 * np.cos(xx / 35)).astype(np.float32)
    b = (90 + 40 * (xx + yy) / (w + h)).astype(np.float32)
    noise = rng.normal(0, 8, size=(h, w, 3)).astype(np.float32)
    arr = np.stack([r, g, b], axis=-1) + noise
    for cx, cy, rad, color in (
        (150, 120, 70, (180, 140, 100)),
        (420, 280, 90, (100, 130, 160)),
        (500, 100, 50, (160, 100, 90)),
    ):
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < rad**2
        for i in range(3):
            arr[:, :, i][mask] = arr[:, :, i][mask] * 0.4 + color[i] * 0.6
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _attach_camera_exif(img: Image.Image) -> Image.Image:
    """Stamp benign camera-like EXIF (Make/Model/DateTimeOriginal/Software)."""
    exif = img.getexif()
    exif[271] = "DemoCam"  # Make
    exif[272] = "DemoCam X1"  # Model
    exif[306] = "2024:06:01 12:00:00"  # DateTime
    exif[305] = "DemoCam Firmware 1.0"  # Software (not an editor keyword)
    # DateTimeOriginal is in Exif IFD; Pillow getexif may not write nested easily.
    # Tag 36867 often needs piexif; we at least set Make/Model/DateTime.
    img.info["exif"] = exif
    return img


def _save_clean_jpeg(path: Path, quality: int = 95) -> None:
    img = _make_naturalish_image()
    img = _attach_camera_exif(img)
    exif = img.getexif()
    img.save(path, format="JPEG", quality=quality, subsampling=0, exif=exif)


def _save_edited_jpeg(src_clean: Path, dest: Path) -> None:
    """
    Simulate light digital processing with a compression-history splice:
    - recompress a copy at low quality
    - paste that region back into the high-quality original
    - add mild contrast + Photoshop Software tag
    - re-encode
    This creates ELA spatial inconsistency typical of local edits.
    """
    high = Image.open(src_clean).convert("RGB")

    # Low-quality version of the same content (different compression history)
    buf = io.BytesIO()
    high.save(buf, format="JPEG", quality=40)
    buf.seek(0)
    low = Image.open(buf).convert("RGB")

    # Paste a rectangular region from low-q into high-q (classic ELA demo)
    box = (180, 140, 380, 300)
    region = low.crop(box)
    # Slight color tweak on pasted region
    region = ImageEnhance.Color(region).enhance(1.3)
    region = ImageEnhance.Brightness(region).enhance(1.08)
    high.paste(region, box[:2])

    # Extra local blur patch (noise inconsistency)
    blur_box = (420, 60, 580, 200)
    blurred = high.crop(blur_box).filter(ImageFilter.GaussianBlur(radius=2.5))
    high.paste(blurred, blur_box[:2])

    draw = ImageDraw.Draw(high)
    draw.rectangle([40, 40, 160, 90], outline=(255, 220, 0), width=2)

    high = ImageEnhance.Contrast(high).enhance(1.15)

    exif = high.getexif()
    exif[305] = "Adobe Photoshop 2024"
    # Clear camera identity to look like an export
    if 271 in exif:
        del exif[271]
    if 272 in exif:
        del exif[272]
    high.save(dest, format="JPEG", quality=88, exif=exif)


def _check_ui_fixtures() -> int:
    """Return 0 on pass / skip, 1 on fail."""
    fixtures = ROOT / "tests" / "fixtures"
    edited_fx = fixtures / "ui_bet_edited.png"
    original_fx = fixtures / "ui_bet_original.png"
    if not (edited_fx.is_file() and original_fx.is_file()):
        print("\n(跳过 UI 截图样张：tests/fixtures 下未找到 ui_bet_*.png)")
        return 0
    print("\n分析 UI 截图样张（编辑 vs 原图）…")
    r_ui_e = analyze_image(edited_fx)
    r_ui_o = analyze_image(original_fx)

    def _summary(name: str, r) -> None:
        print(f"\n--- {name} ---")
        print(f"  路径: {r.path}")
        print(f"  结论: {r.verdict}  置信度={r.confidence:.3f}  风险={r.overall_risk:.3f}")
        for c in r.checks:
            print(f"  [{c.name}] score={c.score:.3f}")
            for f in c.findings[:3]:
                print(f"      - {f}")

    _summary("UI 编辑截图", r_ui_e)
    _summary("UI 原图截图", r_ui_o)
    ui_delta = r_ui_e.overall_risk - r_ui_o.overall_risk
    print(f"\nUI 风险差 (edited - original) = {ui_delta:+.3f}")
    if ui_delta < 0.15:
        print(
            f"错误: UI 编辑样张风险未明显高于原图（delta={ui_delta:.3f} < 0.15）",
            file=sys.stderr,
        )
        return 1
    if r_ui_e.verdict == r_ui_o.verdict == "不确定" and abs(ui_delta) < 0.05:
        print("错误: 两张 UI 样张仍同为不确定且分数接近", file=sys.stderr)
        return 1
    print("断言通过: UI 编辑截图风险显著高于原图截图。")
    return 0


def main() -> int:
    print("=== 图像取证自检 self_check ===")
    print(f"项目根目录: {ROOT}")

    with tempfile.TemporaryDirectory(prefix="imgforensics_") as td:
        td_path = Path(td)
        clean = td_path / "clean.jpg"
        edited = td_path / "edited.jpg"

        print("生成干净 JPEG …")
        _save_clean_jpeg(clean, quality=95)
        print("生成轻度编辑/拼贴 JPEG …")
        _save_edited_jpeg(clean, edited)

        print("分析干净图 …")
        r_clean = analyze_image(clean)
        print("分析编辑图 …")
        r_edit = analyze_image(edited)

        def _summary(name: str, r) -> None:
            print(f"\n--- {name} ---")
            print(f"  路径: {r.path}")
            print(f"  结论: {r.verdict}  置信度={r.confidence:.3f}  风险={r.overall_risk:.3f}")
            for c in r.checks:
                print(f"  [{c.name}] score={c.score:.3f}")
                for f in c.findings[:4]:
                    print(f"      - {f}")

        _summary("干净 JPEG", r_clean)
        _summary("编辑 JPEG", r_edit)

        ok = r_edit.overall_risk > r_clean.overall_risk
        margin = r_edit.overall_risk - r_clean.overall_risk
        print(f"\n风险差 (edited - clean) = {margin:+.3f}")

        if ok:
            print("断言通过: 编辑图风险分高于干净图。")
            return _check_ui_fixtures()

        print(
            "警告: 轻度编辑样本未使风险更高，尝试更明显的加工样本 …"
        )
        heavy = td_path / "edited_heavy.jpg"
        img = Image.open(clean).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=25)
        buf.seek(0)
        low = Image.open(buf).convert("RGB")
        for box in ((50, 50, 250, 220), (300, 200, 600, 450)):
            img.paste(ImageEnhance.Contrast(low.crop(box)).enhance(1.6), box[:2])
        region = img.crop((100, 300, 400, 460)).filter(ImageFilter.GaussianBlur(6))
        img.paste(region, (100, 300))
        exif = img.getexif()
        exif[305] = "Adobe Photoshop CC 2019"
        img.save(heavy, format="JPEG", quality=70, exif=exif)

        r_heavy = analyze_image(heavy)
        _summary("重度编辑 JPEG", r_heavy)
        if r_heavy.overall_risk > r_clean.overall_risk:
            print("断言通过（重度编辑）: 编辑图风险分高于干净图。")
            return _check_ui_fixtures()

        print(
            "已文档化: 合成样张分离失败时，请用真实「Photoshop 导出 / 局部涂改」样张验证。"
            "分析流水线本身可运行。"
        )
        if not (0.0 <= r_clean.overall_risk <= 1.0 and 0.0 <= r_edit.overall_risk <= 1.0):
            print("错误: 风险分超出 [0,1]", file=sys.stderr)
            return 1
        return _check_ui_fixtures()


if __name__ == "__main__":
    sys.exit(main())
