"""EXIF / metadata forensic heuristics via Pillow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from PIL.ExifTags import TAGS


EDITING_SOFTWARE_KEYWORDS = (
    "photoshop",
    "gimp",
    "lightroom",
    "affinity",
    "snapseed",
    "picsart",
    "meitu",
    "美图",
    "pixelmator",
    "capture one",
    "darktable",
    "rawtherapee",
    "acdsee",
    "paint.net",
    "corel",
    "illustrator",
    "canva",
    "fotor",
    "photopea",
    "luminar",
    "dxo",
)

CAMERA_FIELDS = (
    "Make",
    "Model",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "LensModel",
)


@dataclass
class ExifCheckResult:
    score: float  # 0..1 higher = more suspicious
    findings: List[str] = field(default_factory=list)
    metadata_summary: Dict[str, str] = field(default_factory=dict)
    has_exif: bool = False


def _decode_exif(img: Image.Image) -> Dict[str, Any]:
    raw = img.getexif()
    if not raw:
        return {}
    out: Dict[str, Any] = {}
    for tag_id, value in raw.items():
        name = TAGS.get(tag_id, str(tag_id))
        try:
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = repr(value)
            out[name] = value
        except Exception:
            out[name] = str(value)
    return out


def _parse_exif_datetime(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    text = str(s).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def check_exif(image_path: Path, img: Optional[Image.Image] = None) -> ExifCheckResult:
    """Heuristic EXIF/metadata risk score in [0, 1]."""
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    score = 0.0
    summary: Dict[str, str] = {}

    try:
        exif = _decode_exif(img)
        fmt = (img.format or Path(image_path).suffix.lstrip(".").upper() or "UNKNOWN")
        summary["格式"] = str(fmt)
        summary["尺寸"] = f"{img.width}×{img.height}"
        summary["模式"] = str(img.mode)

        software = exif.get("Software") or exif.get("ProcessingSoftware")
        is_editor_software = False
        if software:
            summary["软件"] = str(software)[:120]
            sw_lower = str(software).lower()
            for kw in EDITING_SOFTWARE_KEYWORDS:
                if kw.lower() in sw_lower:
                    findings.append(f"元数据含编辑软件痕迹: {software}")
                    score += 0.50
                    is_editor_software = True
                    break

        artist = exif.get("Artist") or exif.get("Copyright")
        if artist:
            summary["作者/版权"] = str(artist)[:80]

        has_exif = bool(exif)
        if not has_exif:
            # Common for exports/screenshots — weak signal alone
            findings.append("几乎无 EXIF 信息（常见于截图、导出图或经工具剥离元数据）")
            score += 0.18
        else:
            present_camera = [
                f for f in CAMERA_FIELDS if f in exif and exif[f] not in (None, "", b"")
            ]
            missing = [f for f in ("Make", "Model", "DateTimeOriginal") if f not in present_camera]
            # Only penalize missing camera fields when Software suggests an editor,
            # or when some but not all camera fields exist (partial strip)
            if is_editor_software and missing:
                findings.append(f"编辑软件标记下仍缺少相机字段: {', '.join(missing)}")
                score += 0.10 * min(3, len(missing))
            elif len(present_camera) == 0 and not software:
                findings.append("有 EXIF 但无典型相机拍摄字段")
                score += 0.12
            elif missing and len(present_camera) > 0:
                findings.append(f"部分相机字段缺失: {', '.join(missing)}")
                score += 0.08 * len(missing)

            make = exif.get("Make")
            model = exif.get("Model")
            if make:
                summary["相机制造商"] = str(make)[:60]
            if model:
                summary["相机型号"] = str(model)[:60]

            dto = _parse_exif_datetime(exif.get("DateTimeOriginal"))
            dtd = _parse_exif_datetime(exif.get("DateTimeDigitized"))
            dt = _parse_exif_datetime(exif.get("DateTime"))
            if dto:
                summary["拍摄时间"] = dto.strftime("%Y-%m-%d %H:%M:%S")
            if dt and dto and abs((dt - dto).total_seconds()) > 86400 * 2:
                findings.append(
                    f"修改时间与拍摄时间相差较大 "
                    f"({dt.strftime('%Y-%m-%d')} vs {dto.strftime('%Y-%m-%d')})"
                )
                score += 0.25
            if dto and dtd and abs((dto - dtd).total_seconds()) > 3600 * 24:
                findings.append("DateTimeOriginal 与 DateTimeDigitized 不一致")
                score += 0.15

        if fmt.upper() in ("PNG", "WEBP", "BMP", "GIF") and not has_exif:
            score = max(0.0, score - 0.08)
            findings.append(f"{fmt} 格式本身较少保留相机 EXIF，缺失属常见情况")

        if not findings:
            findings.append("未发现明显元数据异常")

        score = max(0.0, min(1.0, score))
        return ExifCheckResult(
            score=score,
            findings=findings,
            metadata_summary=summary,
            has_exif=has_exif,
        )
    finally:
        if close_after:
            img.close()
