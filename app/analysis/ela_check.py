"""Error Level Analysis (ELA) for JPEG re-compression artifacts."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class ElaCheckResult:
    score: float  # 0..1 higher = more anomalous / possible local edit
    findings: List[str] = field(default_factory=list)
    ela_image: Optional[Image.Image] = None  # RGB visualization
    mean_diff: float = 0.0
    std_diff: float = 0.0
    max_diff: float = 0.0
    block_cv: float = 0.0
    applicable: bool = True


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img.copy()
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            background.paste(img, mask=img.split()[-1])
            return background
    return img.convert("RGB")


def compute_ela(
    img: Image.Image,
    quality: int = 90,
    scale: float = 15.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Re-save at given JPEG quality and measure absolute difference.
    Returns (visualization RGB image, float32 difference map HxWx3).
    """
    rgb = _to_rgb(img)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    orig = np.asarray(rgb, dtype=np.float32)
    comp = np.asarray(compressed, dtype=np.float32)
    diff = np.abs(orig - comp)

    vis = np.clip(diff * scale, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(vis, mode="RGB")
    ela_img = ImageEnhance.Brightness(ela_img).enhance(1.1)
    ela_img = ImageEnhance.Contrast(ela_img).enhance(1.3)
    return ela_img, diff


def check_ela(
    image_path: Path,
    img: Optional[Image.Image] = None,
    quality: int = 90,
) -> ElaCheckResult:
    """
    ELA anomaly score emphasizing spatial inconsistency (local edits),
    not raw residual magnitude (which is high for any detailed first-save JPEG).
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        fmt = (img.format or Path(image_path).suffix.lstrip(".").upper() or "").upper()
        applicable = fmt in ("JPEG", "JPG", "") or Path(image_path).suffix.lower() in (
            ".jpg",
            ".jpeg",
        )

        ela_img, diff = compute_ela(img, quality=quality)
        mag = diff.mean(axis=2)
        mean_d = float(mag.mean())
        std_d = float(mag.std())
        max_d = float(mag.max())

        h, w = mag.shape
        bh, bw = 32, 32
        block_means: List[float] = []
        for y in range(0, max(h - bh, 1), bh):
            for x in range(0, max(w - bw, 1), bw):
                block_means.append(float(mag[y : y + bh, x : x + bw].mean()))
        block_arr = np.asarray(block_means, dtype=np.float64) if block_means else np.array([0.0])
        block_std = float(block_arr.std()) if block_arr.size else 0.0
        block_mean = float(block_arr.mean()) if block_arr.size else 0.0
        block_cv = (block_std / block_mean) if block_mean > 1e-6 else 0.0

        # Hotspot: top 10% blocks vs median
        if block_arr.size >= 4:
            p90 = float(np.percentile(block_arr, 90))
            p50 = float(np.percentile(block_arr, 50))
            hotspot_ratio = (p90 / p50) if p50 > 1e-6 else 1.0
        else:
            hotspot_ratio = 1.0

        # Score: spatial inconsistency dominates
        score = 0.0
        score += min(0.50, max(0.0, (block_cv - 0.18) * 1.6))
        score += min(0.35, max(0.0, (hotspot_ratio - 1.25) * 0.7))
        # Mild contribution from very high local peaks relative to mean
        peak_ratio = (max_d / mean_d) if mean_d > 1e-6 else 1.0
        score += min(0.25, max(0.0, (peak_ratio - 6.0) * 0.04))

        if not applicable:
            findings.append("非 JPEG 格式：ELA 参考价值有限，已按弱信号计入")
            score *= 0.55

        if block_cv < 0.22 and hotspot_ratio < 1.35:
            findings.append(
                f"ELA 空间分布较均匀（块 CV={block_cv:.2f}, 热点比={hotspot_ratio:.2f}），"
                f"残差均值={mean_d:.2f}，较符合整图一致压缩特征"
            )
        elif block_cv > 0.40 or hotspot_ratio > 1.8:
            findings.append(
                f"ELA 存在局部高差异区域（块 CV={block_cv:.2f}, 热点比={hotspot_ratio:.2f}），"
                "可能经历局部编辑或来源不一致的拼贴"
            )
            score = max(score, 0.55)
        else:
            findings.append(
                f"ELA 轻度空间不均（均值={mean_d:.2f}, 块 CV={block_cv:.2f}, "
                f"热点比={hotspot_ratio:.2f}），可能经过局部处理或混合压缩"
            )

        score = max(0.0, min(1.0, score))
        return ElaCheckResult(
            score=score,
            findings=findings,
            ela_image=ela_img,
            mean_diff=mean_d,
            std_diff=std_d,
            max_diff=max_d,
            block_cv=block_cv,
            applicable=applicable,
        )
    finally:
        if close_after:
            img.close()
