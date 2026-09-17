"""Detect UI / webpage screenshot characteristics (flat fills, limited palette)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class ScreenshotCheckResult:
    score: float  # 0..1 likelihood this is a UI/screenshot (not a photo)
    findings: List[str] = field(default_factory=list)
    flat_frac: float = 0.0
    color_count_est: int = 0
    edge_band_density: float = 0.0
    is_screenshot: bool = False


def check_screenshot(
    image_path: Path,
    img: Optional[Image.Image] = None,
) -> ScreenshotCheckResult:
    """
    Heuristic UI-screenshot detector.

    Large flat color regions + limited unique colors + edge energy
    concentrated in horizontal text/table bands → screenshot-like.
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        rgb = img.convert("RGB")
        max_side = max(rgb.size)
        if max_side > 1600:
            rgb = rgb.copy()
            rgb.thumbnail((1600, 1600), Image.Resampling.BILINEAR)

        arr = np.asarray(rgb, dtype=np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        h, w = gray.shape

        bs = 16
        flat = 0
        total = 0
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                total += 1
                if float(gray[y : y + bs, x : x + bs].std()) < 2.5:
                    flat += 1
        flat_frac = flat / max(total, 1)

        sample = arr.reshape(-1, 3)[:: max(1, (h * w) // 80000)]
        q = (sample // 12).astype(np.int16)
        keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
        color_count_est = int(np.unique(keys).size)

        gy = np.zeros_like(gray)
        gx = np.zeros_like(gray)
        gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
        gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
        edge = (gx + gy) > 28
        edge_frac = float(edge.mean())

        row_e = edge.mean(axis=1)
        if row_e.sum() > 1e-9:
            order = np.argsort(row_e)[::-1]
            top_n = max(1, h // 5)
            band_share = float(row_e[order[:top_n]].sum() / row_e.sum())
        else:
            band_share = 0.0

        score = 0.0
        # Flat fills (tables / panels). Dense tables may be only moderately flat.
        if flat_frac > 0.18:
            score += min(0.40, (flat_frac - 0.18) * 1.0)
        # Limited palette — UI chrome + text
        if color_count_est < 1600:
            score += min(0.35, (1600 - color_count_est) / 1600 * 0.45)
        # Horizontal text/table banding
        if band_share > 0.32 and edge_frac > 0.015:
            score += min(0.40, (band_share - 0.32) * 1.05)
        # Extra: very strong banding alone (dense tables)
        if band_share > 0.55 and edge_frac > 0.04:
            score += 0.12

        if flat_frac < 0.08 and color_count_est > 2200:
            score *= 0.35
            findings.append(
                f"平坦色块较少（{flat_frac:.0%}）且颜色丰富（约 {color_count_est}），"
                "更接近照片而非界面截图"
            )
        elif score >= 0.50:
            findings.append(
                f"疑似界面/网页截图：平坦色块占比 {flat_frac:.0%}，"
                f"估计颜色数约 {color_count_est}，边缘呈水平带状（带占比 {band_share:.0%}）"
            )
        elif score >= 0.30:
            findings.append(
                f"部分截图特征（平坦块 {flat_frac:.0%}，颜色约 {color_count_est}，"
                f"带状边缘 {band_share:.0%}），截图置信中等"
            )
        else:
            findings.append(
                f"截图特征不明显（平坦块 {flat_frac:.0%}，颜色约 {color_count_est}）"
            )

        score = float(max(0.0, min(1.0, score)))
        return ScreenshotCheckResult(
            score=score,
            findings=findings,
            flat_frac=float(flat_frac),
            color_count_est=color_count_est,
            edge_band_density=float(band_share),
            is_screenshot=score >= 0.50,
        )
    finally:
        if close_after:
            img.close()
