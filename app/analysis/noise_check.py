"""Noise / residual inconsistency across image blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class NoiseCheckResult:
    score: float  # 0..1 higher = more inconsistent noise
    findings: List[str] = field(default_factory=list)
    global_noise: float = 0.0
    block_cv: float = 0.0  # coefficient of variation of block noise


def _highpass_residual(gray: np.ndarray) -> np.ndarray:
    """Simple residual via blur subtraction (noise-ish high frequency)."""
    # gray float32 HxW
    from PIL import Image as PILImage

    pil = PILImage.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=1.2))
    blur_arr = np.asarray(blurred, dtype=np.float32)
    return gray - blur_arr


def check_noise(
    image_path: Path,
    img: Optional[Image.Image] = None,
    block_size: int = 64,
) -> NoiseCheckResult:
    """
    Measure high-frequency residual energy per block; inconsistency may
    indicate splicing or region-wise processing.
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        residual = _highpass_residual(gray)
        # Local energy = mean abs residual
        h, w = residual.shape
        bs = block_size
        energies: List[float] = []
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block = residual[y : y + bs, x : x + bs]
                energies.append(float(np.mean(np.abs(block))))

        if len(energies) < 4:
            # Tiny image — use smaller blocks
            bs = max(16, min(h, w) // 4)
            energies = []
            for y in range(0, max(1, h - bs + 1), bs):
                for x in range(0, max(1, w - bs + 1), bs):
                    block = residual[y : y + bs, x : x + bs]
                    energies.append(float(np.mean(np.abs(block))))

        arr = np.array(energies, dtype=np.float64) if energies else np.array([0.0])
        global_noise = float(np.mean(np.abs(residual)))
        mean_e = float(arr.mean()) if arr.size else 0.0
        std_e = float(arr.std()) if arr.size else 0.0
        cv = (std_e / mean_e) if mean_e > 1e-6 else 0.0

        # Natural photos: moderate noise, relatively stable CV (~0.15-0.45)
        # Heavy smoothing / mixed regions: very low global noise OR high CV
        score = 0.0
        if global_noise < 1.2:
            findings.append(f"整体高频残差偏低（{global_noise:.2f}），可能经过强降噪或过度平滑")
            score += 0.35
        elif global_noise > 12.0:
            findings.append(f"整体高频残差偏高（{global_noise:.2f}），可能含强锐化或压缩噪声")
            score += 0.2

        if cv > 0.55:
            findings.append(
                f"各区块噪声水平差异较大（变异系数={cv:.2f}），"
                "可能存在拼接或局部处理"
            )
            score += min(0.55, (cv - 0.55) * 1.2 + 0.25)
        elif cv < 0.08 and mean_e > 0.5:
            findings.append(f"噪声分布异常均匀（CV={cv:.2f}），偶见于生成图或强统一滤镜")
            score += 0.2
        else:
            findings.append(
                f"噪声分布大致一致（全局={global_noise:.2f}, CV={cv:.2f}）"
            )

        score = max(0.0, min(1.0, score))
        return NoiseCheckResult(
            score=score,
            findings=findings,
            global_noise=global_noise,
            block_cv=cv,
        )
    finally:
        if close_after:
            img.close()
