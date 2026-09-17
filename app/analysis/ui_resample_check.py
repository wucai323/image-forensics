"""Resample / double-render cues inside flat UI cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class UiResampleCheckResult:
    score: float
    findings: List[str] = field(default_factory=list)
    island_rate: float = 0.0
    ring_score: float = 0.0
    hf_cv: float = 0.0


def check_ui_resample(
    image_path: Path,
    img: Optional[Image.Image] = None,
    screenshot_likeness: float = 0.0,
) -> UiResampleCheckResult:
    """
    Look for isolated high-frequency islands and mild bicubic-style ringing
    inside otherwise flat UI cells — typical of pasted / re-rasterized digits.
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        gray_img = img.convert("L")
        if max(gray_img.size) > 1800:
            gray_img = gray_img.copy()
            gray_img.thumbnail((1800, 1800), Image.Resampling.BILINEAR)

        gray = np.asarray(gray_img, dtype=np.float32)
        h, w = gray.shape

        lap_img = gray_img.filter(
            ImageFilter.Kernel(
                (3, 3),
                [0, -1, 0, -1, 4, -1, 0, -1, 0],
                scale=1,
                offset=128,
            )
        )
        lap = np.abs(np.asarray(lap_img, dtype=np.float32) - 128.0)

        bs = 32
        island = 0
        text_flat_cells = 0
        cell_hf: List[float] = []
        ring_hits = 0
        ring_trials = 0

        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                cell = gray[y : y + bs, x : x + bs]
                cell_lap = lap[y : y + bs, x : x + bs]
                std = float(cell.std())
                if std >= 12.0:
                    continue
                dark_frac = float((cell < 95).mean())
                # Only cells that contain some ink (text/numbers)
                if dark_frac < 0.02:
                    continue
                text_flat_cells += 1
                hf = float(cell_lap.mean())
                cell_hf.append(hf)

                thr = max(25.0, float(np.percentile(cell_lap, 85)))
                hot = cell_lap > thr
                hot_frac = float(hot.mean())
                if float(cell_lap.max()) > 55 and 0.01 < hot_frac < 0.18:
                    island += 1

                core = cell < 100
                if core.any() and core.mean() < 0.45:
                    dil = core.copy()
                    dil[1:, :] |= core[:-1, :]
                    dil[:-1, :] |= core[1:, :]
                    dil[:, 1:] |= core[:, :-1]
                    dil[:, :-1] |= core[:, 1:]
                    ring = dil & ~core
                    ring_trials += 1
                    if ring.any():
                        bg = cell[cell > 200]
                        ring_vals = cell[ring]
                        if bg.size > 5 and ring_vals.size > 3:
                            if float(ring_vals.std()) > float(bg.std()) + 4.0:
                                ring_hits += 1

        island_rate = island / max(text_flat_cells, 1)
        ring_score = ring_hits / max(ring_trials, 1)
        hf_cv = 0.0
        if len(cell_hf) >= 8:
            a = np.asarray(cell_hf, dtype=np.float64)
            # robust CV via median
            med = float(np.median(a)) + 1e-6
            hf_cv = float(np.median(np.abs(a - med)) * 1.4826 / med)

        score = 0.0
        if island_rate > 0.05:
            score += min(0.40, island_rate * 3.0)
            findings.append(
                f"含文字的平坦单元格内存在孤立高频岛"
                f"（占比 {island_rate:.1%}，{island}/{text_flat_cells}），"
                "可能为局部重绘/贴入的文字或数字"
            )
        if ring_score > 0.15:
            score += min(0.30, ring_score * 1.4)
            findings.append(
                f"文字边缘出现疑似重采样振铃（命中率 {ring_score:.0%}），"
                "常见于二次缩放或粘贴后的双三次插值"
            )
        if hf_cv > 0.55 and screenshot_likeness >= 0.45 and text_flat_cells >= 8:
            score += min(0.20, (hf_cv - 0.55) * 0.4 + 0.06)
            findings.append(
                f"含文字平坦块的高频能量分布不均（稳健 CV={hf_cv:.2f}），"
                "界面截图中可能提示局部处理"
            )

        if screenshot_likeness < 0.30:
            score *= 0.40
            findings.append("非截图场景：重采样/双渲染项降权")
        elif screenshot_likeness >= 0.50:
            score *= 1.08

        if score < 0.12 and not any(("高频岛" in f or "振铃" in f) for f in findings):
            findings.append(
                f"未发现明显的单元格高频岛或振铃（岛占比 {island_rate:.1%}，"
                f"振铃命中 {ring_score:.0%}）"
            )

        score = float(max(0.0, min(1.0, score)))
        return UiResampleCheckResult(
            score=score,
            findings=findings,
            island_rate=float(island_rate),
            ring_score=float(ring_score),
            hf_cv=float(hf_cv),
        )
    finally:
        if close_after:
            img.close()
