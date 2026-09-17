"""Mobile receipt amount-region edit cues (local ELA vs neighbors, chroma wipe)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import io

import numpy as np
from PIL import Image


@dataclass
class AmountEditCheckResult:
    score: float  # 0..1 higher = more likely amount/UI paste edit
    findings: List[str] = field(default_factory=list)
    dig_neigh_ratio: float = 0.0
    dig_ela: float = 0.0
    neigh_ela: float = 0.0
    chroma_flat_med: float = 0.0
    chroma_flat_mean: float = 0.0
    n_flat_panels: int = 0
    amount_y: int = 0


def _ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    comp = Image.open(buf).convert("RGB")
    diff = np.abs(
        np.asarray(rgb, dtype=np.float32) - np.asarray(comp, dtype=np.float32)
    )
    return diff.mean(axis=2)


def _find_amount_row(gray: np.ndarray) -> int:
    """Pick a vertical position likely containing large amount digits."""
    h, w = gray.shape
    gx = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    row_score = gx.mean(axis=1) * gray.std(axis=1)
    # Prefer upper-mid of portrait UI screenshots (status bar / chrome excluded)
    y0 = int(h * 0.08)
    y1 = int(h * 0.55) if h > w else int(h * 0.70)
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 2, min(y1, h))
    local = row_score[y0:y1]
    if local.size == 0:
        return h // 3
    return int(np.argmax(local) + y0)


def _flat_panel_chroma(
    rgb: np.ndarray, gray: np.ndarray
) -> Tuple[float, float, int]:
    """Chroma std in near-flat UI panels. Paste/heal often wipes chroma noise."""
    h, w = gray.shape
    cr = rgb[:, :, 0] - gray
    cb = rgb[:, :, 2] - gray
    chroma = np.sqrt(cr * cr + cb * cb)
    bs = 24
    vals: List[float] = []
    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            g = gray[y : y + bs, x : x + bs]
            s = float(g.std())
            mean = float(g.mean())
            if 0.8 < s < 5.5 and 30 < mean < 250:
                vals.append(float(chroma[y : y + bs, x : x + bs].std()))
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), len(vals)


def _amount_ela_ratio(
    ela: np.ndarray, gray: np.ndarray, y_best: int
) -> Tuple[float, float, float]:
    h, w = gray.shape
    y0 = max(0, y_best - 30)
    y1 = min(h, y_best + 40)
    band = ela[y0:y1]
    band_g = gray[y0:y1]
    cstd = band_g.std(axis=0)
    cx0, cx1 = int(w * 0.20), int(w * 0.80)
    if cx1 <= cx0 + 8:
        return 1.0, float(ela.mean()), float(ela.mean())
    thr = float(np.percentile(cstd[cx0:cx1], 65))
    dig = np.zeros(w, dtype=bool)
    dig[cx0:cx1] = cstd[cx0:cx1] >= thr
    if not dig.any():
        return 1.0, float(ela.mean()), float(ela.mean())

    dig_ela = float(band[:, dig].mean())
    neigh: List[float] = []
    for yy in list(range(max(0, y0 - 100), y0, 12)) + list(
        range(y1, min(h, y1 + 100), 12)
    ):
        neigh.append(float(ela[yy : yy + 10, cx0:cx1].mean()))
    neigh_ela = float(np.median(neigh)) if neigh else float(ela.mean())
    ratio = dig_ela / (neigh_ela + 1e-6)
    return ratio, dig_ela, neigh_ela


def check_amount_edit(
    image_path: Path,
    img: Optional[Image.Image] = None,
    screenshot_likeness: float = 0.0,
) -> AmountEditCheckResult:
    """
    Detect mobile bank/Alipay-style amount paste edits on JPEG screenshots.

    Strong cues observed on labeled fixtures:
    - Amount digit band ELA >> neighboring flat UI ELA (paste / re-encode)
    - Near-flat UI panels with wiped chroma noise (heal / paint / re-export)
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        work = img
        if max(img.size) > 1600:
            work = img.copy()
            work.thumbnail((1200, 1200), Image.Resampling.BILINEAR)

        rgb = np.asarray(work.convert("RGB"), dtype=np.float32)
        gray = (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        )
        ela = _ela_map(work, quality=90)
        y_best = _find_amount_row(gray)
        ratio, dig_ela, neigh_ela = _amount_ela_ratio(ela, gray, y_best)
        chroma_mean, chroma_med, n_flat = _flat_panel_chroma(rgb, gray)

        score = 0.0

        # Dig/neigh ELA: reals ~0.8–1.0, fakes ~3–6 on labeled set
        if ratio >= 2.2:
            boost = min(0.55, (ratio - 2.2) * 0.12 + 0.28)
            score += boost
            findings.append(
                f"金额区 ELA 相对邻域偏高（比值 {ratio:.2f}，"
                f"金额={dig_ela:.3f} / 邻域={neigh_ela:.3f}），"
                "常见于金额数字被粘贴或局部重编码"
            )
        elif ratio >= 1.6:
            score += min(0.22, (ratio - 1.6) * 0.25 + 0.08)
            findings.append(
                f"金额区 ELA 轻度高于邻域（比值 {ratio:.2f}），"
                "需结合其他项判断"
            )
        else:
            findings.append(
                f"金额区与邻域 ELA 较一致（比值 {ratio:.2f}）"
            )

        # Chroma wipe in flat panels: reals ~0.7–1.1, fakes ~0.0–0.3
        if n_flat >= 12 and chroma_med < 0.25:
            boost = min(0.45, (0.25 - chroma_med) * 1.2 + 0.22)
            score += boost
            findings.append(
                f"平坦界面色块色度噪声被抹平（中位数 {chroma_med:.3f}，"
                f"{n_flat} 个面板），常见于修图/局部填充后重导出"
            )
        elif n_flat >= 12 and chroma_mean < 0.45:
            score += min(0.20, (0.45 - chroma_mean) * 0.5 + 0.06)
            findings.append(
                f"平坦色块色度噪声偏低（均值 {chroma_mean:.3f}），"
                "可能经过平滑或重压缩"
            )
        elif n_flat >= 8:
            findings.append(
                f"平坦色块保留一定色度纹理（中位数 {chroma_med:.3f}）"
            )

        # Neighbor ELA very low with high ratio: painted backdrop around digits
        if neigh_ela < 0.35 and ratio >= 2.0:
            score += 0.12
            findings.append(
                f"金额邻域 ELA 过低（{neigh_ela:.3f}）且金额热点突出，"
                "疑似金额周围被填充/清理"
            )

        # Desktop PNG table screenshots are handled by ClearType/AA checks;
        # amount-band heuristics are tuned for mobile JPEG receipts.
        suffix = Path(image_path).suffix.lower()
        fmt = (getattr(img, "format", None) or "").upper()
        is_jpeg = fmt in ("JPEG", "JPG") or suffix in (".jpg", ".jpeg")
        if not is_jpeg:
            score *= 0.30
            findings.append("非 JPEG：金额区局部编辑项降权（桌面 PNG 更依赖文字 AA 检测）")

        if screenshot_likeness >= 0.50:
            score *= 1.08
        elif screenshot_likeness < 0.30:
            # Photo / non-UI: still allow chroma+ratio but dampen
            score *= 0.55
            findings.append("非截图场景：金额区编辑项降权")

        if score < 0.12 and not any("偏高" in f or "抹平" in f for f in findings):
            findings.append("未发现明显的金额区粘贴/色度抹平痕迹")

        score = float(max(0.0, min(1.0, score)))
        return AmountEditCheckResult(
            score=score,
            findings=findings,
            dig_neigh_ratio=float(ratio),
            dig_ela=float(dig_ela),
            neigh_ela=float(neigh_ela),
            chroma_flat_med=float(chroma_med),
            chroma_flat_mean=float(chroma_mean),
            n_flat_panels=int(n_flat),
            amount_y=int(y_best),
        )
    finally:
        if close_after:
            img.close()
