"""Local UI text / digit patch consistency (AA fringe, sharpness, bg residue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class UiTextCheckResult:
    score: float  # 0..1 higher = more local text inconsistency / reprocess cues
    findings: List[str] = field(default_factory=list)
    gray_aa_frac: float = 0.0
    cleartype_frac: float = 0.0
    aa_strip_cv: float = 0.0
    peer_sharp_cv: float = 0.0
    bg_residue_cv: float = 0.0
    rgba_opaque: bool = False


def _black_ink_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Near-black ink (exclude blue/purple UI accents)."""
    return (
        (gray < 75)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 28)
        & (np.abs(r.astype(np.int16) - b.astype(np.int16)) < 28)
        & (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 28)
    )


def _fringe_chromas(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    gray: np.ndarray,
    black: np.ndarray,
    max_samples: int = 9000,
) -> Tuple[np.ndarray, List[float]]:
    """Return fringe |R-B| samples and per-vertical-strip gray-AA fractions."""
    h, w = gray.shape
    ys, xs = np.where(black)
    chromas: List[float] = []
    if len(ys) == 0:
        return np.asarray([], dtype=np.float64), []

    step = max(1, len(ys) // max_samples)
    for y, x in zip(ys[::step], xs[::step]):
        if y < 1 or x < 1 or y >= h - 1 or x >= w - 1:
            continue
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            yy, xx = y + dy, x + dx
            v = gray[yy, xx]
            if 95 < v < 225:
                chromas.append(float(abs(int(r[yy, xx]) - int(b[yy, xx]))))

    strip_w = max(48, w // 20)
    strip_fracs: List[float] = []
    for x0 in range(0, w - strip_w, strip_w):
        m = black[:, x0 : x0 + strip_w]
        if int(m.sum()) < 50:
            continue
        sy, sx = np.where(m)
        local: List[float] = []
        sstep = max(1, len(sy) // 800)
        for y, x in zip(sy[::sstep], sx[::sstep]):
            xa = x0 + int(x)
            if xa < 1 or xa >= w - 1:
                continue
            for dx in (-1, 1):
                v = gray[y, xa + dx]
                if 95 < v < 225:
                    local.append(float(abs(int(r[y, xa + dx]) - int(b[y, xa + dx]))))
        if len(local) >= 40:
            loc = np.asarray(local, dtype=np.float64)
            strip_fracs.append(float((loc < 10).mean()))

    return np.asarray(chromas, dtype=np.float64), strip_fracs


def _peer_patch_stats(gray: np.ndarray) -> Tuple[float, float]:
    """Sharpness CV and background-residue CV across text-like patches."""
    h, w = gray.shape
    laps: List[float] = []
    bg_stds: List[float] = []
    for y in range(40, h - 24, 14):
        for x in range(40, w - 20, 16):
            patch = gray[y : y + 18, x : x + 14]
            dfrac = float((patch < 90).mean())
            lfrac = float((patch > 200).mean())
            if dfrac < 0.08 or dfrac > 0.55 or lfrac < 0.22:
                continue
            lap = float(
                np.abs(
                    patch[1:-1, 1:-1] * 4
                    - patch[:-2, 1:-1]
                    - patch[2:, 1:-1]
                    - patch[1:-1, :-2]
                    - patch[1:-1, 2:]
                ).mean()
            )
            bg = patch[patch > 190]
            if bg.size < 8:
                continue
            laps.append(lap)
            bg_stds.append(float(bg.std()))

    def _cv(vals: List[float]) -> float:
        if len(vals) < 12:
            return 0.0
        a = np.asarray(vals, dtype=np.float64)
        return float(a.std() / (a.mean() + 1e-6))

    return _cv(laps), _cv(bg_stds)


def check_ui_text(
    image_path: Path,
    img: Optional[Image.Image] = None,
    screenshot_likeness: float = 0.0,
) -> UiTextCheckResult:
    """
    Detect UI text/number paste / re-render cues.

    Stronger when image already looks like a screenshot:
    - black-text anti-alias fringe shifted to grayscale (ClearType destroyed)
    - strip-to-strip AA inconsistency
    - peer glyph sharpness / bg micro-residue inconsistency
    - opaque RGBA (common Photoshop PNG export of screenshots)
    """
    close_after = False
    if img is None:
        img = Image.open(image_path)
        close_after = True

    findings: List[str] = []
    try:
        mode = img.mode
        rgba_opaque = False
        if mode in ("RGBA", "LA"):
            alpha = np.asarray(img.split()[-1])
            rgba_opaque = bool(alpha.min() >= 250)
            if rgba_opaque:
                findings.append(
                    "PNG 为不透明 RGBA（常见于 Photoshop 等编辑器导出截图），"
                    "相对原生 RGB 截图更可疑"
                )

        arr = np.asarray(img.convert("RGB"))
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        black = _black_ink_mask(r, g, b, gray)

        chromas, strip_fracs = _fringe_chromas(r, g, b, gray, black)
        if chromas.size >= 80:
            gray_aa_frac = float((chromas < 10).mean())
            cleartype_frac = float((chromas > 25).mean())
            fringe_mean = float(chromas.mean())
        else:
            gray_aa_frac = 0.0
            cleartype_frac = 0.0
            fringe_mean = 0.0

        if strip_fracs:
            sf = np.asarray(strip_fracs, dtype=np.float64)
            aa_strip_cv = float(sf.std() / (sf.mean() + 1e-6))
            aa_strip_std = float(sf.std())
        else:
            aa_strip_cv = 0.0
            aa_strip_std = 0.0

        peer_sharp_cv, bg_residue_cv = _peer_patch_stats(gray)

        score = 0.0

        # Mobile (Android/iOS) screenshots use grayscale AA natively — not ClearType.
        # Only treat gray-AA dominance as suspicious for PNG/desktop exports,
        # mixed AA strips, or when some ClearType remnants remain (partial re-render).
        fmt = (getattr(img, "format", None) or Path(image_path).suffix.lstrip(".").upper() or "")
        is_jpeg = fmt.upper() in ("JPEG", "JPG") or Path(image_path).suffix.lower() in (".jpg", ".jpeg")
        uniform_mobile_aa = (
            is_jpeg
            and gray_aa_frac > 0.70
            and cleartype_frac < 0.15
            and aa_strip_std < 0.15
        )

        if gray_aa_frac > 0.55 and cleartype_frac < 0.35:
            if uniform_mobile_aa:
                findings.append(
                    f"JPEG 截图黑色文字为均匀灰度抗锯齿（灰度AA {gray_aa_frac:.0%}，"
                    f"ClearType 样 {cleartype_frac:.0%}），移动端常见，不作 ClearType 缺失扣分"
                )
            else:
                boost = min(0.42, (gray_aa_frac - 0.55) * 0.9 + 0.18)
                score += boost
                findings.append(
                    f"黑色文字抗锯齿偏灰度（灰度AA占比 {gray_aa_frac:.0%}，"
                    f"ClearType 样占比 {cleartype_frac:.0%}，边缘色差均值 {fringe_mean:.1f}），"
                    "常见于截图被重采样、重绘或编辑器重导出"
                )
        elif cleartype_frac > 0.7:
            findings.append(
                f"黑色文字保留彩色亚像素抗锯齿（ClearType 样占比 {cleartype_frac:.0%}），"
                "更接近原生屏幕截图渲染"
            )
            score = max(0.0, score - 0.08)
        else:
            findings.append(
                f"文字抗锯齿混合或样本不足（灰度AA {gray_aa_frac:.0%}，"
                f"ClearType 样 {cleartype_frac:.0%}）"
            )

        if aa_strip_std > 0.18 and len(strip_fracs) >= 4:
            score += min(0.28, aa_strip_std * 0.7)
            findings.append(
                f"不同竖条区域黑色文字 AA 风格不一致（灰度AA条带标准差 {aa_strip_std:.2f}），"
                "可能存在局部粘贴/重绘数字"
            )

        if peer_sharp_cv > 0.38:
            score += min(0.22, (peer_sharp_cv - 0.38) * 0.8 + 0.08)
            findings.append(
                f"同类文字块清晰度离散偏高（CV={peer_sharp_cv:.2f}），"
                "疑似局部字体/锐度来源不一致"
            )

        if bg_residue_cv > 0.45:
            score += min(0.18, (bg_residue_cv - 0.45) * 0.5 + 0.06)
            findings.append(
                f"文字邻域背景微残差不一致（CV={bg_residue_cv:.2f}），"
                "可能残留贴图或清理底色痕迹"
            )

        if rgba_opaque:
            score += 0.12

        if screenshot_likeness >= 0.55:
            score *= 1.05
        elif screenshot_likeness < 0.30:
            score *= 0.35
            findings.append("非截图场景：界面文字一致性项降权")

        if not findings:
            findings.append("未发现明显的界面文字局部不一致")

        score = float(max(0.0, min(1.0, score)))
        return UiTextCheckResult(
            score=score,
            findings=findings,
            gray_aa_frac=gray_aa_frac,
            cleartype_frac=cleartype_frac,
            aa_strip_cv=aa_strip_cv,
            peer_sharp_cv=peer_sharp_cv,
            bg_residue_cv=bg_residue_cv,
            rgba_opaque=rgba_opaque,
        )
    finally:
        if close_after:
            img.close()
