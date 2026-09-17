"""Aggregate forensic heuristics into a single analysis result."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from app.analysis.ela_check import ElaCheckResult, check_ela
from app.analysis.exif_check import ExifCheckResult, check_exif
from app.analysis.noise_check import NoiseCheckResult, check_noise
from app.analysis.screenshot_check import ScreenshotCheckResult, check_screenshot
from app.analysis.ui_resample_check import UiResampleCheckResult, check_ui_resample
from app.analysis.ui_text_check import UiTextCheckResult, check_ui_text
from app.analysis.amount_edit_check import AmountEditCheckResult, check_amount_edit


VERDICT_LIKELY_ORIGINAL = "可能原图"
VERDICT_LIKELY_TAMPERED = "可能被加工"
VERDICT_UNCERTAIN = "不确定"

DISCLAIMER = (
    "本工具仅基于启发式规则与图像统计特征给出参考判断，"
    "不能作为司法鉴定或 100% 确定结论。请结合其他证据综合研判。"
)


@dataclass
class CheckDetail:
    name: str
    score: float
    findings: List[str]
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    path: str
    verdict: str
    confidence: float  # 0..1 confidence in the verdict label
    overall_risk: float  # 0..1 tampering / processing risk
    explanation: str
    disclaimer: str
    checks: List[CheckDetail]
    ela_image: Optional[Image.Image] = None
    original_preview: Optional[Image.Image] = None
    metadata_summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "overall_risk": self.overall_risk,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
            "checks": [
                {
                    "name": c.name,
                    "score": c.score,
                    "findings": c.findings,
                    "extras": {
                        k: v
                        for k, v in c.extras.items()
                        if not hasattr(v, "size")
                    },
                }
                for c in self.checks
            ],
            "metadata_summary": self.metadata_summary,
        }


def _decide_verdict(risk: float) -> tuple[str, float, str]:
    """Map overall risk to Chinese verdict + confidence + short explanation."""
    if risk < 0.32:
        verdict = VERDICT_LIKELY_ORIGINAL
        confidence = min(0.9, 0.55 + (0.32 - risk) * 1.1)
        explanation = (
            "各项启发式指标总体偏低，未发现强烈的编辑软件痕迹、"
            "ELA 局部异常、噪声分布明显不一致或界面文字局部异常，"
            "更接近未经明显数字加工的图像。"
        )
    elif risk > 0.55:
        verdict = VERDICT_LIKELY_TAMPERED
        confidence = min(0.9, 0.55 + (risk - 0.55) * 1.0)
        explanation = (
            "多项指标偏高（如元数据编辑痕迹、ELA 残差异常、"
            "界面文字抗锯齿/局部残差不一致或重采样痕迹），"
            "提示图像可能经过重保存、局部编辑或其他数字处理。"
        )
    else:
        verdict = VERDICT_UNCERTAIN
        confidence = 0.45 + abs(risk - 0.43) * 0.3
        confidence = min(0.7, confidence)
        explanation = (
            "指标处于中间地带：可能是正常二次压缩、社交平台转码，"
            "也可能是轻度编辑。建议结合拍摄来源与其他证据判断。"
        )
    return verdict, float(confidence), explanation


def analyze_image(image_path: str | Path) -> AnalysisResult:
    """Run all forensic checks on an image file."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到图像文件: {path}")

    with Image.open(path) as img:
        img.load()
        working = img.copy()
        working.format = img.format  # type: ignore[attr-defined]
        preview = working.copy()
        if max(preview.size) > 1200:
            preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

        exif_r: ExifCheckResult = check_exif(path, working)
        ela_r: ElaCheckResult = check_ela(path, working)
        noise_r: NoiseCheckResult = check_noise(path, working)
        shot_r: ScreenshotCheckResult = check_screenshot(path, working)
        ui_r: UiTextCheckResult = check_ui_text(
            path, working, screenshot_likeness=shot_r.score
        )
        rs_r: UiResampleCheckResult = check_ui_resample(
            path, working, screenshot_likeness=shot_r.score
        )
        amt_r: AmountEditCheckResult = check_amount_edit(
            path, working, screenshot_likeness=shot_r.score
        )

    # Screenshot-aware fusion:
    # For UI screenshots, classic photo ELA/EXIF are weak; up-weight local
    # text AA / patch consistency and resample cues.
    if shot_r.is_screenshot or shot_r.score >= 0.50:
        # Mobile UI JPEGs: global ELA/noise fire on ALL authentic bank UIs
        # (text vs flat panels). Emphasize amount-local + AA cues instead.
        w_exif, w_ela, w_noise = 0.04, 0.04, 0.04
        w_ui, w_rs, w_amt = 0.36, 0.10, 0.42
        fusion_note = (
            "截图模式：大幅下调全局 ELA/噪声，上调金额区局部一致性与界面文字检测"
        )
    else:
        w_exif, w_ela, w_noise = 0.24, 0.30, 0.26
        w_ui, w_rs, w_amt = 0.05, 0.05, 0.10
        fusion_note = "照片模式：以 EXIF / ELA / 噪声为主，界面/金额项弱权重"

    overall = (
        w_exif * exif_r.score
        + w_ela * ela_r.score
        + w_noise * noise_r.score
        + w_ui * ui_r.score
        + w_rs * rs_r.score
        + w_amt * amt_r.score
    )
    # Strong amount-paste evidence on screenshots should clear the tamper band
    if (shot_r.is_screenshot or shot_r.score >= 0.50) and amt_r.score >= 0.85:
        overall = max(overall, 0.58)
        fusion_note += "；金额区强异常，提升综合风险"
    overall = float(max(0.0, min(1.0, overall)))

    verdict, confidence, explanation = _decide_verdict(overall)
    explanation = fusion_note + "。" + explanation

    checks = [
        CheckDetail(
            name="EXIF / 元数据",
            score=exif_r.score,
            findings=exif_r.findings,
            extras={"has_exif": exif_r.has_exif},
        ),
        CheckDetail(
            name="误差级分析 (ELA)",
            score=ela_r.score,
            findings=ela_r.findings,
            extras={
                "mean_diff": round(ela_r.mean_diff, 3),
                "std_diff": round(ela_r.std_diff, 3),
                "max_diff": round(ela_r.max_diff, 3),
                "block_cv": round(ela_r.block_cv, 3),
                "applicable": ela_r.applicable,
            },
        ),
        CheckDetail(
            name="噪声 / 残差一致性",
            score=noise_r.score,
            findings=noise_r.findings,
            extras={
                "global_noise": round(noise_r.global_noise, 3),
                "block_cv": round(noise_r.block_cv, 3),
            },
        ),
        CheckDetail(
            name="截图特征",
            score=shot_r.score,
            findings=shot_r.findings,
            extras={
                "flat_frac": round(shot_r.flat_frac, 3),
                "color_count_est": shot_r.color_count_est,
                "edge_band_density": round(shot_r.edge_band_density, 3),
                "is_screenshot": shot_r.is_screenshot,
            },
        ),
        CheckDetail(
            name="界面文字局部一致性",
            score=ui_r.score,
            findings=ui_r.findings,
            extras={
                "gray_aa_frac": round(ui_r.gray_aa_frac, 3),
                "cleartype_frac": round(ui_r.cleartype_frac, 3),
                "aa_strip_cv": round(ui_r.aa_strip_cv, 3),
                "peer_sharp_cv": round(ui_r.peer_sharp_cv, 3),
                "bg_residue_cv": round(ui_r.bg_residue_cv, 3),
                "rgba_opaque": ui_r.rgba_opaque,
            },
        ),
        CheckDetail(
            name="重采样 / 双渲染痕迹",
            score=rs_r.score,
            findings=rs_r.findings,
            extras={
                "island_rate": round(rs_r.island_rate, 3),
                "ring_score": round(rs_r.ring_score, 3),
                "hf_cv": round(rs_r.hf_cv, 3),
            },
        ),
        CheckDetail(
            name="金额区局部编辑痕迹",
            score=amt_r.score,
            findings=amt_r.findings,
            extras={
                "dig_neigh_ratio": round(amt_r.dig_neigh_ratio, 3),
                "dig_ela": round(amt_r.dig_ela, 3),
                "neigh_ela": round(amt_r.neigh_ela, 3),
                "chroma_flat_med": round(amt_r.chroma_flat_med, 3),
                "chroma_flat_mean": round(amt_r.chroma_flat_mean, 3),
                "n_flat_panels": amt_r.n_flat_panels,
                "amount_y": amt_r.amount_y,
            },
        ),
    ]

    return AnalysisResult(
        path=str(path.resolve()),
        verdict=verdict,
        confidence=confidence,
        overall_risk=overall,
        explanation=explanation,
        disclaimer=DISCLAIMER,
        checks=checks,
        ela_image=ela_r.ela_image,
        original_preview=preview,
        metadata_summary=exif_r.metadata_summary,
    )
