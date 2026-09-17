"""Aggregate forensic heuristics into a single analysis result."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from app.analysis.ela_check import ElaCheckResult, check_ela
from app.analysis.exif_check import ExifCheckResult, check_exif
from app.analysis.noise_check import NoiseCheckResult, check_noise


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
                    "extras": {k: v for k, v in c.extras.items() if not hasattr(v, "size")},
                }
                for c in self.checks
            ],
            "metadata_summary": self.metadata_summary,
        }


def _decide_verdict(risk: float) -> tuple[str, float, str]:
    """Map overall risk to Chinese verdict + confidence + short explanation."""
    if risk < 0.32:
        verdict = VERDICT_LIKELY_ORIGINAL
        # confidence grows as risk goes toward 0
        confidence = min(0.9, 0.55 + (0.32 - risk) * 1.1)
        explanation = (
            "各项启发式指标总体偏低，未发现强烈的编辑软件痕迹、"
            "ELA 局部异常或噪声分布明显不一致，更接近未经明显数字加工的图像。"
        )
    elif risk > 0.55:
        verdict = VERDICT_LIKELY_TAMPERED
        confidence = min(0.9, 0.55 + (risk - 0.55) * 1.0)
        explanation = (
            "多项指标偏高（如元数据编辑痕迹、ELA 残差异常或区块噪声不一致），"
            "提示图像可能经过重保存、局部编辑或其他数字处理。"
        )
    else:
        verdict = VERDICT_UNCERTAIN
        # peak uncertainty around mid risk
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
        # Keep a copy for preview / checks that need pixel data after close
        working = img.copy()
        working.format = img.format  # type: ignore[attr-defined]
        preview = working.copy()
        if max(preview.size) > 1200:
            preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

        exif_r: ExifCheckResult = check_exif(path, working)
        ela_r: ElaCheckResult = check_ela(path, working)
        noise_r: NoiseCheckResult = check_noise(path, working)

    # Weighted fusion — ELA & noise matter more for visual tampering;
    # EXIF is strong when editing software is present but weak alone for stripped files
    w_exif, w_ela, w_noise = 0.28, 0.40, 0.32
    overall = (
        w_exif * exif_r.score + w_ela * ela_r.score + w_noise * noise_r.score
    )
    overall = float(max(0.0, min(1.0, overall)))

    verdict, confidence, explanation = _decide_verdict(overall)

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
