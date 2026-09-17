"""Fixture test: Photoshopped mobile receipts must rank above real ones."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.engine import (
    VERDICT_LIKELY_ORIGINAL,
    VERDICT_LIKELY_TAMPERED,
    VERDICT_UNCERTAIN,
    analyze_image,
)

FIXTURES = ROOT / "tests" / "fixtures"
REALS = [FIXTURES / f"receipt_real_{i}.jpg" for i in (1, 2, 3)]
FAKES = [FIXTURES / f"receipt_fake_{i}.jpg" for i in (1, 2, 3)]


class ReceiptFixtureRankingTest(unittest.TestCase):
    def test_fixtures_exist(self):
        for p in REALS + FAKES:
            self.assertTrue(p.is_file(), f"missing {p}")

    def test_fakes_rank_above_reals(self):
        real_results = [analyze_image(p) for p in REALS]
        fake_results = [analyze_image(p) for p in FAKES]
        real_risks = [r.overall_risk for r in real_results]
        fake_risks = [r.overall_risk for r in fake_results]

        for label, r in zip(
            [f"real{i}" for i in (1, 2, 3)] + [f"fake{i}" for i in (1, 2, 3)],
            real_results + fake_results,
        ):
            print(f"{label}: {r.verdict} risk={r.overall_risk:.3f}")

        mean_gap = sum(fake_risks) / 3 - sum(real_risks) / 3
        self.assertGreaterEqual(
            mean_gap,
            0.12,
            f"mean(fake)-mean(real)={mean_gap:.3f} < 0.12",
        )

        max_real = max(real_risks)
        min_fake = min(fake_risks)
        margin = min_fake - max_real
        ordered = min_fake > max_real and margin >= 0.05
        pairwise = all(
            fr > rr + 0.05 for fr in fake_risks for rr in real_risks
        )
        self.assertTrue(
            ordered or pairwise,
            f"ranking failed: max_real={max_real:.3f} min_fake={min_fake:.3f} "
            f"margin={margin:.3f}",
        )

        # Reals must not be labeled tampered unless all fakes are too and ranking holds
        real_tampered = [r for r in real_results if r.verdict == VERDICT_LIKELY_TAMPERED]
        if real_tampered:
            self.assertTrue(
                all(r.verdict == VERDICT_LIKELY_TAMPERED for r in fake_results),
                "a real receipt was labeled 可能被加工 without all fakes also being so",
            )

        for r in real_results:
            self.assertNotEqual(
                r.verdict,
                VERDICT_LIKELY_TAMPERED,
                f"real receipt unexpectedly 可能被加工 (risk={r.overall_risk:.3f})",
            )

        # Prefer fakes clearly elevated
        for r in fake_results:
            self.assertNotEqual(
                r.verdict,
                VERDICT_LIKELY_ORIGINAL,
                f"fake receipt labeled 可能原图 (risk={r.overall_risk:.3f})",
            )
            self.assertGreaterEqual(r.overall_risk, max_real + 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
