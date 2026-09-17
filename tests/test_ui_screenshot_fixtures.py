"""Fixture test: edited UI screenshot must score higher risk than untouched."""

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
EDITED = FIXTURES / "ui_bet_edited.png"
ORIGINAL = FIXTURES / "ui_bet_original.png"


class UiScreenshotFixtureTest(unittest.TestCase):
    def test_fixtures_exist(self):
        self.assertTrue(EDITED.is_file(), f"missing {EDITED}")
        self.assertTrue(ORIGINAL.is_file(), f"missing {ORIGINAL}")

    def test_edited_risk_higher_than_original(self):
        r_edit = analyze_image(EDITED)
        r_orig = analyze_image(ORIGINAL)
        delta = r_edit.overall_risk - r_orig.overall_risk
        print(
            f"\nedited: {r_edit.verdict} risk={r_edit.overall_risk:.3f} "
            f"conf={r_edit.confidence:.3f}"
        )
        print(
            f"original: {r_orig.verdict} risk={r_orig.overall_risk:.3f} "
            f"conf={r_orig.confidence:.3f}"
        )
        print(f"delta={delta:.3f}")
        for name, r in (("edited", r_edit), ("original", r_orig)):
            for c in r.checks:
                print(f"  [{name}] {c.name}={c.score:.3f}")

        self.assertGreaterEqual(
            delta,
            0.15,
            f"edited-original risk delta {delta:.3f} < 0.15",
        )
        same_stuck = (
            r_edit.verdict == VERDICT_UNCERTAIN
            and r_orig.verdict == VERDICT_UNCERTAIN
            and abs(r_edit.overall_risk - r_orig.overall_risk) < 0.05
        )
        self.assertFalse(same_stuck, "both fixtures stuck at identical 不确定")
        self.assertNotEqual(
            r_edit.verdict,
            r_orig.verdict,
            "edited and original should not share the same verdict",
        )
        self.assertNotEqual(r_edit.verdict, VERDICT_LIKELY_ORIGINAL)
        self.assertNotEqual(r_orig.verdict, VERDICT_LIKELY_TAMPERED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
