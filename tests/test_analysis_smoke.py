"""Minimal smoke test (unittest) wrapping scripts/self_check logic."""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SelfCheckTest(unittest.TestCase):
    def test_self_check_exits_zero(self):
        path = ROOT / "scripts" / "self_check.py"
        # Run as __main__
        ns = runpy.run_path(str(path), run_name="not_main")
        code = ns["main"]()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    unittest.main()
