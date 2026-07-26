from __future__ import annotations

import unittest

from src.premium_calibration import (
    damp_shares,
    geometric_mean,
    max_share_delta,
    shared_premiums,
    target_token_shares,
)


class PremiumCalibrationTests(unittest.TestCase):
    def test_three_arm_geometric_mean(self) -> None:
        shared = shared_premiums(
            {
                "bpe": {"eng_Latn": 1.0, "amh_Ethi": 8.0},
                "superbpe": {"eng_Latn": 1.0, "amh_Ethi": 2.0},
                "parity": {"eng_Latn": 1.0, "amh_Ethi": 4.0},
            }
        )
        self.assertAlmostEqual(shared["eng_Latn"], 1.0)
        self.assertAlmostEqual(shared["amh_Ethi"], geometric_mean([8.0, 2.0, 4.0]))

    def test_damp_and_freeze_threshold(self) -> None:
        prior = {"eng_Latn": 0.5, "amh_Ethi": 0.5}
        target = target_token_shares({"eng_Latn": 1.0, "amh_Ethi": 3.0})
        updated = damp_shares(prior, target, alpha=0.5)
        self.assertAlmostEqual(sum(updated.values()), 1.0)
        self.assertLess(max_share_delta(prior, updated), 0.5)


if __name__ == "__main__":
    unittest.main()
