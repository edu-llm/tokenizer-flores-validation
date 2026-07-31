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
    def test_two_arm_geometric_mean(self) -> None:
        shared = shared_premiums(
            {
                "bpe": {"eng_Latn": 1.0, "swh_Latn": 8.0},
                "superbpe": {"eng_Latn": 1.0, "swh_Latn": 2.0},
            }
        )
        self.assertAlmostEqual(shared["eng_Latn"], 1.0)
        self.assertAlmostEqual(shared["swh_Latn"], geometric_mean([8.0, 2.0]))
        self.assertAlmostEqual(shared["swh_Latn"], 4.0)

    def test_missing_language_in_one_arm_is_an_error(self) -> None:
        with self.assertRaises(KeyError):
            shared_premiums(
                {
                    "bpe": {"eng_Latn": 1.0, "swh_Latn": 8.0},
                    "superbpe": {"eng_Latn": 1.0},
                }
            )

    def test_damp_and_freeze_threshold(self) -> None:
        prior = {"eng_Latn": 0.5, "swh_Latn": 0.5}
        target = target_token_shares({"eng_Latn": 1.0, "swh_Latn": 3.0})
        updated = damp_shares(prior, target, alpha=0.5)
        self.assertAlmostEqual(sum(updated.values()), 1.0)
        self.assertLess(max_share_delta(prior, updated), 0.5)


if __name__ == "__main__":
    unittest.main()
