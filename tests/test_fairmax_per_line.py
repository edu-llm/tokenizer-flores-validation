from __future__ import annotations

import unittest
from collections import Counter

from src.bpe_train import _corpus_token_count, train_parity_bpe_from_lang_freqs


def _word(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in text.encode("utf-8"))


class FairMaxPerLineTests(unittest.TestCase):
    def test_unequal_cr_dev_sizes_score_by_tokens_per_line(self) -> None:
        # Short CR-dev has higher tokens/line; long CR-dev has more raw tokens.
        # Fair-max must follow the per-line rate, not raw CTC.
        rich = _word("qq")  # 2 tokens each
        poor = _word("r")  # 1 token each
        short_dev = Counter({rich: 10})  # 20 tokens / 1 line = 20
        long_dev = Counter({poor: 100})  # 100 tokens / 50 lines = 2
        self.assertGreater(
            _corpus_token_count(long_dev),
            _corpus_token_count(short_dev),
        )
        self.assertGreater(
            _corpus_token_count(short_dev) / 1.0,
            _corpus_token_count(long_dev) / 50.0,
        )

        train = {
            "aaa_short": Counter({_word("xy"): 100}),
            "zzz_long": Counter({_word("xz"): 100}),
        }
        vocab, merges = train_parity_bpe_from_lang_freqs(
            train,
            {"aaa_short": short_dev, "zzz_long": long_dev},
            target_vocab_size=260,
            dev_n_lines={"aaa_short": 1, "zzz_long": 50},
        )
        self.assertGreaterEqual(len(vocab), 256)
        self.assertTrue(merges)
        # Worst-rate language is aaa_short → first merge is its top train pair.
        self.assertEqual(merges[0], (b"x", b"y"))


if __name__ == "__main__":
    unittest.main()
