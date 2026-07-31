from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.official_bpe_encode import load_lang_text_dir


class LoadLangTextDirTests(unittest.TestCase):
    """Coverage for the per-language loader shared by the premium computation.

    Previously exercised via ``tests/test_parity_official.py``; that module went
    away with the Plan A parity arm, but ``scripts/compute_arm_premiums.py``
    still depends on this loader.
    """

    def test_reads_one_language_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "eng_Latn.txt").write_text(
                "hello world\nhello there\n", encoding="utf-8"
            )
            (root / "amh_Ethi.txt").write_text("ሰላም አለም\n", encoding="utf-8")

            by_lang = load_lang_text_dir(root)

        self.assertEqual(sorted(by_lang), ["amh_Ethi", "eng_Latn"])
        self.assertEqual(by_lang["eng_Latn"], ["hello world", "hello there"])
        self.assertEqual(by_lang["amh_Ethi"], ["ሰላም አለም"])

    def test_skips_blank_lines_and_unknown_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "eng_Latn.txt").write_text("a\n\n   \nb\n", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            by_lang = load_lang_text_dir(root)

        self.assertEqual(by_lang, {"eng_Latn": ["a", "b"]})

    def test_reads_jsonl_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "eng_Latn.jsonl").write_text(
                json.dumps({"text": "first"}) + "\n" + json.dumps({"text": "second"}) + "\n",
                encoding="utf-8",
            )

            by_lang = load_lang_text_dir(root)

        self.assertEqual(by_lang, {"eng_Latn": ["first", "second"]})

    def test_rejects_missing_and_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                load_lang_text_dir(root / "absent")
            with self.assertRaises(ValueError):
                load_lang_text_dir(root)


if __name__ == "__main__":
    unittest.main()
