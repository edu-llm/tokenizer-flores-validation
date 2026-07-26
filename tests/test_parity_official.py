from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.parity_official import (
    encode_token_bytes,
    export_official_bpe_artifacts,
    load_lang_text_dir,
    train_parity_official,
)


class ParityOfficialTests(unittest.TestCase):
    def test_export_and_train_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            dev = root / "dev"
            out = root / "parity"
            train.mkdir()
            dev.mkdir()
            (train / "eng_Latn.txt").write_text(
                "hello world\nhello there\n",
                encoding="utf-8",
            )
            (train / "amh_Ethi.txt").write_text(
                "ሰላም አለም\nሰላም ነው\n",
                encoding="utf-8",
            )
            (dev / "eng_Latn.txt").write_text("hello world\n", encoding="utf-8")
            (dev / "amh_Ethi.txt").write_text("ሰላም አለም\n", encoding="utf-8")

            meta = train_parity_official(
                train_by_lang=load_lang_text_dir(train),
                dev_by_lang=load_lang_text_dir(dev),
                target_vocab_size=300,
                output_dir=out,
                train_bytes=64,
                train_files=["eng_Latn.txt", "amh_Ethi.txt"],
            )
            self.assertEqual(meta["arm"], "parity")
            self.assertTrue((out / "vocab.json").is_file())
            self.assertTrue((out / "merges.txt").is_file())
            vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(vocab), 256)
            self.assertEqual(encode_token_bytes(b"a"), "a")


if __name__ == "__main__":
    unittest.main()
