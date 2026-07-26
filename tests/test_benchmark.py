from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.benchmark import (
    build_round_robin_corpus,
    run_monitored_command,
    sha256_file,
)


class BenchmarkCorpusTests(unittest.TestCase):
    def test_round_robin_corpus_is_bounded_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            second.write_text("uno\ndos\ntres\n", encoding="utf-8")

            output_a = root / "sample-a.txt"
            output_b = root / "sample-b.txt"
            manifest_a = build_round_robin_corpus([first, second], output_a, 24)
            manifest_b = build_round_robin_corpus([second, first], output_b, 24)

            self.assertLessEqual(manifest_a["actual_bytes"], 24)
            self.assertEqual(output_a.read_bytes().decode("utf-8"), output_a.read_text("utf-8"))
            self.assertEqual(manifest_a["sha256"], sha256_file(output_a))
            self.assertEqual(manifest_a["sha256"], manifest_b["sha256"])
            self.assertTrue(all(item["records_written"] > 0 for item in manifest_a["sources"]))

    def test_monitored_command_writes_portable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "result.json"
            log_path = root / "run.log"
            result = run_monitored_command(
                name="unit-smoke",
                command=[sys.executable, "-c", "print('ok')"],
                result_path=result_path,
                log_path=log_path,
                input_bytes=100,
                projection_bytes=1000,
                poll_seconds=0.01,
                min_available_gb=0.01,
                metadata={"arm": "test"},
            )

            saved = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(saved["schema_version"], 1)
            self.assertEqual(saved["metadata"]["arm"], "test")
            self.assertEqual(saved["projection"]["target_bytes"], 1000)
            self.assertIn("ok", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

