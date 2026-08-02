#!/usr/bin/env python3
"""Train one arm with gigatoken and write the official artifact set.

Runs as a **subprocess** of ``run_gigatoken_tokenizer_benchmark.py``, for two
reasons:

1. ``src/benchmark.py:run_monitored_command`` samples peak RSS over a process
   tree. Calling ``train_superbpe`` in-process would leave the trainer's memory
   unmeasured, and CLAUDE.md requires memory be measured at every tier and
   never extrapolated.
2. gigatoken lives in its own Python 3.13 + Rust environment, separate from
   both the repo env and the 3.11 ``.venv-benchmark`` that the official
   trainer's patched ``tokenizers`` fork pins.

Arguments arrive as a single JSON blob on argv[1] so the parent controls the
whole spec and the result JSON can record it verbatim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# gigatoken byte-level tokens are stored in the GPT-2 unicode alphabet, the
# same encoding merges.txt and vocab.json use.
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def build_pre_tokenizer(arm: str, stage1_regex: str):
    """The pre-tokenizer this arm must be *encoded* with.

    It has to match what the arm was trained with, or the measured premiums
    describe a tokenizer nobody trained. The SuperBPE arm lifts whitespace
    entirely (gigatoken's stage 2 applies no regex), which is exactly
    ``ByteLevel(use_regex=False)`` -- and is also the only form gigatoken's
    fast ``Superword`` encoder can load.
    """
    byte_level = pre_tokenizers.ByteLevel(
        add_prefix_space=False, trim_offsets=True, use_regex=False
    )
    if arm == "superbpe":
        return byte_level
    # BPE arm: the stage-1 split, mirroring the official trainer's own
    # construction in .cache/superbpe/utils.py:57-68.
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(stage1_regex), behavior="isolated", invert=False
            ),
            byte_level,
        ]
    )


def main() -> int:
    spec = json.loads(sys.argv[1])
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(spec["corpus_file"])

    import gigatoken as gt

    separator = spec["separator"].encode("utf-8")
    started = time.time()
    if spec["arm"] == "superbpe":
        vocab, merges = gt.train_superbpe(
            corpus,
            vocab_size=spec["vocab_size"],
            transition_point=spec["transition_vocab_size"],
            special_tokens=[],
            separator=separator,
            max_unit_len=spec["max_unit_len"],
            pretokenizer=spec["pretokenizer"],
        )
    else:
        vocab, merges = gt.train_bpe(
            corpus,
            vocab_size=spec["vocab_size"],
            special_tokens=[],
            separator=separator,
            pretokenizer=spec["pretokenizer"],
        )
    train_seconds = time.time() - started
    print(f"gigatoken train time: {train_seconds:.3f}s", flush=True)

    b2u = gpt2_bytes_to_unicode()

    def enc(token: bytes) -> str:
        return "".join(b2u[b] for b in token)

    # vocab.json / merges.txt in the official layout, so
    # verify_official_tokenizer_pair.py reads them unchanged.
    hf_vocab = {enc(tok): tid for tid, tok in vocab.items()}
    hf_merges = [(enc(a), enc(b)) for a, b in merges]

    (out / "vocab.json").write_text(
        json.dumps(hf_vocab, ensure_ascii=False), encoding="utf-8"
    )
    (out / "merges.txt").write_text(
        "\n".join(["#version: 0.2", *(f"{a} {b}" for a, b in hf_merges)]) + "\n",
        encoding="utf-8",
    )

    tokenizer = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    tokenizer.pre_tokenizer = build_pre_tokenizer(spec["arm"], spec["stage1_regex"])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(out / "tokenizer.json"))

    # The official trainer records what it trained on in meta.json; mirror the
    # field names so downstream tooling does not have to branch on trainer.
    (out / "meta.json").write_text(
        json.dumps(
            {
                "total_bytes": corpus.stat().st_size,
                "train_files": [str(corpus)],
                "trainer": "gigatoken",
                "pretokenizer": spec["pretokenizer"],
                "separator": spec["separator"],
                "max_unit_len": spec["max_unit_len"],
                "train_seconds": train_seconds,
            },
            indent=5,
        ),
        encoding="utf-8",
    )

    if spec["arm"] == "superbpe":
        # gigatoken's stage 2 continues from stage 1 in-process, so the
        # inherited prefix is exactly the first transition_vocab_size -
        # initial_vocab_size merges of this arm's own list. Writing it lets
        # verify_official_tokenizer_pair.py check the prefix identically for
        # both trainers.
        initial_vocab_size = len(hf_vocab) - len(hf_merges)
        inherited = spec["transition_vocab_size"] - initial_vocab_size
        if inherited <= 0 or inherited > len(hf_merges):
            raise ValueError(
                f"transition vocab must inherit between 1 and all merges; "
                f"initial={initial_vocab_size}, "
                f"transition={spec['transition_vocab_size']}, "
                f"merges={len(hf_merges)}"
            )
        (out / "requested_initial_merges.txt").write_text(
            "\n".join(
                ["#version: 0.2", *(f"{a} {b}" for a, b in hf_merges[:inherited])]
            )
            + "\n",
            encoding="utf-8",
        )

    n_superwords = sum(1 for t in hf_vocab if "Ġ" in t[1:])
    print(
        f"vocab={len(hf_vocab)} merges={len(hf_merges)} superwords={n_superwords}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
