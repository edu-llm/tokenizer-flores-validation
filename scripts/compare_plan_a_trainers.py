#!/usr/bin/env python3
"""Cross-check the gigatoken and official trainers on the same corpus.

The gate for trusting gigatoken as the Plan A trainer. Trains are done
elsewhere; this reads the finished artifacts and reports, per language:

* **bytes/token** -- compression, higher is better.
* **token premium vs English** -- the quantity Plan A actually ships. It feeds
  the UniMax allocation and the byte<->token conversion for the 20B target, and
  it is a cross-*language* ratio, so a stage-1 pretokenizer difference does
  **not** cancel out of it the way it cancels out of an arm-vs-arm delta.

Evaluated on FLORES **dev**, leaving ``devtest`` held out for the study proper.

Runs the plan's five configurations:

===============================  ===============================================
official-bpe / official-superbpe the reference implementation
gigatoken-bpe / -superbpe        same stage-1 regex (``superbpe_stage1``)
gigatoken-bpe-gpt2               diagnostic: gigatoken's own default stage 1,
                                 whose letter runs exclude ``\\p{M}``
===============================  ===============================================

``official-bpe`` vs ``gigatoken-bpe`` and ``official-superbpe`` vs
``gigatoken-superbpe`` are the parity comparisons. ``gigatoken-bpe`` vs
``gigatoken-bpe-gpt2`` quantifies the Devanagari fragmentation the
``superbpe_stage1`` scheme exists to avoid -- ``hin_Deva`` is the diagnostic
language.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json
from src.load_flores import REFERENCE_LANG, load_flores_sentences
from src.official_bpe_encode import load_official_bpe_tokenizer
from src.plan_a_langs import PLAN_A_CODES

# label -> artifact directory, relative to --tokenizers-root
DEFAULT_CONFIGS: dict[str, str] = {
    "official-bpe": "off_bpe_4k",
    "official-superbpe": "off_superbpe_4k_t3072",
    "gigatoken-bpe": "gt_bpe_4k",
    "gigatoken-superbpe": "gt_superbpe_4k_t3072",
    "gigatoken-bpe-gpt2": "gt_bpe_4k_gpt2",
}

PARITY_PAIRS = [
    ("official-bpe", "gigatoken-bpe"),
    ("official-superbpe", "gigatoken-superbpe"),
]
DIAGNOSTIC_PAIR = ("gigatoken-bpe", "gigatoken-bpe-gpt2")
DIAGNOSTIC_LANG = "hin_Deva"

# Languages allowed to exceed --tolerance on the **SuperBPE arm only**, with
# the mechanism that explains it. This is an exemption from a specific,
# understood cause -- not a blanket loosening. Anything else, on either arm,
# still fails at --tolerance.
#
# The reference STAGE2_REGEX alternative `[^\s\p{L}\p{N}]{2,}` was written to
# catch punctuation runs (`...`, `--`). Devanagari combining marks are neither
# \p{L} nor \p{N}, so a two-mark vowel cluster matches it and becomes a split
# point: official stage 2 cuts `हिन्दी भाषा में कई शब्द हैं` into four pieces
# while leaving English and Swahili whole. gigatoken's stage 2 applies no regex
# and so does not inherit that behaviour, which is why it compresses Hindi
# better. Verified directly with tokenizers.pre_tokenizers.Split.
#
# The gap therefore GROWS with vocab size (-3.13% at 4k, -6.61% at 16k), so it
# must be re-read at every tier and reported, never assumed stable.
STAGE2_DEVIATION_EXPECTED: dict[str, str] = {
    "hin_Deva": (
        "Reference STAGE2_REGEX splits Devanagari multi-mark clusters via "
        "[^\\s\\p{L}\\p{N}]{2,}; gigatoken's regex-free stage 2 does not. "
        "gigatoken compresses Hindi better. Grows with vocab size."
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizers-root", type=Path, default=ROOT / "artifacts" / "plan_a" / "tokenizers"
    )
    p.add_argument(
        "--configs",
        nargs="*",
        metavar="LABEL=SUBDIR",
        help="Override the artifact directory for each label. Directory names "
        "carry the vocab size, so a non-smoke tier must pass these.",
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "plan_a" / "results" / "trainer_cross_check.json")
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Max allowed relative premium difference per language between "
        "corresponding official and gigatoken arms. Tight by design: the "
        "BPE arm agrees to within 0.05%%, so anything looser would stop "
        "detecting a stage-1 regression.",
    )
    p.add_argument(
        "--stage2-deviation-tolerance",
        type=float,
        default=0.15,
        help="Bound for the SuperBPE-arm languages in "
        "STAGE2_DEVIATION_EXPECTED. Generous because the deviation grows "
        "with vocab size; it is a tripwire against the mechanism changing, "
        "not a parity claim.",
    )
    return p.parse_args(argv)


def measure(tokenizer, by_lang: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lang, sentences in by_lang.items():
        tokens = sum(len(tokenizer.encode(s).ids) for s in sentences)
        nbytes = sum(len(s.encode("utf-8")) for s in sentences)
        out[lang] = {
            "tokens": tokens,
            "bytes": nbytes,
            "bytes_per_token": nbytes / tokens if tokens else 0.0,
        }
    ref = out[REFERENCE_LANG]["tokens"]
    if ref <= 0:
        raise ValueError(f"{REFERENCE_LANG} produced no tokens")
    for lang in out:
        out[lang]["token_premium"] = out[lang]["tokens"] / ref
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    configs = dict(DEFAULT_CONFIGS)
    for item in args.configs or []:
        label, _, subdir = item.partition("=")
        if label not in DEFAULT_CONFIGS:
            raise ValueError(
                f"Unknown config label {label!r}; expected one of "
                f"{sorted(DEFAULT_CONFIGS)}"
            )
        configs[label] = subdir

    # Every configuration must be present. Skipping absent ones and still
    # reporting PASS would make an empty run indistinguishable from a clean
    # one -- the gate would be loudest exactly when it had measured nothing.
    missing = {
        label: str(args.tokenizers_root / subdir)
        for label, subdir in configs.items()
        if not (args.tokenizers_root / subdir).is_dir()
    }
    if missing:
        raise FileNotFoundError(
            "Missing tokenizer artifacts, refusing to report a verdict on a "
            f"partial comparison: {json.dumps(missing, indent=2)}\n"
            "Directory names carry the vocab size; pass --configs "
            "LABEL=SUBDIR for a non-smoke tier."
        )

    measurements: dict[str, dict[str, dict[str, float]]] = {}
    by_lang = load_flores_sentences(PLAN_A_CODES, split=args.split)

    for label, subdir in configs.items():
        tokenizer = load_official_bpe_tokenizer(args.tokenizers_root / subdir)
        measurements[label] = measure(tokenizer, by_lang)
        print(f"  measured {label}")

    # --- bytes/token -------------------------------------------------------
    labels = list(measurements)
    print(f"\n=== bytes/token on FLORES {args.split} (higher = better) ===")
    print(f"{'language':12s}" + "".join(f"{l:>21s}" for l in labels))
    for lang in PLAN_A_CODES:
        row = "".join(
            f"{measurements[l][lang]['bytes_per_token']:>21.4f}" for l in labels
        )
        print(f"{lang:12s}{row}")

    # --- token premium -----------------------------------------------------
    print(f"\n=== token premium vs {REFERENCE_LANG} (Plan A's deliverable) ===")
    print(f"{'language':12s}" + "".join(f"{l:>21s}" for l in labels))
    for lang in PLAN_A_CODES:
        row = "".join(f"{measurements[l][lang]['token_premium']:>21.4f}" for l in labels)
        print(f"{lang:12s}{row}")

    # eng_Latn premium must be exactly 1.0 in any premium table.
    reference_failures = {
        label: m[REFERENCE_LANG]["token_premium"]
        for label, m in measurements.items()
        if m[REFERENCE_LANG]["token_premium"] != 1.0
    }

    # --- parity verdict ----------------------------------------------------
    parity: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    documented: dict[str, float] = {}
    worst = 0.0
    for official, giga in PARITY_PAIRS:
        is_superbpe = "superbpe" in giga
        deltas = {}
        for lang in PLAN_A_CODES:
            a = measurements[official][lang]["token_premium"]
            b = measurements[giga][lang]["token_premium"]
            d = (b - a) / a if a else 0.0
            deltas[lang] = d
            exempt = is_superbpe and lang in STAGE2_DEVIATION_EXPECTED
            if exempt:
                documented[lang] = d
                if abs(d) > args.stage2_deviation_tolerance:
                    violations.append(
                        f"{giga}/{lang}: {d:+.2%} exceeds the documented-deviation "
                        f"bound {args.stage2_deviation_tolerance:.0%}; the "
                        "stage-2 mechanism may have changed"
                    )
            else:
                worst = max(worst, abs(d))
                if abs(d) > args.tolerance:
                    violations.append(
                        f"{giga}/{lang}: {d:+.2%} exceeds tolerance "
                        f"{args.tolerance:.0%}"
                    )
        parity[f"{official} vs {giga}"] = deltas

    print("\n=== premium parity: relative difference, gigatoken vs official ===")
    for pair, deltas in parity.items():
        is_superbpe = "superbpe" in pair.split(" vs ")[-1]
        print(f"  {pair}")
        for lang, d in deltas.items():
            if is_superbpe and lang in STAGE2_DEVIATION_EXPECTED:
                note = "  <-- DOCUMENTED DEVIATION (not parity)"
            elif abs(d) > args.tolerance:
                note = "  <-- OVER TOLERANCE"
            else:
                note = ""
            print(f"    {lang:12s} {d:+8.2%}{note}")

    if documented:
        print("\n=== documented stage-2 deviations (NOT parity; must be reported) ===")
        for lang, d in documented.items():
            print(f"  {lang}: {d:+.2%}")
            print(f"    {STAGE2_DEVIATION_EXPECTED[lang]}")
        print(
            "  The premium table is gigatoken-SuperBPE's own. Its SuperBPE-arm\n"
            "  numbers for these languages are NOT directly comparable to\n"
            "  published SuperBPE (Liu et al.) results."
        )

    # --- diagnostic: what the stage-1 fix bought ---------------------------
    diagnostic = {}
    good, bad = DIAGNOSTIC_PAIR
    if good in measurements and bad in measurements:
        print(f"\n=== diagnostic: {good} vs {bad} (stage-1 scheme only) ===")
        for lang in PLAN_A_CODES:
            g = measurements[good][lang]["bytes_per_token"]
            b = measurements[bad][lang]["bytes_per_token"]
            gain = (g - b) / b if b else 0.0
            diagnostic[lang] = gain
            mark = "  <-- diagnostic language" if lang == DIAGNOSTIC_LANG else ""
            print(
                f"    {lang:12s} {b:7.4f} -> {g:7.4f} bytes/token  "
                f"({gain:+.2%}){mark}"
            )

    # A verdict requires that both parity pairs were actually compared.
    if len(parity) != len(PARITY_PAIRS):
        raise AssertionError(
            f"Compared {len(parity)} of {len(PARITY_PAIRS)} parity pairs; "
            "refusing to report a verdict."
        )
    passed = not violations and not reference_failures
    payload = {
        "schema_version": 2,
        "kind": "plan_a_trainer_cross_check",
        "split": args.split,
        "tolerance": args.tolerance,
        "stage2_deviation_tolerance": args.stage2_deviation_tolerance,
        "passed": passed,
        "violations": violations,
        "worst_premium_delta_excluding_documented": worst,
        "documented_stage2_deviations": {
            lang: {"relative_premium_delta": d, "mechanism": STAGE2_DEVIATION_EXPECTED[lang]}
            for lang, d in documented.items()
        },
        "comparability_note": (
            "The SuperBPE-arm premium table is gigatoken-SuperBPE's own. For "
            "languages in documented_stage2_deviations it is NOT directly "
            "comparable to published SuperBPE (Liu et al.) numbers."
        ),
        "reference_premium_failures": reference_failures,
        "configs": {k: str(args.tokenizers_root / v) for k, v in configs.items()},
        "measurements": measurements,
        "premium_parity": parity,
        "stage1_diagnostic_bytes_per_token_gain": diagnostic,
        "notes": [
            "Premium is a cross-language ratio, so a stage-1 pretokenizer "
            "difference does not cancel out of it.",
            "gigatoken stage 2 applies no regex, so its superword space is a "
            "strict superset of the official STAGE2_REGEX. Expect the SuperBPE "
            "arms to differ by more than the BPE arms.",
        ],
    }
    atomic_write_json(args.out, payload)
    print(
        f"\nworst premium delta excluding documented deviations: {worst:.2%} "
        f"(tolerance {args.tolerance:.0%})"
    )
    for v in violations:
        print(f"  VIOLATION {v}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}   -> {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
