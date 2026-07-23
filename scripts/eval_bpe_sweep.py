#!/usr/bin/env python3
"""Evaluate BPE sweep arms on FLORES devtest; emit A/B tables and gap plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpe_encoder import load_bpe_spec
from src.load_flores import CONTINENT, LANGUAGES, REFERENCE_LANG, load_flores_sentences
from src.metrics import (
    attach_token_premiums,
    gini_for_tokenizer,
    metrics_for_language,
    rows_to_dicts,
)
from src.tokenizers_registry import TokenizerSpec, _tiktoken_surface

DEFAULT_UNITS = ("byte", "grapheme")
SIZES = (8000, 16000, 32000)
FOCUS_LANGS = (
    "ory_Orya",
    "amh_Ethi",
    "arz_Arab",
    "ary_Arab",
    "quy_Latn",
    "grn_Latn",
)

# Subdirectory naming per unit (grapheme_constrained -> gconstr)
UNIT_DIR = {
    "byte": "byte",
    "grapheme": "grapheme",
    "grapheme_constrained": "gconstr",
    "parity": "parity",
}

# Tokenizer id prefix per unit
UNIT_ID = {
    "byte": "byte",
    "grapheme": "grapheme",
    "grapheme_constrained": "gconstr",
    "parity": "parity",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/bpe"),
        help="Directory containing trained tokenizer subfolders",
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional separate directory for baseline arm artifacts",
    )
    p.add_argument(
        "--baseline-unit",
        default="byte",
        choices=tuple(UNIT_DIR.keys()),
        help="Baseline arm unit name (default: byte)",
    )
    p.add_argument(
        "--compare-unit",
        default="grapheme",
        choices=tuple(UNIT_DIR.keys()),
        help="Comparison arm unit name (default: grapheme)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to --artifact-dir)",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap per language for smoke tests",
    )
    return p.parse_args(argv)


def size_label(size: int) -> str:
    return f"{size // 1000}k"


def tokenizer_id(unit: str, size: int) -> str:
    return f"bpe_{UNIT_ID[unit]}_{size_label(size)}"


def artifact_subdir(unit: str, size: int) -> str:
    return f"{UNIT_DIR[unit]}_{size_label(size)}"


def load_arm_specs(
    artifact_dir: Path,
    unit: str,
) -> dict[str, TokenizerSpec]:
    specs: dict[str, TokenizerSpec] = {}
    for size in SIZES:
        tid = tokenizer_id(unit, size)
        subdir = artifact_dir / artifact_subdir(unit, size)
        display_unit = UNIT_ID[unit]
        specs[tid] = load_bpe_spec(
            subdir,
            tokenizer_id=tid,
            name=f"BPE {display_unit} {size_label(size)} (o200k pretok)",
        )
    return specs


def o200k_reference_spec() -> TokenizerSpec:
    enc = tiktoken.get_encoding("o200k_base")
    return TokenizerSpec(
        id="o200k",
        name="OpenAI o200k_base (reference)",
        source="o200k_base",
        is_frontier=True,
        encode=lambda text: enc.encode(text),
        surface=_tiktoken_surface(enc),
        leading_space_for_words=True,
    )


def macro_means(df: pd.DataFrame, metric: str) -> pd.Series:
    """Macro-average a metric across languages (skip NaN / None)."""
    tmp = df.dropna(subset=[metric])
    return tmp.groupby("tokenizer_id")[metric].mean()


def build_ab_summary(
    df: pd.DataFrame,
    rows,
    *,
    baseline_unit: str,
    compare_unit: str,
) -> pd.DataFrame:
    rows_out: list[dict] = []
    baseline_key = UNIT_ID[baseline_unit]
    compare_key = UNIT_ID[compare_unit]
    for size in SIZES:
        sl = size_label(size)
        base_id = f"bpe_{baseline_key}_{sl}"
        cmp_id = f"bpe_{compare_key}_{sl}"
        for metric in ("fertility", "chars_per_token", "token_premium", "stfr", "strr"):
            base_val = macro_means(df[df["tokenizer_id"] == base_id], metric)
            cmp_val = macro_means(df[df["tokenizer_id"] == cmp_id], metric)
            b = float(base_val.get(base_id, float("nan")))
            c = float(cmp_val.get(cmp_id, float("nan")))
            rows_out.append(
                {
                    "vocab_size": size,
                    "metric": metric,
                    baseline_key: b,
                    compare_key: c,
                    f"delta_{compare_key}_minus_{baseline_key}": c - b,
                    "pct_change": ((c - b) / b * 100.0) if b else float("nan"),
                }
            )
        # Gini of tokens-per-line (paper primary fairness metric)
        b_gini = gini_for_tokenizer(rows, base_id)
        c_gini = gini_for_tokenizer(rows, cmp_id)
        rows_out.append(
            {
                "vocab_size": size,
                "metric": "gini",
                baseline_key: b_gini,
                compare_key: c_gini,
                f"delta_{compare_key}_minus_{baseline_key}": c_gini - b_gini,
                "pct_change": (
                    ((c_gini - b_gini) / b_gini * 100.0) if b_gini else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows_out)


def build_focus_ab(
    df: pd.DataFrame,
    *,
    baseline_unit: str,
    compare_unit: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    focus = df[df["language"].isin(FOCUS_LANGS)]
    baseline_key = UNIT_ID[baseline_unit]
    compare_key = UNIT_ID[compare_unit]
    for size in SIZES:
        sl = size_label(size)
        base_id = f"bpe_{baseline_key}_{sl}"
        cmp_id = f"bpe_{compare_key}_{sl}"
        for lang in FOCUS_LANGS:
            for metric in ("fertility", "token_premium", "stfr", "strr"):
                b_row = focus[(focus["tokenizer_id"] == base_id) & (focus["language"] == lang)]
                c_row = focus[(focus["tokenizer_id"] == cmp_id) & (focus["language"] == lang)]
                if b_row.empty or c_row.empty:
                    continue
                b = float(b_row.iloc[0][metric]) if pd.notna(b_row.iloc[0][metric]) else float("nan")
                c = float(c_row.iloc[0][metric]) if pd.notna(c_row.iloc[0][metric]) else float("nan")
                rows.append(
                    {
                        "vocab_size": size,
                        "language": lang,
                        "metric": metric,
                        baseline_key: b,
                        compare_key: c,
                        "delta": c - b,
                    }
                )
    return pd.DataFrame(rows)


def per_size_tables(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    baseline_unit: str,
    compare_unit: str,
) -> None:
    metrics = ("fertility", "chars_per_token", "token_premium", "stfr", "strr")
    baseline_key = UNIT_ID[baseline_unit]
    compare_key = UNIT_ID[compare_unit]
    for size in SIZES:
        sl = size_label(size)
        subset = df[
            df["tokenizer_id"].isin(
                [f"bpe_{baseline_key}_{sl}", f"bpe_{compare_key}_{sl}", "o200k"]
            )
        ]
        for metric in metrics:
            wide = subset.pivot(index="language", columns="tokenizer_id", values=metric)
            wide.to_csv(out_dir / f"table_{metric}_{sl}.csv")


def plot_gap_vs_vocab(
    ab_summary: pd.DataFrame,
    out_dir: Path,
    *,
    baseline_unit: str,
    compare_unit: str,
) -> None:
    baseline_key = UNIT_ID[baseline_unit]
    compare_key = UNIT_ID[compare_unit]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for metric, ax, title in (
        ("token_premium", axes[0], "Token premium (macro mean)"),
        ("stfr", axes[1], "STFR (macro mean)"),
        ("gini", axes[2], "Gini (tokens/line)"),
    ):
        sub = ab_summary[ab_summary["metric"] == metric].sort_values("vocab_size")
        if sub.empty:
            continue
        xs = [s // 1000 for s in sub["vocab_size"]]
        ax.plot(xs, sub[baseline_key], marker="o", label=f"{baseline_key} BPE")
        ax.plot(xs, sub[compare_key], marker="s", label=f"{compare_key} BPE")
        ax.set_xlabel("Vocab size (thousands)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{compare_key} vs {baseline_key} gap vs vocab size (FLORES devtest)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gap_vs_vocab_size.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir: Path = args.artifact_dir
    baseline_dir = args.baseline_dir or artifact_dir
    out_dir = args.out_dir or artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_unit = args.baseline_unit
    compare_unit = args.compare_unit
    baseline_key = UNIT_ID[baseline_unit]
    compare_key = UNIT_ID[compare_unit]

    print("Loading FLORES devtest...")
    sentences = load_flores_sentences(split="devtest")
    if args.max_sentences:
        sentences = {k: v[: args.max_sentences] for k, v in sentences.items()}
    n = len(next(iter(sentences.values())))
    print(f"Evaluating on {len(sentences)} languages × {n} sentences")

    specs: dict[str, TokenizerSpec] = {}
    specs.update(load_arm_specs(baseline_dir, baseline_unit))
    if compare_unit != baseline_unit:
        specs.update(load_arm_specs(artifact_dir, compare_unit))
    specs["o200k"] = o200k_reference_spec()

    rows = []
    for tid, spec in specs.items():
        print(f"Encoding with {tid}...")
        for lang, texts in sentences.items():
            rows.append(metrics_for_language(spec, lang, texts))

    rows = attach_token_premiums(rows)
    records = rows_to_dicts(rows)
    df = pd.DataFrame(records)
    df["language_name"] = df["language"].map(LANGUAGES)
    df["continent"] = df["language"].map(CONTINENT)

    metrics_path = out_dir / "eval_metrics.json"
    csv_path = out_dir / "eval_metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    df.to_csv(csv_path, index=False)

    ab_summary = build_ab_summary(
        df, rows, baseline_unit=baseline_unit, compare_unit=compare_unit
    )
    focus_ab = build_focus_ab(
        df, baseline_unit=baseline_unit, compare_unit=compare_unit
    )
    ab_summary.to_csv(out_dir / "ab_summary_macro.csv", index=False)
    focus_ab.to_csv(out_dir / "ab_summary_focus_langs.csv", index=False)
    per_size_tables(
        df, out_dir, baseline_unit=baseline_unit, compare_unit=compare_unit
    )
    plot_gap_vs_vocab(
        ab_summary, out_dir, baseline_unit=baseline_unit, compare_unit=compare_unit
    )

    print(f"\nMacro A/B delta ({compare_key} - {baseline_key}):")
    for size in SIZES:
        sl = size_label(size)
        print(f"\n  {sl}:")
        chunk = ab_summary[ab_summary["vocab_size"] == size]
        for _, r in chunk.iterrows():
            print(
                f"    {r['metric']:16s}  {baseline_key}={r[baseline_key]:.4f}  "
                f"{compare_key}={r[compare_key]:.4f}  "
                f"delta={r[f'delta_{compare_key}_minus_{baseline_key}']:+.4f}"
            )

    print(f"\nWrote results to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
