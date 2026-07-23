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
from src.metrics import attach_token_premiums, metrics_for_language, rows_to_dicts
from src.tokenizers_registry import TokenizerSpec, _tiktoken_surface

UNITS = ("byte", "grapheme")
SIZES = (8000, 16000, 32000)
FOCUS_LANGS = (
    "ory_Orya",
    "amh_Ethi",
    "arz_Arab",
    "ary_Arab",
    "quy_Latn",
    "grn_Latn",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/bpe"),
        help="Directory containing trained tokenizer subfolders",
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


def load_sweep_specs(artifact_dir: Path) -> dict[str, TokenizerSpec]:
    specs: dict[str, TokenizerSpec] = {}
    for unit in UNITS:
        for size in SIZES:
            tid = f"bpe_{unit}_{size_label(size)}"
            subdir = artifact_dir / f"{unit}_{size_label(size)}"
            specs[tid] = load_bpe_spec(
                subdir,
                tokenizer_id=tid,
                name=f"BPE {unit} {size_label(size)} (o200k pretok)",
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


def build_ab_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for size in SIZES:
        sl = size_label(size)
        byte_id = f"bpe_byte_{sl}"
        graph_id = f"bpe_grapheme_{sl}"
        for metric in ("fertility", "chars_per_token", "token_premium", "stfr", "strr"):
            byte_val = macro_means(df[df["tokenizer_id"] == byte_id], metric)
            graph_val = macro_means(df[df["tokenizer_id"] == graph_id], metric)
            b = float(byte_val.get(byte_id, float("nan")))
            g = float(graph_val.get(graph_id, float("nan")))
            rows.append(
                {
                    "vocab_size": size,
                    "metric": metric,
                    "byte": b,
                    "grapheme": g,
                    "delta_grapheme_minus_byte": g - b,
                    "pct_change": ((g - b) / b * 100.0) if b else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_focus_ab(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    focus = df[df["language"].isin(FOCUS_LANGS)]
    for size in SIZES:
        sl = size_label(size)
        byte_id = f"bpe_byte_{sl}"
        graph_id = f"bpe_grapheme_{sl}"
        for lang in FOCUS_LANGS:
            for metric in ("fertility", "token_premium", "stfr", "strr"):
                b_row = focus[(focus["tokenizer_id"] == byte_id) & (focus["language"] == lang)]
                g_row = focus[(focus["tokenizer_id"] == graph_id) & (focus["language"] == lang)]
                if b_row.empty or g_row.empty:
                    continue
                b = float(b_row.iloc[0][metric]) if pd.notna(b_row.iloc[0][metric]) else float("nan")
                g = float(g_row.iloc[0][metric]) if pd.notna(g_row.iloc[0][metric]) else float("nan")
                rows.append(
                    {
                        "vocab_size": size,
                        "language": lang,
                        "metric": metric,
                        "byte": b,
                        "grapheme": g,
                        "delta": g - b,
                    }
                )
    return pd.DataFrame(rows)


def per_size_tables(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ("fertility", "chars_per_token", "token_premium", "stfr", "strr")
    for size in SIZES:
        sl = size_label(size)
        subset = df[
            df["tokenizer_id"].isin(
                [f"bpe_byte_{sl}", f"bpe_grapheme_{sl}", "o200k"]
            )
        ]
        for metric in metrics:
            wide = subset.pivot(index="language", columns="tokenizer_id", values=metric)
            wide.to_csv(out_dir / f"table_{metric}_{sl}.csv")


def plot_gap_vs_vocab(ab_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for metric, ax, title in (
        ("token_premium", axes[0], "Token premium (macro mean)"),
        ("stfr", axes[1], "STFR (macro mean)"),
    ):
        sub = ab_summary[ab_summary["metric"] == metric].sort_values("vocab_size")
        xs = [s // 1000 for s in sub["vocab_size"]]
        ax.plot(xs, sub["byte"], marker="o", label="byte BPE")
        ax.plot(xs, sub["grapheme"], marker="s", label="grapheme BPE")
        ax.set_xlabel("Vocab size (thousands)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Baseline vs grapheme gap vs vocab size (FLORES devtest)")
    fig.tight_layout()
    fig.savefig(out_dir / "gap_vs_vocab_size.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir: Path = args.artifact_dir
    out_dir = args.out_dir or artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES devtest...")
    sentences = load_flores_sentences(split="devtest")
    if args.max_sentences:
        sentences = {k: v[: args.max_sentences] for k, v in sentences.items()}
    n = len(next(iter(sentences.values())))
    print(f"Evaluating on {len(sentences)} languages × {n} sentences")

    specs = load_sweep_specs(artifact_dir)
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

    ab_summary = build_ab_summary(df)
    focus_ab = build_focus_ab(df)
    ab_summary.to_csv(out_dir / "ab_summary_macro.csv", index=False)
    focus_ab.to_csv(out_dir / "ab_summary_focus_langs.csv", index=False)
    per_size_tables(df, out_dir)
    plot_gap_vs_vocab(ab_summary, out_dir)

    print("\nMacro A/B delta (grapheme - byte):")
    for size in SIZES:
        sl = size_label(size)
        print(f"\n  {sl}:")
        chunk = ab_summary[ab_summary["vocab_size"] == size]
        for _, r in chunk.iterrows():
            print(
                f"    {r['metric']:16s}  byte={r['byte']:.4f}  "
                f"grapheme={r['grapheme']:.4f}  delta={r['delta_grapheme_minus_byte']:+.4f}"
            )

    print(f"\nWrote results to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
