"""CLI: run FLORES-200 tokenizer efficiency validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .load_flores import CONTINENT, LANGUAGES, REFERENCE_LANG, load_flores_sentences
from .metrics import (
    attach_token_premiums,
    evaluate_decision_rule,
    metrics_for_language,
    rows_to_dicts,
)
from .tokenizers_registry import load_tokenizers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV/JSON/plots/report inputs",
    )
    p.add_argument(
        "--tokenizers",
        nargs="*",
        default=None,
        help="Subset of tokenizer ids (default: all)",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap for a smoke test",
    )
    return p.parse_args(argv)


def plot_premium_heatmap(df: pd.DataFrame, path: Path) -> None:
    pivot = df.pivot(index="language", columns="tokenizer_id", values="token_premium")
    # Stable language order by continent then code
    order = sorted(pivot.index, key=lambda c: (CONTINENT.get(c, "Z"), c))
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    labels = [f"{LANGUAGES.get(c, c)} ({c})" for c in pivot.index]
    ax.set_yticklabels(labels)
    ax.set_title("Token premium vs English (CTC_lang / CTC_eng)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def continent_means(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["continent"] = tmp["language"].map(CONTINENT)
    # Exclude English from continent means for a cleaner "tax" view, but keep Europe others
    non_en = tmp[tmp["language"] != REFERENCE_LANG]
    return (
        non_en.groupby(["continent", "tokenizer_id"])["token_premium"]
        .mean()
        .reset_index()
        .rename(columns={"token_premium": "mean_token_premium"})
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES-200 sentences...")
    sentences = load_flores_sentences()
    if args.max_sentences:
        sentences = {k: v[: args.max_sentences] for k, v in sentences.items()}
    n = len(next(iter(sentences.values())))
    print(f"Loaded {len(sentences)} languages × {n} sentences")

    print("Loading tokenizers...")
    specs = load_tokenizers(args.tokenizers)

    rows = []
    for tid, spec in specs.items():
        print(f"Encoding with {tid}...")
        for lang, texts in sentences.items():
            rows.append(metrics_for_language(spec, lang, texts))

    rows = attach_token_premiums(rows)
    decision = evaluate_decision_rule(rows)

    records = rows_to_dicts(rows)
    df = pd.DataFrame(records)
    df["language_name"] = df["language"].map(LANGUAGES)
    df["continent"] = df["language"].map(CONTINENT)

    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    decision_path = out_dir / "decision.json"
    heatmap_path = out_dir / "token_premium_heatmap.png"
    continent_path = out_dir / "continent_mean_premium.csv"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with decision_path.open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    plot_premium_heatmap(df, heatmap_path)
    cont = continent_means(df)
    cont.to_csv(continent_path, index=False)

    # Wide tables for the report
    for metric, fname in [
        ("token_premium", "table_token_premium.csv"),
        ("fertility", "table_fertility.csv"),
        ("chars_per_token", "table_chars_per_token.csv"),
        ("strr", "table_strr.csv"),
        ("stfr", "table_stfr.csv"),
    ]:
        wide = df.pivot(index="language", columns="tokenizer_id", values=metric)
        wide.to_csv(out_dir / fname)

    print(json.dumps(decision, indent=2))
    print(f"Wrote results to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
