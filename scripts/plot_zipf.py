"""Figures for the language-specific Zipf deviation study.

Reads the Stage 1 and Stage 2 outputs and writes PNGs beside them:

``zipf_script_allocation.png``   how o200k's vocabulary splits across scripts
``zipf_rank_frequency.png``      log-log rank-frequency per language, fitted
                                 Zipf-Mandelbrot drawn over the empirical curve
``zipf_allocation_vs_deviation.png``  the headline scatter
``zipf_deviation_heatmap.png``   language x tokenizer
``zipf_token_vs_baseline.png``   token units against their own word/grapheme
                                 baseline, which is the attribution argument

Usage::

    python scripts/plot_zipf.py --results-dir results/zipf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vocab_profile import BYTE_FRAGMENT, MIXED_SCRIPT, NO_LETTER  # noqa: E402
from src.zipf import counts_from_interned, fit_zipf_mandelbrot, zm_probabilities  # noqa: E402
from src.zipf_langs import PRIMARY_TOKENIZER, SPLITS, STUDY_CODES  # noqa: E402

DPI = 160


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results/zipf"))
    p.add_argument("--view", default="matched_token", choices=["matched_token", "matched_sentence"])
    p.add_argument("--tokenizer", default=PRIMARY_TOKENIZER)
    return p.parse_args(argv)


def plot_script_allocation(alloc: pd.DataFrame, out: Path) -> None:
    special = {BYTE_FRAGMENT, MIXED_SCRIPT, NO_LETTER}
    top = alloc.head(20).iloc[::-1]
    colors = ["#b0b0b0" if s in special else "#c2410c" for s in top["script"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["script"], 100 * top["share"], color=colors)
    for y, (share, count) in enumerate(zip(top["share"], top["n_tokens"])):
        ax.text(100 * share + 0.4, y, f"{100 * share:.2f}%  ({count:,})", va="center", fontsize=8)
    ax.set_xlabel("share of o200k_base mergeable ranks (%)")
    ax.set_title(
        "o200k_base vocabulary allocation by Unicode script\n"
        "grey = not a script (byte fragments, mixed, no-letter)",
        fontsize=11,
    )
    ax.set_xlim(0, max(100 * top["share"]) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_word_coverage(profile: pd.DataFrame, out: Path) -> None:
    """Stage 1 headline: what share of a language's words survive as one token,
    and what share of its tokens are raw bytes.

    These are the two most directly legible numbers in the study and they answer
    the original question -- what percentage of each language does o200k actually
    have vocabulary for.
    """
    df = profile[profile["in_study"]].sort_values("whole_word_coverage", ascending=True)
    labels = [f"{r.language_name} ({r.code.split('_')[0]})" for r in df.itertuples()]
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6))

    ax = axes[0]
    ax.barh(y, 100 * df["whole_word_coverage"], color="#1d4ed8")
    for i, v in enumerate(100 * df["whole_word_coverage"]):
        ax.text(v + 0.6, i, f"{v:.2f}%", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% of distinct word types that encode to exactly one token")
    ax.set_title("Whole-word coverage (type-level)", fontsize=10.5)
    ax.set_xlim(0, 100 * df["whole_word_coverage"].max() * 1.22)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.barh(y, 100 * df["share_fragment_mass"], color="#c2410c")
    for i, v in enumerate(100 * df["share_fragment_mass"]):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% of emitted tokens that are partial-UTF-8 byte fragments")
    ax.set_title("Byte-fragment mass", fontsize=10.5)
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "What share of each language o200k_base actually has vocabulary for\n"
        f"FLORES {'+'.join(SPLITS)}, 2,009 parallel sentences per language",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_rank_frequency(caches: dict, fits: pd.DataFrame, out: Path, tokenizer: str) -> None:
    """Empirical rank-frequency with the fitted Zipf-Mandelbrot over the top.

    Full corpus, so the truncation each language actually suffers is visible
    rather than clipped by a matched budget.
    """
    codes = [c for c in STUDY_CODES if c in caches]
    ncol = 3
    nrow = math.ceil(len(codes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)

    row_lookup = fits[
        (fits["tokenizer_id"] == tokenizer)
        & (fits["unit"] == "token")
        & (fits["view"] == "matched_sentence")
    ].set_index("language")

    for i, code in enumerate(codes):
        ax = axes[i // ncol][i % ncol]
        desc = counts_from_interned(caches[code])
        ranks = np.arange(1, desc.size + 1)
        ax.loglog(ranks, desc, color="#1f2937", lw=1.4, label="empirical")

        fit = fit_zipf_mandelbrot(desc)
        if math.isfinite(fit.alpha):
            model = zm_probabilities(desc.size, fit.alpha, fit.b) * desc.sum()
            ax.loglog(ranks, model, color="#c2410c", lw=1.1, ls="--", label="fitted ZM")
        # Pure Zipf reference, the law being tested against.
        zipf = zm_probabilities(desc.size, 1.0, 0.0) * desc.sum()
        ax.loglog(ranks, zipf, color="#2563eb", lw=1.0, ls=":", label="Zipf (a=1)")

        name = row_lookup.loc[code, "language_name"] if code in row_lookup.index else code
        ax.set_title(
            f"{name} ({code})\nV={desc.size:,}  a={fit.alpha:.2f}  b={fit.b:.2f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(codes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        f"Token rank-frequency, {tokenizer}, FLORES {'+'.join(SPLITS)} full corpus",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_allocation_vs_deviation(
    merged: pd.DataFrame, out: Path, view: str, tokenizer: str
) -> None:
    """The headline: does vocabulary allocation predict distributional collapse?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    for ax, ycol, ylabel in (
        (axes[0], "log_effective_vocab", "log effective vocabulary  ln exp(H)"),
        (axes[1], "ks_zipf", "KS distance from Zipf's law"),
    ):
        x = 100 * merged["share_of_vocab"]
        y = merged[ycol]
        sizes = 40 + 260 * merged["share_fragment_mass"]
        sc = ax.scatter(
            x,
            y,
            s=sizes,
            c=merged["share_fragment_mass"],
            cmap="OrRd",
            vmin=0,
            vmax=1,
            edgecolor="#374151",
            linewidth=0.6,
            zorder=3,
        )
        for _, r in merged.iterrows():
            ax.annotate(
                r["language"].split("_")[0],
                (100 * r["share_of_vocab"], r[ycol]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("active vocabulary share of o200k (%, log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, zorder=0)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="share of tokens that are byte fragments")

    fig.suptitle(
        f"Vocabulary allocation vs distributional collapse - {tokenizer}, {view} view\n"
        "point size and colour: fraction of the language encoded as raw byte fragments",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_deviation_heatmap(fits: pd.DataFrame, out: Path, view: str) -> None:
    sub = fits[(fits["unit"] == "token") & (fits["view"] == view)]
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index="language", columns="tokenizer_id", values="log_effective_vocab"
    )
    order = (
        sub[sub["tokenizer_id"] == sub["tokenizer_id"].iloc[0]]
        .sort_values("log_effective_vocab", ascending=False)["language"]
        .tolist()
    )
    pivot = pivot.reindex([c for c in order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(1.7 * len(pivot.columns) + 4, 0.42 * len(pivot) + 2.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color="w")
    ax.set_title(f"log effective vocabulary (higher is healthier) - token units, {view}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# Which delta metric is legitimate against which baseline.
#
# Word supports (~7-16k types) are broadly comparable to token supports
# (~200-10k), so the exponent shift is meaningful there. Grapheme supports are
# ~100-2,500 types, where alpha and b are jointly near-degenerate and alpha is
# not comparable to a token-scale fit -- so the grapheme panel uses effective
# vocabulary, which is well defined at any support size.
_BASELINE_DELTA = {
    "word": ("delta_alpha", "alpha(token) - alpha(word)", "Zipf exponent shift"),
    "grapheme": (
        "delta_log_effective_vocab",
        "ln exp(H) token - ln exp(H) grapheme",
        "Effective-vocabulary shift",
    ),
}


def plot_token_vs_baseline(deltas: pd.DataFrame, out: Path, view: str, tokenizer: str) -> None:
    """Token units against the same text's own word/grapheme distribution.

    This is the attribution argument: word frequencies are the language's own
    property, so a gap here is the tokenizer's contribution rather than the
    language's morphology.
    """
    sub = deltas[(deltas["view"] == view) & (deltas["tokenizer_id"] == tokenizer)]
    if sub.empty:
        return
    baselines = [b for b in ("word", "grapheme") if b in set(sub["baseline"])]
    fig, axes = plt.subplots(1, len(baselines), figsize=(7.4 * len(baselines), 6.2), squeeze=False)

    for ax, baseline in zip(axes[0], baselines):
        col, xlabel, title = _BASELINE_DELTA[baseline]
        part = sub[sub["baseline"] == baseline].sort_values(col)
        y = np.arange(len(part))
        ax.barh(y, part[col], color="#c2410c")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r.language_name} ({r.language.split('_')[0]})" for r in part.itertuples()],
            fontsize=8,
        )
        ax.axvline(0, color="#374151", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title} vs {baseline} baseline", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"How far the tokenizer moves the distribution away from the text's own\n"
        f"unit distribution - {tokenizer}, {view} view\n"
        "(exponent shift vs word; effective-vocabulary shift vs grapheme, where "
        "alpha is not comparable across support sizes)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    d: Path = args.results_dir
    fits = pd.read_csv(d / "zipf_fits.csv")
    deltas_path = d / "zipf_deltas.csv"
    deltas = pd.read_csv(deltas_path) if deltas_path.is_file() else pd.DataFrame()

    alloc_path = d / "vocab_allocation_by_script.csv"
    profile_path = d / "lang_vocab_profile.csv"

    written: List[str] = []

    if alloc_path.is_file():
        plot_script_allocation(pd.read_csv(alloc_path), d / "zipf_script_allocation.png")
        written.append("zipf_script_allocation.png")

    if profile_path.is_file():
        profile = pd.read_csv(profile_path)
        if "whole_word_coverage" in profile.columns:
            plot_word_coverage(profile, d / "zipf_word_coverage.png")
            written.append("zipf_word_coverage.png")
        sub = fits[
            (fits["tokenizer_id"] == args.tokenizer)
            & (fits["unit"] == "token")
            & (fits["view"] == args.view)
        ]
        merged = sub.merge(
            profile[["code", "share_of_vocab", "share_fragment_mass", "share_mass_not_in_control"]],
            left_on="language",
            right_on="code",
        )
        if not merged.empty:
            plot_allocation_vs_deviation(
                merged, d / "zipf_allocation_vs_deviation.png", args.view, args.tokenizer
            )
            written.append("zipf_allocation_vs_deviation.png")

    plot_deviation_heatmap(fits, d / "zipf_deviation_heatmap.png", args.view)
    written.append("zipf_deviation_heatmap.png")

    if not deltas.empty:
        plot_token_vs_baseline(
            deltas, d / "zipf_token_vs_baseline.png", args.view, args.tokenizer
        )
        written.append("zipf_token_vs_baseline.png")

    # Rank-frequency needs the raw unit streams, so re-encode. Cheap for one
    # tokenizer and it keeps the figure independent of the Stage 2 run.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_vocab_profile import load_concatenated

    from src.tokenizers_registry import load_tokenizers
    from src.zipf import intern_units

    spec = load_tokenizers([args.tokenizer])[args.tokenizer]
    corpora = load_concatenated(list(STUDY_CODES))
    caches = {
        code: intern_units([spec.encode(s) for s in sentences])
        for code, sentences in corpora.items()
    }
    plot_rank_frequency(caches, fits, d / "zipf_rank_frequency.png", args.tokenizer)
    written.append("zipf_rank_frequency.png")

    print(f"Wrote {len(written)} figures to {d.resolve()}:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
