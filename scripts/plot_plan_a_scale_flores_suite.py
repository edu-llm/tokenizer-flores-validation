#!/usr/bin/env python3
"""Plot Plan A scale FLORES suite: premium, fertility, Zipf ks_zipf.

Reads wide CSVs from ``eval_plan_a_scale_flores_suite.py`` and writes PNGs under
``--suite-dir/plots/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plan_a_langs import LANG_NAMES, PLAN_A_CODES

# Colorblind-friendly pair (BPE / SuperBPE)
COLOR_BPE = "#0072B2"
COLOR_SUPER = "#E69F00"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite-dir",
        type=Path,
        default=ROOT / "artifacts" / "plan_a" / "scale" / "flores_suite",
    )
    return p.parse_args()


def _load_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("language")
    return df.reindex([c for c in PLAN_A_CODES if c in df.index])


def _labels(codes: list[str]) -> list[str]:
    return [LANG_NAMES.get(c, c) for c in codes]


def plot_grouped_bars(
    df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    out: Path,
    hline: float | None = None,
) -> None:
    codes = list(df.index)
    x = np.arange(len(codes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, df["bpe"], width, label="BPE", color=COLOR_BPE)
    ax.bar(x + width / 2, df["superbpe"], width, label="SuperBPE", color=COLOR_SUPER)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(codes), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, color="#666666", linestyle="--", linewidth=1, label=f"ref={hline:g}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_summary_figure(
    premium: pd.DataFrame,
    fertility: pd.DataFrame,
    ks: pd.DataFrame,
    out: Path,
) -> None:
    """One composition: three panels, shared language axis."""
    codes = list(premium.index)
    labels = _labels(codes)
    x = np.arange(len(codes))
    width = 0.36

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    panels = [
        (axes[0], premium, "Token premium vs English (CTC_lang / CTC_eng)", "Premium", 1.0),
        (axes[1], fertility, "Fertility (tokens / word)", "Tokens / word", None),
        (axes[2], ks, "Zipf deviation (ks_zipf, matched_sentence)", "ks_zipf", None),
    ]
    for ax, df, title, ylabel, href in panels:
        ax.bar(x - width / 2, df["bpe"], width, label="BPE", color=COLOR_BPE)
        ax.bar(x + width / 2, df["superbpe"], width, label="SuperBPE", color=COLOR_SUPER)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if href is not None:
            ax.axhline(href, color="#666666", linestyle="--", linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")

    # Mandarin fertility is ~10× others — annotate so the panel stays readable.
    fert_ax = axes[1]
    ymax = max(fertility["bpe"].max(), fertility["superbpe"].max())
    if ymax > 4:
        fert_ax.set_ylim(0, 3.2)
        for i, code in enumerate(codes):
            if code == "zho_Hans":
                fert_ax.annotate(
                    f"zho BPE {fertility.loc[code, 'bpe']:.1f} / "
                    f"SBPE {fertility.loc[code, 'superbpe']:.1f}",
                    xy=(i, 3.05),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#333333",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle(
        "Plan A scale gigatoken — FLORES-200 devtest (6 languages)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    suite = args.suite_dir
    plots = suite / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    premium = _load_wide(suite / "token_premium_wide.csv")
    fertility = _load_wide(suite / "fertility_wide.csv")
    ks = _load_wide(suite / "zipf" / "ks_zipf_wide.csv")

    plot_grouped_bars(
        premium,
        title="Token premium vs English",
        ylabel="CTC_lang / CTC_eng",
        out=plots / "token_premium.png",
        hline=1.0,
    )
    plot_grouped_bars(
        fertility,
        title="Fertility (tokens / word)",
        ylabel="Tokens / word",
        out=plots / "fertility.png",
    )
    plot_grouped_bars(
        ks,
        title="Zipf deviation (ks_zipf)",
        ylabel="ks_zipf (lower = closer to Zipf)",
        out=plots / "ks_zipf.png",
    )
    plot_summary_figure(
        premium,
        fertility,
        ks,
        out=plots / "flores_suite_summary.png",
    )
    print(f"Wrote plots under {plots}")
    for p in sorted(plots.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
