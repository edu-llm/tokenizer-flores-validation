"""Stage 2: language-specific Zipf-law deviation across tokenizers.

For every (language, tokenizer, unit) combination, fits Zipf-Mandelbrot by MLE
over the language's own active support and measures how far the distribution sits
from Zipf's law.

Three unit types, because the word and grapheme baselines are what attribute a
deviation to the *tokenizer* rather than to the language:

``token``     the tokenizer's output.
``word``      whitespace words. Unavailable for zho_Hans and tha_Thai.
``grapheme``  UAX #29 clusters -- defined for every script, so it covers the two
              languages with no word baseline.

Two views, both reported because they answer different questions and silently
picking one would mislead:

``matched_token``     every language subsampled to an identical unit budget.
                      Matched statistical power, unmatched content.
``matched_sentence``  the whole 2,009-sentence parallel corpus. Matched content,
                      unmatched sample size.

Usage::

    python scripts/run_zipf_eval.py --tokenizers o200k --max-sentences 50 --bootstrap 5 \
        --out-dir results/zipf/_smoke
    python scripts/run_zipf_eval.py --out-dir results/zipf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizers_registry import load_tokenizers  # noqa: E402
from src.zipf import (  # noqa: E402
    DEFAULT_RENYI_ORDER,
    counts_from_interned,
    distribution_stats,
    draw_counts_matched,
    draw_counts_resample,
    fit_zipf_mandelbrot,
    grapheme_units,
    intern_units,
    summarize_bootstrap,
    word_units,
)
from src.zipf_langs import (  # noqa: E402
    BY_CODE,
    CONTROL_LANG,
    NO_WORD_BOUNDARY,
    PRIMARY_TOKENIZER,
    STUDY_CODES,
    STUDY_TOKENIZERS,
    UNITS,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_vocab_profile import load_concatenated  # noqa: E402

# Metrics carried through every fit. alpha/b/ks are the Zipf-specific ones;
# the entropy family stays meaningful where a power-law exponent does not.
METRICS: Sequence[str] = (
    "alpha",
    "b",
    "ks",
    "ks_zipf",
    "loglog_rmse",
    "loglog_ols_slope",
    "n_types",
    "entropy_norm",
    "effective_vocab",
    "log_effective_vocab",
    "renyi_efficiency_active",
)

# Metrics for which token-minus-baseline deltas are reported. These are the ones
# where "the tokenizer moved the distribution this far" is meaningful.
DELTA_METRICS: Sequence[str] = (
    "alpha",
    "ks_zipf",
    "entropy_norm",
    "log_effective_vocab",
)

VIEWS: Sequence[str] = ("matched_token", "matched_sentence")

# Fraction of the smallest available corpus used as the matched budget. Slightly
# under 1.0 so that even the smallest language is subsampled and therefore
# carries sampling variability like every other.
BUDGET_FRACTION = 0.95


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/zipf"))
    p.add_argument("--tokenizers", nargs="*", default=list(STUDY_TOKENIZERS))
    p.add_argument("--languages", nargs="*", default=list(STUDY_CODES))
    p.add_argument("--units", nargs="*", default=list(UNITS))
    p.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Draws per (language, tokenizer, unit, view) for the interval",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Cap sentences per language for a smoke run",
    )
    p.add_argument(
        "--profile-csv",
        type=Path,
        default=None,
        help=(
            "Stage 1 lang_vocab_profile.csv. Defaults to <out-dir>/../"
            "lang_vocab_profile.csv then <out-dir>/lang_vocab_profile.csv. "
            "Required for the allocation-vs-deviation hypotheses."
        ),
    )
    p.add_argument("--seed", type=int, default=20260731)
    return p.parse_args(argv)


def measure(counts, full_vocab_size: int | None, start_hint) -> Dict[str, float]:
    """Fit and summarize one unit distribution."""
    fit = fit_zipf_mandelbrot(counts, starts=start_hint)
    stats = distribution_stats(counts, full_vocab_size=full_vocab_size)
    effective = stats.effective_vocab
    return {
        "alpha": fit.alpha,
        "b": fit.b,
        "ks": fit.ks,
        "ks_zipf": fit.ks_zipf,
        "loglog_rmse": fit.loglog_rmse,
        "loglog_ols_slope": fit.loglog_ols_slope,
        "n_types": float(fit.n_types),
        "entropy_norm": stats.entropy_norm,
        "effective_vocab": effective,
        "log_effective_vocab": (
            math.log(effective) if effective and math.isfinite(effective) and effective > 0 else float("nan")
        ),
        "renyi_efficiency_active": stats.renyi_efficiency_active,
        "converged": float(fit.converged),
        "at_bound": float(fit.at_bound),
    }


def build_unit_cache(sentences: Sequence[str], unit: str, spec=None) -> List[np.ndarray]:
    if unit == "word":
        raw = [word_units(s) for s in sentences]
    elif unit == "grapheme":
        raw = [grapheme_units(s) for s in sentences]
    elif unit == "token":
        raw = [spec.encode(s) for s in sentences]
    else:
        raise KeyError(f"unknown unit {unit!r}")
    return intern_units(raw)


def total_units(cache: Sequence[np.ndarray]) -> int:
    return int(sum(len(a) for a in cache))


def evaluate_cell(
    cache: Sequence[np.ndarray],
    view: str,
    budget: int,
    n_boot: int,
    seed: int,
    full_vocab_size: int | None,
) -> dict:
    """Point estimate plus bootstrap interval for one cell."""
    reference = measure(counts_from_interned(cache), full_vocab_size, start_hint=None)
    hint = None
    if math.isfinite(reference["alpha"]):
        hint = [(reference["alpha"], reference["b"])]

    draws: List[Dict[str, float]] = []
    budget_met = True
    used: List[int] = []
    for i in range(n_boot):
        rng = np.random.default_rng(seed + i)
        if view == "matched_token":
            counts, drawn, met = draw_counts_matched(cache, budget, rng)
            budget_met = budget_met and met
            used.append(drawn)
        else:
            counts = draw_counts_resample(cache, rng)
            used.append(int(counts.sum()))
        draws.append(measure(counts, full_vocab_size, start_hint=hint))

    row: Dict[str, object] = {
        "view": view,
        "budget": budget if view == "matched_token" else total_units(cache),
        "budget_met": budget_met,
        "n_units_full": total_units(cache),
        "n_units_used_mean": float(np.mean(used)) if used else float("nan"),
        "n_bootstrap": n_boot,
        "converged_share": float(np.mean([d["converged"] for d in draws])) if draws else float("nan"),
        # Share of draws whose optimum was cut off by an artificial bound. Any
        # cell above 0 must not have its alpha or b read as an estimate.
        "at_bound_share": float(np.mean([d["at_bound"] for d in draws])) if draws else float("nan"),
        "reference_at_bound": bool(reference["at_bound"]),
    }
    for metric in METRICS:
        summary = summarize_bootstrap([d[metric] for d in draws])
        # matched_sentence resamples the full corpus, so its point estimate is
        # the full-corpus fit; matched_token has no such anchor and uses the
        # mean across subsamples.
        row[metric] = reference[metric] if view == "matched_sentence" else summary["mean"]
        row[f"{metric}_mean"] = summary["mean"]
        row[f"{metric}_lo"] = summary["lo"]
        row[f"{metric}_hi"] = summary["hi"]
        row[f"{metric}_sd"] = summary["sd"]
        row[f"{metric}_n"] = summary["n"]
    row["full_corpus_alpha"] = reference["alpha"]
    row["full_corpus_ks_zipf"] = reference["ks_zipf"]
    return row


def resolve_profile_csv(args: argparse.Namespace) -> Path | None:
    if args.profile_csv:
        return args.profile_csv if args.profile_csv.is_file() else None
    for candidate in (
        args.out_dir.parent / "lang_vocab_profile.csv",
        args.out_dir / "lang_vocab_profile.csv",
    ):
        if candidate.is_file():
            return candidate
    return None


def evaluate_hypotheses(fits: pd.DataFrame, deltas: pd.DataFrame, profile: pd.DataFrame | None) -> dict:
    """Score the four pre-registered hypotheses against the measured numbers."""
    out: dict = {}

    def spearman(x, y) -> dict:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4:
            return {"rho": None, "p": None, "n": int(ok.sum())}
        rho, p = spearmanr(x[ok], y[ok])
        return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}

    if profile is None:
        out["note"] = (
            "lang_vocab_profile.csv not found; allocation-linked hypotheses "
            "H1-H3 could not be evaluated. Run scripts/run_vocab_profile.py first."
        )
    else:
        prof = profile[profile["in_study"]].set_index("code")

        # H1: raw allocation share vs how much of that is the language's own.
        out["H1_allocation_vs_exclusivity"] = {
            "claim": (
                "Latin-script low-resource languages have high raw active-vocab "
                "share but low exclusive share -- they ride English subwords."
            ),
            "spearman_share_vs_mass_exclusivity": spearman(
                prof["share_of_vocab"], prof["share_mass_not_in_control"]
            ),
            "latin_low_resource": {
                code: {
                    "share_of_vocab": float(prof.loc[code, "share_of_vocab"]),
                    "share_mass_not_in_control": float(prof.loc[code, "share_mass_not_in_control"]),
                }
                for code in ("quy_Latn", "grn_Latn", "hau_Latn", "swh_Latn")
                if code in prof.index
            },
        }

        primary = fits[
            (fits["tokenizer_id"] == PRIMARY_TOKENIZER)
            & (fits["unit"] == "token")
            & (fits["view"] == "matched_token")
        ].set_index("language")
        shared = [c for c in primary.index if c in prof.index]

        # H2: deviation worsens as allocation share falls.
        out["H2_allocation_vs_deviation"] = {
            "claim": (
                "At a matched token budget, Zipf deviation worsens monotonically "
                "as active-vocabulary share falls."
            ),
            "tokenizer": PRIMARY_TOKENIZER,
            "spearman_share_vs_ks_zipf": spearman(
                prof.loc[shared, "share_of_vocab"], primary.loc[shared, "ks_zipf"]
            ),
            "spearman_share_vs_log_effective_vocab": spearman(
                prof.loc[shared, "share_of_vocab"], primary.loc[shared, "log_effective_vocab"]
            ),
            "spearman_fragment_mass_vs_ks_zipf": spearman(
                prof.loc[shared, "share_fragment_mass"], primary.loc[shared, "ks_zipf"]
            ),
        }

        # H3: the token-minus-baseline delta tracks allocation, attributing the
        # distortion to the tokenizer rather than the language.
        h3: dict = {
            "claim": (
                "Token-minus-baseline deltas track allocation share, attributing "
                "the distortion to the tokenizer rather than the language."
            ),
            "tokenizer": PRIMARY_TOKENIZER,
        }
        for baseline in ("word", "grapheme"):
            sub = deltas[
                (deltas["tokenizer_id"] == PRIMARY_TOKENIZER)
                & (deltas["baseline"] == baseline)
                & (deltas["view"] == "matched_token")
            ].set_index("language")
            codes = [c for c in sub.index if c in prof.index]
            h3[f"vs_{baseline}"] = {
                "spearman_share_vs_delta_ks_zipf": spearman(
                    prof.loc[codes, "share_of_vocab"], sub.loc[codes, "delta_ks_zipf"]
                ),
                "spearman_share_vs_delta_log_effective_vocab": spearman(
                    prof.loc[codes, "share_of_vocab"], sub.loc[codes, "delta_log_effective_vocab"]
                ),
                "n_languages": len(codes),
            }
        out["H3_delta_vs_allocation"] = h3

    # H4: the multilingual-by-design tokenizer should spread languages less.
    spread = (
        fits[(fits["unit"] == "token") & (fits["view"] == "matched_token")]
        .groupby("tokenizer_id")["ks_zipf"]
        .agg(["std", "min", "max", "count"])
    )
    spread["range"] = spread["max"] - spread["min"]
    frontier = [t for t in ("o200k", "llama", "qwen") if t in spread.index]
    verdict = None
    if "multi" in spread.index and frontier:
        verdict = bool(
            spread.loc["multi", "std"] < min(spread.loc[t, "std"] for t in frontier)
        )
    out["H4_multilingual_spread"] = {
        "claim": "NLLB-200 shows a smaller cross-language deviation spread than the frontier three.",
        "per_tokenizer_ks_zipf_spread": {
            tid: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            for tid, row in spread.iterrows()
        },
        "supported": verdict,
    }
    return out


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = load_concatenated(list(args.languages))
    missing = [c for c in args.languages if c not in corpora]
    if missing:
        print(f"[warn] FLORES files not found, skipping: {missing}")
    if args.max_sentences:
        corpora = {k: v[: args.max_sentences] for k, v in corpora.items()}
    codes = sorted(corpora)
    print(f"Loaded {len(codes)} languages x {len(next(iter(corpora.values())))} sentences")

    specs = load_tokenizers(list(args.tokenizers))

    # Word and grapheme units are tokenizer-independent, so build them once.
    print("\nBuilding tokenizer-independent unit caches (word, grapheme)...")
    caches: Dict[tuple[str, str, str], List[np.ndarray]] = {}
    for unit in [u for u in args.units if u in ("word", "grapheme")]:
        for code in codes:
            if unit == "word" and code in NO_WORD_BOUNDARY:
                continue
            caches[("-", code, unit)] = build_unit_cache(corpora[code], unit)

    if "token" in args.units:
        for tid, spec in specs.items():
            print(f"Encoding 18-language set with {tid}...")
            for code in codes:
                caches[(tid, code, "token")] = build_unit_cache(
                    corpora[code], "token", spec
                )

    # One budget per unit type, from the global minimum across every cell using
    # that unit, so within-unit cross-language comparison is size-matched.
    budgets: Dict[str, int] = {}
    for unit in args.units:
        totals = [total_units(c) for (_, _, u), c in caches.items() if u == unit]
        if not totals:
            continue
        budgets[unit] = max(1, int(BUDGET_FRACTION * min(totals)))
    print("\nMatched budgets (95% of the smallest corpus for that unit):")
    for unit, budget in budgets.items():
        print(f"  {unit:10s} {budget:,} units")

    # full_vocab_size is left unset here: normalizing entropy by the whole
    # tokenizer vocabulary is a utilization measure and belongs to Stage 1, which
    # reports it per language. Stage 2 is about distribution *shape* over the
    # active support.
    rows: List[dict] = []
    total_cells = len([1 for _ in caches]) * len(VIEWS)
    done = 0
    for (tid, code, unit), cache in sorted(caches.items()):
        tokenizer_ids = list(specs) if tid == "-" else [tid]
        for view in VIEWS:
            result = evaluate_cell(
                cache,
                view=view,
                budget=budgets[unit],
                n_boot=args.bootstrap,
                seed=args.seed,
                full_vocab_size=None,
            )
            for owner in tokenizer_ids:
                study = BY_CODE.get(code)
                rows.append(
                    {
                        "tokenizer_id": owner,
                        "language": code,
                        "language_name": study.name if study else code,
                        "script": study.script if study else "",
                        "tier": study.tier if study else "",
                        "unit": unit,
                        "tokenizer_independent": tid == "-",
                        **result,
                    }
                )
            done += 1
            if done % 20 == 0 or done == total_cells:
                print(f"  {done}/{total_cells} cells fitted")

    fits = pd.DataFrame(rows)
    fits.to_csv(out_dir / "zipf_fits.csv", index=False)

    # Deltas: token minus each available baseline, same language and view.
    delta_rows: List[dict] = []
    token_rows = fits[fits["unit"] == "token"]
    for _, tok in token_rows.iterrows():
        for baseline in ("word", "grapheme"):
            match = fits[
                (fits["unit"] == baseline)
                & (fits["language"] == tok["language"])
                & (fits["view"] == tok["view"])
                & (fits["tokenizer_id"] == tok["tokenizer_id"])
            ]
            if match.empty:
                continue
            base = match.iloc[0]
            entry = {
                "tokenizer_id": tok["tokenizer_id"],
                "language": tok["language"],
                "language_name": tok["language_name"],
                "script": tok["script"],
                "view": tok["view"],
                "baseline": baseline,
            }
            for metric in DELTA_METRICS:
                entry[f"delta_{metric}"] = tok[metric] - base[metric]
                entry[f"token_{metric}"] = tok[metric]
                entry[f"baseline_{metric}"] = base[metric]
            delta_rows.append(entry)

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(out_dir / "zipf_deltas.csv", index=False)

    profile_path = resolve_profile_csv(args)
    profile = pd.read_csv(profile_path) if profile_path else None
    if profile_path:
        print(f"\nUsing Stage 1 profile: {profile_path}")
    hypotheses = evaluate_hypotheses(fits, deltas, profile)
    (out_dir / "hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    payload = {
        "tokenizers": list(specs),
        "languages": codes,
        "units": list(args.units),
        "views": list(VIEWS),
        "budgets": budgets,
        "budget_fraction": BUDGET_FRACTION,
        "n_bootstrap": args.bootstrap,
        "renyi_order": DEFAULT_RENYI_ORDER,
        "control_language": CONTROL_LANG,
        "metrics": list(METRICS),
        "fits": fits.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
        "hypotheses": hypotheses,
    }
    (out_dir / "zipf_fits.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    headline = fits[
        (fits["tokenizer_id"] == PRIMARY_TOKENIZER)
        & (fits["unit"] == "token")
        & (fits["view"] == "matched_token")
    ].sort_values("ks_zipf")
    if not headline.empty:
        print(f"\n{PRIMARY_TOKENIZER} token units, matched-token view, by distance from Zipf's law:")
        print(
            headline[
                [
                    "language",
                    "language_name",
                    "script",
                    "n_types",
                    "alpha",
                    "ks_zipf",
                    "ks_zipf_lo",
                    "ks_zipf_hi",
                    "effective_vocab",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
    print(f"\nWrote Stage 2 outputs to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
