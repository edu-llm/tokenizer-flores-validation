"""Zipf-law deviation statistics for tokenizer output distributions.

Pure statistics: numpy + scipy only, no tokenizer or dataset imports, so the
estimator can be pinned against synthetic ground truth in isolation. Callers
supply unit streams; :func:`word_units` and :func:`grapheme_units` cover the two
text-derived baselines, and token units come from a
``src.tokenizers_registry.TokenizerSpec``.

Two distinct quantities are computed and must not be conflated:

``ks``
    Kolmogorov-Smirnov distance from the *best-fit* Zipf-Mandelbrot
    distribution. Answers "is this distribution power-law shaped at all".

``ks_zipf``
    KS distance from *pure Zipf* (alpha=1, b=0) over the same support. Answers
    "how far is this from Zipf's law", which is the headline for a deviation
    study.

A uniform distribution shows the difference: it is Zipf-Mandelbrot with
alpha=0, so its ``ks`` is near zero while its ``ks_zipf`` is large.

The Zipf-Mandelbrot form is ``p(r) proportional to (r + b)^-alpha`` over the
observed support, fit by maximum likelihood. OLS on the log-log rank-frequency
curve is biased (Clauset, Shalizi & Newman 2009) and is exposed only as an
interpretability aid via :func:`loglog_ols_slope`.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import regex as re
from scipy.optimize import minimize
from scipy.special import logsumexp

# Order used for Renyi efficiency. 2.5 follows Zouhar et al., "Tokenization and
# the Noiseless Channel", which reports this order as the best-correlating
# tokenizer quality proxy.
DEFAULT_RENYI_ORDER = 2.5

# b > -1 keeps (r + b) positive at rank 1.
#
# The alpha ceiling is 50, not 10. A ceiling of 10 truncated real optima: on
# English grapheme clusters (V=110) the unconstrained optimum sits at alpha ~17
# with b ~133, and 40 of 144 grapheme fits were pinned at exactly 10.0. Any fit
# that still lands on a bound is flagged by ``ZMFit.at_bound`` and must not be
# read as an estimate.
#
# Note that at small support alpha and b are jointly near-degenerate: moving from
# (10, 70) to (17, 133) changes the log-likelihood by 0.03%. The *shape* is
# identified, the individual parameters are not. Compare alpha only between
# distributions of broadly similar support size.
_B_BOUNDS = (-0.9, 1.0e4)
_ALPHA_BOUNDS = (0.0, 50.0)

# Multiple starts guard against local minima in the (alpha, b) surface, which is
# shallow when the support is small. The high-alpha/high-b starts matter for
# grapheme-scale supports, where the optimum is far from Zipf.
_STARTS: Sequence[tuple[float, float]] = (
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.0),
    (1.5, 2.0),
    (0.1, 0.0),
    (2.0, 5.0),
    (5.0, 30.0),
    (17.0, 130.0),
)


@dataclass(frozen=True)
class ZMFit:
    """Zipf-Mandelbrot maximum-likelihood fit over an observed support."""

    alpha: float
    b: float
    ks: float
    ks_zipf: float
    loglog_rmse: float
    loglog_ols_slope: float
    nll: float
    n_types: int
    n_tokens: int
    converged: bool
    free_b: bool
    # True when alpha or b landed on an optimizer bound. Such a fit is a
    # truncation of the real optimum, not an estimate of it, and must be excluded
    # from parameter comparisons rather than read at face value.
    at_bound: bool = False


@dataclass(frozen=True)
class DistributionStats:
    """Shape measures that stay meaningful when the support is tiny.

    ``alpha`` and the KS statistics degrade on a few-hundred-type support, which
    is exactly the regime byte-level-tokenized languages occupy, so these are
    reported alongside every fit.
    """

    n_tokens: int
    n_types: int
    entropy_nats: float
    entropy_bits: float
    # Normalized by the *active* support: a shape measure.
    entropy_norm: float
    # Normalized by the whole tokenizer vocabulary: a utilization measure.
    entropy_norm_full: float | None
    # exp(H): the number of equally-likely units the distribution behaves like.
    effective_vocab: float
    renyi_order: float
    renyi_entropy_nats: float
    renyi_efficiency_active: float
    renyi_efficiency_full: float | None


# ---------------------------------------------------------------------------
# Support handling
# ---------------------------------------------------------------------------


def counts_descending(counts: Mapping[object, float] | Counter) -> np.ndarray:
    """Positive counts as a descending float array. Zero counts are dropped."""
    vals = [float(v) for v in counts.values() if v > 0]
    vals.sort(reverse=True)
    return np.asarray(vals, dtype=float)


def _as_desc(data: Mapping[object, float] | Counter | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data[data > 0].astype(float)
        return np.sort(arr)[::-1]
    if isinstance(data, Mapping):
        return counts_descending(data)
    arr = np.asarray([float(v) for v in data if v > 0], dtype=float)
    return np.sort(arr)[::-1]


def zm_probabilities(n_types: int, alpha: float, b: float) -> np.ndarray:
    """Normalized Zipf-Mandelbrot probabilities over ranks 1..n_types."""
    if n_types <= 0:
        return np.empty(0, dtype=float)
    if b <= -1.0:
        raise ValueError(f"b must exceed -1 to keep (rank + b) positive; got {b}")
    ranks = np.arange(1, n_types + 1, dtype=float)
    log_w = -alpha * np.log(ranks + b)
    # Subtract the max before exponentiating so extreme alpha cannot underflow
    # the whole weight vector to zero.
    weights = np.exp(log_w - log_w.max())
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Entropy family
# ---------------------------------------------------------------------------


def shannon_entropy(desc: np.ndarray) -> float:
    """Shannon entropy in nats."""
    arr = _as_desc(desc)
    if arr.size == 0:
        return float("nan")
    p = arr / arr.sum()
    return float(-np.sum(p * np.log(p)))


def renyi_entropy(desc: np.ndarray, order: float) -> float:
    """Renyi entropy of the given order, in nats. Order 1 falls back to Shannon."""
    arr = _as_desc(desc)
    if arr.size == 0:
        return float("nan")
    if math.isclose(order, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return shannon_entropy(arr)
    p = arr / arr.sum()
    return float(np.log(np.sum(p**order)) / (1.0 - order))


def effective_vocab_size(desc: np.ndarray) -> float:
    """exp(H) -- the equally-likely-unit count the distribution behaves like.

    Bounded above by the support size, with equality only for a uniform
    distribution. Stays informative where power-law exponents do not.
    """
    h = shannon_entropy(desc)
    return float("nan") if math.isnan(h) else float(math.exp(h))


def distribution_stats(
    counts: Mapping[object, float] | Counter | np.ndarray,
    full_vocab_size: int | None = None,
    renyi_order: float = DEFAULT_RENYI_ORDER,
) -> DistributionStats:
    """Sample-size-robust shape measures for one unit distribution."""
    desc = _as_desc(counts)
    n_types = int(desc.size)
    n_tokens = int(desc.sum()) if n_types else 0

    if n_types == 0:
        nan = float("nan")
        return DistributionStats(
            n_tokens=0,
            n_types=0,
            entropy_nats=nan,
            entropy_bits=nan,
            entropy_norm=nan,
            entropy_norm_full=None if full_vocab_size is None else nan,
            effective_vocab=nan,
            renyi_order=renyi_order,
            renyi_entropy_nats=nan,
            renyi_efficiency_active=nan,
            renyi_efficiency_full=None if full_vocab_size is None else nan,
        )

    h = shannon_entropy(desc)
    h_renyi = renyi_entropy(desc, renyi_order)
    # A single type carries no information and has no meaningful normalizer.
    log_active = math.log(n_types) if n_types > 1 else 0.0
    norm = (h / log_active) if log_active > 0 else 0.0
    renyi_active = (h_renyi / log_active) if log_active > 0 else 0.0

    if full_vocab_size is not None and full_vocab_size > 1:
        log_full = math.log(full_vocab_size)
        norm_full: float | None = h / log_full
        renyi_full: float | None = h_renyi / log_full
    elif full_vocab_size is not None:
        norm_full, renyi_full = 0.0, 0.0
    else:
        norm_full, renyi_full = None, None

    return DistributionStats(
        n_tokens=n_tokens,
        n_types=n_types,
        entropy_nats=h,
        entropy_bits=h / math.log(2.0),
        entropy_norm=norm,
        entropy_norm_full=norm_full,
        effective_vocab=effective_vocab_size(desc),
        renyi_order=renyi_order,
        renyi_entropy_nats=h_renyi,
        renyi_efficiency_active=renyi_active,
        renyi_efficiency_full=renyi_full,
    )


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------


def ks_distance(desc: np.ndarray, alpha: float, b: float) -> float:
    """KS distance between the empirical rank CDF and Zipf-Mandelbrot(alpha, b)."""
    arr = _as_desc(desc)
    if arr.size == 0:
        return float("nan")
    emp = np.cumsum(arr) / arr.sum()
    model = np.cumsum(zm_probabilities(arr.size, alpha, b))
    return float(np.max(np.abs(emp - model)))


def ks_vs_pure_zipf(desc: np.ndarray) -> float:
    """KS distance from Zipf's law itself (alpha=1, b=0) over the same support."""
    return ks_distance(desc, 1.0, 0.0)


def loglog_rmse(desc: np.ndarray, alpha: float, b: float) -> float:
    """RMSE of log-frequency residuals against the fitted model.

    Computed over the whole support, so the hapax tail contributes heavily. That
    is intentional -- truncation and tail collapse are the effects of interest --
    but it means the value is not comparable across different support sizes
    unless the token budget is matched.
    """
    arr = _as_desc(desc)
    if arr.size == 0:
        return float("nan")
    p_emp = arr / arr.sum()
    p_mod = zm_probabilities(arr.size, alpha, b)
    resid = np.log(p_emp) - np.log(p_mod)
    return float(math.sqrt(float(np.mean(resid**2))))


def loglog_ols_slope(desc: np.ndarray) -> float:
    """OLS slope of log frequency on log rank, returned as a positive exponent.

    Biased (Clauset et al. 2009) and reported only for interpretability against
    the older literature; :func:`fit_zipf_mandelbrot` is the real estimator.
    """
    arr = _as_desc(desc)
    if arr.size < 2:
        return float("nan")
    log_rank = np.log(np.arange(1, arr.size + 1, dtype=float))
    log_freq = np.log(arr)
    slope = np.polyfit(log_rank, log_freq, 1)[0]
    return float(-slope)


# ---------------------------------------------------------------------------
# Maximum-likelihood fit
# ---------------------------------------------------------------------------


def _negative_log_likelihood(desc: np.ndarray, alpha: float, b: float) -> float:
    n_types = desc.size
    total = desc.sum()
    log_ranks = np.log(np.arange(1, n_types + 1, dtype=float) + b)
    log_norm = logsumexp(-alpha * log_ranks)
    return float(alpha * float(np.dot(desc, log_ranks)) + total * log_norm)


def fit_zipf_mandelbrot(
    counts: Mapping[object, float] | Counter | np.ndarray,
    free_b: bool = True,
    starts: Sequence[tuple[float, float]] | None = None,
) -> ZMFit:
    """Fit ``p(r) ~ (r + b)^-alpha`` by MLE over the observed support.

    With ``free_b=False`` this is the pure Zipf fit (b pinned to 0). Support size
    is taken from the data rather than estimated: truncation is a property of the
    tokenizer's allocation for that language and is reported, not fit away.

    ``starts`` overrides the default multi-start grid. Bootstrap draws pass the
    full-corpus estimate as a single start, which is both faster and adequate:
    a resample of the same corpus lands near the same optimum.

    Degenerate inputs (fewer than two types) return ``converged=False`` with NaN
    parameters rather than raising, so a run over 200 languages is not derailed
    by one collapsed distribution.
    """
    desc = _as_desc(counts)
    n_types = int(desc.size)
    n_tokens = int(desc.sum()) if n_types else 0
    nan = float("nan")

    if n_types < 2:
        return ZMFit(
            alpha=nan,
            b=nan,
            ks=nan,
            ks_zipf=nan,
            loglog_rmse=nan,
            loglog_ols_slope=nan,
            nll=nan,
            n_types=n_types,
            n_tokens=n_tokens,
            converged=False,
            free_b=free_b,
        )

    grid = tuple(starts) if starts is not None else _STARTS
    if free_b:
        def objective(params: np.ndarray) -> float:
            return _negative_log_likelihood(desc, float(params[0]), float(params[1]))

        bounds = [_ALPHA_BOUNDS, _B_BOUNDS]
        start_points = [np.array(s, dtype=float) for s in grid]
    else:
        def objective(params: np.ndarray) -> float:
            return _negative_log_likelihood(desc, float(params[0]), 0.0)

        bounds = [_ALPHA_BOUNDS]
        start_points = [np.array([s[0]], dtype=float) for s in grid]

    best = None
    for start in start_points:
        try:
            res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        except (ValueError, FloatingPointError):
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        return ZMFit(
            alpha=nan,
            b=nan,
            ks=nan,
            ks_zipf=nan,
            loglog_rmse=nan,
            loglog_ols_slope=loglog_ols_slope(desc),
            nll=nan,
            n_types=n_types,
            n_tokens=n_tokens,
            converged=False,
            free_b=free_b,
        )

    alpha = float(best.x[0])
    b = float(best.x[1]) if free_b else 0.0

    # Only *artificial* bounds count as truncation. alpha = 0 is not flagged: it
    # is the uniform limit and a natural edge of the parameter space, since a
    # negative exponent cannot fit a descending count vector. The alpha ceiling
    # and both b bounds are ours, so hitting them means the optimum was cut off.
    tol = 1e-6
    at_bound = alpha >= _ALPHA_BOUNDS[1] - tol or (
        free_b and (b <= _B_BOUNDS[0] + tol or b >= _B_BOUNDS[1] - tol)
    )

    return ZMFit(
        alpha=alpha,
        b=b,
        ks=ks_distance(desc, alpha, b),
        ks_zipf=ks_vs_pure_zipf(desc),
        loglog_rmse=loglog_rmse(desc, alpha, b),
        loglog_ols_slope=loglog_ols_slope(desc),
        nll=float(best.fun),
        n_types=n_types,
        n_tokens=n_tokens,
        converged=bool(best.success),
        free_b=free_b,
        at_bound=at_bound,
    )


# ---------------------------------------------------------------------------
# Matched-budget sampling and bootstrap
# ---------------------------------------------------------------------------


def matched_token_draw(
    per_sentence_units: Sequence[Sequence[object]],
    budget: int,
    rng: np.random.Generator,
) -> tuple[Counter, int, bool]:
    """Draw sentences without replacement until ``budget`` units are collected.

    Returns ``(counts, n_drawn, budget_met)``. The final sentence is truncated so
    ``n_drawn`` equals the budget exactly, which is what makes cross-language
    comparison of sample-size-sensitive statistics legitimate. A corpus too small
    to supply the budget returns ``budget_met=False`` rather than a silently
    short draw.
    """
    counts: Counter = Counter()
    drawn = 0
    for idx in rng.permutation(len(per_sentence_units)):
        units = per_sentence_units[int(idx)]
        # len(), not truthiness: unit streams arrive as numpy arrays from
        # intern_units, and `not array` raises for length > 1.
        if len(units) == 0:
            continue
        if drawn + len(units) >= budget:
            counts.update(units[: budget - drawn])
            drawn = budget
            break
        counts.update(units)
        drawn += len(units)
    return counts, drawn, drawn >= budget


def counts_from_interned(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Descending count vector from interned id arrays, via ``np.bincount``.

    Every statistic in this module depends only on the multiset of counts, never
    on which unit produced them, so this skips building a Counter entirely. That
    is what makes 200 bootstrap draws over 50,000-unit budgets tractable.
    """
    if not len(arrays):
        return np.empty(0, dtype=float)
    flat = np.concatenate(arrays) if len(arrays) > 1 else np.asarray(arrays[0])
    if flat.size == 0:
        return np.empty(0, dtype=float)
    counts = np.bincount(flat)
    counts = counts[counts > 0].astype(float)
    return np.sort(counts)[::-1]


def draw_counts_matched(
    cache: Sequence[np.ndarray],
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, bool]:
    """Vectorized :func:`matched_token_draw` returning a descending count vector."""
    lengths = np.fromiter((len(a) for a in cache), dtype=np.int64, count=len(cache))
    order = rng.permutation(len(cache))
    cumulative = np.cumsum(lengths[order])
    if cumulative.size == 0 or cumulative[-1] < budget:
        return counts_from_interned([cache[int(i)] for i in order]), int(cumulative[-1] if cumulative.size else 0), False

    # First sentence at which the running total reaches the budget.
    cutoff = int(np.searchsorted(cumulative, budget, side="left"))
    chosen = [cache[int(i)] for i in order[:cutoff]]
    taken = int(cumulative[cutoff - 1]) if cutoff else 0
    remainder = budget - taken
    if remainder > 0:
        chosen.append(np.asarray(cache[int(order[cutoff])])[:remainder])
    return counts_from_interned(chosen), budget, True


def draw_counts_resample(
    cache: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized :func:`resample_sentences` returning a descending count vector."""
    if not len(cache):
        return np.empty(0, dtype=float)
    idx = rng.integers(0, len(cache), size=len(cache))
    return counts_from_interned([cache[int(i)] for i in idx])


def resample_sentences(
    per_sentence_units: Sequence[Sequence[object]],
    rng: np.random.Generator,
) -> Counter:
    """Classic bootstrap: draw ``n`` sentences with replacement from ``n``.

    Used for the matched-sentence view, where the whole parallel corpus is the
    estimate and resampling only supplies the confidence interval.
    """
    n = len(per_sentence_units)
    counts: Counter = Counter()
    for idx in rng.integers(0, n, size=n):
        counts.update(per_sentence_units[int(idx)])
    return counts


def intern_units(per_sentence: Sequence[Sequence[object]]) -> List[np.ndarray]:
    """Replace unit values with compact int ids, one array per sentence.

    Every statistic here depends only on unit *identity* and frequency, never on
    the surface form, so interning is lossless for this purpose. It keeps the
    grapheme and word caches small enough to hold all 18 languages in memory at
    once, which is what makes 200 bootstrap draws cheap.
    """
    lookup: dict = {}
    out: List[np.ndarray] = []
    for units in per_sentence:
        ids = np.empty(len(units), dtype=np.int32)
        for i, unit in enumerate(units):
            got = lookup.get(unit)
            if got is None:
                got = len(lookup)
                lookup[unit] = got
            ids[i] = got
        out.append(ids)
    return out


def summarize_bootstrap(values: Iterable[float], lo_pct: float = 2.5, hi_pct: float = 97.5) -> dict:
    """Mean, percentile interval, and SD over bootstrap draws.

    Non-finite draws (a collapsed fit in one resample) are excluded and the
    surviving count is reported, so a partially-degenerate metric is visible
    rather than silently averaged as NaN.
    """
    arr = np.asarray([float(v) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "lo": float(np.percentile(finite, lo_pct)),
        "hi": float(np.percentile(finite, hi_pct)),
        "n": int(finite.size),
    }


# ---------------------------------------------------------------------------
# Text-derived unit streams
# ---------------------------------------------------------------------------


def _nfkc(text: str) -> str:
    """NFKC, matching the convention in src/metrics.py:normalize."""
    return unicodedata.normalize("NFKC", text)


def word_units(text: str) -> List[str]:
    """Whitespace-delimited words after NFKC.

    Matches ``src/metrics.py:count_words`` so word counts agree with the
    efficiency experiment. Meaningless for languages written without word
    spacing -- see ``src/zipf_langs.py:NO_WORD_BOUNDARY``.
    """
    stripped = _nfkc(text).strip()
    return stripped.split() if stripped else []


def grapheme_units(text: str) -> List[str]:
    """UAX #29 extended grapheme clusters after NFKC, excluding whitespace.

    Well defined for every script, so this is the universal reference baseline
    and the only one available for ``zho_Hans`` and ``tha_Thai``.
    """
    normalized = _nfkc(text)
    if not normalized:
        return []
    return [
        m.group()
        for m in re.finditer(r"\X", normalized, flags=re.VERSION1)
        if m.group() and not m.group().isspace()
    ]
