"""Tests for src/zipf.py.

The estimator is the piece most likely to be subtly wrong, so it is pinned
against synthetic data with known ground truth before it ever sees a corpus.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from src.zipf import (
    DEFAULT_RENYI_ORDER,
    counts_descending,
    counts_from_interned,
    distribution_stats,
    draw_counts_matched,
    draw_counts_resample,
    fit_zipf_mandelbrot,
    grapheme_units,
    ks_distance,
    intern_units,
    ks_vs_pure_zipf,
    loglog_ols_slope,
    loglog_rmse,
    matched_token_draw,
    renyi_entropy,
    resample_sentences,
    shannon_entropy,
    summarize_bootstrap,
    word_units,
    zm_probabilities,
)


def sample_zm(alpha: float, b: float, n_types: int, n_draws: int, seed: int) -> Counter:
    """Draw ``n_draws`` samples from Zipf-Mandelbrot(alpha, b) over n_types ranks."""
    probs = zm_probabilities(n_types, alpha, b)
    rng = np.random.default_rng(seed)
    draws = rng.choice(n_types, size=n_draws, p=probs)
    return Counter(int(d) for d in draws)


# --------------------------------------------------------------------------
# zm_probabilities
# --------------------------------------------------------------------------


def test_zm_probabilities_normalized():
    p = zm_probabilities(1000, 1.1, 2.0)
    assert p.shape == (1000,)
    assert math.isclose(float(p.sum()), 1.0, rel_tol=1e-12)
    # Strictly decreasing in rank for positive alpha.
    assert np.all(np.diff(p) < 0)


def test_zm_probabilities_alpha_zero_is_uniform():
    p = zm_probabilities(50, 0.0, 0.0)
    assert np.allclose(p, 1.0 / 50)


# --------------------------------------------------------------------------
# Zipf-Mandelbrot MLE recovery
# --------------------------------------------------------------------------


def test_mle_recovers_known_alpha_and_b():
    truth_alpha, truth_b = 1.2, 2.7
    counts = sample_zm(truth_alpha, truth_b, n_types=5000, n_draws=200_000, seed=11)
    fit = fit_zipf_mandelbrot(counts)
    assert fit.converged
    assert fit.alpha == pytest.approx(truth_alpha, abs=0.08)
    assert fit.b == pytest.approx(truth_b, abs=0.8)


def test_mle_recovers_pure_zipf():
    counts = sample_zm(1.0, 0.0, n_types=4000, n_draws=200_000, seed=23)
    fit = fit_zipf_mandelbrot(counts)
    assert fit.alpha == pytest.approx(1.0, abs=0.08)
    assert fit.b == pytest.approx(0.0, abs=0.5)
    # Data really is Zipf, so distance from the law must be small.
    assert fit.ks_zipf < 0.05


def test_mle_recovers_steep_exponent():
    counts = sample_zm(1.8, 0.0, n_types=3000, n_draws=200_000, seed=31)
    fit = fit_zipf_mandelbrot(counts)
    assert fit.alpha == pytest.approx(1.8, abs=0.10)


def test_fixed_b_fit_pins_b_to_zero():
    counts = sample_zm(1.3, 0.0, n_types=2000, n_draws=100_000, seed=41)
    fit = fit_zipf_mandelbrot(counts, free_b=False)
    assert fit.b == 0.0
    assert fit.alpha == pytest.approx(1.3, abs=0.08)


# --------------------------------------------------------------------------
# The uniform case: alpha -> 0, well fit by ZM, but far from Zipf's law.
# These are two different questions and the code must separate them.
# --------------------------------------------------------------------------


def test_uniform_input_fits_alpha_zero_and_is_close_to_its_own_zm():
    counts = Counter({i: 200 for i in range(500)})
    fit = fit_zipf_mandelbrot(counts)
    assert fit.alpha == pytest.approx(0.0, abs=0.05)
    # ZM with alpha=0 IS the uniform distribution, so goodness-of-fit is good.
    assert fit.ks < 0.02
    # alpha = 0 is the uniform limit and a natural edge of the parameter space,
    # not an artificial truncation, so it must NOT raise the bound flag.
    assert not fit.at_bound


# --------------------------------------------------------------------------
# Bound saturation must be visible, never silently read as an estimate
# --------------------------------------------------------------------------


def test_steep_small_support_fit_stays_interior():
    """Grapheme-scale supports need headroom above alpha = 10.

    English grapheme clusters (V=110) optimize at alpha ~17, b ~133. A ceiling of
    10 truncated that and pinned 40 of 144 grapheme fits at exactly 10.0.
    """
    # A flat-then-cliff shape of the kind letter frequencies produce.
    probs = zm_probabilities(110, 17.0, 133.0)
    counts = Counter({i: max(1, int(round(p * 220_000))) for i, p in enumerate(probs)})
    fit = fit_zipf_mandelbrot(counts)
    assert fit.alpha > 10.0, "must not be capped at the old ceiling"
    assert not fit.at_bound, f"optimum should be interior, got alpha={fit.alpha}"


def test_at_bound_flag_raised_when_alpha_hits_the_ceiling():
    """A distribution steeper than the ceiling can express must be flagged."""
    probs = zm_probabilities(60, 120.0, 400.0)
    counts = Counter({i: max(1, int(round(p * 5_000_000))) for i, p in enumerate(probs)})
    fit = fit_zipf_mandelbrot(counts)
    if fit.alpha >= 49.9:
        assert fit.at_bound, "ceiling reached but not reported"
    else:
        assert not fit.at_bound


def test_at_bound_false_for_ordinary_fits():
    counts = sample_zm(1.1, 1.0, n_types=3000, n_draws=150_000, seed=97)
    fit = fit_zipf_mandelbrot(counts)
    assert not fit.at_bound
    assert 0.5 < fit.alpha < 3.0


def test_uniform_input_is_far_from_pure_zipf():
    counts = Counter({i: 200 for i in range(500)})
    fit = fit_zipf_mandelbrot(counts)
    # Deviation from the *law* is the study's headline and must be large here.
    assert fit.ks_zipf > 0.20


def test_ks_vs_pure_zipf_is_zero_for_exact_zipf_expectation():
    # Expected counts under exact Zipf: deviation from the law should vanish.
    n_types, total = 400, 1_000_000
    probs = zm_probabilities(n_types, 1.0, 0.0)
    counts = Counter({i: int(round(p * total)) for i, p in enumerate(probs)})
    assert ks_vs_pure_zipf(counts_descending(counts)) < 1e-3


# --------------------------------------------------------------------------
# Degenerate / truncated support -- the Santali and Amharic regime
# --------------------------------------------------------------------------


def test_tiny_truncated_support_fits_without_error():
    # 219 types is Santali's measured active vocabulary under o200k.
    counts = sample_zm(1.1, 1.0, n_types=219, n_draws=719_000, seed=53)
    fit = fit_zipf_mandelbrot(counts)
    assert fit.converged
    assert fit.n_types == 219
    for value in (fit.alpha, fit.b, fit.ks, fit.ks_zipf, fit.loglog_rmse):
        assert math.isfinite(value), "no NaN may leak into the results CSVs"


def test_single_type_is_flagged_not_crashed():
    fit = fit_zipf_mandelbrot(Counter({7: 1000}))
    assert fit.n_types == 1
    assert not fit.converged, "a single type cannot support a power-law fit"
    stats = distribution_stats(Counter({7: 1000}))
    assert stats.entropy_nats == pytest.approx(0.0)
    assert stats.entropy_norm == 0.0
    assert stats.effective_vocab == pytest.approx(1.0)


def test_empty_input_is_flagged_not_crashed():
    fit = fit_zipf_mandelbrot(Counter())
    assert fit.n_types == 0
    assert not fit.converged
    stats = distribution_stats(Counter())
    assert stats.n_tokens == 0
    assert stats.n_types == 0


# --------------------------------------------------------------------------
# Entropy family
# --------------------------------------------------------------------------


def test_uniform_entropy_is_log_v():
    counts = Counter({i: 10 for i in range(64)})
    stats = distribution_stats(counts)
    assert stats.entropy_nats == pytest.approx(math.log(64))
    assert stats.entropy_bits == pytest.approx(6.0)
    assert stats.entropy_norm == pytest.approx(1.0)
    assert stats.effective_vocab == pytest.approx(64.0)


def test_effective_vocab_never_exceeds_support():
    """Invariant that also guards the real results: exp(H) <= V_active."""
    for seed, alpha in enumerate([0.3, 0.8, 1.0, 1.5, 2.2]):
        counts = sample_zm(alpha, 0.0, n_types=800, n_draws=60_000, seed=100 + seed)
        stats = distribution_stats(counts)
        assert stats.effective_vocab <= stats.n_types + 1e-9
        assert 0.0 <= stats.entropy_norm <= 1.0 + 1e-9


def test_all_distinct_types_is_uniform():
    counts = Counter({i: 1 for i in range(500)})
    stats = distribution_stats(counts)
    assert stats.n_tokens == 500
    assert stats.n_types == 500
    assert stats.entropy_norm == pytest.approx(1.0)


def test_skewed_distribution_has_low_effective_vocab():
    # One token dominates: effective vocab collapses far below support size.
    counts = Counter({0: 100_000})
    counts.update({i: 1 for i in range(1, 501)})
    stats = distribution_stats(counts)
    assert stats.n_types == 501
    assert stats.effective_vocab < 10.0


def test_renyi_entropy_order_one_matches_shannon():
    counts = sample_zm(1.1, 0.5, n_types=600, n_draws=50_000, seed=61)
    assert renyi_entropy(counts_descending(counts), 1.0) == pytest.approx(
        shannon_entropy(counts_descending(counts)), rel=1e-9
    )


def test_renyi_entropy_decreases_with_order():
    counts = sample_zm(1.1, 0.5, n_types=600, n_draws=50_000, seed=67)
    desc = counts_descending(counts)
    orders = [0.5, 1.0, 2.0, DEFAULT_RENYI_ORDER, 4.0]
    values = [renyi_entropy(desc, a) for a in orders]
    assert all(x >= y - 1e-9 for x, y in zip(values, values[1:]))


def test_renyi_efficiency_reported_against_both_denominators():
    counts = sample_zm(1.0, 0.0, n_types=1000, n_draws=80_000, seed=71)
    stats = distribution_stats(counts, full_vocab_size=199_998)
    # Normalizing by the active support gives a shape measure; normalizing by
    # the whole tokenizer vocab gives a utilization measure. The latter must be
    # smaller, since the support is a subset.
    assert stats.renyi_efficiency_active > stats.renyi_efficiency_full
    assert 0.0 <= stats.renyi_efficiency_full <= 1.0


def test_full_vocab_denominator_optional():
    stats = distribution_stats(sample_zm(1.0, 0.0, 200, 10_000, seed=73))
    assert stats.renyi_efficiency_full is None
    assert stats.entropy_norm_full is None


# --------------------------------------------------------------------------
# Goodness-of-fit helpers
# --------------------------------------------------------------------------


def test_ks_distance_zero_for_exact_model_expectation():
    n_types, total = 300, 2_000_000
    probs = zm_probabilities(n_types, 1.15, 1.5)
    desc = np.array([p * total for p in probs])
    assert ks_distance(desc, 1.15, 1.5) < 1e-6


def test_loglog_rmse_zero_for_exact_model_expectation():
    n_types, total = 300, 2_000_000
    probs = zm_probabilities(n_types, 1.15, 1.5)
    desc = np.array([p * total for p in probs])
    assert loglog_rmse(desc, 1.15, 1.5) < 1e-6


def test_counts_descending_sorts_and_drops_zeros():
    desc = counts_descending(Counter({"a": 3, "b": 10, "c": 0, "d": 7}))
    assert list(desc) == [10.0, 7.0, 3.0]


def test_loglog_ols_slope_recovers_exponent_on_noiseless_curve():
    n_types, total = 2000, 5_000_000
    probs = zm_probabilities(n_types, 1.25, 0.0)
    desc = np.array([p * total for p in probs])
    assert loglog_ols_slope(desc) == pytest.approx(1.25, abs=0.01)


def test_hapax_dominated_sample_biases_alpha_downward():
    """Pins why word-level fits land below 1 on a 29k-word corpus.

    When a sample is small relative to the true support, most types are seen
    once, the observed rank-frequency curve is truncated at frequency 1, and MLE
    over that observed support returns a flatter exponent than the truth. The
    real run shows exactly this: word-unit alpha runs 0.64-0.94 rather than ~1,
    Spearman(types-per-token, alpha) = -0.95, and alpha rises toward 1 for all
    16 languages when the full corpus is used instead of the matched budget.

    This is a property of small corpora, not of the estimator, and it is why the
    study's attribution rests on token-minus-word *deltas* -- both units share
    the corpus and its sample size, so the bias largely cancels.
    """
    truth = 1.0
    sizes = [30_000, 100_000, 500_000, 2_000_000]
    fits = [
        fit_zipf_mandelbrot(sample_zm(truth, 0.0, n_types=40_000, n_draws=n, seed=77))
        for n in sizes
    ]
    alphas = [f.alpha for f in fits]

    # Every estimate is at or below the truth, and they rise monotonically toward
    # it as the sample grows: 0.90 -> 0.95 -> 0.99 -> 1.00 for these sizes.
    assert all(a <= truth + 1e-3 for a in alphas)
    assert all(x < y for x, y in zip(alphas, alphas[1:])), alphas
    assert alphas[0] < truth - 0.05, "small sample must be materially flatter"
    assert alphas[-1] == pytest.approx(truth, abs=0.01), "large sample recovers truth"

    # The mechanism: the small sample is hapax dominated, so its curve is
    # truncated at frequency 1. Type-to-token ratio falls as the sample grows.
    ratios = [f.n_types / n for f, n in zip(fits, sizes)]
    assert all(x > y for x, y in zip(ratios, ratios[1:])), ratios
    assert ratios[0] > 10 * ratios[-1]


@pytest.mark.parametrize("truth", [0.8, 1.0, 1.3])
@pytest.mark.parametrize("seed", [83, 84, 85])
def test_mle_beats_ols_on_sampled_data(truth: float, seed: int):
    """Documents why OLS is not the primary estimator (Clauset et al. 2009).

    The *direction* of OLS bias is regime-dependent, so only the magnitude claim
    is asserted: MLE recovers the true exponent an order of magnitude more
    accurately. Measured across these nine configurations, MLE error stays under
    0.007 while OLS error runs 0.03-0.07.
    """
    counts = sample_zm(truth, 0.0, n_types=4000, n_draws=200_000, seed=seed)
    mle_err = abs(fit_zipf_mandelbrot(counts).alpha - truth)
    ols_err = abs(loglog_ols_slope(counts_descending(counts)) - truth)
    assert mle_err < 0.01
    assert ols_err > mle_err * 3.0


def test_loglog_ols_slope_needs_two_points():
    assert math.isnan(loglog_ols_slope(np.array([5.0])))


# --------------------------------------------------------------------------
# Matched-token sampling
# --------------------------------------------------------------------------


def _per_sentence(n_sentences: int, per: int) -> list[list[int]]:
    return [[s * per + i for i in range(per)] for s in range(n_sentences)]


def test_matched_token_draw_hits_budget_exactly():
    counts, drawn, met = matched_token_draw(
        _per_sentence(500, 40), budget=5000, rng=np.random.default_rng(3)
    )
    assert met
    assert drawn == 5000
    assert sum(counts.values()) == 5000


def test_matched_token_draw_is_deterministic_under_seed():
    units = _per_sentence(400, 25)
    a, _, _ = matched_token_draw(units, 3000, np.random.default_rng(9))
    b, _, _ = matched_token_draw(units, 3000, np.random.default_rng(9))
    c, _, _ = matched_token_draw(units, 3000, np.random.default_rng(10))
    assert a == b
    assert a != c, "different seeds must give different draws"


def test_matched_token_draw_flags_insufficient_corpus():
    counts, drawn, met = matched_token_draw(
        _per_sentence(10, 5), budget=5000, rng=np.random.default_rng(1)
    )
    assert not met, "must report, not silently return a short draw"
    assert drawn == 50
    assert sum(counts.values()) == 50


def test_matched_token_draw_accepts_interned_numpy_arrays():
    """The real pipeline passes intern_units output, not Python lists.

    Guards against truthiness checks on arrays, which raise for length > 1.
    """
    cache = intern_units([[f"w{s}_{i}" for i in range(30)] for s in range(200)])
    counts, drawn, met = matched_token_draw(
        cache, budget=2000, rng=np.random.default_rng(2)
    )
    assert met
    assert drawn == 2000
    assert sum(counts.values()) == 2000


def test_resample_sentences_accepts_interned_numpy_arrays():
    cache = intern_units([[f"w{i}" for i in range(10)] for _ in range(50)])
    counts = resample_sentences(cache, np.random.default_rng(2))
    assert sum(counts.values()) == 500


def test_matched_token_draw_skips_empty_sentences():
    cache = intern_units([[], ["a", "b"], [], ["c", "d"]])
    counts, drawn, met = matched_token_draw(
        cache, budget=4, rng=np.random.default_rng(1)
    )
    assert met and drawn == 4


def test_matched_token_draw_samples_without_replacement():
    # Every unit id here is globally unique, so any duplicate count proves a
    # sentence was drawn twice.
    counts, _, _ = matched_token_draw(
        _per_sentence(300, 20), budget=2000, rng=np.random.default_rng(5)
    )
    assert set(counts.values()) == {1}


def test_fit_accepts_a_start_hint_and_matches_the_default_grid():
    """Bootstrap draws pass the full-corpus estimate as their only start."""
    counts = sample_zm(1.15, 1.5, n_types=2000, n_draws=120_000, seed=91)
    full = fit_zipf_mandelbrot(counts)
    hinted = fit_zipf_mandelbrot(counts, starts=[(full.alpha, full.b)])
    assert hinted.alpha == pytest.approx(full.alpha, abs=1e-3)
    assert hinted.b == pytest.approx(full.b, abs=1e-2)


# --------------------------------------------------------------------------
# Sentence resampling and interning
# --------------------------------------------------------------------------


def test_resample_sentences_preserves_total_sentence_count():
    units = _per_sentence(200, 10)
    counts = resample_sentences(units, np.random.default_rng(7))
    # 200 sentences drawn with replacement, 10 units each.
    assert sum(counts.values()) == 2000


def test_resample_sentences_draws_with_replacement():
    """With replacement, some sentence must repeat -- unique ids get count > 1."""
    units = _per_sentence(100, 5)
    counts = resample_sentences(units, np.random.default_rng(7))
    assert max(counts.values()) > 1


def test_resample_sentences_deterministic_under_seed():
    units = _per_sentence(100, 5)
    a = resample_sentences(units, np.random.default_rng(4))
    b = resample_sentences(units, np.random.default_rng(4))
    assert a == b


def test_intern_units_preserves_frequency_distribution():
    per_sentence = [["the", "cat"], ["the", "dog", "the"]]
    interned = intern_units(per_sentence)
    assert [len(a) for a in interned] == [2, 3]
    original = Counter(u for s in per_sentence for u in s)
    mapped = Counter(int(v) for a in interned for v in a)
    # Same multiset of frequencies, just relabelled.
    assert sorted(original.values()) == sorted(mapped.values())
    assert len(mapped) == len(original)


def test_intern_units_reuses_ids_across_sentences():
    interned = intern_units([["a"], ["a"], ["b"]])
    assert int(interned[0][0]) == int(interned[1][0])
    assert int(interned[2][0]) != int(interned[0][0])


def test_intern_units_handles_empty_sentences():
    interned = intern_units([[], ["x"], []])
    assert [len(a) for a in interned] == [0, 1, 0]


# --------------------------------------------------------------------------
# Vectorized count paths -- must agree exactly with the Counter versions,
# since only the fast ones are used on the real corpus.
# --------------------------------------------------------------------------


def test_counts_from_interned_matches_counter():
    cache = intern_units([["a", "b", "a"], ["b", "c"], ["a"]])
    fast = counts_from_interned(cache)
    slow = counts_descending(Counter(int(v) for a in cache for v in a))
    assert list(fast) == list(slow)


def test_counts_from_interned_empty():
    assert counts_from_interned([]).size == 0
    assert counts_from_interned(intern_units([[], []])).size == 0


@pytest.mark.parametrize("seed", [1, 2, 3, 17])
def test_draw_counts_matched_agrees_with_counter_version(seed: int):
    cache = intern_units([[f"w{(s * 7 + i) % 90}" for i in range(30)] for s in range(120)])
    fast, drawn_fast, met_fast = draw_counts_matched(
        cache, 1500, np.random.default_rng(seed)
    )
    slow_counts, drawn_slow, met_slow = matched_token_draw(
        cache, 1500, np.random.default_rng(seed)
    )
    slow = counts_descending(slow_counts)
    assert (drawn_fast, met_fast) == (drawn_slow, met_slow)
    assert list(fast) == list(slow)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_draw_counts_resample_agrees_with_counter_version(seed: int):
    cache = intern_units([[f"w{(s + i) % 40}" for i in range(12)] for s in range(80)])
    fast = draw_counts_resample(cache, np.random.default_rng(seed))
    slow = counts_descending(resample_sentences(cache, np.random.default_rng(seed)))
    assert list(fast) == list(slow)


def test_draw_counts_matched_flags_insufficient_corpus():
    cache = intern_units([["a", "b"], ["c"]])
    counts, drawn, met = draw_counts_matched(cache, 500, np.random.default_rng(1))
    assert not met
    assert drawn == 3
    assert counts.sum() == 3


def test_draw_counts_matched_hits_budget_exactly():
    cache = intern_units([[f"u{s}_{i}" for i in range(17)] for s in range(300)])
    counts, drawn, met = draw_counts_matched(cache, 2000, np.random.default_rng(8))
    assert met and drawn == 2000
    assert counts.sum() == 2000


# --------------------------------------------------------------------------
# Bootstrap summary
# --------------------------------------------------------------------------


def test_summarize_bootstrap_reports_mean_and_interval():
    summary = summarize_bootstrap([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["lo"] <= summary["mean"] <= summary["hi"]
    assert summary["n"] == 5


def test_summarize_bootstrap_ignores_non_finite_values():
    summary = summarize_bootstrap([1.0, float("nan"), 3.0, float("inf")])
    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(2.0)


def test_summarize_bootstrap_empty_is_nan_not_crash():
    summary = summarize_bootstrap([])
    assert summary["n"] == 0
    assert math.isnan(summary["mean"])


# --------------------------------------------------------------------------
# Unit extraction
# --------------------------------------------------------------------------


def test_word_units_split_on_whitespace_after_nfkc():
    assert word_units("the  quick\tbrown\nfox") == ["the", "quick", "brown", "fox"]


def test_word_units_empty_text():
    assert word_units("   ") == []


def test_grapheme_units_keep_combining_marks_attached():
    # Devanagari क + ि is one extended grapheme cluster, not two.
    assert grapheme_units("कि") == ["कि"]


def test_grapheme_units_exclude_whitespace():
    assert grapheme_units("a b") == ["a", "b"]


def test_grapheme_units_handle_emoji_zwj_sequence():
    assert len(grapheme_units("👩‍💻")) == 1
