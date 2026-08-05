"""Tests for Plan B UniMax allocation (plans/03-model-pretraining.md §5 / §9)."""

from __future__ import annotations

import math

import pytest

from src.plan_b_mixture import (
    PLANNING_AVAILABLE_BYTES,
    PLANNING_EPOCH_CAP,
    Allocation,
    acquire_pool_bytes,
    allocation_shares,
    passes_over_pool,
    planning_budget_bytes,
    unimax_allocation,
    unique_pool_bytes,
)


def test_allocation_sums_to_budget():
    avail = {"a": 100.0, "b": 100.0, "c": 100.0}
    budget = 90.0
    alloc = unimax_allocation(avail, budget, epoch_cap=4.0)
    assert sum(a.bytes for a in alloc.values()) == pytest.approx(budget)


def test_no_language_exceeds_epoch_cap():
    avail = {"scarce": 10.0, "rich": 1_000.0}
    alloc = unimax_allocation(avail, budget_bytes=100.0, epoch_cap=4.0)
    for lang, a in alloc.items():
        assert a.epochs_of_available <= 4.0 + 1e-9
        assert a.bytes <= 4.0 * avail[lang] + 1e-6


def test_unconstrained_set_is_exactly_uniform():
    avail = {"x": 1e12, "y": 1e12, "z": 1e12}
    budget = 300.0
    alloc = unimax_allocation(avail, budget, epoch_cap=4.0)
    shares = allocation_shares(alloc)
    for s in shares.values():
        assert s == pytest.approx(1.0 / 3.0)
    assert all(not a.capped for a in alloc.values())


def test_production_six_language_hat_capped_others_uniform():
    budget = planning_budget_bytes()  # 20B × 3.9 = 78e9
    alloc = unimax_allocation(
        PLANNING_AVAILABLE_BYTES, budget, epoch_cap=PLANNING_EPOCH_CAP
    )
    assert alloc["hat_Latn"].capped is True
    assert alloc["hat_Latn"].epochs_of_available == pytest.approx(4.0)
    assert alloc["hat_Latn"].bytes == pytest.approx(
        PLANNING_EPOCH_CAP * PLANNING_AVAILABLE_BYTES["hat_Latn"]
    )

    others = [lang for lang in alloc if lang != "hat_Latn"]
    shares = allocation_shares(alloc)
    other_shares = [shares[lang] for lang in others]
    assert max(other_shares) == pytest.approx(min(other_shares), rel=1e-9)
    assert shares["hat_Latn"] == pytest.approx(0.0396, abs=5e-4)

    pools = unique_pool_bytes(alloc, PLANNING_AVAILABLE_BYTES)
    assert sum(pools.values()) == pytest.approx(65.3e9, rel=5e-3)

    acquire = acquire_pool_bytes(alloc, PLANNING_AVAILABLE_BYTES, headroom=0.15)
    assert acquire["hat_Latn"] == pytest.approx(PLANNING_AVAILABLE_BYTES["hat_Latn"])
    assert acquire["swh_Latn"] == pytest.approx(PLANNING_AVAILABLE_BYTES["swh_Latn"])
    assert acquire["eng_Latn"] == pytest.approx(alloc["eng_Latn"].bytes * 1.15)

    pops = passes_over_pool(alloc, pools)
    assert pops["hat_Latn"] == pytest.approx(4.0)
    assert pops["eng_Latn"] == pytest.approx(1.0)


def test_budget_above_cap_times_sum_raises():
    avail = {"a": 10.0, "b": 10.0}
    with pytest.raises(ValueError, match="exceeds"):
        unimax_allocation(avail, budget_bytes=100.0, epoch_cap=4.0)


def test_single_language_degenerates():
    alloc = unimax_allocation({"only": 50.0}, budget_bytes=40.0, epoch_cap=4.0)
    assert set(alloc) == {"only"}
    assert alloc["only"].bytes == pytest.approx(40.0)
    assert alloc["only"].capped is False


def test_single_language_at_exact_cap_is_not_marked_capped():
    """``capped`` is set only when the uniform share *exceeds* the epoch cap.

    A budget that lands exactly on ``epoch_cap × available`` takes the uniform
    branch (epochs == cap, capped == False). Overshoot forces the cap branch.
    """
    at_cap = unimax_allocation({"only": 10.0}, budget_bytes=40.0, epoch_cap=4.0)
    assert at_cap["only"].bytes == pytest.approx(40.0)
    assert at_cap["only"].epochs_of_available == pytest.approx(4.0)
    assert at_cap["only"].capped is False

    # Two languages: scarce would want more than 4×available under a naive split.
    over = unimax_allocation(
        {"scarce": 10.0, "rich": 1_000.0}, budget_bytes=100.0, epoch_cap=4.0
    )
    assert over["scarce"].capped is True
    assert over["scarce"].bytes == pytest.approx(40.0)


def test_input_order_does_not_change_result():
    avail_a = {"zho_Hans": 1000.0, "hat_Latn": 10.0, "eng_Latn": 500.0}
    avail_b = {"eng_Latn": 500.0, "hat_Latn": 10.0, "zho_Hans": 1000.0}
    a = unimax_allocation(avail_a, 100.0, 4.0)
    b = unimax_allocation(avail_b, 100.0, 4.0)
    assert {k: (v.bytes, v.capped) for k, v in a.items()} == {
        k: (v.bytes, v.capped) for k, v in b.items()
    }


def test_unbounded_language_epochs_zero():
    alloc = unimax_allocation(
        {"eng_Latn": math.inf, "hat_Latn": 1.0},
        budget_bytes=10.0,
        epoch_cap=4.0,
    )
    assert alloc["hat_Latn"].capped is True
    assert alloc["eng_Latn"].epochs_of_available == 0.0
    assert alloc["eng_Latn"].capped is False


def test_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        unimax_allocation({}, 10.0, 4.0)
    with pytest.raises(ValueError):
        unimax_allocation({"a": 1.0}, 0.0, 4.0)
    with pytest.raises(ValueError):
        unimax_allocation({"a": 0.0}, 10.0, 4.0)
