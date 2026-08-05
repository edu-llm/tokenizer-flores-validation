"""UniMax allocation for Plan B pretraining mixtures.

Pure functions, no I/O. Spec: ``plans/03-model-pretraining.md`` §5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

# Planning constants from plans/03 §5.2 and plans/00 §4.2 (bytes, not GB).
PLANNING_BYTES_PER_TOKEN = 3.9
PLANNING_TARGET_TOKENS = 20_000_000_000
PLANNING_EPOCH_CAP = 4.0
PLANNING_HEADROOM = 0.15

# Availability used for planning (FineWeb / FineWeb-2 + ladder totals for
# hat/swh). English is unbounded at this scale.
PLANNING_AVAILABLE_BYTES: dict[str, float] = {
    "hat_Latn": 0.772e9,
    "swh_Latn": 4.577e9,
    "eng_Latn": math.inf,
    "hin_Deva": 34.4e9,
    "hun_Latn": 98.6e9,
    "zho_Hans": 1622e9,
}


@dataclass(frozen=True)
class Allocation:
    bytes: float
    epochs_of_available: float
    capped: bool


def planning_budget_bytes(
    *,
    target_tokens: float = PLANNING_TARGET_TOKENS,
    bytes_per_token: float = PLANNING_BYTES_PER_TOKEN,
) -> float:
    if target_tokens <= 0 or bytes_per_token <= 0:
        raise ValueError("target_tokens and bytes_per_token must be positive")
    return float(target_tokens) * float(bytes_per_token)


def unimax_allocation(
    available_bytes: Mapping[str, float],
    budget_bytes: float,
    epoch_cap: float,
) -> dict[str, Allocation]:
    """Maximize uniformity subject to a per-language epoch cap (UniMax).

    Sort languages ascending by availability. Walk them: if a language's
    uniform share of the *remaining* budget exceeds ``epoch_cap × available``,
    allocate the capped amount and continue; otherwise allocate that uniform
    share to every language not yet assigned and stop.
    """
    if not available_bytes:
        raise ValueError("available_bytes must be non-empty")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if epoch_cap <= 0:
        raise ValueError("epoch_cap must be positive")

    for lang, avail in available_bytes.items():
        if not isinstance(lang, str) or not lang:
            raise ValueError(f"invalid language key: {lang!r}")
        if avail is None or float(avail) <= 0:
            raise ValueError(f"available_bytes[{lang!r}] must be positive; got {avail!r}")

    avail = {lang: float(v) for lang, v in available_bytes.items()}
    finite_sum = sum(v for v in avail.values() if math.isfinite(v))
    # Unbounded languages (inf) do not increase the satisfiable ceiling by themselves;
    # the raise below only fires when every language is finite and the budget is too large.
    if all(math.isfinite(v) for v in avail.values()):
        max_satisfiable = epoch_cap * finite_sum
        if budget_bytes > max_satisfiable + 1e-6:
            raise ValueError(
                f"budget_bytes {budget_bytes} exceeds epoch_cap×sum(available) "
                f"= {max_satisfiable}; UniMax cannot satisfy this budget"
            )

    # Stable sort: availability ascending, then language code.
    ordered = sorted(avail.keys(), key=lambda lang: (avail[lang], lang))
    remaining_budget = float(budget_bytes)
    remaining = list(ordered)
    out: dict[str, Allocation] = {}

    while remaining:
        n = len(remaining)
        uniform = remaining_budget / n
        lang = remaining[0]
        cap_bytes = epoch_cap * avail[lang]
        if uniform > cap_bytes + 1e-9 and math.isfinite(avail[lang]):
            allocated = cap_bytes
            epochs = epoch_cap
            out[lang] = Allocation(bytes=allocated, epochs_of_available=epochs, capped=True)
            remaining_budget -= allocated
            remaining = remaining[1:]
            continue
        # First uncapped language: give every remaining language the uniform share.
        for lang in remaining:
            allocated = uniform
            epochs = (
                0.0
                if not math.isfinite(avail[lang])
                else allocated / avail[lang]
            )
            out[lang] = Allocation(
                bytes=allocated,
                epochs_of_available=epochs,
                capped=False,
            )
        break

    if set(out) != set(avail):
        raise RuntimeError("internal error: allocation missing languages")
    return out


def unique_pool_bytes(
    allocation: Mapping[str, Allocation],
    available_bytes: Mapping[str, float],
) -> dict[str, float]:
    """Unique text to hold per language (min of allocation and availability)."""
    out: dict[str, float] = {}
    for lang, alloc in allocation.items():
        avail = float(available_bytes[lang])
        if math.isfinite(avail):
            out[lang] = min(alloc.bytes, avail)
        else:
            out[lang] = alloc.bytes
    return out


def acquire_pool_bytes(
    allocation: Mapping[str, Allocation],
    available_bytes: Mapping[str, float],
    *,
    headroom: float = PLANNING_HEADROOM,
) -> dict[str, float]:
    """Bytes to pull: full ladder for availability-bound langs; allocation×(1+headroom) else."""
    if headroom < 0:
        raise ValueError("headroom must be >= 0")
    pools = unique_pool_bytes(allocation, available_bytes)
    out: dict[str, float] = {}
    for lang, unique in pools.items():
        avail = float(available_bytes[lang])
        alloc = allocation[lang].bytes
        if math.isfinite(avail) and unique >= avail - 1e-6:
            # Drawn in full — no headroom exists.
            out[lang] = avail
        else:
            target = alloc * (1.0 + headroom)
            out[lang] = min(target, avail) if math.isfinite(avail) else target
    return out


def passes_over_pool(
    allocation: Mapping[str, Allocation],
    pool_bytes: Mapping[str, float],
) -> dict[str, float]:
    """Training passes over the *acquired* pool (sampler property, not UniMax)."""
    out: dict[str, float] = {}
    for lang, alloc in allocation.items():
        pool = float(pool_bytes[lang])
        if pool <= 0:
            raise ValueError(f"pool_bytes[{lang!r}] must be positive")
        out[lang] = alloc.bytes / pool
    return out


def allocation_shares(allocation: Mapping[str, Allocation]) -> dict[str, float]:
    total = sum(a.bytes for a in allocation.values())
    if total <= 0:
        raise ValueError("allocation total must be positive")
    return {lang: a.bytes / total for lang, a in allocation.items()}
