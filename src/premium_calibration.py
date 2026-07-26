"""Three-arm shared token-premium calibration for Plan A mixing."""

from __future__ import annotations

import math
from typing import Mapping


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError("premiums must be positive")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def shared_premiums(
    arm_premiums: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute shared premium as the geometric mean across arms for each language."""
    if not arm_premiums:
        raise ValueError("arm_premiums must be non-empty")
    languages = sorted({lang for premiums in arm_premiums.values() for lang in premiums})
    shared: dict[str, float] = {}
    for lang in languages:
        values = []
        for arm, premiums in arm_premiums.items():
            if lang not in premiums:
                raise KeyError(f"language {lang} missing from arm {arm}")
            values.append(float(premiums[lang]))
        shared[lang] = geometric_mean(values)
    return shared


def target_token_shares(shared: Mapping[str, float]) -> dict[str, float]:
    total = sum(shared.values())
    if total <= 0:
        raise ValueError("shared premiums must sum to a positive value")
    return {lang: value / total for lang, value in shared.items()}


def damp_shares(
    previous: Mapping[str, float],
    target: Mapping[str, float],
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    languages = sorted(set(previous) | set(target))
    mixed = {
        lang: (1.0 - alpha) * float(previous.get(lang, 0.0))
        + alpha * float(target.get(lang, 0.0))
        for lang in languages
    }
    total = sum(mixed.values())
    if total <= 0:
        raise ValueError("damped shares must sum to a positive value")
    return {lang: value / total for lang, value in mixed.items()}


def max_share_delta(previous: Mapping[str, float], updated: Mapping[str, float]) -> float:
    languages = set(previous) | set(updated)
    return max(
        abs(float(updated.get(lang, 0.0)) - float(previous.get(lang, 0.0)))
        for lang in languages
    )
