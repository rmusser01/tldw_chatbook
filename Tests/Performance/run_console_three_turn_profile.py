"""Revision-pinned three-turn Console latency benchmark for TASK-19641."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Mapping


ARMS = ("control", "disabled", "enabled")


def balanced_arm_order(iteration: int) -> tuple[str, str, str]:
    """Return one complete arm triple with a rotating starting arm."""
    offset = iteration % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    """Return the one-based nearest-rank percentile for ``values``."""
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires values and 0 < fraction <= 1")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def paired_p95_ratio_bounds(
    blocks: Sequence[Mapping[str, float]],
    candidate: str,
    *,
    resamples: int = 10_000,
    seed: int = 19_641,
) -> dict[str, tuple[float, float] | float]:
    """Bootstrap candidate/control p95 ratios by complete iteration block."""
    if len(blocks) < 2:
        raise ValueError("paired bootstrap requires at least two blocks")
    if candidate not in ARMS or candidate == "control":
        raise ValueError("candidate must identify a non-control arm")
    if any(set(block) != set(ARMS) for block in blocks):
        raise ValueError("paired bootstrap requires complete blocks")
    if resamples < 1:
        raise ValueError("paired bootstrap requires at least one resample")

    control_p95 = nearest_rank_percentile(
        [float(block["control"]) for block in blocks], 0.95
    )
    if control_p95 <= 0:
        raise ValueError("paired bootstrap requires a positive control p95")

    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        sampled = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        sampled_control = nearest_rank_percentile(
            [float(block["control"]) for block in sampled], 0.95
        )
        if sampled_control <= 0:
            raise ValueError("paired bootstrap requires a positive control p95")
        sampled_candidate = nearest_rank_percentile(
            [float(block[candidate]) for block in sampled], 0.95
        )
        ratios.append(sampled_candidate / sampled_control)

    return {
        "two_sided_95": (
            nearest_rank_percentile(ratios, 0.025),
            nearest_rank_percentile(ratios, 0.975),
        ),
        "one_sided_lower_95": nearest_rank_percentile(ratios, 0.05),
        "one_sided_upper_95": nearest_rank_percentile(ratios, 0.95),
    }


def sample_heartbeat_p95_ns(tick_lateness_ns: Sequence[int]) -> float:
    """Reduce one sample's raw heartbeat ticks to one equally weighted p95."""
    if not tick_lateness_ns:
        raise ValueError("heartbeat vector must not be empty")
    return nearest_rank_percentile(tick_lateness_ns, 0.95)
