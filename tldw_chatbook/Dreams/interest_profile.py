"""[dreams] Interest-profile weight math and the profile snapshot.

Pure decay/merge functions plus the ``snapshot()`` read the discovery cycle
assembles its queries from (spec §interest profile). Weight semantics:
0.0-1.0, decayed toward a floor with a half-life (14 days by default) so
last month's obsession doesn't dominate forever; goals are excluded here —
they never decay and change only by direct user edit.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Dreams.settings import dreams_setting

if TYPE_CHECKING:
    from tldw_chatbook.DB.Dreams_DB import DreamsDB


def decay_weights(
    topics: list[dict],
    *,
    now_epoch: float,
    half_life_days: float = 14.0,
    floor: float = 0.05,
) -> list[dict]:
    """Decay each topic's ``weight`` toward ``floor`` by time since its boost.

    Pure: returns new row dicts, never mutates the input. A topic's
    ``last_boosted_at`` is an epoch float (0.0/missing = never boosted, i.e.
    maximally stale). Each elapsed half-life halves the distance between the
    current weight and the floor; the result is clamped to
    ``[floor, 1.0]`` so a weight never reaches zero or exceeds one.

    Args:
        topics: Profile rows carrying ``weight`` and optionally
            ``last_boosted_at``.
        now_epoch: Current time as seconds since the epoch.
        half_life_days: Decay half-life in days (guarded to >= 0.5 so a
            zero/negative value cannot divide by zero).
        floor: Minimum weight a decayed topic converges to.

    Returns:
        Copies of the input rows with ``weight`` decayed.
    """
    half_life_s = max(half_life_days, 0.5) * 86400.0
    out: list[dict] = []
    for topic in topics:
        age = max(0.0, now_epoch - float(topic.get("last_boosted_at") or 0.0))
        factor = math.pow(0.5, age / half_life_s)
        weight = floor + (min(float(topic.get("weight", 0.0)), 1.0) - floor) * factor
        row = dict(topic)
        row["weight"] = min(max(weight, floor), 1.0)
        out.append(row)
    return out


def merge_signals(signal_lists: list[list[dict]], *, top_n: int = 30) -> list[dict]:
    """Sum signal weights per normalized ``(facet, text)`` key, ranked, top-N.

    Normalization is strip + lowercase, and the surviving row's ``text`` is
    the normalized form: the first spelling seen may carry stray casing or
    whitespace ("  Rust  " vs "rust") and callers downstream compare on the
    normalized identity. Empty texts are dropped. Per-topic sums are capped
    at 1.0.

    Args:
        signal_lists: Signal rows (from the readers in ``profile_sources``)
            carrying ``facet``, ``text``, ``weight``.
        top_n: Maximum number of rows to return.

    Returns:
        Rows ``{"facet", "text", "weight"}`` sorted by weight, descending.
    """
    totals: dict[str, dict] = {}
    for signals in signal_lists:
        for entry in signals:
            key = (entry.get("facet", "topic"),
                   str(entry.get("text", "")).strip().lower())
            if not key[1]:
                continue
            agg = totals.setdefault(
                key[0] + "|" + key[1],
                {"facet": key[0], "text": key[1], "weight": 0.0},
            )
            agg["weight"] += float(entry.get("weight", 0.0))
    ranked = sorted(totals.values(), key=lambda t: t["weight"], reverse=True)
    for row in ranked:
        row["weight"] = min(row["weight"], 1.0)
    return ranked[:top_n]


#: Feedback offset per net reaction (ruling R19; spec §feedback loop) and
#: the bounds a feedback-adjusted weight is clamped to -- the same ones decay
#: converges to.
FEEDBACK_STEP = 0.1
WEIGHT_FLOOR = 0.05
WEIGHT_CEILING = 1.0

#: Topics a snapshot carries into query synthesis and every story prompt.
#: Profile rows are never pruned, so without a cap the egress (and token
#: cost) grows with every cycle's merged signals.
SNAPSHOT_TOPIC_CAP = 20


def snapshot(
    db: DreamsDB,
    *,
    now_epoch: float,
    feedback: dict[str, int] | None = None,
) -> dict:
    """Read the profile's topics, decay them, and pair them with the region.

    Only searchable ``facet == "topic"`` rows join the snapshot (goals ride
    along in later cycle steps, undecayed); ``last_boosted_at`` is stored as
    an ISO string or NULL and is converted to epoch floats for the decay
    math. Feedback is applied here as an OFFSET over the decayed weight
    (``FEEDBACK_STEP`` per net reaction, clamped) instead of being written
    back into the stored weight: a stored write would re-apply the same
    reactions every cycle of their window, and the signal refresh would
    clobber it on derived rows.

    Args:
        db: Dreams database to read the interest profile from.
        now_epoch: Current time as seconds since the epoch.
        feedback: Net reaction count per normalized topic text (see
            ``cycle_service._feedback_net``); None applies no offset.

    Returns:
        ``{"topics": [decayed topic rows, heaviest first, at most
        SNAPSHOT_TOPIC_CAP], "region": str}`` where ``region`` comes from
        ``dreams_setting("region")`` (empty when unset).
    """
    rows = [
        row for row in db.list_profile()
        if row.get("facet") == "topic" and int(row.get("searchable") or 0)
    ]
    for row in rows:
        row["last_boosted_at"] = _epoch(row.get("last_boosted_at"))
    topics = decay_weights(rows, now_epoch=now_epoch)
    for topic in topics:
        net = (feedback or {}).get(str(topic.get("text", "")).strip().lower(), 0)
        if net:
            topic["weight"] = min(WEIGHT_CEILING, max(
                WEIGHT_FLOOR, float(topic["weight"]) + FEEDBACK_STEP * net))
    topics.sort(key=lambda topic: float(topic["weight"]), reverse=True)
    return {"topics": topics[:SNAPSHOT_TOPIC_CAP],
            "region": str(dreams_setting("region", ""))}


def _epoch(value: Any) -> float:
    """Convert an ISO timestamp to epoch seconds; blank/unparseable -> 0.0.

    0.0 means "never boosted", which decay treats as maximally stale.
    Naive timestamps are read as UTC (the DB writes UTC everywhere).
    """
    if not value:
        return 0.0
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return 0.0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()
