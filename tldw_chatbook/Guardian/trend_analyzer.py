"""Guardian trend analyzer: rolling aggregates over recorded alert hits.

Spec §Trend analyzer, v1 deliberately conservative definitions:

* **Fixation proxy** (sustained-topic share): one topic holds at least
  ``fixation_share_threshold`` (default 0.6) of ALL rule-generated alert
  hits inside ``fixation_window_days`` (default 7), with at least
  ``fixation_min_hits`` (default 30) hits of its own.
* **Doomloop proxy** (repetitive volume): at least
  ``doomloop_hits_per_day`` (default 20) hits on one topic for at least
  three consecutive UTC days.

Loop exclusion is structural (spec §Trend analyzer): inputs are
``rule_id IS NOT NULL`` rows only -- trend-generated rows are excluded so
analyzer output never feeds analyzer input. Every DETECTED trend is itself
inserted as a ``guardian_alerts`` row (``rule_id=None``, topic
``fixation:<topic>`` / ``doomloop:<topic>``), frequency-capped to one row
per trend label per visit, and the returned list carries only the NEWLY
inserted notices (capped repeats return nothing) -- so both callers (the
visit-end cadence and the daily task) dispatch at most once per label per
visit. All thresholds are read live from ``[guardian]`` so tuning applies
without a restart. Local analysis only: no LLM, no network, budget-free.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import settings as guardian_settings

#: Doomloop streak floor (spec: ">= 3 consecutive days").
DOOMLOOP_MIN_CONSECUTIVE_DAYS = 3

FIXATION_KIND = "fixation"
DOOMLOOP_KIND = "doomloop"


@dataclass(frozen=True)
class TrendNotice:
    """One detected (and newly recorded) trend.

    Attributes:
        kind: ``"fixation"`` or ``"doomloop"``.
        topic: The rule topic the trend was detected over.
        hits: The qualifying count (fixation: window hits; doomloop: the
            peak per-day count inside the streak).
        message: A ready-to-surface, markup-free sentence (notification
            body and summary text).
    """

    kind: str
    topic: str
    hits: int
    message: str

    @property
    def label(self) -> str:
        """The stored alert topic for this trend, ``<kind>:<topic>``."""
        return f"{self.kind}:{self.topic}"


def analyze(
    guardian_db: Any,
    *,
    now: str,
    visit_id: str = "trends",
    session_id: str = "",
) -> list[TrendNotice]:
    """Run both trend detections and record every NEW notice.

    Args:
        guardian_db: The :class:`~tldw_chatbook.DB.Guardian_DB.GuardianDB`
            to read rule-generated alerts from and insert trend rows into.
        now: Current UTC ISO timestamp (the injected clock's shape; trend
            rows are stamped with it).
        visit_id: The visit scope for both the inserted rows and the
            frequency cap (the visit-end cadence passes the Console visit
            id; the daily task passes a date-scoped id so the cap bounds it
            to one notice per label per day).
        session_id: Provenance label for the inserted rows (the Console
            session, or ``"daily"`` for the scheduler task).

    Returns:
        The notices inserted by THIS run -- a label already recorded in
        this visit is capped and absent from the list.
    """
    share_threshold = float(
        guardian_settings.guardian_setting("fixation_share_threshold", 0.6)
        or 0.6
    )
    window_days = int(
        guardian_settings.guardian_setting("fixation_window_days", 7) or 7
    )
    min_hits = int(
        guardian_settings.guardian_setting("fixation_min_hits", 30) or 30
    )
    daily_hits = int(
        guardian_settings.guardian_setting("doomloop_hits_per_day", 20) or 20
    )

    now_dt = _parse_iso(now) or datetime.now(timezone.utc)
    with guardian_db.connection() as conn:
        rows = conn.execute(
            "SELECT topic, ts FROM guardian_alerts WHERE rule_id IS NOT NULL"
        ).fetchall()

    window_cutoff = now_dt - timedelta(days=window_days)
    window_counts: dict[str, int] = {}
    window_total = 0
    per_day_counts: dict[tuple[str, date], int] = {}
    for row in rows:
        moment = _parse_iso(row["ts"])
        if moment is None:
            continue
        topic = str(row["topic"])
        day = (
            moment.astimezone(timezone.utc).date()
            if moment.tzinfo is not None
            else moment.date()
        )
        per_day_counts[(topic, day)] = per_day_counts.get((topic, day), 0) + 1
        if moment >= window_cutoff:
            window_counts[topic] = window_counts.get(topic, 0) + 1
            window_total += 1

    detected: list[TrendNotice] = []

    if window_total:
        for topic, count in sorted(window_counts.items()):
            if count >= min_hits and (count / window_total) >= share_threshold:
                detected.append(
                    TrendNotice(
                        kind=FIXATION_KIND,
                        topic=topic,
                        hits=count,
                        message=(
                            f"Sustained attention on '{topic}': {count} of "
                            f"{window_total} Guardian notices in the last "
                            f"{window_days} days."
                        ),
                    )
                )

    qualifying_days: dict[str, set[date]] = {}
    for (topic, day), count in per_day_counts.items():
        if count >= daily_hits:
            qualifying_days.setdefault(topic, set()).add(day)
    for topic in sorted(qualifying_days):
        days = qualifying_days[topic]
        streak = _longest_consecutive_run(days)
        if streak >= DOOMLOOP_MIN_CONSECUTIVE_DAYS:
            peak = max(
                count
                for (t, _day), count in per_day_counts.items()
                if t == topic
            )
            detected.append(
                TrendNotice(
                    kind=DOOMLOOP_KIND,
                    topic=topic,
                    hits=peak,
                    message=(
                        f"Repetitive daily pattern on '{topic}': at least "
                        f"{daily_hits} Guardian notices a day for {streak} "
                        "consecutive days."
                    ),
                )
            )

    inserted: list[TrendNotice] = []
    for notice in detected:
        if guardian_db.count_topic_alerts(notice.label, visit_id=visit_id) > 0:
            continue  # frequency cap: one notice per label per visit
        guardian_db.insert_alert(
            rule_id=None,
            session_id=session_id,
            visit_id=visit_id,
            topic=notice.label,
            message_digest=hashlib.sha256(notice.label.encode()).hexdigest(),
            ts=now,
        )
        inserted.append(notice)
    return inserted


def _longest_consecutive_run(days: set[date]) -> int:
    """Return the longest run of consecutive calendar days in ``days``."""
    longest = 0
    for day in days:
        if day - timedelta(days=1) in days:
            continue  # not the start of a run
        run = 1
        while day + timedelta(days=run) in days:
            run += 1
        longest = max(longest, run)
    return longest


def _parse_iso(value: Any) -> datetime | None:
    """Parse a stored timestamp tolerantly; None when unparseable."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
