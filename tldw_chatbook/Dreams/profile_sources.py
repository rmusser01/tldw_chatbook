"""[dreams] Interest-profile signal readers: notes, media, Personal Context.

Readers degrade, never block (spec §interest profile): notes and media are
plain DB reads over the existing keyword tables, and the Personal Context
reader is opt-in and lock-aware — when the profile is passphrase-locked it
returns the distillate cached by the last successful read instead of
raising.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Utils.private_paths import (
    atomic_private_write_text,
    open_private_binary,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


#: Weight contributed per touch inside the window: four touches saturate a
#: keyword at 1.0 before ``interest_profile.merge_signals`` sums and caps.
_WEIGHT_PER_TOUCH = 0.25


def _cutoff_iso(window_days: int) -> str:
    """Return the window cutoff in the stores' lexical DATETIME idiom.

    Both DBs keep DATETIME text columns and compare them as strings (the
    same idiom as their own window queries, e.g. Client_Media_DB_v2's
    hard-delete cutoff), so the cutoff is formatted to match: space-
    separated UTC. Rows written with the ``T``-separated ISO variant differ
    only on the boundary day itself (``T`` sorts above a space), at most a
    day of skew on a 14-day window.
    """
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def read_note_topics(
    chachanotes_db: CharactersRAGDB, *, window_days: int = 14
) -> list[dict]:
    """Aggregate keywords of notes touched inside the window.

    ``notes.last_modified >= cutoff`` reuses the indexed read the Notes
    screens build on (``idx_notes_last_modified``); no LIKE scans.

    Args:
        chachanotes_db: Notes database to read from.
        window_days: How many days back a note's last modification still
            counts as a signal.

    Returns:
        Topic rows ``{"facet": "topic", "text", "weight", "searchable",
        "source": "notes"}``, heaviest first.
    """
    cutoff = _cutoff_iso(window_days)
    rows = chachanotes_db.execute_query(
        """
        SELECT k.keyword AS keyword, COUNT(*) AS uses
        FROM note_keywords AS nk
        JOIN notes AS n ON n.id = nk.note_id
        JOIN keywords AS k ON k.id = nk.keyword_id
        WHERE n.deleted = 0
          AND n.last_modified >= ?
          AND k.deleted = 0
        GROUP BY k.id, k.keyword
        ORDER BY uses DESC, k.keyword COLLATE NOCASE
        """,
        (cutoff,),
    ).fetchall()
    return [
        {
            "facet": "topic",
            "text": str(row["keyword"]).strip(),
            "weight": min(1.0, _WEIGHT_PER_TOUCH * int(row["uses"])),
            "searchable": 1,
            "source": "notes",
        }
        for row in rows
    ]


def read_media_topics(
    media_db: MediaDatabase, *, window_days: int = 14
) -> list[dict]:
    """Aggregate keywords of media touched inside the window.

    A media row is "touched" when it was ingested/modified inside the
    window, and touched again when its ReadingProgress was updated inside
    the window — reading progress counts double, because going back to an
    item is a stronger interest signal than merely having saved it.

    Args:
        media_db: Media database to read from.
        window_days: How many days back a touch still counts.

    Returns:
        Topic rows ``{"facet": "topic", "text", "weight", "searchable",
        "source": "media"}``, heaviest first.
    """
    cutoff = _cutoff_iso(window_days)
    rows = media_db.execute_query(
        """
        SELECT k.keyword AS keyword, SUM(touch.score) AS score
        FROM (
            SELECT m.id AS media_id,
                   (CASE WHEN COALESCE(m.ingestion_date, '') >= ?
                            OR m.last_modified >= ? THEN 1 ELSE 0 END)
                 + (CASE WHEN rp.last_modified >= ? THEN 1 ELSE 0 END) AS score
            FROM Media AS m
            LEFT JOIN ReadingProgress AS rp ON rp.media_id = m.id
            WHERE m.deleted = 0 AND m.is_trash = 0
        ) AS touch
        JOIN MediaKeywords AS mk ON mk.media_id = touch.media_id
        JOIN Keywords AS k ON k.id = mk.keyword_id
        WHERE touch.score > 0 AND k.deleted = 0
        GROUP BY k.id, k.keyword
        ORDER BY score DESC, k.keyword COLLATE NOCASE
        """,
        (cutoff, cutoff, cutoff),
    ).fetchall()
    return [
        {
            "facet": "topic",
            "text": str(row["keyword"]).strip(),
            "weight": min(1.0, _WEIGHT_PER_TOUCH * float(row["score"])),
            "searchable": 1,
            "source": "media",
        }
        for row in rows
    ]


def read_personal_context_topics(pc_service: Any, *, cache_path: Path) -> list[dict]:
    """Opt-in signal: degrade to the cached distillate on any read failure.

    Personal Context key custody may be passphrase-wrapped
    (Personal_Context/key_protector.py), so an unattended read raises
    ProfileLockedError -- the cache written by the last successful read is
    the whole point (spec §interest profile, cache-on-unlock). Never
    raises. The cache itself goes through the private-path discipline
    (``open_private_binary`` for reads, its atomic write counterpart for
    writes), and is only ever written after a successful read.
    """
    try:
        records = pc_service.list_records(scope_ids=_dreams_scope_ids(pc_service))
        topics = [
            topic
            for topic in (_record_to_topic(record) for record in records)
            if topic is not None
        ]
    except Exception:  # noqa: BLE001 - locked, unreadable, or unconfigured
        return _load_cache(cache_path)
    _store_cache(cache_path, topics)
    return topics


def _dreams_scope_ids(pc_service: Any) -> tuple[str, ...]:
    """Pick the interests-like Personal Context scope ids for a Dreams read.

    ``Personal_Context/service.py`` ``list_scopes`` returns a fixed enum of
    kinds — ``global``/``workspace`` only (packages/tldw_profile_core/
    src/tldw_profile_core/enums.py) — so there is no dedicated interests
    scope to pick; the user-level GLOBAL scope is the interests-like one,
    and workspace scopes (job-local context) are left out -- always. A
    profile with no GLOBAL scope yet reads nothing: falling back to every
    scope would send workspace context into discovery queries and persist
    it in the distillate cache. A service without ``list_scopes`` (narrow
    test doubles) reads with no scope ids, which ``list_records`` treats as
    a successful empty read.
    """
    list_scopes = getattr(pc_service, "list_scopes", None)
    if list_scopes is None:
        return ()
    try:
        scopes = tuple(list_scopes())
    except Exception:  # noqa: BLE001 - scope discovery must not block the read
        return ()
    return tuple(
        str(scope.scope_id)
        for scope in scopes
        if str(getattr(scope, "kind", "")) == "global"
    )


def _record_to_topic(record: Any) -> dict | None:
    """Map one ProfileRecord to a profile-entry dict, or None if content-free.

    Payload subjects are the short user-level concepts ("visit japan");
    ``goal`` records feed the never-decayed goal facet, everything else the
    topic facet. Weight starts at 1.0 — weighting belongs to the boost step
    that consumes these signals, not to the reader.
    """
    payload = getattr(record, "payload", None)
    if payload is None:
        return None
    text = str(
        getattr(payload, "subject", None) or getattr(payload, "text", "") or ""
    ).strip()
    if not text:
        return None
    facet = "goal" if str(getattr(record, "kind", "")) == "goal" else "topic"
    return {
        "facet": facet,
        "text": text,
        "weight": 1.0,
        "searchable": 1,
        "source": "personal_context",
    }


def _load_cache(cache_path: Path) -> list[dict]:
    """Read the distillate cache through the private-path discipline.

    The file is outside input: every entry is re-validated to the shape
    ``_record_to_topic`` writes, and malformed ones are dropped, so a
    corrupt cache degrades to less signal instead of crashing the merge.
    """
    try:
        with open_private_binary(cache_path) as pinned:
            cached = json.loads(pinned.stream.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - missing or corrupt cache == no signal
        return []
    if not isinstance(cached, list):
        return []
    return [entry for entry in cached if _valid_cached_topic(entry)]


def _valid_cached_topic(entry: Any) -> bool:
    """Whether one cached entry has the shape ``_record_to_topic`` writes."""
    return (
        isinstance(entry, dict)
        and entry.get("facet") in ("topic", "goal")
        and isinstance(entry.get("text"), str)
        and bool(entry["text"].strip())
        and isinstance(entry.get("weight"), (int, float))
        and not isinstance(entry.get("weight"), bool)
    )


def _store_cache(cache_path: Path, topics: list[dict]) -> None:
    """Best-effort cache write; a failure never discards the fresh topics."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_private_write_text(cache_path, json.dumps(topics))
    except Exception as exc:  # noqa: BLE001 - the cache is an optimization
        # Type name only: the cache path sits under the private data root.
        logger.debug("Dreams PC distillate cache write failed: {}",
                     type(exc).__name__)
