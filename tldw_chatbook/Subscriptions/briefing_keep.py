"""Copy a Subscriptions_DB briefing (and its complete scripts) into
ChaChaNotes, so it survives watchlist deletion (task-1780).

`Subscriptions_DB` chains a briefing's lifecycle to its watchlist --
`briefings` -> `briefing_scripts` -> `briefing_audio` all cascade on
``ON DELETE CASCADE`` when the watchlist goes. That is right for working
data and wrong for content a user has decided to keep. This module is the
one writer for the ``kept_briefings``/``kept_scripts`` tables in
ChaChaNotes (see
``Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`` and Task 1's
CRUD, ``ChaChaNotes_DB.create_kept_briefing``/``create_kept_script`` and
friends): every field a kept row carries is denormalized at keep time, so
the kept copy remains self-interpreting even after the Subscriptions_DB
side (watchlist, briefing, scripts) is entirely gone.

Two rules carry this module, both from the spec's "Keep service" section:

**1. Refuse before writing, never after.** A missing briefing, a briefing
that has not reached ``complete``, or a ``complete`` briefing with an empty
(or whitespace-only) body all raise :class:`KeepRefused` before any
``kept_briefings`` row exists. This is what keeps auto-keep (Task 3, which
fires on every scheduled-generation completion) from mirroring ``empty``
scheduled rows -- a scheduled run only auto-keeps when it actually produced
something.

**2. Keep is additive-idempotent.** Re-keeping an already-kept briefing
never creates a second ``kept_briefings`` row (the idempotency key is
``kept_briefings.source_briefing_id UNIQUE``) and never rewrites the
existing row's fields -- ``origin`` in particular passes through to a
*newly created* row only; a later re-keep (e.g. an auto-keep after the user
already pressed Keep manually) must not silently relabel a manual keep as
scheduled, or vice versa. Scripts are diffed by ``source_script_id``
against :meth:`CharactersRAGDB.kept_script_source_ids`, so a re-keep adds
only scripts that were cast *after* the previous keep (or missed because
they were not yet ``complete``) and never duplicates or overwrites one
already kept.

Cross-DB datetime boundary (load-bearing, see Task 1's report): every
``CharactersRAGDB`` connection opens with ``sqlite3.PARSE_DECLTYPES`` plus
a process-wide ``DATETIME`` converter (``DB/sqlite_datetime_fix.py``) that
only assumes UTC when a stored string ends in ``Z`` -- otherwise a naive
string round-trips as a naive ``datetime``. ``Subscriptions_DB``'s own
``DATETIME DEFAULT CURRENT_TIMESTAMP`` columns (``briefings.created_at``,
``briefing_scripts.created_at``) are SQLite's own naive-UTC string format,
with no offset at all; ``covers_from_ts`` is usually already an
offset-bearing ISO string (built from a tz-aware ``datetime.now(timezone.
utc)`` in ``briefing_selection``), but is not guaranteed to be. Passing
either shape straight through to a ChaChaNotes ``DATETIME`` column would
silently produce a *naive* ``datetime`` on read-back for the naive case --
losing "this is UTC" instead of surfacing it. :func:`_to_chacha_datetime`
is the one boundary-crossing point: it parses the incoming value and
attaches UTC explicitly only when the value has no tzinfo of its own, so
every kept datetime field reads back tz-aware, at the correct instant,
regardless of which of Subscriptions_DB's two DATETIME string shapes
supplied it.

Thread-safety (spec-mandated plan-time verification, see the Task 2
report): ``CharactersRAGDB._get_thread_connection`` hands each calling
thread its own private connection via ``threading.local()`` -- opened
lazily on first use, reused afterward -- exactly the same shape
``SubscriptionsDB.conn`` already uses (also a ``threading.local`` slot).
Both databases are therefore equally safe to call from a worker thread
dispatched by ``asyncio.to_thread``: each thread that ever calls in gets
its own connection, and neither class shares one connection across
threads. This function is itself a plain **synchronous** call -- it does
no ``asyncio`` of its own -- so an ``async`` caller (Task 3's scheduler
handler, a future UI Keep button) must wrap the whole call in
``asyncio.to_thread(keep_briefing, ...)`` rather than call it directly on
the event loop.

Nothing here logs briefing/script content -- only ids, statuses and counts
-- matching ``briefing_service``/``briefing_cast``'s own logging rule.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from .briefing_cast import STATUS_COMPLETE as _SCRIPT_COMPLETE
from .briefing_service import STATUS_COMPLETE as _BRIEFING_COMPLETE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from ..DB.Subscriptions_DB import SubscriptionsDB

#: Fallback for `kept_briefings.watchlist_name` when the source watchlist
#: row no longer exists at keep time. Denormalized text, not a foreign key
#: (the spec's whole point) -- so a gone watchlist degrades to this literal
#: rather than a lookup failure.
DELETED_WATCHLIST_NAME = "(deleted watchlist)"

#: Page size for walking every page of `list_briefing_scripts` (which
#: itself defaults its own `limit` to 200). A briefing with more cast
#: scripts than one page would otherwise silently lose the overflow to a
#: single unpaginated call.
_SCRIPT_PAGE_SIZE = 200


class KeepRefused(RuntimeError):
    """Raised when a briefing must not be kept; no `kept_briefings` row is written.

    Every raise site names both the briefing id and the specific reason
    (missing, not `complete`, or an empty body) -- never a generic "keep
    failed" -- mirroring `ScriptCastError`'s naming discipline in
    `briefing_cast`.
    """


def _to_chacha_datetime(value: str | None) -> str | None:
    """Normalize a Subscriptions_DB DATETIME string for a ChaChaNotes column.

    See the module docstring's "Cross-DB datetime boundary" section for the
    full reasoning. A value that already carries an explicit offset (or a
    trailing ``Z``) is returned with its instant unchanged; a naive value
    (Subscriptions_DB's own ``DEFAULT CURRENT_TIMESTAMP`` format) has UTC
    attached explicitly before being re-serialized.

    Args:
        value: A Subscriptions_DB DATETIME column's string value, or None.

    Returns:
        An ISO-8601 string with an explicit UTC offset, ready to be written
        into a ChaChaNotes DATETIME column so it reads back as a tz-aware
        `datetime.datetime` at the correct instant. `None` if `value` is
        `None`.
    """
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _watchlist_name(subs_db: "SubscriptionsDB", watchlist_id: int) -> str:
    """Resolve a watchlist's display name at keep time, tolerating deletion.

    Args:
        subs_db: An open `SubscriptionsDB`.
        watchlist_id: The `watchlists.id` the briefing being kept belongs
            to.

    Returns:
        The watchlist's current `name`, or `DELETED_WATCHLIST_NAME` if no
        such watchlist row exists. `briefings.watchlist_id` carries a real
        `ON DELETE CASCADE` foreign key in Subscriptions_DB, so this branch
        is not reachable through that database's own delete path today --
        it exists because a kept row must remain self-interpreting even if
        that invariant ever changes, and is exercised directly in tests.
    """
    with subs_db.transaction() as conn:
        row = conn.execute(
            "SELECT name FROM watchlists WHERE id = ?", (watchlist_id,)
        ).fetchone()
    return row["name"] if row is not None else DELETED_WATCHLIST_NAME


def _all_briefing_scripts(
    subs_db: "SubscriptionsDB", briefing_id: int
) -> list[dict[str, Any]]:
    """Every `briefing_scripts` row for `briefing_id`, walking all pages.

    Args:
        subs_db: An open `SubscriptionsDB`.
        briefing_id: The briefing to list every cast script for.

    Returns:
        All `briefing_scripts` rows for `briefing_id`, of any status, in
        `list_briefing_scripts`'s own (newest-first) order.
    """
    scripts: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = subs_db.list_briefing_scripts(
            briefing_id, limit=_SCRIPT_PAGE_SIZE, offset=offset
        )
        scripts.extend(page)
        if len(page) < _SCRIPT_PAGE_SIZE:
            break
        offset += _SCRIPT_PAGE_SIZE
    return scripts


def _copy_missing_scripts(
    subs_db: "SubscriptionsDB",
    chacha_db: "CharactersRAGDB",
    briefing_id: int,
    kept_briefing_id: int,
) -> int:
    """Copy every `complete` script not yet kept under `kept_briefing_id`.

    The additive-idempotency diff: a script already present by
    `source_script_id` (per `kept_script_source_ids`) is skipped entirely
    -- never touched, never re-inserted -- and a script that has not
    reached `complete` yet is left for a future keep to pick up once it
    has.

    Args:
        subs_db: An open `SubscriptionsDB`.
        chacha_db: An open `CharactersRAGDB`.
        briefing_id: The Subscriptions_DB briefing whose scripts are being
            mirrored.
        kept_briefing_id: The owning `kept_briefings.id` to copy scripts
            under.

    Returns:
        The number of `kept_scripts` rows newly created.
    """
    already_kept = chacha_db.kept_script_source_ids(kept_briefing_id)
    added = 0
    for script in _all_briefing_scripts(subs_db, briefing_id):
        if script["status"] != _SCRIPT_COMPLETE:
            continue
        if script["id"] in already_kept:
            continue
        chacha_db.create_kept_script(
            kept_briefing_id,
            source_script_id=script["id"],
            preset_name=script["preset_name"],
            roster_snapshot_json=script["roster_snapshot_json"],
            turns_json=script["turns_json"],
            model_used=script.get("model_used"),
            original_created_at=_to_chacha_datetime(script.get("created_at")),
        )
        added += 1
    return added


def keep_briefing(
    subs_db: "SubscriptionsDB",
    chacha_db: "CharactersRAGDB",
    briefing_id: int,
    *,
    origin: str,
) -> dict[str, Any]:
    """Keep a briefing (and its complete scripts) in ChaChaNotes.

    Synchronous -- this function does blocking SQLite I/O against both
    databases and no `asyncio` of its own. An `async` caller must offload
    the whole call, e.g. ``await asyncio.to_thread(keep_briefing, ...)``;
    see the module docstring's "Thread-safety" section for why this is
    safe.

    Additive-idempotent: calling this again for a briefing that is already
    kept never creates a second `kept_briefings` row and never rewrites
    its fields (including `origin` -- it is set only when this call is the
    one that creates the row). It only ever *adds* `kept_scripts` rows for
    scripts that were not kept yet, keyed by `source_script_id`.

    Args:
        subs_db: An open `SubscriptionsDB` -- the source of the briefing
            and its scripts.
        chacha_db: An open `CharactersRAGDB` -- where the kept copy lives.
        briefing_id: The Subscriptions_DB `briefings.id` to keep.
        origin: `"manual"` (a user pressed Keep) or `"scheduled"` (an
            auto-mirror on a scheduled generation's completion). Stored on
            the `kept_briefings` row only when this call creates it.

    Returns:
        A dict with:

        - `kept_id`: The `kept_briefings.id` (whether just created or
          already existing).
        - `created`: `True` only if this call inserted the `kept_briefings`
          row; `False` if the briefing was already kept.
        - `scripts_added`: How many `kept_scripts` rows this call newly
          created (0 on a re-keep that finds nothing new to add).

    Raises:
        KeepRefused: If `briefing_id` does not exist in `subs_db`, is not
            `status == "complete"`, or its `body_markdown` is empty or
            whitespace-only. No `kept_briefings` row is written in any of
            these cases.
    """
    briefing = subs_db.get_briefing(briefing_id)
    if briefing is None:
        raise KeepRefused(f"briefing {briefing_id} does not exist; refusing to keep it")
    if briefing["status"] != _BRIEFING_COMPLETE:
        raise KeepRefused(
            f"briefing {briefing_id} is {briefing['status']!r}, not "
            f"{_BRIEFING_COMPLETE!r}; refusing to keep it"
        )
    if not (briefing.get("body_markdown") or "").strip():
        raise KeepRefused(
            f"briefing {briefing_id} has an empty body; refusing to keep it"
        )

    existing = chacha_db.get_kept_briefing_by_source(briefing_id)
    if existing is None:
        kept_id = chacha_db.create_kept_briefing(
            source_briefing_id=briefing_id,
            watchlist_name=_watchlist_name(subs_db, briefing["watchlist_id"]),
            body_markdown=briefing["body_markdown"],
            covers_through_item_id=briefing.get("covers_through_item_id"),
            covers_from_ts=_to_chacha_datetime(briefing.get("covers_from_ts")),
            selection_mode=briefing.get("selection_mode"),
            model_used=briefing.get("model_used"),
            item_count=briefing.get("item_count") or 0,
            featured_count=briefing.get("featured_count") or 0,
            overflow_count=briefing.get("overflow_count") or 0,
            origin=origin,
            original_created_at=_to_chacha_datetime(briefing.get("created_at")),
        )
        created = True
    else:
        kept_id = existing["id"]
        created = False

    scripts_added = _copy_missing_scripts(subs_db, chacha_db, briefing_id, kept_id)

    logger.info(
        f"kept briefing {briefing_id} as kept_briefings.id={kept_id} "
        f"(created={created}, scripts_added={scripts_added})"
    )
    return {"kept_id": kept_id, "created": created, "scripts_added": scripts_added}
