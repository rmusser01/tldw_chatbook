"""Tests for the keep service (task-1780, Task 2).

`briefing_keep.keep_briefing` copies a `complete` Subscriptions_DB briefing
(and its `complete` scripts) into ChaChaNotes' `kept_briefings`/
`kept_scripts` tables, so the copy survives the source watchlist's
deletion. Every test uses two REAL databases rooted at pytest's `tmp_path`
-- a `SubscriptionsDB` (the source) and a `CharactersRAGDB` (the kept
copy) -- never `:memory:` for either (matches `test_briefing_cast.py`'s own
rule: `generate_script`'s `asyncio.to_thread` hops need a real, file-backed
`SubscriptionsDB`; `keep_briefing` itself is synchronous, but the same
real-DB convention is followed throughout this stream) and never the live
user config/data directory. No probes or ad-hoc script execution outside
pytest.

Two rules shape most of these tests, both load-bearing enough to be named
invariants in the plan:

- **Refuse before writing, never after.** A missing briefing, a
  non-`complete` briefing, or a `complete` briefing with an empty body all
  raise `KeepRefused` with NO `kept_briefings` row ever created.
- **Additive-idempotent.** Re-keeping an already-kept briefing never
  duplicates or overwrites an existing `kept_briefings`/`kept_scripts` row
  -- pinned by full-dict equality assertions, not just row counts -- and
  only ever adds scripts missing by `source_script_id`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_keep
from tldw_chatbook.Subscriptions.briefing_keep import (
    DELETED_WATCHLIST_NAME,
    KeepRefused,
    _to_chacha_datetime,
    keep_briefing,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


# --- Fixtures / helpers ------------------------------------------------------


def _subs_db(tmp_path: Path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- see the module docstring."""
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _chacha_db(tmp_path: Path) -> CharactersRAGDB:
    """A real, file-backed `CharactersRAGDB` -- see the module docstring."""
    return CharactersRAGDB(tmp_path / "chacha.sqlite", client_id="keep-service-test")


def _watchlist(subs_db: SubscriptionsDB, *, name: str = "Tech Watch") -> int:
    return WatchlistBundleService(subs_db).create(name=name)["id"]


def _complete_briefing(
    subs_db: SubscriptionsDB,
    watchlist_id: int,
    *,
    body: str = "# Digest\n\nSomething happened this week.\n",
    covers_through_item_id: int | None = 42,
    covers_from_ts: str | None = None,
    selection_mode: str | None = "auto",
    model_used: str | None = "gpt-test",
    item_count: int = 5,
    featured_count: int = 2,
    overflow_count: int = 0,
) -> int:
    """A `complete` `briefings` row with a body -- the only status a keep may start from."""
    briefing_id = subs_db.insert_briefing(watchlist_id)
    subs_db.update_briefing(
        briefing_id,
        status="complete",
        body_markdown=body,
        covers_through_item_id=covers_through_item_id,
        covers_from_ts=covers_from_ts,
        selection_mode=selection_mode,
        model_used=model_used,
        item_count=item_count,
        featured_count=featured_count,
        overflow_count=overflow_count,
    )
    return briefing_id


def _script(
    subs_db: SubscriptionsDB,
    briefing_id: int,
    *,
    status: str = "complete",
    preset_name: str = "Duo",
    roster_snapshot_json: str = '[{"name": "Host"}]',
    turns_json: str | None = '[{"speaker": "Host", "text": "Welcome back."}]',
    model_used: str | None = "gpt-test",
) -> int:
    script_id = subs_db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name=preset_name,
        roster_snapshot_json=roster_snapshot_json,
    )
    subs_db.update_briefing_script(
        script_id, status=status, turns_json=turns_json, model_used=model_used
    )
    return script_id


def _delete_watchlist_bypassing_cascade(subs_db: SubscriptionsDB, watchlist_id: int) -> None:
    """Delete a watchlist WITHOUT cascading its briefings.

    `briefings.watchlist_id` carries a real `ON DELETE CASCADE` foreign key
    in `Subscriptions_DB`, and that database always runs with `PRAGMA
    foreign_keys = ON` -- so `WatchlistBundleService.delete` (the real,
    only app-level delete path) can never leave a `briefings` row whose
    watchlist is gone; deleting the watchlist always takes the briefing
    with it. This helper reproduces the one way that state CAN exist (a
    delete that happened to run with FK enforcement off), purely so
    `_watchlist_name`'s "already gone" fallback -- defensive code for a
    kept row's self-interpreting guarantee -- has something real to be
    tested against, mirroring this stream's precedent of bypassing an
    app-level guard with raw SQL to exercise a schema-layer edge (Task 1's
    `origin` CHECK/NOT NULL tests).
    """
    conn = subs_db.conn
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


# --- Refusal (no row ever written) -------------------------------------------


def test_keep_refuses_a_missing_briefing(tmp_path: Path) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        with pytest.raises(KeepRefused, match="404"):
            keep_briefing(subs_db, chacha_db, 404, origin="manual")
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


@pytest.mark.parametrize("status", ["generating", "failed", "empty"])
def test_keep_refuses_a_non_complete_briefing(tmp_path: Path, status: str) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = subs_db.insert_briefing(watchlist_id)
        subs_db.update_briefing(briefing_id, status=status)

        with pytest.raises(KeepRefused, match=status):
            keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


@pytest.mark.parametrize("body", [None, "", "   \n\t  "])
def test_an_empty_briefing_is_refused_not_kept(tmp_path: Path, body) -> None:
    """Named invariant test (plan Task 2, AC): auto-keep must never mirror
    an `empty` scheduled row, and a `complete` briefing with a blank body
    is refused exactly the same way."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = subs_db.insert_briefing(watchlist_id)
        fields = {"status": "complete"}
        if body is not None:
            fields["body_markdown"] = body
        subs_db.update_briefing(briefing_id, **fields)

        with pytest.raises(KeepRefused, match="empty"):
            keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


# --- First keep: denormalized fields + provenance ----------------------------


def test_keep_creates_a_new_kept_briefing_with_denormalized_fields(tmp_path: Path) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db, name="Security Watch")
        briefing_id = _complete_briefing(
            subs_db,
            watchlist_id,
            body="# Digest\n\nAcme shipped a thing.\n",
            covers_from_ts="2026-07-25 00:00:00",  # naive -- see the datetime tests
        )

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert result["created"] is True
        assert isinstance(result["kept_id"], int)
        assert result["scripts_added"] == 0

        kept = chacha_db.get_kept_briefing(result["kept_id"])
        assert kept is not None
        assert kept["source_briefing_id"] == briefing_id
        assert kept["watchlist_name"] == "Security Watch"
        assert kept["body_markdown"] == "# Digest\n\nAcme shipped a thing.\n"
        assert kept["covers_through_item_id"] == 42
        assert kept["selection_mode"] == "auto"
        assert kept["model_used"] == "gpt-test"
        assert kept["item_count"] == 5
        assert kept["featured_count"] == 2
        assert kept["overflow_count"] == 0
        assert kept["origin"] == "manual"
        assert kept["kept_at"]

        # By-source lookup finds the same row (the idempotency key).
        assert chacha_db.get_kept_briefing_by_source(briefing_id) == kept
    finally:
        chacha_db.close_connection()


def test_keep_passes_origin_through_only_on_creation(tmp_path: Path) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)

        first = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        assert first["created"] is True

        second = keep_briefing(subs_db, chacha_db, briefing_id, origin="scheduled")
        assert second["created"] is False
        assert second["kept_id"] == first["kept_id"]

        # A later auto-keep must never relabel an existing manual keep.
        kept = chacha_db.get_kept_briefing(first["kept_id"])
        assert kept["origin"] == "manual"
    finally:
        chacha_db.close_connection()


# --- Concurrent keep race -----------------------------------------------------


def test_keep_survives_a_racing_create_kept_briefing_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1 (Important): two concurrent callers -- Task 3's
    auto-keep and a manual Keep press, say -- can both pass the "does a
    kept row already exist?" check before either has inserted one. The
    loser's `create_kept_briefing` call then hits the real
    `source_briefing_id UNIQUE` constraint (the table's only UNIQUE, so a
    `ConflictError` from it is unambiguous) and must land as a friendly
    re-keep result, not a raw exception.

    Forced deterministically rather than with real threads: the kept row
    is pre-created directly via `chacha_db.create_kept_briefing` (exactly
    what "another caller won the race" would have done), then this
    briefing's *first* `get_kept_briefing_by_source` call within the
    `keep_briefing` call below is monkeypatched to still report `None` --
    reproducing the exact TOCTOU window a real race would hit -- so the
    real `create_kept_briefing` call underneath it collides with the
    pre-created row for real.
    """
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db, name="Tech Watch")
        briefing_id = _complete_briefing(subs_db, watchlist_id)

        raced_kept_id = chacha_db.create_kept_briefing(
            source_briefing_id=briefing_id,
            watchlist_name="Tech Watch",
            body_markdown="# Digest\n\nAnother caller already kept this.\n",
            origin="scheduled",
        )

        real_lookup = chacha_db.get_kept_briefing_by_source
        calls = {"n": 0}

        def _first_call_reports_absent(source_briefing_id: int):
            calls["n"] += 1
            return None if calls["n"] == 1 else real_lookup(source_briefing_id)

        monkeypatch.setattr(
            chacha_db, "get_kept_briefing_by_source", _first_call_reports_absent
        )

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert result == {
            "kept_id": raced_kept_id,
            "created": False,
            "scripts_added": 0,
        }
        assert len(chacha_db.list_kept_briefings()) == 1  # no duplicate row
        kept = chacha_db.get_kept_briefing(raced_kept_id)
        assert kept["origin"] == "scheduled"  # the racing caller's origin stands
    finally:
        chacha_db.close_connection()


def test_copy_missing_scripts_survives_a_racing_create_kept_script_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qodo (PR #1197): `_copy_missing_scripts` snapshots
    `kept_script_source_ids` ONCE, then inserts every script missing from
    that snapshot. Two concurrent `keep_briefing` calls -- the scheduled
    auto-keep and a manual Keep press landing together, say, both
    surviving the briefing-row race fold above -- can both pass that
    snapshot for the same `source_script_id` before either has inserted.
    The loser's `create_kept_script` call then hits the real
    `source_script_id UNIQUE` constraint for real and must be folded into
    "already kept" rather than failing this caller's *entire* keep even
    though every script ended up copied.

    Forced deterministically rather than with real threads: one script
    (`raced_script_id`) is pre-copied directly via
    `chacha_db.create_kept_script` (exactly what "another caller already
    kept this one" would have done), then `kept_script_source_ids` is
    monkeypatched to still report the STALE (pre-race) empty set --
    reproducing the exact TOCTOU window a real race would hit -- so the
    real `create_kept_script` call underneath collides with the
    pre-created row for real.
    """
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db, name="Tech Watch")
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        raced_script_id = _script(subs_db, briefing_id, preset_name="Duo")
        other_script_id = _script(subs_db, briefing_id, preset_name="Solo")

        kept_id = chacha_db.create_kept_briefing(
            source_briefing_id=briefing_id,
            watchlist_name="Tech Watch",
            body_markdown="# Digest\n\nSomething happened this week.\n",
            origin="manual",
        )
        # "Another caller" already copied `raced_script_id` first.
        chacha_db.create_kept_script(
            kept_id,
            source_script_id=raced_script_id,
            preset_name="Duo",
            roster_snapshot_json='[{"name": "Host"}]',
            turns_json='[{"speaker": "Host", "text": "Welcome back."}]',
            model_used="gpt-test",
        )

        monkeypatch.setattr(
            chacha_db, "kept_script_source_ids", lambda kept_briefing_id: set()
        )

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert result["created"] is False
        assert result["kept_id"] == kept_id
        assert result["scripts_added"] == 1  # only `other_script_id` counted
        kept_scripts = chacha_db.list_kept_scripts(kept_id)
        assert len(kept_scripts) == 2  # no duplicate row for raced_script_id
        kept_source_ids = {row["source_script_id"] for row in kept_scripts}
        assert kept_source_ids == {raced_script_id, other_script_id}
    finally:
        chacha_db.close_connection()


# --- Scripts: complete-only, additive-idempotent -----------------------------


def test_keep_copies_only_complete_scripts(tmp_path: Path) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        complete_a = _script(subs_db, briefing_id, preset_name="Duo")
        complete_b = _script(subs_db, briefing_id, preset_name="Solo")
        generating = _script(subs_db, briefing_id, status="generating", turns_json=None)
        failed = _script(subs_db, briefing_id, status="failed", turns_json=None)

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert result["scripts_added"] == 2
        kept_scripts = chacha_db.list_kept_scripts(result["kept_id"])
        assert len(kept_scripts) == 2
        kept_source_ids = {row["source_script_id"] for row in kept_scripts}
        assert kept_source_ids == {complete_a, complete_b}
        assert generating not in kept_source_ids
        assert failed not in kept_source_ids

        by_preset = {row["preset_name"]: row for row in kept_scripts}
        assert by_preset["Duo"]["roster_snapshot_json"] == '[{"name": "Host"}]'
        assert (
            by_preset["Duo"]["turns_json"]
            == '[{"speaker": "Host", "text": "Welcome back."}]'
        )
        assert by_preset["Duo"]["model_used"] == "gpt-test"
    finally:
        chacha_db.close_connection()


def test_rekeeping_never_duplicates_an_already_kept_script(tmp_path: Path) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        _script(subs_db, briefing_id)

        first = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        assert first["scripts_added"] == 1

        second = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        third = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert second["scripts_added"] == 0
        assert third["scripts_added"] == 0
        assert len(chacha_db.list_kept_scripts(first["kept_id"])) == 1
    finally:
        chacha_db.close_connection()


def test_rekeeping_is_byte_identical_when_nothing_new(tmp_path: Path) -> None:
    """Additive idempotency, direction 1: nothing changed subs-side ->
    nothing changes kept-side, asserted by full-dict equality (not just
    counts) of both the kept briefing row and its kept script rows."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        _script(subs_db, briefing_id, preset_name="Duo")

        first = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept_before = chacha_db.get_kept_briefing(first["kept_id"])
        scripts_before = chacha_db.list_kept_scripts(first["kept_id"])

        second = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert second == {
            "kept_id": first["kept_id"],
            "created": False,
            "scripts_added": 0,
        }
        assert chacha_db.get_kept_briefing(first["kept_id"]) == kept_before
        assert chacha_db.list_kept_scripts(first["kept_id"]) == scripts_before
    finally:
        chacha_db.close_connection()


def test_all_briefing_scripts_walks_every_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins `_all_briefing_scripts`'s own pagination loop (task-1780
    whole-branch review, FIX 4) -- previously untested. Shrinking the
    module's own `_SCRIPT_PAGE_SIZE` (rather than seeding hundreds of real
    rows to exceed its real 200-row default) is the established idiom for
    this class of test.

    Mutation target: break the loop so it stops after the first page and
    this REDs (too few scripts get copied -- `scripts_added == 2`, not
    `5`). A second mutation -- `offset` never advancing -- is a genuine
    infinite loop (every page keeps re-reading the same first
    `_SCRIPT_PAGE_SIZE` rows forever) that this synchronous function has
    no `asyncio` hook to bound from inside the test; verify that one with
    an external process timeout, never by running it unbounded.
    """
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        monkeypatch.setattr(briefing_keep, "_SCRIPT_PAGE_SIZE", 2)
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        script_ids = [
            _script(subs_db, briefing_id, preset_name=f"Preset {i}")
            for i in range(5)
        ]

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert result["scripts_added"] == 5
        kept_scripts = chacha_db.list_kept_scripts(result["kept_id"])
        assert {row["source_script_id"] for row in kept_scripts} == set(script_ids)
    finally:
        chacha_db.close_connection()


def test_rekeep_adds_a_script_cast_after_the_first_keep(tmp_path: Path) -> None:
    """Additive idempotency, direction 2: a scheduled briefing is
    auto-kept scriptless, then a script cast later from the ORIGINAL
    briefing is picked up by keeping again -- the earlier kept row is
    untouched (byte-identical), and only the new script is added."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        first_script_id = _script(subs_db, briefing_id, preset_name="Duo")

        first = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        assert first["scripts_added"] == 1
        kept_before = chacha_db.get_kept_briefing(first["kept_id"])
        first_kept_script = chacha_db.list_kept_scripts(first["kept_id"])[0]

        later_script_id = _script(subs_db, briefing_id, preset_name="Solo")
        second = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")

        assert second["created"] is False
        assert second["scripts_added"] == 1

        # The briefing row and the previously-kept script are untouched.
        assert chacha_db.get_kept_briefing(first["kept_id"]) == kept_before
        kept_scripts = chacha_db.list_kept_scripts(first["kept_id"])
        assert len(kept_scripts) == 2
        by_source = {row["source_script_id"]: row for row in kept_scripts}
        assert by_source[first_script_id] == first_kept_script
        assert by_source[later_script_id]["preset_name"] == "Solo"
    finally:
        chacha_db.close_connection()


# --- AC #3: kept rows survive watchlist deletion -----------------------------


def test_kept_rows_survive_watchlist_deletion(tmp_path: Path) -> None:
    """Named invariant test (plan Task 2 / spec AC #3): keep a briefing and
    its scripts, delete the watchlist through the REAL subscriptions path
    (`WatchlistBundleService.delete`, which cascades the briefing and its
    scripts away on the Subscriptions_DB side), then re-read every kept
    field and confirm it is intact."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db, name="Doomed Watch")
        briefing_id = _complete_briefing(
            subs_db, watchlist_id, body="# Digest\n\nLast body before deletion.\n"
        )
        _script(subs_db, briefing_id, preset_name="Duo")

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept_before = chacha_db.get_kept_briefing(result["kept_id"])
        scripts_before = chacha_db.list_kept_scripts(result["kept_id"])
        assert kept_before["watchlist_name"] == "Doomed Watch"
        assert len(scripts_before) == 1

        WatchlistBundleService(subs_db).delete(watchlist_id)

        # Prove the deletion actually cascaded on the Subscriptions_DB side
        # -- otherwise this test would not be exercising anything.
        assert subs_db.get_briefing(briefing_id) is None

        kept_after = chacha_db.get_kept_briefing(result["kept_id"])
        scripts_after = chacha_db.list_kept_scripts(result["kept_id"])
        assert kept_after == kept_before
        assert scripts_after == scripts_before
        assert kept_after["watchlist_name"] == "Doomed Watch"
        assert kept_after["body_markdown"] == "# Digest\n\nLast body before deletion.\n"
    finally:
        chacha_db.close_connection()


def test_watchlist_name_falls_back_when_watchlist_already_gone(tmp_path: Path) -> None:
    """The deleted-watchlist name fallback. `WatchlistBundleService.delete`
    can never leave this state (its cascade always takes the briefing with
    it); see `_delete_watchlist_bypassing_cascade`'s docstring for why this
    is the one way to construct it."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db, name="About To Vanish")
        briefing_id = _complete_briefing(subs_db, watchlist_id)

        _delete_watchlist_bypassing_cascade(subs_db, watchlist_id)
        assert subs_db.get_briefing(briefing_id) is not None  # orphaned, not cascaded

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept = chacha_db.get_kept_briefing(result["kept_id"])
        assert kept["watchlist_name"] == DELETED_WATCHLIST_NAME
    finally:
        chacha_db.close_connection()


# --- Cross-DB datetime boundary ----------------------------------------------


def test_to_chacha_datetime_attaches_utc_to_a_naive_string() -> None:
    assert _to_chacha_datetime("2026-01-15 10:30:00") == (
        datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc).isoformat()
    )


def test_to_chacha_datetime_leaves_an_offset_bearing_string_alone() -> None:
    assert _to_chacha_datetime("2026-01-15T10:30:00+05:00") == (
        datetime(2026, 1, 15, 10, 30, tzinfo=timezone(timedelta(hours=5))).isoformat()
    )


def test_to_chacha_datetime_passes_none_through() -> None:
    assert _to_chacha_datetime(None) is None


def test_kept_briefing_original_created_at_round_trips_as_tz_aware_utc(
    tmp_path: Path,
) -> None:
    """The named cross-DB round-trip: `briefings.created_at` is
    Subscriptions_DB's own naive-UTC `DEFAULT CURRENT_TIMESTAMP` string (no
    offset at all). Read back through ChaChaNotes -- which registers a
    `DATETIME` converter that only assumes UTC for a trailing `Z` (`DB/
    sqlite_datetime_fix.py`) -- the kept `original_created_at` must still
    be the CORRECT instant, tz-aware, not a naive datetime silently
    stripped of "this is UTC"."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)

        raw_created_at = subs_db.get_briefing(briefing_id)["created_at"]
        assert isinstance(raw_created_at, str)
        naive = datetime.fromisoformat(raw_created_at)
        assert naive.tzinfo is None  # pins the "naive-UTC-string" assumption itself

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept = chacha_db.get_kept_briefing(result["kept_id"])

        assert kept["original_created_at"] == naive.replace(tzinfo=timezone.utc)
        assert kept["original_created_at"].tzinfo is not None
    finally:
        chacha_db.close_connection()


def test_kept_script_original_created_at_round_trips_as_tz_aware_utc(
    tmp_path: Path,
) -> None:
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(subs_db, watchlist_id)
        script_id = _script(subs_db, briefing_id)

        raw_created_at = subs_db.get_briefing_script(script_id)["created_at"]
        naive = datetime.fromisoformat(raw_created_at)
        assert naive.tzinfo is None

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept_script = chacha_db.list_kept_scripts(result["kept_id"])[0]

        assert kept_script["original_created_at"] == naive.replace(tzinfo=timezone.utc)
    finally:
        chacha_db.close_connection()


def test_kept_briefing_covers_from_ts_round_trips_when_naive(tmp_path: Path) -> None:
    """`covers_from_ts` is usually already offset-bearing in real usage
    (`briefing_selection` builds it from a tz-aware `datetime.now(timezone.
    utc)`), but the boundary conversion must handle a naive value too --
    this seeds one explicitly via `update_briefing`, which accepts any
    string for this column."""
    subs_db = _subs_db(tmp_path)
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = _watchlist(subs_db)
        briefing_id = _complete_briefing(
            subs_db, watchlist_id, covers_from_ts="2026-07-25 00:00:00"
        )

        result = keep_briefing(subs_db, chacha_db, briefing_id, origin="manual")
        kept = chacha_db.get_kept_briefing(result["kept_id"])

        assert kept["covers_from_ts"] == datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc)
    finally:
        chacha_db.close_connection()
