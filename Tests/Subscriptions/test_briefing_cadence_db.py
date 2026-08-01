"""Tests for per-watchlist briefing cadence (briefings phase 4, Task 2).

`briefing_cadence_seconds` is an additive `watchlists` column: `NULL` means
"never scheduled" (Locked Decision 4 of
`Docs/superpowers/plans/2026-08-01-watchlists-briefings-phase-4.md` --
scheduled briefings are opt-in per watchlist, off by default).
`set_watchlist_briefing_settings` grows a third `_UNSET`-sentinel keyword
following the `default_preset_id` pattern verbatim; `list_briefing_schedules`
is the read side the phase 4 scheduler projection will consume.

Same harness as `test_briefing_presets_db.py`: a real `SubscriptionsDB` on
`:memory:`, `WatchlistBundleService` for watchlist creation (there is no
`SubscriptionsDB.create_watchlist`).
"""

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _make_watchlist(db, name="w"):
    return WatchlistBundleService(db).create(name=name)["id"]


# --- set_watchlist_briefing_settings: briefing_cadence_seconds sentinel trio --


def test_briefing_cadence_seconds_unset_leaves_alone_and_none_clears():
    """The sentinel trio: set once, leave-alone (the one that catches
    sentinel inversion -- if `_UNSET` were mistaken for "clear", the second
    call here would wipe the value it never mentioned), then explicit
    `None` clears it back to NULL."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    # Set it once.
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)
    row = db.conn.execute(
        "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["briefing_cadence_seconds"] == 3600

    # Calling again without the argument (the `_UNSET` sentinel default)
    # must leave it alone -- a call that only wants to change
    # `selection_mode` must not accidentally clear the cadence.
    db.set_watchlist_briefing_settings(watchlist_id, selection_mode="auto")
    row = db.conn.execute(
        "SELECT briefing_cadence_seconds, briefing_selection_mode FROM watchlists "
        "WHERE id = ?",
        (watchlist_id,),
    ).fetchone()
    assert row["briefing_cadence_seconds"] == 3600
    assert row["briefing_selection_mode"] == "auto"

    # Passing `None` explicitly clears it.
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=None)
    row = db.conn.execute(
        "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["briefing_cadence_seconds"] is None


def test_briefing_cadence_seconds_defaults_to_null_on_a_fresh_watchlist():
    """Locked Decision 4: scheduled briefings are opt-in -- a watchlist that
    never called `set_watchlist_briefing_settings` at all must read back
    NULL, not some non-opt-in default cadence."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    row = db.conn.execute(
        "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["briefing_cadence_seconds"] is None


@pytest.mark.parametrize("bad_value", [0, -1, -3600])
def test_briefing_cadence_seconds_rejects_non_positive_values_naming_the_value(bad_value):
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    with pytest.raises(ValueError, match=str(bad_value)):
        db.set_watchlist_briefing_settings(
            watchlist_id, briefing_cadence_seconds=bad_value
        )


def test_briefing_cadence_seconds_rejected_value_is_not_written():
    """A rejected call must not partially apply -- the column stays
    whatever it was before the raise."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=1800)

    with pytest.raises(ValueError):
        db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=-1)

    row = db.conn.execute(
        "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["briefing_cadence_seconds"] == 1800


# --- list_briefing_schedules -----------------------------------------------


def _force_created_at(db, briefing_id, timestamp):
    """Overwrite a `briefings` row's `created_at` directly.

    `insert_briefing` always stamps `CURRENT_TIMESTAMP`, which has only
    second resolution -- rows inserted in the same test within the same
    wall-clock second would otherwise share a `created_at` and make the
    status-allowlist tests below pass by accident (the MAX would land on
    the right value even if the WHERE clause leaked). Forcing a
    deliberately later timestamp on the row that must be *excluded* makes
    the assertion load-bearing: if the filter leaked, the excluded row's
    later timestamp would win the MAX and the test would catch it.
    """
    db.conn.execute(
        "UPDATE briefings SET created_at = ? WHERE id = ?", (timestamp, briefing_id)
    )
    db.conn.commit()


def test_list_briefing_schedules_excludes_null_cadence_watchlists_by_identity():
    """A count assertion alone would pass even if the NULL-cadence filter
    leaked (e.g. one cadenced + one un-cadenced watchlist both showing up
    would still yield len==1 if the filter dropped the wrong one by
    coincidence in a smaller fixture) -- assert on identity instead, and
    seed a second cadenced watchlist so the filter has something real to
    exclude between."""
    db = SubscriptionsDB(":memory:", "test")
    cadenced_a = _make_watchlist(db, name="cadenced-a")
    cadenced_b = _make_watchlist(db, name="cadenced-b")
    never_scheduled = _make_watchlist(db, name="never-scheduled")

    db.set_watchlist_briefing_settings(cadenced_a, briefing_cadence_seconds=3600)
    db.set_watchlist_briefing_settings(cadenced_b, briefing_cadence_seconds=86400)
    # never_scheduled: no cadence ever set -- stays NULL.

    schedules = db.list_briefing_schedules()

    assert {row["watchlist_id"] for row in schedules} == {cadenced_a, cadenced_b}
    assert never_scheduled not in {row["watchlist_id"] for row in schedules}


def test_list_briefing_schedules_reports_name_and_cadence():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="My Watchlist")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=7200)

    schedules = db.list_briefing_schedules()

    assert len(schedules) == 1
    row = schedules[0]
    assert row["watchlist_id"] == watchlist_id
    assert row["name"] == "My Watchlist"
    assert row["briefing_cadence_seconds"] == 7200


def test_list_briefing_schedules_never_briefed_cadenced_watchlist_has_null_last_completed_at():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    schedules = db.list_briefing_schedules()

    assert len(schedules) == 1
    assert schedules[0]["last_completed_at"] is None


def test_list_briefing_schedules_last_completed_at_ignores_failed_and_generating_rows():
    """THE schedule-side mirror of the coverage invariant: a failed or
    still-generating briefing must never advance the schedule. Forces both
    excluded rows to a later `created_at` than the real `complete` row, so
    if the status allowlist leaked either one in, the MAX would jump to
    the excluded row's later timestamp and this assertion would catch it."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    complete_id = db.insert_briefing(watchlist_id, status="complete")
    _force_created_at(db, complete_id, "2020-01-01 00:00:00")

    failed_id = db.insert_briefing(watchlist_id, status="failed")
    _force_created_at(db, failed_id, "2099-01-01 00:00:00")

    generating_id = db.insert_briefing(watchlist_id, status="generating")
    _force_created_at(db, generating_id, "2099-06-01 00:00:00")

    schedules = db.list_briefing_schedules()

    assert len(schedules) == 1
    assert schedules[0]["last_completed_at"] == "2020-01-01 00:00:00"


def test_list_briefing_schedules_an_empty_briefing_does_advance_last_completed_at():
    """`empty` is in the same allowlist as `complete` -- a briefing run
    that found nothing new still counts as having run, and must advance
    the schedule like any other completed attempt."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    empty_id = db.insert_briefing(watchlist_id, status="empty")
    _force_created_at(db, empty_id, "2021-06-15 12:00:00")

    schedules = db.list_briefing_schedules()

    assert len(schedules) == 1
    assert schedules[0]["last_completed_at"] == "2021-06-15 12:00:00"


def test_list_briefing_schedules_reads_inside_a_transaction(monkeypatch):
    """Qodo rule 1011851: reads must go through `self.transaction()`, not a
    bare connection call, so rollback-on-exception is consistently wired
    even for read paths."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    called = {"transaction": False}
    real_transaction = SubscriptionsDB.transaction

    def spy_transaction(self):
        called["transaction"] = True
        return real_transaction(self)

    monkeypatch.setattr(SubscriptionsDB, "transaction", spy_transaction)

    db.list_briefing_schedules()

    assert called["transaction"] is True
