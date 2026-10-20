"""A Console Character display check opens one SQLite handle (TASK-33620.15.6).

Every fresh file-backed private connection starts an ADR-125 validation helper
process, and its fork holds the GIL for milliseconds. Live, a send window ran
the stock Character display check about once a second; each check that found
the scope changed paid two finite callbacks (the scope probe, then the reload
batch, which reads the same scope again), so two helpers. These tests pin one
finite callback, one handle and one helper per display check, changed or not,
and that the merged check keeps the original admission, maintenance fence,
retirement and database-switch behaviour.

The spawn-counting seam is ``private_sqlite.prepare_in_helper`` (one call per
helper process), observed by code object through ``_original_calls`` without
replacing any guarded identity.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from types import CodeType

import pytest

from Tests.Backup_Recovery.test_character_presentation_cadence import (
    _character,
    _display,
)
from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _retired,
)
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]

_LOAD_ERROR = "Could not load local character chats · Retry"


def _assert_one_handle(observed) -> None:
    assert observed["callbacks"] == 1, (
        "a display check made more than one finite callback (one helper each)"
    )
    assert len(observed["connections"]) == 1
    if os.name != "nt":
        assert observed["helpers"] == 1, observed["helpers"]


@contextmanager
def _loop_checks():
    """Count the UI-loop owner checks a display check makes, on every thread.

    ``midpoints`` is the scope pair's on-loop ambient check (the probe's one
    loop handoff from its worker); ``owner_handoffs`` and ``owner_checks``
    are the batch readers' post-pair handoff and their on-loop owner check
    (reader provenance plus the display owner key).
    """
    readers = character_context._CharacterRefreshReaders
    pair = character_context.ConsoleCharacterContextController._read_database_scope_metadata_pair
    midpoint = [
        item
        for item in pair.__code__.co_consts
        if type(item) is CodeType and item.co_name == "validate"
    ]
    assert len(midpoint) == 1
    codes = {
        midpoint[0]: "midpoints",
        readers.ambient_on_loop.__code__: "owner_handoffs",
        readers.ambient_current.__code__: "owner_checks",
    }
    stats = dict.fromkeys(codes.values(), 0)

    def observe(frame, event, _argument):
        name = codes.get(frame.f_code)
        if name is not None and event == "call":
            stats[name] += 1

    old_thread, old_main = threading.getprofile(), sys.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        yield stats
    finally:
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)


async def _settled(database, *, run_active: bool):
    controller, screen, active, current, conversation = _character(database)
    await controller.refresh()
    if run_active:
        controller._run_active = lambda: True
    else:
        assert await _display(controller, screen) is False  # takes the memo
        controller._presentation_scope_at -= 2.01  # ... and expires it
    return controller, screen, active


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
@pytest.mark.parametrize("change", ["revision", "invalidated"])
async def test_changed_display_check_reloads_through_one_helper(
    tmp_path, run_active, change
):
    database = CharactersRAGDB(tmp_path / "changed.sqlite", "one-helper")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        if change == "revision":
            revision = database.increment_character_conversation_search_revision()
        else:
            controller.invalidate_scope()
            revision = database.get_character_conversation_search_revision()

        with _original_calls(database) as observed:
            assert await _display(controller, screen) is True

        _retired(database, observed)
        state = controller.state
        assert not state.loading and not state.error
        assert state.data_revision == revision
        assert state.scope_fingerprint.data_revision == revision
        assert observed["recent"] == 1
        # One scope read to decide, one fence read after the load.
        assert observed["pairs"] == 2
        _assert_one_handle(observed)
        if not run_active:
            # The reloaded display is current, so the memo serves the next one.
            with _original_calls(database) as warm:
                assert await _display(controller, screen) is False
            assert warm["pairs"] == warm["callbacks"] == 0
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_unchanged_display_check_stops_after_its_scope_read(tmp_path, run_active):
    database = CharactersRAGDB(tmp_path / "unchanged.sqlite", "one-helper")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        before = controller.state

        with _original_calls(database) as observed:
            assert await _display(controller, screen) is False

        _retired(database, observed)
        assert controller.state is before
        assert observed["pairs"] == 1 and observed["recent"] == 0
        _assert_one_handle(observed)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_unchanged_display_check_keeps_the_probe_loop_cost(tmp_path, run_active):
    """Folding the reload in adds no UI-loop handoff to an unchanged check.

    The unchanged check is the common one (about once a second during a
    run). Like the original probe it makes one on-loop midpoint inside its
    pair and confirms the ambient scope on the loop after retirement; it
    makes no reader owner check or extra loop handoff. (Qualifying the stock
    readers still runs on the loop; this pins handoffs, not CPU.)
    """
    database = CharactersRAGDB(tmp_path / "loop-cost.sqlite", "one-helper")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        before = controller.state

        with _loop_checks() as observed:
            assert await _display(controller, screen) is False

        assert controller.state is before
        assert observed == {"midpoints": 1, "owner_handoffs": 0, "owner_checks": 0}
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flap", ["owner-first-read", "owner-fence-read", "reader-first-read"]
)
async def test_unpublished_changed_check_leaves_no_idle_memo(
    tmp_path, monkeypatch, flap
):
    """A changed check that publishes nothing does not keep the idle memo.

    The original route reached ``refresh()`` whenever the scope changed, and
    ``refresh()`` drops the last-successful-refresh marker on entry. Here a
    run's check finds the scope changed, but its display owner (or a reader)
    changes during the read and changes back afterwards, so it publishes
    nothing. If the marker survived, the next idle display would take the
    still-armed memo and keep showing the old revision until the next sync.
    """
    # Pin the memo window so a slow helper start cannot expire it first.
    monkeypatch.setattr(
        character_context, "_CHARACTER_PRESENTATION_TTL_SECONDS", 3600.0
    )
    database = CharactersRAGDB(tmp_path / "flap.sqlite", "one-helper")
    try:
        controller, screen, *_ = _character(database)
        await controller.refresh()
        assert await _display(controller, screen) is False  # arms the idle memo
        shown = controller.state.data_revision
        revision = database.increment_character_conversation_search_revision()
        assert revision != shown
        app = screen.app_instance
        factory = controller._service_factory
        target = 2 if flap == "owner-fence-read" else 1
        pairs = []

        def change(_frame, name, event, _argument):
            if name != "pairs" or event != "call":
                return
            pairs.append(name)
            if len(pairs) != target:
                return
            if flap.startswith("owner"):
                app.chachanotes_db = object()  # another display owner
            else:
                controller._service_factory = object  # another reader

        controller._run_active = lambda: True
        try:
            with _original_calls(database, barrier=change):
                await _display(controller, screen)
        finally:
            app.chachanotes_db = database
            controller._service_factory = factory
        assert len(pairs) >= target
        assert not worker_leases(database)

        controller._run_active = lambda: False  # the reply ended
        assert await _display(controller, screen) is True
        assert controller.state.data_revision == revision
        assert not controller.state.loading and not controller.state.error
    finally:
        database.close()


@pytest.mark.asyncio
async def test_paused_storage_still_refuses_the_display_check(tmp_path):
    """Per-use storage admission applies to the merged check's one handle."""
    database = CharactersRAGDB(tmp_path / "paused.sqlite", "one-helper")
    try:
        controller, screen, _ = await _settled(database, run_active=True)
        database.increment_character_conversation_search_revision()
        database.close_connection()
        pause = storage._begin_local_pause()
        try:
            assert await _display(controller, screen) is True
        finally:
            pause.resume()
        assert controller.state.error == _LOAD_ERROR
        assert controller.state.scope_fingerprint is None
        assert not worker_leases(database)

        # Admission restored: the next check reloads through one handle.
        controller._scope_checked = None
        with _original_calls(database) as observed:
            assert await _display(controller, screen) is True
        _retired(database, observed)
        assert not controller.state.error
        _assert_one_handle(observed)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_database_maintenance_still_fences_the_display_check(tmp_path):
    """A check during a quiescence window opens no handle around the fence."""
    database = CharactersRAGDB(tmp_path / "quiesced.sqlite", "one-helper")
    try:
        controller, screen, _ = await _settled(database, run_active=True)
        database.increment_character_conversation_search_revision()
        with database.quiesce_connections(timeout_seconds=5):
            assert await _display(controller, screen) is True
            assert controller.state.error == _LOAD_ERROR
            assert not worker_leases(database)

        controller._scope_checked = None
        with _original_calls(database) as observed:
            assert await _display(controller, screen) is True
        _retired(database, observed)
        assert not controller.state.error
        assert (
            controller.state.data_revision
            == database.get_character_conversation_search_revision()
        )
        _assert_one_handle(observed)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_database_switch_mid_check_reads_the_new_database_its_own_way(
    tmp_path,
):
    """A handle serves one database: a switch during the read is not reused."""
    old = CharactersRAGDB(tmp_path / "old.sqlite", "old-profile")
    new = CharactersRAGDB(tmp_path / "new.sqlite", "new-profile")
    try:
        new_authority = new.get_local_authority_id()
        assert new_authority != old.get_local_authority_id()
        new.close_connection()
        controller, screen, active = await _settled(old, run_active=True)
        old.increment_character_conversation_search_revision()
        old.close_connection()
        switched = []

        def switch(_frame, name, event, _argument):
            if name == "pairs" and event == "call" and not switched:
                switched.append(True)
                active[0] = new  # the profile switches during the first read

        with _original_calls(old, barrier=switch) as observed:
            await _display(controller, screen)

        assert switched
        # The old profile's handle read only the old profile and is retired.
        assert observed["leases"]
        assert all(item.resource_path == old.db_path for item in observed["leases"])
        assert not worker_leases(old)
        assert not worker_leases(new)

        # A display owner that changed mid-check yields to the next check,
        # which reads the new profile through its own single handle. The
        # switch itself defeats the run's one-second recheck interval.
        with _original_calls(new) as fresh:
            assert await _display(controller, screen) is True
        _retired(new, fresh)
        fingerprint = controller.state.scope_fingerprint
        assert fingerprint is not None and not controller.state.error
        assert fingerprint.database_identity == id(new)
        assert fingerprint.data_authority_id == new_authority
        assert not worker_leases(old)
        _assert_one_handle(fresh)
    finally:
        old.close()
        new.close()
