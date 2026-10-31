"""A settled Console Character display check costs the loop what the probe did (TASK-35961).

TASK-33620.15.6 folded the display check's scope probe and reload into one
finite callback. Its common case, an unchanged scope (about once a second
during a reply), then paid more UI-loop CPU than the original probe route:
before the read it qualified every stock reader on the loop (the selector,
then the readers' own check), and the pair's on-loop midpoint qualified them
all twice more. Measured in process on the 15.6 head, an unchanged check made
252 reader qualifications on the loop against the probe route's 94.

A settled check now qualifies on the loop only what the loop runs (the
display entry's own checks, as on the original route, and the readers the
midpoint calls); the worker qualifies every reader before each read, as it
did. A first scope read that does not settle, or fails, is the probe's
outcome: the check reloads without probing the scope again. A reader
customized before a settled check is still refused before any read, and the
check then takes the original route.

Two of the loop's remaining checks are pinned on their own (review of the
first cut): an identity reader changed as a settled pair's on-loop midpoint
starts never runs its changed body there (``require_loop_readers``; dev
holds this too, with its whole-set check), and a ``_await_database_work``
changed after the display entry's checks never receives the merged callback
(it was handed ``read_current`` before the selection qualified it).
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
from contextlib import contextmanager
from types import CodeType, FunctionType

import pytest

from Tests.Backup_Recovery.test_character_presentation_cadence import _display
from Tests.Backup_Recovery.test_character_refresh_batch_entry_controls import (
    _body_fault,
)
from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _retired,
    _source_hashes,
)
from Tests.Backup_Recovery.test_character_scope_check_one_helper import _settled
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]

_ENTRY_CODES = frozenset(
    {
        character_context.ConsoleCharacterContextController.refresh_presentation_if_scope_changed.__code__,
        character_context._CHARACTER_PRESENTATION_OWNER_CODE,
    }
)


@contextmanager
def _loop_qualifications():
    """Count, on the UI loop's thread only, the reader qualifications a check makes.

    ``full`` counts every qualification of the whole stock reader set:
    ``presentation`` (``_stock_character_presentation``) by its caller,
    ``selector`` (``_stock_character_refresh_readers``) and ``readers``
    (``_CharacterRefreshReaders.require_current``). ``other`` counts single
    reader checks (``_character_reader_current``) made outside a
    presentation or scope-pair check.
    """
    codes = {
        character_context._stock_character_presentation.__code__: "presentation",
        character_context._stock_character_refresh_readers.__code__: "selector",
        character_context._CharacterRefreshReaders.require_current.__code__: "readers",
    }
    grouped = {
        character_context._stock_character_presentation.__code__,
        character_context._stock_character_scope_pair.__code__,
    }
    single = character_context._character_reader_current.__code__
    stats = {"presentation_entry": 0, "presentation_other": 0, "selector": 0}
    stats |= {"readers": 0, "other": 0}

    def observe(frame, event, _argument):
        if event != "call":
            return
        code = frame.f_code
        name = codes.get(code)
        if name == "presentation":
            caller = frame.f_back.f_code if frame.f_back is not None else None
            key = (
                "presentation_entry" if caller in _ENTRY_CODES else "presentation_other"
            )
            stats[key] += 1
        elif name is not None:
            stats[name] += 1
        elif code is single:
            parent = frame.f_back
            for _ in range(3):  # Through the genexpr feeding all()/any().
                if parent is None or parent.f_code in grouped:
                    break
                parent = parent.f_back
            if parent is None or parent.f_code not in grouped:
                stats["other"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)  # This thread only: the loop the check runs on.
    try:
        yield stats
    finally:
        sys.setprofile(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_unchanged_settled_check_qualifies_on_the_loop_only_at_its_entry(
    tmp_path, run_active
):
    """No whole reader-set qualification on the loop beyond the display entry's.

    The display entry's own presentation checks are the ones the original
    probe route made. On the 15.6 head an unchanged check also ran the
    selector, its readers' check and two more in the pair's midpoint, and a
    run's check one more presentation check: 252 single-reader checks on the
    loop against the probe route's 94.
    """
    database = CharactersRAGDB(tmp_path / "settled.sqlite", "loop-cost")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        before = controller.state
        if run_active:
            controller._scope_checked = None  # The 1 s recheck interval is over.

        with _original_calls(database) as observed, _loop_qualifications() as loop:
            assert await _display(controller, screen) is False

        _retired(database, observed)
        assert controller.state is before
        assert observed["callbacks"] == 1 and observed["pairs"] == 1
        if os.name != "nt":
            assert observed["helpers"] == 1
        assert loop["selector"] == 0, loop
        assert loop["readers"] == 0, loop
        assert loop["presentation_other"] == 0, loop
        # The readers the midpoint calls are checked one by one: fewer than
        # one whole reader set in all.
        assert loop["other"] < len(character_context._CHARACTER_REFRESH_READERS), loop
    finally:
        database.close()


@contextmanager
def _unsettled_first_reads(database, unsettled: int):
    """Move the revision between the two reads of the first ``unsettled`` pairs.

    The bump runs in the pair's on-loop midpoint, which sits between its two
    metadata reads, so each of those pairs reads two different revisions.
    Also counts finite callbacks, pairs and helper processes for ``database``.
    """
    controller_type = character_context.ConsoleCharacterContextController
    pair = controller_type._read_database_scope_metadata_pair
    (midpoint,) = (
        item
        for item in pair.__code__.co_consts
        if type(item) is CodeType and item.co_name == "validate"
    )
    (invoke,) = (
        item
        for item in controller_type._await_database_work.__code__.co_consts
        if type(item) is CodeType and item.co_name == "invoke"
    )
    stats = {"callbacks": 0, "pairs": 0, "helpers": 0, "bumps": 0}

    def observe(frame, event, _argument):
        if event != "call":
            return
        code = frame.f_code
        if code is invoke and frame.f_locals.get("database") is database:
            stats["callbacks"] += 1
        elif code is pair.__code__ and frame.f_locals.get("database") is database:
            stats["pairs"] += 1
        elif code is private_sqlite.prepare_in_helper.__code__:
            stats["helpers"] += 1
        elif code is midpoint and stats["bumps"] < unsettled:
            stats["bumps"] += 1
            database.increment_character_conversation_search_revision()

    old_thread, old_main = threading.getprofile(), sys.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        yield stats
    finally:
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_unsettled_first_read_reloads_without_probing_again(tmp_path, run_active):
    """A first scope read that does not settle is the probe's outcome.

    The merged read tries the pair three times on its one handle, as the
    original probe tried three times, and does not settle. The original route
    then reloads. On the 15.6 head the check handed that to the whole
    original probe-then-refresh route, which probed the scope again on a
    handle of its own first: three helpers where the read and the reload
    need two.
    """
    database = CharactersRAGDB(tmp_path / "unsettled.sqlite", "unsettled")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        if run_active:
            controller._scope_checked = None
        attempts = character_context._SCOPE_CAPTURE_ATTEMPTS

        with _unsettled_first_reads(database, attempts) as observed:
            assert await _display(controller, screen) is True

        assert observed["bumps"] == attempts
        assert not worker_leases(database)
        state = controller.state
        assert not state.loading and not state.error
        revision = database.get_character_conversation_search_revision()
        assert state.data_revision == revision
        assert state.scope_fingerprint.data_revision == revision
        # The unsettled read (three pairs), then the reload (a pair, the
        # groups and the fence pair): two finite callbacks, two helpers.
        assert observed["pairs"] == attempts + 2, observed
        assert observed["callbacks"] == 2, observed
        if os.name != "nt":
            assert observed["helpers"] == 2, observed
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
@pytest.mark.parametrize(
    "method", ["_read_recent_scope_sync", "_read_refresh_scope_sync"]
)
async def test_reader_changed_after_selection_refuses_and_takes_the_probe_route(
    tmp_path, run_active, method
):
    """A settled check still refuses a reader changed after its selection.

    A reader changed before the check is refused at the display entry, which
    qualifies every reader on the loop as the original route did. One changed
    after the loop selected the check's readers (here as its finite callback
    starts) is refused by the worker's qualification before any read: that
    callback opens no handle, and the check takes the original probe route,
    which runs neither body and finds the scope unchanged. On the 15.6 head
    the same refusal ended the check as changed without reading the scope.
    """
    database = CharactersRAGDB(tmp_path / "custom.sqlite", "custom")
    function = vars(character_context.ConsoleCharacterContextController)[method]
    original_code, before = function.__code__, _source_hashes()
    calls, restores = [], []
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        state = controller.state
        if run_active:
            controller._scope_checked = None

        def change(_frame, name, event, _argument):
            if name == "callbacks" and event == "call" and not restores:
                restores.append(_body_fault(function, calls))

        with _original_calls(database, barrier=change) as observed:
            changed = await _display(controller, screen)

        assert restores, "the check's finite callback never started"
        assert not calls, "a changed Character reader body ran on a settled check"
        assert observed["recent"] == 0
        # The refused callback read nothing; the original probe read the
        # scope once, through the check's only handle.
        assert observed["callbacks"] == 2 and observed["pairs"] == 1, observed
        _retired(database, observed)
        assert len(observed["connections"]) == 1
        if os.name != "nt":
            assert observed["helpers"] == 1
        assert changed is False
        assert controller.state is state
        assert before == _source_hashes()
    finally:
        for restore in restores:
            restore()
        assert function.__code__ is original_code
        database.close()


def _recording_body(function, record):
    """Change ``function``'s body to one that records each call, then runs the original.

    The change is a new code object in the original defining namespace, so
    every reader qualification sees a customized reader. Returns the restore.
    """
    assert type(function) is FunctionType and function.__closure__ is None
    clone = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    namespace = function.__globals__
    names = ("_character_settled_original", "_character_settled_record")
    assert all(name not in namespace for name in names)
    namespace[names[0]], namespace[names[1]] = clone, record
    asynchronous = inspect.iscoroutinefunction(function)
    exec(  # noqa: S102 - fixed test body in the original defining namespace.
        f"{'async ' if asynchronous else ''}def _character_settled_body(self, *args, **kwargs):\n"
        "    _character_settled_record(args)\n"
        f"    return {'await ' if asynchronous else ''}"
        "_character_settled_original(self, *args, **kwargs)\n",
        namespace,
    )
    original_code = function.__code__
    function.__code__ = namespace.pop("_character_settled_body").__code__

    def restore():
        function.__code__ = original_code
        for name in names:
            namespace.pop(name)

    return restore


@contextmanager
def _loop_change(code, function, calls, *, until_return):
    """Change ``function``'s body when ``code`` first starts on this (the loop's) thread.

    With ``until_return`` the original body is back when that first frame
    returns; otherwise it stays changed until the context exits. ``calls``
    records each changed-body call as ``(thread name, arguments)``.
    """
    state = {"changed": 0, "restore": None}

    def record(args):
        calls.append((threading.current_thread().name, args))

    def observe(frame, event, _argument):
        if frame.f_code is not code or event not in {"call", "return"}:
            return
        if event == "call" and not state["changed"]:
            state["changed"] = 1
            state["restore"] = _recording_body(function, record)
        elif event == "return" and until_return and state["restore"] is not None:
            state.pop("restore")()
            state["restore"] = None

    previous = sys.getprofile()
    sys.setprofile(observe)  # This thread only: the loop the check runs on.
    try:
        yield state
    finally:
        sys.setprofile(previous)
        if state.get("restore") is not None:
            state.pop("restore")()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
@pytest.mark.parametrize(
    "method", ["_current_character_identity", "_open_conversation_identity"]
)
async def test_reader_changed_at_the_settled_midpoint_never_runs_on_the_loop(
    tmp_path, run_active, method
):
    """The settled pair's on-loop midpoint still refuses a changed identity reader.

    A settled check's midpoint no longer qualifies the whole reader set on
    the loop; it qualifies what it runs there (``require_loop_readers``):
    the two identity readers ``_ambient_scope_matches`` calls. One changed as
    the midpoint starts, after the worker's own check, is refused there:
    its changed body never runs on the UI loop. Dev qualifies the whole set
    there, so this holds on dev too; it pins the one on-loop check left in
    the settled first read, which no other Character test catches.
    """
    database = CharactersRAGDB(tmp_path / "midpoint.sqlite", "midpoint")
    function = vars(character_context.ConsoleCharacterContextController)[method]
    original_code, calls = function.__code__, []
    pair = character_context.ConsoleCharacterContextController._read_database_scope_metadata_pair
    (midpoint,) = (
        item
        for item in pair.__code__.co_consts
        if type(item) is CodeType and item.co_name == "validate"
    )
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        if run_active:
            controller._scope_checked = None

        with _loop_change(midpoint, function, calls, until_return=True) as change:
            await _display(controller, screen)

        assert change["changed"] == 1, "the settled check's midpoint never ran"
        assert function.__code__ is original_code
        assert not calls, f"a changed {method} body ran: {calls}"
        assert not worker_leases(database)
        assert not controller.state.loading
    finally:
        function.__code__ = original_code
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_dispatch_changed_after_the_entry_never_receives_the_merged_read(
    tmp_path, run_active
):
    """A settled check hands its merged read only to the stock dispatch.

    ``_await_database_work`` runs on the loop and receives the check's
    finite callback. The display entry qualifies it with every reader, but
    ambient accessors run between that and the dispatch. Changed there (here
    as ``_check_recent_batch`` starts), the check takes the original route,
    whose own reads the changed dispatch keeps, as a customized reader
    always has; the merged callback never reaches it.
    """
    database = CharactersRAGDB(tmp_path / "dispatch.sqlite", "dispatch")
    controller_type = character_context.ConsoleCharacterContextController
    function = vars(controller_type)["_await_database_work"]
    original_code, calls = function.__code__, []
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        state = controller.state
        if run_active:
            controller._scope_checked = None

        with _loop_change(
            controller_type._check_recent_batch.__code__,
            function,
            calls,
            until_return=False,
        ) as change:
            changed = await _display(controller, screen)

        assert change["changed"] == 1, "the check never reached its dispatch"
        assert function.__code__ is original_code
        merged = [
            getattr(args[1], "__qualname__", "")
            for _thread, args in calls
            if len(args) > 1
        ]
        assert not any(
            name.endswith("_check_recent_batch.<locals>.read_current") for name in merged
        ), merged
        assert calls, "the original route did not read through the changed dispatch"
        assert changed is False
        assert controller.state is state
        assert not worker_leases(database)
    finally:
        function.__code__ = original_code
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_active", [True, False], ids=["active", "idle"])
async def test_failed_first_read_reloads_without_probing_again(tmp_path, run_active):
    """A first scope read that fails is the probe's outcome, as one that does not settle.

    The original route answers a failed probe by reloading, which presents
    the read error. On the 15.6 head the check handed the failure to the
    whole probe-then-refresh route, which probed the scope again on a handle
    of its own first: three helpers where the read and the reload need two.
    """
    database = CharactersRAGDB(tmp_path / "failed.sqlite", "failed")
    try:
        controller, screen, _ = await _settled(database, run_active=run_active)
        if run_active:
            controller._scope_checked = None
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE character_conversation_search_revision"
                " SET data_revision = ? WHERE singleton_id = 1",
                ("invalid",),
            )

        with _unsettled_first_reads(database, 0) as observed:
            await _display(controller, screen)

        assert not worker_leases(database)
        state = controller.state
        assert not state.loading and state.error
        # The failed read, then the reload that presents its error: two
        # finite callbacks, two helpers.
        assert observed["callbacks"] == 2, observed
        if os.name != "nt":
            assert observed["helpers"] == 2, observed
    finally:
        database.close()
