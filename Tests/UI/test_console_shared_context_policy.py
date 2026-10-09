"""Actual checked display policy sharing never replaces live action authority."""

import asyncio
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.config_participants import operation
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


def _screen(tmp_path):
    database, store, controller, _ = _controller(tmp_path)
    store.active_session_id = "session-1"
    store.append_message(
        "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
    )
    tasks = []

    def schedule(coroutine, **_kwargs):
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    async def refresh():
        return None

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            app_config=config.load_settings(), chachanotes_db=database
        ),
        _console_chat_store=store,
        _console_config_snapshot_is_disk_loaded=lambda _: True,
        _console_derivation_memo=None,
        _console_derivation_scope=lambda: _empty_scope(),
        _sync_native_console_chat_ui=refresh,
        run_worker=schedule,
    )
    return database, store, controller, screen, tasks


@contextmanager
def _empty_scope():
    yield


@contextmanager
def _actual_calls(controller):
    # Observe the actual original guarded/native path without replacing any
    # installed callable identity. Keep generator frames to count entry once.
    config_code = operation.__wrapped__.__code__
    policy_code = controller._global_context_policy_overrides.__func__.__code__
    frames, calls = [], {"config": 0, "live_policy": 0}
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _argument):
        if event != "call":
            return
        if frame.f_code is policy_code:
            calls["live_policy"] += 1
        elif frame.f_code is config_code and all(frame is not old for old in frames):
            frames.append(frame)
            calls["config"] += 1

    threading.setprofile_all_threads(observe)
    try:
        yield calls
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)
        frames.clear()


@pytest.mark.asyncio
async def test_real_context_display_reuses_actual_checked_config_operation(tmp_path):
    database, store, controller, screen, tasks = _screen(tmp_path)
    try:
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
        with _actual_calls(controller) as calls:
            assert projection.run(lambda: None) is False
            await asyncio.gather(*tasks)
            assert await snapshot.warm(controller, "session-1")
            displayed = snapshot.inputs(controller, "session-1")
            assert (
                calls["live_policy"] == 0
            ), "display opened a second live policy scope"
            assert calls["config"] == 1, "display repeated the checked config lifetime"
            live = controller.context_control_inputs("session-1")
            assert (
                calls["live_policy"] == 1
            ), "live action reused disposable display data"
            assert displayed[1] == live[1]
            assert calls["config"] == 2
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_expired_checked_read_failure_cannot_satisfy_modal_warm(tmp_path):
    database, _store, controller, screen, tasks = _screen(tmp_path)
    try:
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        assert await snapshot.warm(controller, "session-1")
        original = projection.read_current
        projection.at -= projection.max_age + 1

        def failed_read():
            raise OSError("checked reader unavailable")

        projection.read_current = failed_read
        assert not await snapshot.warm(
            controller, "session-1"
        ), "expired display data satisfied a modal requiring a checked refresh"
        projection.read_current = original
        assert await snapshot.warm(controller, "session-1")
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["before", "after"])
async def test_context_never_uses_wrong_checked_source_policy(tmp_path, edge):
    from dataclasses import replace

    database, _store, controller, screen, tasks = _screen(tmp_path)
    try:
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
        original = projection.read_current
        actual = config.current_config_identity()
        wrong = actual[0] + 1, actual[1]

        def wrong_source():
            checked = original()
            return replace(checked, **{f"source_{edge}": wrong})

        projection.read_current = wrong_source
        with _actual_calls(controller) as calls:
            assert not await snapshot.warm(controller, "session-1")
            assert calls["live_policy"] == 0, "source refusal fell back to live policy"
        assert projection.value is None and snapshot.value is None
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_context_source_retarget_aba_rejects_actual_other_source_result(
    tmp_path, monkeypatch
):
    from pathlib import Path

    database, _store, controller, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    actual = config.current_config_identity()
    requested = actual[0], str(Path(actual[1]).with_name("requested-A.toml"))
    observed = list(requested)
    monkeypatch.setattr(config, "current_config_identity", lambda: tuple(observed))
    entered, begin, loaded, finish = [threading.Event() for _ in range(4)]
    real_to_thread = asyncio.to_thread
    failures = []

    def held_read():
        entered.set()
        assert begin.wait(5)
        try:
            checked = projection.read_current()
        except Exception as error:
            failures.append(error)
            loaded.set()
            raise
        loaded.set()
        assert finish.wait(5)
        return checked

    async def gate(function, *args, **kwargs):
        if function is projection.read_current:
            return await real_to_thread(held_read)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(spend.asyncio, "to_thread", gate)
    warmer = asyncio.create_task(snapshot.warm(controller, "session-1"))
    try:
        assert await real_to_thread(entered.wait, 5)
        observed[:] = actual
        begin.set()
        assert await real_to_thread(loaded.wait, 5)
        assert not failures, repr(failures)
        # Both tags are the actual native B source, while the cheap UI key
        # returns to requested A. No native source guard is bypassed/replaced.
        observed[:] = requested
        finish.set()
        assert not await warmer
        assert snapshot.value is None and projection.value is None
    finally:
        begin.set()
        finish.set()
        await asyncio.gather(warmer, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["session", "workspace", "settings", "closed"])
async def test_shared_context_cold_read_rejects_changed_actual_owner(
    tmp_path, mutation
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    original = projection.read_current
    entered, release = threading.Event(), threading.Event()

    def held_read():
        checked = original()
        entered.set()
        assert release.wait(5)
        return checked

    projection.read_current = held_read
    warmer = asyncio.create_task(snapshot.warm(controller, "session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if mutation == "session":
            store.create_session(session_id="other")
            store.active_session_id = "other"
        elif mutation == "workspace":
            store.sessions()[0].workspace_id = "other-workspace"
        elif mutation == "settings":
            from Tests.Chat.test_console_first_send_atomicity import (
                _stage_first_send_settings,
            )

            before = store.session_settings_revision("session-1")
            await _stage_first_send_settings(store, expected_staged=False)
            assert store.session_settings_revision("session-1") > before
        else:
            store.close_session("session-1")
        release.set()
        assert not await warmer
        assert projection.value is None and snapshot.value is None
    finally:
        release.set()
        await asyncio.gather(warmer, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_cancelled_cold_modal_wait_does_not_publish_context_or_retire_config_early(
    tmp_path,
):
    database, _store, controller, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    original = projection.read_current
    entered, release, retired = [threading.Event() for _ in range(3)]

    def held_read():
        try:
            checked = original()
            entered.set()
            assert release.wait(5)
            return checked
        finally:
            retired.set()

    projection.read_current = held_read
    warmer = asyncio.create_task(snapshot.warm(controller, "session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        warmer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await warmer
        assert not retired.is_set() and projection.pending
        assert snapshot.value is None
        release.set()
        await asyncio.gather(*tasks)
        assert retired.is_set() and not projection.pending
        assert snapshot.value is None
        assert await snapshot.warm(controller, "session-1")
    finally:
        release.set()
        await asyncio.gather(warmer, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_shared_policy_expiry_coalesces_one_real_checked_refresh(tmp_path):
    database, _store, controller, screen, tasks = _screen(tmp_path)
    try:
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
        with _actual_calls(controller) as calls:
            assert await snapshot.warm(controller, "session-1")
            initial = snapshot.inputs(controller, "session-1")
            assert calls["config"] == 1
            projection.at -= projection.max_age + 1
            snapshot.at -= snapshot.max_age + 1
            for _ in range(6):
                assert snapshot.inputs(controller, "session-1") == initial
            assert calls["config"] == 1
            assert len([task for task in tasks if not task.done()]) == 2
            # One config worker plus one context worker waiting on it.
            await asyncio.gather(*tasks)
            assert calls["config"] == 2 and calls["live_policy"] == 0
            assert snapshot.inputs(controller, "session-1") == initial
            assert projection.max_age == snapshot.max_age == 1
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_changed_config_owner_discards_actual_context_worker_result(
    tmp_path, monkeypatch
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    observed = list(config.current_config_identity())
    monkeypatch.setattr(config, "current_config_identity", lambda: tuple(observed))
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    original = store.persistence.get_message_versions
    entered, release = threading.Event(), threading.Event()

    def held_versions(ids):
        actual = original(ids)
        entered.set()
        assert release.wait(5)
        return actual

    monkeypatch.setattr(store.persistence, "get_message_versions", held_versions)
    warmer = asyncio.create_task(snapshot.warm(controller, "session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        observed[0] += 1
        release.set()
        assert not await warmer
        assert snapshot.value is None
    finally:
        release.set()
        await asyncio.gather(warmer, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_settings_open_waits_for_shared_cold_checked_policy(
    tmp_path, monkeypatch
):
    from Tests.UI.test_console_settings_context_owner import _screen as modal_screen

    database, store, controller, screen, pushed, tasks = modal_screen(
        tmp_path, monkeypatch
    )
    screen.app_instance.app_config = config.load_settings()
    screen.app_instance.chachanotes_db = database
    # This existing modal fixture intentionally omits runtime/view wiring;
    # expose its real store without constructing that unrelated runtime.
    monkeypatch.setattr(
        type(screen), "_console_chat_store", property(lambda _screen: store)
    )
    screen._console_derivation_memo = None
    screen._console_config_snapshot_is_disk_loaded = lambda _: True
    try:
        assert await asyncio.wait_for(screen._open_console_settings(), 5)
        assert len(pushed) == 1
        assert screen._console_readiness_config_projection.value is not None
        assert screen._console_context_read_snapshot.value is not None
        assert screen._console_context_read_snapshot.config_projection is (
            screen._console_readiness_config_projection
        )
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_settings_open_closed_during_shared_checked_read_never_pushes_modal(
    tmp_path, monkeypatch
):
    from Tests.UI.test_console_settings_context_owner import _screen as modal_screen

    database, store, _controller, screen, pushed, tasks = modal_screen(
        tmp_path, monkeypatch
    )
    screen.app_instance.app_config = config.load_settings()
    screen.app_instance.chachanotes_db = database
    # This existing modal fixture intentionally omits runtime/view wiring;
    # expose its real store without constructing that unrelated runtime.
    monkeypatch.setattr(
        type(screen), "_console_chat_store", property(lambda _screen: store)
    )
    screen._console_derivation_memo = None
    screen._console_config_snapshot_is_disk_loaded = lambda _: True
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    original = projection.read_current
    entered, release = threading.Event(), threading.Event()

    def held_read():
        actual = original()
        entered.set()
        assert release.wait(5)
        return actual

    projection.read_current = held_read
    opener = asyncio.create_task(screen._open_console_settings())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        store.close_session("session-1")
        release.set()
        assert not await opener
        assert pushed == []
        assert projection.value is None
        assert screen._console_context_read_snapshot.value is None
    finally:
        release.set()
        await asyncio.gather(opener, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_forged_display_policy_never_becomes_action_or_dispatch_preflight(
    tmp_path,
):
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextCompactionMode,
    )

    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    try:
        actual = controller.context_control_inputs("session-1")[1]
        prior_mode = controller._compaction_mode_for(store.sessions()[0])
        other_mode = (
            ContextCompactionMode.OFF
            if prior_mode is not ContextCompactionMode.OFF
            else ContextCompactionMode.ASK
        )
        forged = ConsoleContextPolicyOverrides(compaction_mode=other_mode)
        with _actual_calls(controller) as calls:
            display = await controller.context_control_presentation_inputs(
                "session-1", _presentation_global_overrides=forged
            )
            assert display[1] is forged
            assert calls["live_policy"] == 0
            assert controller.context_control_inputs("session-1")[1] == actual
            assert controller._compaction_mode_for(store.sessions()[0]) is prior_mode
            assert calls["live_policy"] == 2
            assert calls["config"] == 2
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_actual_same_id_session_replacement_refuses_pending_checked_policy(
    tmp_path,
):
    from dataclasses import replace

    database, store, controller, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    original = projection.read_current
    entered, release = threading.Event(), threading.Event()

    def held_read():
        checked = original()
        entered.set()
        assert release.wait(5)
        return checked

    projection.read_current = held_read
    warmer = asyncio.create_task(snapshot.warm(controller, "session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        prior = store._sessions["session-1"]
        replacement = replace(prior)
        assert replacement == prior and replacement is not prior
        store._sessions["session-1"] = replacement
        release.set()
        assert (
            not await warmer
        ), "field-equal replacement inherited old session's checked display owner"
        assert projection.value is None and snapshot.value is None
    finally:
        release.set()
        await asyncio.gather(warmer, *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_context_memo_rejects_field_equal_same_id_owner_replacement(tmp_path):
    from dataclasses import replace

    database, store, controller, screen, tasks = _screen(tmp_path)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    try:
        assert await snapshot.warm(controller, "session-1")
        prior = snapshot._key(controller, "session-1")
        owner = store._sessions["session-1"]
        replacement = replace(owner)
        assert replacement == owner and replacement is not owner
        store._sessions["session-1"] = replacement
        assert (
            snapshot._key(controller, "session-1") != prior
        ), "context memo omitted exact actual session owner"
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_repeated_readiness_cancellation_waits_for_actual_raw_native_retirement(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.config_participants import (
        checked_config_identity,
    )

    database, _store, _controller_owner, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    entered, release = threading.Event(), threading.Event()
    observed = []
    returns = 0
    code = checked_config_identity.__code__
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _arg):
        nonlocal returns
        if event == "return" and frame.f_code is code:
            returns += 1
            if returns == 2:
                observed.append(raw._states[frame.f_locals["active"]])
                entered.set()
                assert release.wait(10)

    threading.setprofile_all_threads(observe)
    try:
        assert projection.run(lambda: None) is False
        assert await asyncio.to_thread(entered.wait, 10)
        state = observed[0]
        with storage._lock:
            assert state.active and len(state.leases) >= 2
            assert all(lease in storage._live_leases for lease in state.leases)
        task = tasks[0]
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert (
            not task.done()
        ), "second cancellation ended readiness before native retirement"
        assert projection.pending and not projection._settled.is_set()
        assert state.active
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not projection.pending and projection._settled.is_set()
        assert not state.active and projection.value is None
        with storage._lock:
            assert all(lease not in storage._live_leases for lease in state.leases)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)
        database.close()
