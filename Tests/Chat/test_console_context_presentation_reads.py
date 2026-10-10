"""Presentation reads use captured lineage and leave send authority unchanged."""

import asyncio
from contextlib import asynccontextmanager
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_compaction_live_session import (
    _close_live_databases as _close_live_databases,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_context_presentation_batches_reads_off_ui_thread(tmp_path, monkeypatch):
    database, store, controller, _ = _controller(tmp_path)
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        calls = []
        original = store.persistence.get_message_versions

        def versions(ids):
            calls.append(threading.get_ident())
            return original(ids)

        monkeypatch.setattr(store.persistence, "get_message_versions", versions)
        assert hasattr(
            spend, "ConsoleContextReadSnapshot"
        ), "finite presentation read owner is missing"
        snapshot = spend.ConsoleContextReadSnapshot()
        with snapshot.scope():
            await snapshot.warm(controller, "session-1")
            first = snapshot.inputs(controller, "session-1")
            assert snapshot.inputs(controller, "session-1") == first
            await snapshot.warm(controller, "session-1")
        assert calls == [calls[0]]
        assert calls[0] != threading.get_ident()
        # Other presentation callers reuse the same owner outside the tick.
        assert (
            await asyncio.create_task(_presentation_input(snapshot, controller))
            == first
        )
        assert len(calls) == 1
        # Explicit actions use the controller authority seam independently.
        controller.context_control_inputs("session-1")
        assert calls[-1] == threading.get_ident()
    finally:
        database.close()


async def _presentation_input(snapshot, controller):
    return snapshot.inputs(controller, "session-1")


def test_removed_presentation_session_preserves_key_error_contract(tmp_path):
    database, store, controller, _ = _controller(tmp_path)
    try:
        store.close_session("session-1")
        snapshot = spend.ConsoleContextReadSnapshot()
        with pytest.raises(KeyError, match="session-1"):
            snapshot.inputs(controller, "session-1")
    finally:
        database.close()


@pytest.mark.asyncio
async def test_cold_and_expired_presentation_never_fall_back_to_live_ui_reads(
    tmp_path, monkeypatch
):
    database, store, controller, _ = _controller(tmp_path)
    tasks, calls = [], []
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original = store.persistence.get_message_versions

        def versions(ids):
            calls.append(threading.get_ident())
            return original(ids)

        def schedule(operation, **_kwargs):
            task = asyncio.create_task(operation)
            tasks.append(task)
            return task

        monkeypatch.setattr(store.persistence, "get_message_versions", versions)
        monkeypatch.setattr(
            controller,
            "context_control_inputs",
            lambda *_: pytest.fail("presentation fell back to live UI reads"),
        )
        snapshot = spend.ConsoleContextReadSnapshot(schedule=schedule)
        assert snapshot.inputs(controller, "session-1")[2] is None
        assert calls == []
        await asyncio.gather(*tasks)
        first = snapshot.inputs(controller, "session-1")
        assert len(calls) == 1
        snapshot.at -= snapshot.max_age + 1
        for _ in range(6):
            assert snapshot.inputs(controller, "session-1") == first
        assert len(calls) == 1
        await asyncio.gather(*tasks)
        assert len(calls) == 2
        assert all(thread != threading.get_ident() for thread in calls)
        # A changed owner cannot display the previous memory while pending.
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="new", persist=True
        )
        assert snapshot.inputs(controller, "session-1")[2] is None
        await asyncio.gather(*tasks)
        assert len(calls) == 3
    finally:
        database.close()


@pytest.mark.asyncio
async def test_unchanged_presentation_reuses_one_read_across_refresh_ticks(
    tmp_path, monkeypatch
):
    database, store, controller, _ = _controller(tmp_path)
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        calls = []
        original = store.persistence.get_message_versions

        def versions(ids):
            calls.append(threading.get_ident())
            return original(ids)

        monkeypatch.setattr(store.persistence, "get_message_versions", versions)
        snapshot = spend.ConsoleContextReadSnapshot()
        for _ in range(6):
            with snapshot.scope():
                assert await snapshot.warm(controller, "session-1")
                snapshot.inputs(controller, "session-1")
        assert len(calls) == 1
        # Payload changes within the run invalidate immediately.
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="new", persist=True
        )
        with snapshot.scope():
            assert await snapshot.warm(controller, "session-1")
        assert len(calls) == 2
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["session", "workspace", "payload"])
async def test_context_presentation_rejects_changed_owner_after_worker_read(
    tmp_path, monkeypatch, mutation
):
    database, store, controller, _ = _controller(tmp_path)
    release, entered = threading.Event(), threading.Event()
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original = store.persistence.get_message_versions

        def held(ids):
            entered.set()
            assert release.wait(5)
            return original(ids)

        monkeypatch.setattr(store.persistence, "get_message_versions", held)
        assert hasattr(
            spend, "ConsoleContextReadSnapshot"
        ), "finite presentation read owner is missing"
        snapshot = spend.ConsoleContextReadSnapshot()
        with snapshot.scope():
            # Scope ownership belongs to this task; a mutation task only changes
            # inputs while this task awaits its own warm/read.
            async def change():
                assert await asyncio.to_thread(entered.wait, 5)
                if mutation == "session":
                    store.create_session(session_id="session-2", title="other")
                    store.active_session_id = "session-2"
                elif mutation == "workspace":
                    store.sessions()[0].workspace_id = "other-workspace"
                else:
                    store.append_message(
                        "session-1",
                        role=ConsoleMessageRole.USER,
                        content="new input",
                        persist=False,
                    )
                release.set()

            mutation_task = asyncio.create_task(change())
            assert await snapshot.warm(controller, "session-1") is False
            await mutation_task
            assert snapshot.value is None
    finally:
        release.set()
        database.close()


def _context_revision_facts(store, controller):
    """Facts that an actual held-send resume must leave unchanged at resolution."""
    owner = next(row for row in store.sessions() if row.id == "session-1")
    return (
        store.active_session_id,
        owner.persisted_conversation_id,
        owner.workspace_id,
        getattr(owner, "active_run_id", None),
        owner.context_policy_overrides,
        store.payload_revision(owner.id),
        store.display_projection_revision(owner.id),
        store.conversation_context_epoch(owner.id),
        store.session_settings_revision(owner.id),
        store.session_context_summary(owner.id),
        controller.run_state_for(owner.id).status,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["status", "run-id"])
async def test_stock_context_lifecycle_only_change_reuses_original_read(
    tmp_path, monkeypatch, change
):
    import time

    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

    database, store, controller, _ = _controller(tmp_path)
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original, calls = store.persistence.get_message_versions, []

        def versions(ids):
            result = original(ids)
            calls.append(tuple(ids))
            return result

        monkeypatch.setattr(store.persistence, "get_message_versions", versions)
        snapshot = spend.ConsoleContextReadSnapshot()
        assert await snapshot.warm(controller, "session-1")
        value = snapshot.inputs(controller, "session-1")
        assert len(calls) == 1 and calls[0]
        assert controller._held_send_echo_id("session-1") is None
        if change == "status":
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING), session_id="session-1"
            )
        else:
            next(
                row for row in store.sessions() if row.id == "session-1"
            ).active_run_id = "new-display-run"
        assert time.monotonic() - snapshot.at < snapshot.max_age
        assert await snapshot.warm(controller, "session-1")
        assert snapshot.inputs(controller, "session-1") == value
        assert len(calls) == 1
    finally:
        database.close()


@asynccontextmanager
async def _real_compaction_resume(tmp_path, monkeypatch):
    """Keep the actual resumed send at its original provider-resolution await."""
    from Tests.Chat.test_console_compaction_ask_hold import _ASK, _send_until_held
    from Tests.Chat.test_console_compaction_live_session import _live_controller
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsoleTurnPreparationState,
    )

    _database, store, controller, gateway = _live_controller(tmp_path, overrides=_ASK)
    _draft, result = await _send_until_held(controller, gateway)
    preparation = store.preparation_for_session("session-1")
    assert (
        preparation is not None and preparation.preparation_id == result.preparation_id
    )
    assert preparation.state is ConsoleTurnPreparationState.PAUSED
    assert preparation.pause_kind is ConsolePreparationPauseKind.CONTEXT_COMPACTION
    echo = controller._held_send_echo_id("session-1")
    assert echo is not None and store.get_message(echo).persisted_message_id is None
    original = gateway.resolve_for_send
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_resolution(selection):
        resolution = await original(selection)
        entered.set()
        await asyncio.wait_for(release.wait(), 5.0)
        return resolution

    monkeypatch.setattr(gateway, "resolve_for_send", held_resolution)
    case = SimpleNamespace(
        store=store,
        controller=controller,
        preparation=preparation,
        echo=echo,
        entered=entered,
        release=release,
        resume=None,
        before=_context_revision_facts(store, controller),
    )
    try:
        yield case
    finally:
        release.set()
        if case.resume is not None:
            await asyncio.wait_for(
                asyncio.gather(case.resume, return_exceptions=True), 5.0
            )
        # The imported live-controller fixture closes its actual database.


async def _resume_to_resolution(case):
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

    case.resume = asyncio.create_task(
        case.controller.send_without_compacting(case.preparation.preparation_id)
    )
    await asyncio.wait_for(case.entered.wait(), 5.0)
    current = case.store.preparation_for_session("session-1")
    assert current.preparation_id == case.preparation.preparation_id
    assert current.state is ConsoleTurnPreparationState.READY
    assert current.transient_user_message_id == case.echo
    assert case.controller._held_send_echo_id("session-1") is None
    assert _context_revision_facts(case.store, case.controller) == case.before
    assert not case.resume.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("in_flight", [False, True])
async def test_actual_compaction_resume_invalidates_context_echo(
    tmp_path, monkeypatch, in_flight
):
    import time

    async with _real_compaction_resume(tmp_path, monkeypatch) as case:
        original = case.store.persistence.get_message_versions
        calls, entered, release = [], threading.Event(), threading.Event()
        read_task = None

        def versions(ids):
            result = original(ids)
            calls.append(tuple(ids))
            if in_flight and len(calls) == 1:
                entered.set()
                assert release.wait(5.0)
            return result

        monkeypatch.setattr(case.store.persistence, "get_message_versions", versions)
        snapshot = spend.ConsoleContextReadSnapshot()
        try:
            if in_flight:
                read_task = asyncio.create_task(
                    snapshot.warm(case.controller, "session-1")
                )
                assert await asyncio.to_thread(entered.wait, 5.0)
                assert calls and calls[0]
                await _resume_to_resolution(case)
                release.set()
                assert await asyncio.wait_for(read_task, 5.0) is False
                assert snapshot.value is None and snapshot.key is None
            else:
                assert await snapshot.warm(case.controller, "session-1")
                assert len(calls) == 1 and calls[0]
                await _resume_to_resolution(case)
                assert time.monotonic() - snapshot.at < snapshot.max_age
            assert await snapshot.warm(case.controller, "session-1")
            assert len(calls) == 2
        finally:
            # Physical version-reading work must return before resume/database teardown.
            release.set()
            if read_task is not None:
                await asyncio.wait_for(
                    asyncio.gather(read_task, return_exceptions=True), 5.0
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", ["reader", "echo-getter"])
@pytest.mark.parametrize("installed", ["before-warm", "after-warm"])
async def test_custom_context_callback_keeps_conservative_lifecycle(
    tmp_path, monkeypatch, custom, installed
):
    import time

    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

    database, store, controller, _ = _controller(tmp_path)
    try:
        store.active_session_id = "session-1"
        store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original, calls = store.persistence.get_message_versions, []
        custom_calls = []

        def versions(ids):
            result = original(ids)
            calls.append(tuple(ids))
            return result

        reader = controller.context_control_presentation_inputs
        getter = controller._held_send_echo_id

        async def custom_reader(*args, **kwargs):
            custom_calls.append("reader")
            return await reader(*args, **kwargs)

        def custom_getter(session_id):
            custom_calls.append("echo-getter")
            return getter(session_id)

        def install():
            monkeypatch.setattr(
                controller,
                "context_control_presentation_inputs"
                if custom == "reader"
                else "_held_send_echo_id",
                custom_reader if custom == "reader" else custom_getter,
            )

        monkeypatch.setattr(store.persistence, "get_message_versions", versions)
        if installed == "before-warm":
            install()
        snapshot = spend.ConsoleContextReadSnapshot()
        assert await snapshot.warm(controller, "session-1")
        assert len(calls) == 1 and calls[0]
        if installed == "after-warm":
            # Source drift alone cannot reuse the preceding stock result.
            install()
        else:
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING), session_id="session-1"
            )
            next(
                row for row in store.sessions() if row.id == "session-1"
            ).active_run_id = "custom-lifecycle-run"
        assert time.monotonic() - snapshot.at < snapshot.max_age
        assert await snapshot.warm(controller, "session-1")
        assert len(calls) == 2
        assert custom_calls
    finally:
        database.close()
