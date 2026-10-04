"""Presentation reads use captured lineage and leave send authority unchanged."""

import asyncio
import threading

import pytest

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
