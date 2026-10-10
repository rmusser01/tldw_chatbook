"""Conversation settings waits for its exact finite context owner."""

from Tests.UI.console_controller_stubs import context_spend_for_test

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


def _screen(tmp_path, monkeypatch):
    database, store, controller, _ = _controller(
        tmp_path,
        initial_settings=ConsoleSessionSettings(
            provider="llama_cpp", model="test-model"
        ),
    )
    store.active_session_id = "session-1"
    store.append_message(
        "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
    )
    screen = ChatScreen.__new__(ChatScreen)
    context_spend_for_test(screen)
    pushed, tasks = [], []

    async def push(modal, **_kwargs):
        pushed.append(modal)

    def schedule(operation, **_kwargs):
        task = asyncio.create_task(operation)
        tasks.append(task)
        return task

    async def models(*_args, **_kwargs):
        return {"llama_cpp": ["test-model"]}

    monkeypatch.setattr(
        ChatScreen, "app", property(lambda _self: SimpleNamespace(push_screen=push))
    )
    screen.app_instance = SimpleNamespace()
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: controller
    screen._ensure_console_provider_gateway = lambda: SimpleNamespace(
        resolve_context_window=lambda _settings: 4096
    )
    screen._context_spend._console_settings_context_estimate_for_session = (
        lambda *_args, **_kwargs: ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    screen._provider_readiness_app_config = lambda: {
        "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}
    }
    screen._global_chat_display_name = lambda: "Ada"
    screen._console_run_active = lambda: False
    screen._console_default_durability_state = lambda: None
    screen._providers_models_for_console_settings = models
    screen.run_worker = schedule

    async def refresh():
        return None

    screen._sync_native_console_chat_ui = refresh
    return database, store, controller, screen, pushed, tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["cold", "changed"])
async def test_settings_open_does_not_freeze_completion_on_pending_context(
    tmp_path, monkeypatch, prior
):
    database, store, controller, screen, pushed, tasks = _screen(tmp_path, monkeypatch)
    try:
        if prior == "changed":
            snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
            assert await snapshot.warm(controller, "session-1")
            store.append_message(
                "session-1",
                role=ConsoleMessageRole.USER,
                content="changed",
                persist=True,
            )
        assert await screen._open_console_settings() is True
        modal = pushed[0]
        assert modal._context_state.busy is False
        assert modal._context_operation_active() is False
        assert modal._can_save is True
    finally:
        if tasks:
            await asyncio.gather(*tasks)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["session", "payload", "binding", "controller", "deleted"]
)
async def test_settings_open_rejects_owner_changed_during_model_read(
    tmp_path, monkeypatch, change
):
    database, store, controller, screen, pushed, tasks = _screen(tmp_path, monkeypatch)

    async def models(*_args, **_kwargs):
        if change == "session":
            store.create_session(session_id="other")
            store.active_session_id = "other"
        elif change == "deleted":
            store.close_session("session-1")
        elif change == "payload":
            store.append_message(
                "session-1", role=ConsoleMessageRole.USER, content="new", persist=False
            )
        elif change == "binding":
            store.sessions()[0].conversation_binding_revision += 1
        else:
            screen._ensure_console_chat_controller = lambda: object()
        await asyncio.sleep(0)
        return {"llama_cpp": ["test-model"]}

    screen._providers_models_for_console_settings = models
    try:
        assert await screen._open_console_settings() is False
        assert pushed == []
    finally:
        if tasks:
            await asyncio.gather(*tasks)
        database.close()


@pytest.mark.asyncio
async def test_cancelled_settings_open_keeps_worker_owned_until_safe_retirement(
    tmp_path, monkeypatch
):
    import threading

    database, store, controller, screen, pushed, tasks = _screen(tmp_path, monkeypatch)
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    original = store.persistence.get_message_versions

    def held(ids):
        result = original(ids)
        entered.set()
        try:
            assert release.wait(10)
            return result
        finally:
            exited.set()

    monkeypatch.setattr(store.persistence, "get_message_versions", held)
    opener = asyncio.create_task(screen._open_console_settings())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        opener.cancel()
        await asyncio.wait({opener}, timeout=0.05)
        assert not opener.done(), "Stock context opener detached a live read"
        assert worker_leases(database)
        assert pushed == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opener
        assert await asyncio.to_thread(exited.wait, 5)
        for _ in range(200):
            if not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(database)
        assert pushed == []
        assert screen._console_context_read_snapshot.value is None
    finally:
        release.set()
        await asyncio.gather(opener, return_exceptions=True)
        if tasks:
            await asyncio.gather(*tasks)
        database.close()


@pytest.mark.asyncio
async def test_cold_context_modal_mounted_completion_controls_are_enabled(
    tmp_path, monkeypatch
):
    from textual.app import App
    from textual.widgets import Button

    database, store, controller, screen, pushed, tasks = _screen(tmp_path, monkeypatch)
    screen._console_default_readiness = None
    try:
        assert await screen._open_console_settings() is True
        modal = pushed[0]
        async with App().run_test(size=(140, 45)) as pilot:
            await pilot.app.push_screen(modal)
            await pilot.pause()
            assert not modal.query_one("#console-settings-save", Button).disabled
            assert not modal.query_one(
                "#console-settings-save-default", Button
            ).disabled
            assert not modal.query_one(
                "#console-settings-make-default", Button
            ).disabled
            assert not modal._context_operation_active()
    finally:
        if tasks:
            await asyncio.gather(*tasks)
        database.close()


@pytest.mark.asyncio
async def test_settings_open_rejects_chat_closed_during_actual_thinking_resolve(
    tmp_path, monkeypatch
):
    database, store, controller, screen, pushed, tasks = _screen(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    original = controller.provider_gateway.resolve_for_send

    async def held(selection):
        entered.set()
        await release.wait()
        return await original(selection)

    monkeypatch.setattr(controller.provider_gateway, "resolve_for_send", held)
    opener = asyncio.create_task(screen._open_console_settings())
    try:
        await entered.wait()
        store.close_session("session-1")
        release.set()
        assert await opener is False
        assert pushed == []
    finally:
        release.set()
        await asyncio.gather(opener, return_exceptions=True)
        if tasks:
            await asyncio.gather(*tasks)
        database.close()
