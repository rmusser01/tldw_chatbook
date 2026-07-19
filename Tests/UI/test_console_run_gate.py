"""Mid-run gate for Console message actions (TASK-232).

With a run streaming, retry/regenerate/continue on an OLDER message used to
spawn a new ``group="console-run"`` exclusive worker, which cancelled the
in-flight run at worker-creation time — before the controller's
``_active_run_rejection`` could reject the newcomer. Neither stream finalized:
run_state stayed STREAMING and the row stayed status "streaming" (the same
stuck-[streaming] face as TASK-228's V1, user-triggered). The screen must
gate on ``run_state.is_send_allowed`` BEFORE spawning, exactly like the
submit path does.
"""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)

ALREADY_RUNNING_COPY = "A Console run is already running."


def _build_screen():
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "test-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "test-model"}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    screen = ChatScreen(app)
    return app, screen


def _seed_messages(screen):
    """Return (failed_assistant, completed_assistant) messages in one session."""
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    pending = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    failed = store.mark_message_failed(pending.id)
    completed = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="done."
    )
    return failed, completed


def _instrument(app, screen):
    """Capture run_worker spawns and notify copy; return (spawned, notices)."""
    spawned: list[dict] = []
    notices: list[str] = []

    def fake_run_worker(work, **kwargs):
        spawned.append(kwargs)
        if asyncio.iscoroutine(work):
            work.close()  # never awaited by the fake
        return SimpleNamespace(cancel=lambda: None)

    screen.run_worker = fake_run_worker
    app.notify = lambda message, **kwargs: notices.append(str(message))
    return spawned, notices


def _action_event(action_id: str, message_id: str):
    return SimpleNamespace(
        button=SimpleNamespace(id=f"console-message-action-{action_id}-{message_id}"),
        stop=lambda: None,
    )


def _start_fake_run(screen) -> None:
    controller = screen._ensure_console_chat_controller()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id,target", [
    ("retry", "failed"),
    ("regenerate", "completed"),
    ("continue", "completed"),
])
async def test_mid_run_action_notifies_instead_of_spawning(action_id, target):
    """While a run is active, the action must NOT spawn a console-run worker
    (which would cancel the in-flight stream) — it must notify and return."""
    app, screen = _build_screen()
    failed, completed = _seed_messages(screen)
    _start_fake_run(screen)
    spawned, notices = _instrument(app, screen)

    message = failed if target == "failed" else completed
    handled = await screen.handle_console_message_action(
        _action_event(action_id, message.id)
    )

    assert handled is True
    assert spawned == [], (
        f"mid-run {action_id} spawned a console-run worker — this cancels the "
        "in-flight stream before the controller gate can reject the action"
    )
    assert any(ALREADY_RUNNING_COPY in note for note in notices)


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id,target", [
    ("retry", "failed"),
    ("regenerate", "completed"),
    ("continue", "completed"),
])
async def test_idle_action_still_spawns_console_run_worker(action_id, target):
    """Regression guard: with no active run, the actions dispatch exactly one
    worker in the console-run group."""
    app, screen = _build_screen()
    failed, completed = _seed_messages(screen)
    spawned, notices = _instrument(app, screen)

    message = failed if target == "failed" else completed
    handled = await screen.handle_console_message_action(
        _action_event(action_id, message.id)
    )

    assert handled is True
    assert len(spawned) == 1
    assert spawned[0].get("group") == "console-run"
    assert spawned[0].get("exclusive") is True
    assert not any(ALREADY_RUNNING_COPY in note for note in notices)
