"""TASK-33620.1.12 / .18: during an emergency stop, Console run hooks don't run.

Owner ruling, 2026-11-24 (an agent run's hooks: MCP actions and shell
commands are held), extended to the Console's documented ``[[hooks.hook]]``
run hooks (``Agents/run_hooks.py``'s ``RunHooksEngine``).

Every run-hook command reaches its process through ``_run_hook``, on the
engine's pool threads (awaited fires and notifications alike), so the stop is
read there, before launch. A held hook keeps the engine's existing meaning of
a hook that could not be launched (a refused launch, ``HookLaunchRefused``):
a blocking event (PreToolUse, UserPromptSubmit) is denied with the stop's
copy; every other event is skipped. So a held PreToolUse guard still refuses
its tool, and a held UserPromptSubmit hook blocks its prompt rather than
failing open. Each hold logs one line naming the event and the hook's
fingerprint, never its command.

The stop is the per-test file from Tests/conftest.py's redirect, read through
the module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from loguru import logger

from Tests.Agents.hook_test_utils import trusted_hook_engine
from tldw_chatbook import emergency_stop
from tldw_chatbook.Agents.run_hooks import RunHooksEngine, load_hooks_config
from tldw_chatbook.emergency_stop import clear_emergency_stop, set_emergency_stop

HELD = "Emergency stop is active"


def _stop_path() -> Path:
    return emergency_stop.default_emergency_stop_path()


def _engine(tmp_path: Path, event: str, marker: Path, *, code: str | None = None):
    command = [
        sys.executable,
        "-c",
        code or f"open({str(marker)!r},'w').write('ran')",
    ]
    config = load_hooks_config(
        {"hooks": {"hook": [{"event": event, "command": command}]}}
    )
    return trusted_hook_engine(
        config_provider=lambda: config, cwd_provider=lambda: str(tmp_path)
    )


@pytest.fixture(autouse=True)
def close_engines(monkeypatch):
    engines: list[RunHooksEngine] = []
    original = RunHooksEngine.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        engines.append(self)

    monkeypatch.setattr(RunHooksEngine, "__init__", initialize)
    yield
    for engine in engines:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def logs():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    yield lines
    logger.remove(sink)


@pytest.fixture
def stop():
    set_emergency_stop(_stop_path(), reason="run hooks held")


TOOL = {"tool_name": "fs_read", "tool_args": {"path": "notes.txt"}}


def test_a_held_pre_tool_guard_still_refuses_its_tool(tmp_path, stop, logs):
    """The fail-closed pin: a PreToolUse guard that cannot run denies."""
    marker = tmp_path / "guard-ran"
    engine = _engine(tmp_path, "PreToolUse", marker)

    outcome = engine.fire("PreToolUse", session_id="s", data=TOOL)

    assert not marker.exists(), "the run hook ran during the stop"
    assert outcome.blocked is True
    assert outcome.reason.startswith(HELD)
    held = [line for line in logs if "held by the emergency stop" in line]
    assert len(held) == 1 and "event=PreToolUse" in held[0]
    assert sys.executable not in held[0] and "-c" not in held[0]


def test_a_held_user_prompt_hook_blocks_rather_than_failing_open(tmp_path, stop):
    """UserPromptSubmit fails open on an execution failure, never on a hold."""
    marker = tmp_path / "prompt-ran"
    engine = _engine(tmp_path, "UserPromptSubmit", marker)

    outcome = engine.fire(
        "UserPromptSubmit", session_id="s", data={"prompt": "hello"}
    )

    assert not marker.exists()
    assert outcome.blocked is True, "a held prompt hook let the prompt through"
    assert outcome.reason.startswith(HELD)


@pytest.mark.parametrize(
    ("event", "data"),
    [
        ("PostToolUse", {**TOOL, "tool_result": "ok", "is_error": False}),
        ("Stop", {"status": "completed"}),
        ("ApprovalRequested", {"calls": [], "session_active": True}),
        ("SubagentStop", {"child_run_id": "child", "status": "done"}),
    ],
)
def test_a_held_observer_hook_is_skipped(tmp_path, stop, logs, event, data):
    marker = tmp_path / f"{event}-ran"
    engine = _engine(tmp_path, event, marker)

    outcome = engine.fire(event, session_id="s", data=data)

    assert not marker.exists(), f"the {event} run hook ran during the stop"
    assert outcome.blocked is False  # an observer's skip blocks nothing
    assert any(
        f"event={event}" in line and "held by the emergency stop" in line
        for line in logs
    )


def test_a_held_notification_is_skipped(tmp_path, stop):
    """The fire-and-forget path reaches the same spawn point."""
    marker = tmp_path / "notified-ran"
    engine = _engine(tmp_path, "Stop", marker)

    engine.notify("Stop", session_id="s", data={"status": "completed"})
    engine._notify_worker.shutdown(wait=True)
    engine._notification_pool.shutdown(wait=True)

    assert not marker.exists(), "the notified run hook ran during the stop"


@pytest.mark.parametrize("damage", ["directory", "corrupt"])
def test_an_unreadable_stop_holds_run_hooks(tmp_path, damage):
    path = _stop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if damage == "directory":
        path.mkdir()
    else:
        path.write_text("{ not json", encoding="utf-8")
    marker = tmp_path / "ran"
    engine = _engine(tmp_path, "PreToolUse", marker)

    outcome = engine.fire("PreToolUse", session_id="s", data=TOOL)

    assert not marker.exists()
    assert outcome.blocked is True


@pytest.mark.emergency_stop_unavailable
def test_a_stop_that_cannot_be_located_holds_run_hooks(tmp_path, monkeypatch):
    def refuse():
        raise PermissionError("data dir refused")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    marker = tmp_path / "ran"
    engine = _engine(tmp_path, "PostToolUse", marker)

    engine.fire("PostToolUse", session_id="s", data={**TOOL, "tool_result": "ok"})

    assert not marker.exists()


@pytest.mark.emergency_stop_unavailable
def test_an_unlocatable_stop_denies_without_offering_clear(tmp_path, monkeypatch):
    """Clearing writes to the same unreachable folder; the denial says so."""

    def refuse():
        raise PermissionError("data dir refused")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    marker = tmp_path / "ran"
    engine = _engine(tmp_path, "PreToolUse", marker)

    outcome = engine.fire("PreToolUse", session_id="s", data=TOOL)

    assert not marker.exists() and outcome.blocked is True
    assert outcome.reason == emergency_stop.STATE_UNAVAILABLE_COPY


def test_run_hooks_run_again_once_the_stop_is_cleared(tmp_path):
    set_emergency_stop(_stop_path(), reason="brief")
    clear_emergency_stop(_stop_path())
    marker = tmp_path / "ran"
    engine = _engine(tmp_path, "PreToolUse", marker)

    outcome = engine.fire("PreToolUse", session_id="s", data=TOOL)

    assert marker.exists() and outcome.blocked is False


# --- UserPromptSubmit cannot fire during a stop, even for a send admitted first --


# The controller's hook admission reads the profile's hook permissions.
@pytest.mark.bootstrap_profile
async def test_a_send_admitted_before_the_stop_is_refused_before_its_prompt_hook(
    tmp_path,
):
    """The Console gate checks the stop before a send; the send then awaits
    preparation before its UserPromptSubmit hooks fire. A stop landing in that
    window used to let the hook run (and, failing open, let the prompt reach
    the provider). The turn is now refused there with the stop's copy, before
    any prompt hook or provider call."""
    from Tests.Chat.test_console_chat_controller import (
        ConsoleChatController,
        ConsoleChatStore,
        RecordingStreamingGateway,
    )

    marker = tmp_path / "prompt-ran"
    engine = _engine(tmp_path, "UserPromptSubmit", marker)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, ensure_run_hooks=lambda: engine
    )
    # The send passed the Console's gate; the stop lands before its hooks.
    set_emergency_stop(_stop_path(), reason="landed after the gate")

    result = await controller.submit_draft("hello")

    assert not marker.exists(), "UserPromptSubmit fired during the stop"
    assert gateway.messages_seen is None, "the prompt reached the provider"
    assert result.accepted is False and result.should_clear_draft is False
    assert result.visible_copy.startswith(HELD)
