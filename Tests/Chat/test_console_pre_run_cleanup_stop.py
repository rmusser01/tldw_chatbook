"""A turn stopped before its run by an unproven cleanup (TASK-41307).

Drives the real accepted-turn failure handler: the agent reply's setup (MCP
capture, in the Console) is refused because an earlier read on the turn's
owner retained its cleanup.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Agents.execution_capacity import (
    CleanupUnprovenRefused,
    RuntimeCapacity,
    WorkOrigin,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus

pytestmark = pytest.mark.bootstrap_profile

_STOP = (
    "Agent stopped: the cleanup of a remote SSH read could not be proven, so "
    "its workspace and storage stay held until you restart the app."
)


def _refused_setup():
    """The turn owner's next reservation after a retained startup read."""
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.reserve_tool().retain_unproven("a remote SSH read")
    try:
        owner.reserve_tool()
    except CleanupUnprovenRefused as refused:
        return owner, refused
    raise AssertionError("the retained owner admitted another tool")


@pytest.mark.asyncio
async def test_the_stop_names_the_cleanup_instead_of_recovery_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _db, _store, controller, _gateway = _controller(tmp_path)
    owner, refused = _refused_setup()

    async def stream(**_kwargs):
        raise refused

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    result = await controller.submit_draft("go", session_id="session-1")

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert controller.run_state_for("session-1").visible_copy == _STOP
    reply = result.assistant_message_id
    assert controller.run_terminal_rows[reply] == {owner.execution_id}
    # The restore consumed the copy; nothing is left for a later failure.
    assert controller._preflight_block_copies == {}


@pytest.mark.asyncio
async def test_a_closed_chat_leaves_no_stop_copy_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An early return skips the restore, so it must skip the copy too."""
    _db, store, controller, _gateway = _controller(tmp_path)
    _owner, refused = _refused_setup()

    async def stream(**_kwargs):
        store.close_session("session-1")
        raise refused

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    result = await controller.submit_draft("go", session_id="session-1")

    assert result.visible_copy == "Session closed."
    assert controller._preflight_block_copies == {}


@pytest.mark.asyncio
async def test_an_unrelated_setup_failure_keeps_the_recovery_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _db, _store, controller, _gateway = _controller(tmp_path)

    async def stream(**_kwargs):
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    result = await controller.submit_draft("go", session_id="session-1")

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert controller.run_state_for("session-1").visible_copy == (
        "Response recovery failed. Try again or discard."
    )
    assert controller.run_terminal_rows == {}


@pytest.mark.asyncio
async def test_a_failed_recovery_mark_leaves_no_stop_copy_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later, unrelated failure after "Try again" must not inherit the copy."""
    _db, store, controller, _gateway = _controller(tmp_path)
    _owner, refused = _refused_setup()

    async def stream(**_kwargs):
        raise refused

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("recovery mark unavailable")

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    monkeypatch.setattr(store, "mark_dispatch_recovery_needed", unavailable)
    with contextlib.suppress(Exception):
        await controller.submit_draft("go", session_id="session-1")

    assert controller._preflight_block_copies == {}

