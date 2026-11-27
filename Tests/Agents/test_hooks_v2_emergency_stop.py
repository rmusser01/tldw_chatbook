"""TASK-33620.1.12: during an emergency stop, hook-v2 actions do not run.

Owner ruling, 2026-11-24: "during a stop, hook-v2 MCP/command actions don't
run; each is recorded as skipped because of the emergency stop."

Every hook delivery -- awaited (``fire_async``, a tool's PreToolUse gate, the
PostToolUse of a call that was running, Stop) and fire-and-forget
(``notify``) -- reaches its executor through ``HookEngine._deliver_current``.
The stop is checked there, after admission and before the command or MCP
executor starts, and a held action becomes a ``HookFailure`` coded
``emergency_stop``: the same record a disabled, cancelled or refused action
gets today. ``HookEngine._failure`` decides whether that failure blocks, so a
PreToolUse gate that cannot run still refuses its tool (fail closed); an
observer's skip blocks nothing.

The stop is the per-test file from Tests/conftest.py's redirect; a command
action proves it ran by writing a marker file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Plugins import platform_support
from tldw_chatbook.Agents.hooks_v2 import parse_event, parse_handlers
from tldw_chatbook import emergency_stop
from tldw_chatbook.emergency_stop import clear_emergency_stop, set_emergency_stop

# The hook child bootstraps the fixture's own private profile.
pytestmark = pytest.mark.bootstrap_profile

STOP_CODE = "emergency_stop"


def _event(name: str, **data) -> object:
    return parse_event(
        {
            "protocol_version": 2,
            "event_id": f"event-{name}",
            "event": name,
            "timestamp": "2026-11-24T00:00:00Z",
            "runtime_session_id": "runtime-a",
            "initiator": "manual",
            "origin": "test",
            "causal_chain_id": "chain-1",
            "causal_depth": 0,
            "data": data,
        }
    )


TOOL = {"tool_id": "local:test", "provider": "local"}


def _command(marker: Path, name: str, **kwargs):
    from Tests.hooks_v2_process_support import child_argv

    code = f"open({str(marker)!r},'w').write('ran')"
    return parse_handlers(
        [
            {
                "id": f"command-{name}",
                "event": name,
                "type": "command",
                "effects": [],
                "argv": child_argv(code),
                **kwargs,
            }
        ]
    )[0]


async def _engine(*handlers):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    return HookEngine(tuple(handlers), lambda *_: True, HookBudgetOwner())


def _stop_path() -> Path:
    """The redirected per-test stop file (read through the module, as the
    engine reads it: a name imported at collection is the unpatched one)."""
    return emergency_stop.default_emergency_stop_path()


@pytest.fixture
def stop():
    """Set the (per-test) emergency stop; yields its path."""
    path = _stop_path()
    set_emergency_stop(path, reason="hook hold")
    return path


@platform_support.v2_command
@pytest.mark.asyncio
async def test_a_pre_tool_gate_that_cannot_run_still_refuses_its_tool(tmp_path, stop):
    """The fail-closed pin: a held gate is a blocking failure, never an allow."""
    marker = tmp_path / "gate-ran"
    engine = await _engine(_command(marker, "PreToolUse", effects=["deny"]))
    try:
        result = await engine.fire_async(_event("PreToolUse", **TOOL))
    finally:
        await engine.close()

    assert not marker.exists(), "the hook command ran during the stop"
    [failure] = result.failures
    assert (failure.handler_id, failure.code) == ("command-PreToolUse", STOP_CODE)
    assert failure.blocking is True
    assert result.allowed is False, "a gate that could not run turned into an allow"


@platform_support.v2_command
@pytest.mark.asyncio
async def test_a_required_hook_that_cannot_run_blocks(tmp_path, stop):
    marker = tmp_path / "required-ran"
    engine = await _engine(_command(marker, "PreToolUse", required=True))
    try:
        result = await engine.fire_async(_event("PreToolUse", **TOOL))
    finally:
        await engine.close()

    assert not marker.exists()
    assert [f.code for f in result.failures] == [STOP_CODE]
    assert result.allowed is False


@platform_support.v2_command
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("PostToolUse", TOOL),  # the call that was running when the stop landed
        ("Stop", {"status": "done"}),
        ("SessionEnd", {}),
    ],
)
async def test_an_observer_action_is_skipped_and_recorded(tmp_path, stop, name, data):
    marker = tmp_path / f"{name}-ran"
    engine = await _engine(_command(marker, name))
    try:
        result = await engine.fire_async(_event(name, **data))
    finally:
        await engine.close()

    assert not marker.exists(), f"the {name} hook command ran during the stop"
    [failure] = result.failures
    assert failure.code == STOP_CODE
    # An observer's skip blocks nothing: the run's own outcome is unchanged.
    assert failure.blocking is False and result.allowed is True


@platform_support.v2_command
@pytest.mark.asyncio
async def test_a_fire_and_forget_notification_is_skipped_and_counted(tmp_path, stop):
    marker = tmp_path / "notified-ran"
    engine = await _engine(_command(marker, "PostToolUse"))
    try:
        assert engine.notify(_event("PostToolUse", **TOOL)) is True
        for _ in range(200):
            if engine.notification_failures[STOP_CODE]:
                break
            await asyncio.sleep(0.02)
    finally:
        await engine.close()

    assert not marker.exists(), "the notified hook command ran during the stop"
    assert engine.notification_failures[STOP_CODE] == 1


@pytest.mark.asyncio
async def test_an_mcp_hook_action_is_skipped(stop):
    """The MCP executor is never started; the action is recorded as held."""
    handler = parse_handlers(
        [
            {
                "id": "mcp-post",
                "event": "PostToolUse",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "effects": [],
                "input": {},
            }
        ]
    )[0]
    engine = await _engine(handler)
    started: list[object] = []

    def start(event, *_args, **_kwargs):
        from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome

        started.append(event)
        future = asyncio.get_running_loop().create_future()
        future.set_result(HookEventOutcome())
        return SimpleNamespace(task=future, stop=lambda: None)

    async def close(**_kwargs):
        return None

    engine.mcp_executor = SimpleNamespace(
        start=start,
        _context=lambda _event: None,
        retire_teardown=lambda *_args, **_kwargs: None,
        cleanup_pending=False,
        close=close,
    )
    try:
        result = await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()

    assert started == [], "the MCP hook action ran during the stop"
    assert [f.code for f in result.failures] == [STOP_CODE]


@platform_support.v2_command
@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["directory", "corrupt"])
async def test_an_unreadable_stop_holds_hook_actions(tmp_path, damage):
    path = _stop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if damage == "directory":
        path.mkdir()
    else:
        path.write_text("{ not json", encoding="utf-8")
    marker = tmp_path / "ran"
    engine = await _engine(_command(marker, "PostToolUse"))
    try:
        result = await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()

    assert not marker.exists()
    assert [f.code for f in result.failures] == [STOP_CODE]


@platform_support.v2_command
@pytest.mark.asyncio
@pytest.mark.emergency_stop_unavailable
async def test_a_stop_that_cannot_be_located_holds_hook_actions(tmp_path, monkeypatch):
    def refuse():
        raise PermissionError("data dir refused")

    # Resolve the child's argv first: it reads the profile itself.
    handler = _command(tmp_path / "ran", "PostToolUse")
    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    engine = await _engine(handler)
    try:
        result = await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()

    assert not (tmp_path / "ran").exists()
    assert [f.code for f in result.failures] == [STOP_CODE]


@platform_support.v2_command
@pytest.mark.asyncio
@pytest.mark.emergency_stop_unavailable
async def test_a_stop_read_that_fails_holds_as_unavailable_and_says_why(
    tmp_path, monkeypatch
):
    """The read runs on a worker thread, which fails after executor shutdown.

    The hold goes through the fail-safe's one state (so the suite sees it),
    and its log line names the failed read, not an active stop.
    """
    from loguru import logger

    def no_worker():
        raise RuntimeError("cannot schedule new futures after shutdown")

    handler = _command(tmp_path / "ran", "PostToolUse")
    unavailable = emergency_stop.stop_state_unavailable
    holds: list[object] = []
    monkeypatch.setattr(
        emergency_stop,
        "stop_state_unavailable",
        lambda: holds.append(1) or unavailable(),
    )
    # to_thread re-raises what its worker raised.
    monkeypatch.setattr(emergency_stop, "current_emergency_stop", no_worker)
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    engine = await _engine(handler)
    try:
        result = await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()
        logger.remove(sink)

    assert not (tmp_path / "ran").exists()
    assert [f.code for f in result.failures] == [STOP_CODE]
    assert holds == [1], "the hold bypassed the fail-safe's state"
    failed = [line for line in lines if "emergency-stop read failed" in line]
    assert len(failed) == 1 and "error_type=RuntimeError" in failed[0]


@platform_support.v2_command
@pytest.mark.asyncio
async def test_hook_actions_run_again_once_the_stop_is_cleared(tmp_path):
    path = _stop_path()
    set_emergency_stop(path, reason="brief")
    clear_emergency_stop(path)
    marker = tmp_path / "ran"
    engine = await _engine(_command(marker, "PostToolUse"))
    try:
        result = await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()

    assert marker.exists() and result.succeeded


@platform_support.v2_command
@pytest.mark.asyncio
async def test_a_held_action_logs_its_event_and_handler_never_its_command(
    tmp_path, stop
):
    """hooks_v2 keeps no log of its own; a held action leaves one line."""
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    marker = tmp_path / "logged-ran"
    handler = _command(marker, "PostToolUse")
    engine = await _engine(handler)
    try:
        await engine.fire_async(_event("PostToolUse", **TOOL))
    finally:
        await engine.close()
        logger.remove(sink)

    held = [line for line in lines if "held by the emergency stop" in line]
    assert len(held) == 1
    assert "event=PostToolUse" in held[0] and "handler=command-PostToolUse" in held[0]
    assert str(marker) not in held[0] and handler.argv[0] not in held[0]
