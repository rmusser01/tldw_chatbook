"""Structured, session-only Console activity presentation contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Agents.agent_models import STEP_ERROR, STEP_SPAWN, STEP_TOOL_RESULT
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_ROOT_CHANGED_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
)
from tldw_chatbook.Agents.mcp_tool_provider import (
    DENY_REFUSAL as MCP_DENY_REFUSAL,
    KILL_SWITCH_REFUSAL as MCP_KILL_SWITCH_REFUSAL,
    TIMEOUT_REFUSAL as MCP_TIMEOUT_REFUSAL,
    UNRESOLVED_REFUSAL as MCP_UNRESOLVED_REFUSAL,
    USER_DENY_REFUSAL as MCP_USER_DENY_REFUSAL,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    STEP_APPROVAL_TIMEOUT,
    build_step_activity_presentation,
    classify_activity_status,
)
from tldw_chatbook.Chat.console_chat_controller import (
    KILL_SWITCH_REFUSAL as CONTROLLER_KILL_SWITCH_REFUSAL,
    USER_DENIED_REFUSAL as CONTROLLER_USER_DENIED_REFUSAL,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _RecordingPersistence:
    """Small persistence fake that records the durable message payload."""

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    def create_conversation(self, **_kwargs: object) -> str:
        return "conv-1"

    def create_message(self, **kwargs: object) -> str:
        self.created.append(kwargs)
        return f"msg-{len(self.created)}"


def test_activity_presentation_accepts_only_the_bounded_contract() -> None:
    presentation = ConsoleActivityPresentation("tool", "fs_list", "success")

    assert presentation.kind == "tool"
    assert presentation.label == "fs_list"
    assert presentation.status == "success"
    with pytest.raises(FrozenInstanceError):
        presentation.label = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kind", "label", "status"),
    [
        ("unknown", "Label", "done"),
        ("tool", "", "success"),
        ("tool", "   ", "success"),
        ("tool", "two\nlines", "success"),
        ("tool", "carriage\rreturn", "success"),
        ("tool", "x" * 201, "success"),
        ("tool", "Label", "running"),
    ],
)
def test_activity_presentation_rejects_invalid_values(
    kind: str, label: str, status: str
) -> None:
    with pytest.raises(ValueError):
        ConsoleActivityPresentation(kind, label, status)  # type: ignore[arg-type]


def test_activity_presentation_is_session_only_on_tool_markers() -> None:
    persistence = _RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    presentation = ConsoleActivityPresentation("tool", "fs_list", "success")

    marker = store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content="⚙ fs_list → src/",
        persist=True,
        activity_presentation=presentation,
    )

    assert marker.activity_presentation == presentation
    assert marker.persisted_message_id is None
    assert persistence.created == []


def test_activity_presentation_never_enters_persistence_or_restore_payload() -> None:
    persistence = _RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    presentation = ConsoleActivityPresentation("activity", "Working", "done")

    stored = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        persist=True,
        activity_presentation=presentation,
    )

    assert stored.activity_presentation == presentation
    assert "activity_presentation" not in persistence.created[-1]

    durable = persistence.created[-1]
    restored_node = ConsoleChatMessage(
        role=ConsoleMessageRole(str(durable["sender"])),
        content=str(durable["content"]),
        persisted_message_id=stored.persisted_message_id,
    )
    restored_store = ConsoleChatStore()
    restored_session = restored_store.restore_persisted_session(
        title="Restored",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[restored_node],
    )

    restored = restored_store.messages_for_session(restored_session.id)
    assert restored[0].activity_presentation is None


def test_legacy_tool_message_can_omit_activity_presentation() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="legacy marker",
    )

    assert message.activity_presentation is None


@pytest.mark.parametrize(
    ("kind", "result", "expected"),
    [
        (STEP_TOOL_RESULT, "42", "success"),
        (STEP_TOOL_RESULT, "", "success"),
        (STEP_SPAWN, "spawned", "done"),
        (STEP_ERROR, "boom", "failed"),
        (STEP_APPROVAL_TIMEOUT, None, "blocked"),
    ],
)
def test_activity_status_classifies_success_error_timeout_and_non_tool_steps(
    kind: str, result: object, expected: str
) -> None:
    assert classify_activity_status(kind, result) == expected


@pytest.mark.parametrize(
    "verdict",
    [
        CONTROLLER_USER_DENIED_REFUSAL.format(name="fs_list"),
        CONTROLLER_KILL_SWITCH_REFUSAL,
    ],
)
def test_direct_controller_review_results_are_blocked(verdict: str) -> None:
    assert classify_activity_status(STEP_TOOL_RESULT, verdict) == "blocked"


@pytest.mark.parametrize(
    "refusal",
    [
        # Builtin gate copy: exact kill-switch text, plus pinned prefixes
        # whose provider-owned suffix is the runtime tool name.
        "tool execution is disabled by the kill switch",
        "tool is set to Off: calculator",
        "tool call denied by the user: calculator",
        "tool requires approval and none was granted: calculator",
        LOCAL_DENY_REFUSAL,
        LOCAL_TIMEOUT_REFUSAL,
        LOCAL_KILL_SWITCH_REFUSAL,
        LOCAL_GATE_ERROR_REFUSAL,
        LOCAL_ROOT_CHANGED_REFUSAL,
        MCP_DENY_REFUSAL,
        MCP_USER_DENY_REFUSAL,
        MCP_UNRESOLVED_REFUSAL,
        MCP_TIMEOUT_REFUSAL,
        MCP_KILL_SWITCH_REFUSAL,
    ],
)
def test_error_wrapped_provider_refusals_are_blocked(refusal: str) -> None:
    assert (
        classify_activity_status(STEP_TOOL_RESULT, f"ERROR: {refusal}")
        == "blocked"
    )


def test_unknown_error_wrapped_tool_failure_is_failed() -> None:
    assert (
        classify_activity_status(STEP_TOOL_RESULT, "ERROR: disk exploded")
        == "failed"
    )


@pytest.mark.parametrize(
    ("kind", "tool_name", "result", "expected"),
    [
        (STEP_TOOL_RESULT, "calculator", "42", ("tool", "calculator", "success")),
        (STEP_SPAWN, None, None, ("spawn", "Sub-agent", "done")),
        (STEP_ERROR, None, None, ("warning", "Error", "failed")),
        (
            STEP_APPROVAL_TIMEOUT,
            "fs_write",
            None,
            ("warning", "fs_write", "blocked"),
        ),
    ],
)
def test_step_activity_builder_uses_structured_step_facts(
    kind: str,
    tool_name: str | None,
    result: object,
    expected: tuple[str, str, str],
) -> None:
    presentation = build_step_activity_presentation(
        kind, tool_name=tool_name, result=result
    )

    assert (presentation.kind, presentation.label, presentation.status) == expected


def test_step_activity_builder_sanitizes_and_bounds_untrusted_labels() -> None:
    presentation = build_step_activity_presentation(
        STEP_TOOL_RESULT,
        tool_name="unsafe\n\x00" + "x" * 300,
        result="ok",
    )

    assert "\n" not in presentation.label
    assert "\x00" not in presentation.label
    assert len(presentation.label) == 200
