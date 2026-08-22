"""Structured, session-only Console activity presentation contracts."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from tldw_chatbook.Agents.agent_models import (
    STEP_ERROR,
    STEP_SPAWN,
    STEP_TOOL_RESULT,
    ToolResult,
)
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
    build_intermediate_thinking_marker,
    build_step_activity_presentation,
    classify_activity_status,
    safe_intermediate_thinking_summary,
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
from tldw_chatbook.Chat.console_turn_grouping import visual_messages


class _RecordingPersistence:
    """Small persistence fake that records the durable message payload."""

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    def create_conversation(self, **_kwargs: object) -> str:
        return "conv-1"

    def create_message(self, **kwargs: object) -> str:
        self.created.append(kwargs)
        return f"msg-{len(self.created)}"


@pytest.mark.parametrize(
    "public_callable",
    [
        safe_intermediate_thinking_summary,
        build_intermediate_thinking_marker,
        visual_messages,
        ToolResult.blocked,
    ],
)
def test_new_public_activity_helpers_use_google_style_docstrings(
    public_callable,
) -> None:
    docstring = inspect.getdoc(public_callable) or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


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


def test_restore_state_clears_incoming_session_only_activity_presentation() -> None:
    source = ConsoleChatStore()
    session = source.ensure_session()
    presentation = ConsoleActivityPresentation("activity", "Working", "done")
    incoming = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        activity_presentation=presentation,
    )
    restored_store = ConsoleChatStore()

    restored_store.restore_state(
        sessions=[session],
        messages_by_session={session.id: [incoming]},
        active_session_id=session.id,
    )

    restored = restored_store.messages_for_session(session.id)
    assert incoming.activity_presentation == presentation
    assert restored[0].activity_presentation is None


def test_persisted_restore_clears_incoming_session_only_activity_presentation() -> None:
    presentation = ConsoleActivityPresentation("tool", "fs_list", "success")
    incoming = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        persisted_message_id="message-1",
        activity_presentation=presentation,
    )
    restored_store = ConsoleChatStore()

    session = restored_store.restore_persisted_session(
        title="Restored",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[incoming],
        active_leaf_persisted_id="message-1",
    )

    restored = restored_store.messages_for_session(session.id)
    assert incoming.activity_presentation == presentation
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
    assert classify_activity_status(STEP_TOOL_RESULT, f"ERROR: {refusal}") == "blocked"


def test_unknown_error_wrapped_tool_failure_is_failed() -> None:
    assert (
        classify_activity_status(STEP_TOOL_RESULT, "ERROR: disk exploded") == "failed"
    )


@pytest.mark.parametrize(
    "collision",
    [
        "ERROR: harmless successful payload",
        CONTROLLER_USER_DENIED_REFUSAL.format(name="fs_list"),
    ],
)
def test_structured_success_outcome_overrides_payload_collision(collision: str) -> None:
    assert (
        classify_activity_status(
            STEP_TOOL_RESULT,
            collision,
            tool_outcome="success",
        )
        == "success"
    )


def test_legacy_step_without_structured_outcome_keeps_safe_fallback() -> None:
    refusal = f"ERROR: {LOCAL_DENY_REFUSAL}"

    assert classify_activity_status(STEP_TOOL_RESULT, refusal) == "blocked"
    assert (
        classify_activity_status(STEP_TOOL_RESULT, "ERROR: disk exploded") == "failed"
    )


def test_malformed_persisted_outcome_falls_back_without_raising() -> None:
    assert (
        classify_activity_status(
            STEP_TOOL_RESULT,
            f"ERROR: {LOCAL_DENY_REFUSAL}",
            tool_outcome="unknown",  # type: ignore[arg-type]
        )
        == "blocked"
    )


def test_safe_intermediate_thinking_summary_retains_only_safe_prefence_preamble(
    monkeypatch,
) -> None:
    from tldw_chatbook.config import MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS

    monkeypatch.setenv(
        "TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS",
        str(MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS),
    )
    preamble = "I will inspect the relevant files before choosing the smallest fix."
    summary = (
        f"{preamble}\n```tool_call\n"
        '{"name":"fs_read","arguments":{"path":"secret"}}\n```'
    )

    safe = bridge_module.safe_intermediate_thinking_summary(summary)

    assert safe == bridge_module._truncate_step_text(
        preamble,
        limit=MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
    )
    assert "secret" not in safe


@pytest.mark.parametrize(
    "summary",
    [
        "<thinking>private chain</thinking>",
        "<ANALYSIS>private chain</ANALYSIS>",
        "</reasoning>",
        "<reasoning_content>private chain</reasoning_content>",
        "[analysis] private chain [/analysis]",
        "BEGIN THINKING\nprivate chain\nEND THINKING",
        "Safe-looking text\n```analysis\nprivate chain\n```",
        "<|channel|>analysis private chain",
        "<|reasoning|>private chain",
        'Safe-looking text {"tool_call": {"name": "fs_read"}}',
        'Safe-looking text {"tool_calls": []}',
        'Safe-looking text {"function_call": {}}',
        'Safe-looking text {"arguments": {"path": "private"}}',
        '```json\n{"name":"fs_read","arguments":{"path":"private"}}\n```',
        "",
        " \n\t ",
        None,
    ],
)
def test_safe_intermediate_thinking_summary_rejects_private_or_payload_shapes(
    summary: str | None,
) -> None:
    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


def test_safe_intermediate_thinking_summary_rejects_non_line_controls() -> None:
    summary = "Inspect\nthese\tfiles\x00before\x1b continuing."

    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize(
    "control",
    ["\x00", "\x01", "\x1b", "\x1f", "\x7f", "\x80", "\x9f"],
    ids=["nul", "soh", "esc", "unit-separator", "del", "c1-start", "c1-end"],
)
@pytest.mark.parametrize(
    "template",
    [
        "{control}Thinking: PRIVATE",
        "Safe preamble\n{control}Analysis: PRIVATE",
        "Reason{control}ing: PRIVATE",
    ],
    ids=["prefix", "after-boundary", "inside-header"],
)
def test_safe_intermediate_thinking_summary_rejects_c0_c1_header_evasion(
    control: str,
    template: str,
) -> None:
    summary = template.format(control=control)

    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize(
    "separator",
    ["\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\u2029"],
    ids=["cr", "crlf", "vt", "ff", "nel", "line-separator", "paragraph-separator"],
)
@pytest.mark.parametrize("header", ["Thinking", "Analysis", "Reasoning"])
def test_safe_intermediate_thinking_summary_rejects_private_headers_after_any_splitline(
    separator: str,
    header: str,
) -> None:
    summary = f"Safe preamble{separator}{header}: PRIVATE"

    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize(
    "summary",
    [
        '<tool_call>{"name":"fs_read","parameters":{"path":"PRIVATE"}}</tool_call>',
        '<FUNCTION_CALL>{"name":"fs_read","parameters":{}}</FUNCTION_CALL>',
        '[tool_call] {"name":"fs_read","parameters":{}} [/tool_call]',
        'Calling function fs_read({"path":"PRIVATE"})',
        'invoking tool fs_write with {"path":"PRIVATE"}',
        'Calling tool fs_read with arguments {"path":"PRIVATE"}',
        'Invoking function fs_read with args {"path":"PRIVATE"}',
        '<tool_use>{"input":{"path":"PRIVATE"}}</tool_use>',
        '[tool_use] {"input":{"path":"PRIVATE"}} [/tool_use]',
    ],
)
def test_safe_intermediate_thinking_summary_rejects_explicit_call_shapes(
    summary: str,
) -> None:
    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r", "\r\n", "\u2028", "\u2029"],
    ids=["lf", "cr", "crlf", "line-separator", "paragraph-separator"],
)
@pytest.mark.parametrize(
    "signal",
    [
        "Invoking function fs_read with args",
        "Calling tool fs_read with arguments",
    ],
    ids=["invoking-function", "calling-tool"],
)
def test_safe_intermediate_thinking_summary_rejects_multiline_call_payloads(
    signal: str,
    separator: str,
) -> None:
    summary = f'{signal}{separator}{{"path":"PRIVATE"}}'

    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize("gap_kind", ["over-old-limit", "near-display-cap"])
def test_safe_intermediate_thinking_summary_rejects_long_call_payload_gaps(
    monkeypatch,
    gap_kind: str,
) -> None:
    from tldw_chatbook.config import MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS

    cap = MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS
    monkeypatch.setenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", str(cap))
    signal = "Calling tool fs_read with arguments "
    payload = '{"path":"PRIVATE"}'
    gap = 81 if gap_kind == "over-old-limit" else cap - len(signal) - len(payload) - 1
    summary = f"{signal}{'x' * gap}{payload}"

    assert gap > 80
    assert len(summary) < cap
    assert bridge_module.safe_intermediate_thinking_summary(summary) is None


@pytest.mark.parametrize(
    "summary",
    [
        "Calling attention to the tool selection without a payload object.",
        "Invoking a function can be useful after validation.",
        "Calling the tool fs_read later remains a possible option.",
    ],
)
def test_safe_intermediate_thinking_summary_allows_call_prose_without_object(
    summary: str,
) -> None:
    assert bridge_module.safe_intermediate_thinking_summary(summary) == summary


def test_safe_intermediate_thinking_summary_allows_non_call_shaped_tool_prose() -> None:
    summary = "I will use the fs_read tool after checking the path."

    assert bridge_module.safe_intermediate_thinking_summary(summary) == summary


def test_thinking_marker_without_safe_summary_has_no_expandable_detail() -> None:
    marker = bridge_module.build_intermediate_thinking_marker(
        "<thinking>private chain</thinking>"
    )

    assert marker.role is ConsoleMessageRole.TOOL
    assert marker.content == ""
    assert marker.tool_output_full is None
    assert marker.activity_presentation == ConsoleActivityPresentation(
        "thinking", "Thinking", "done"
    )


def test_thinking_marker_with_safe_summary_uses_bounded_content_as_its_detail() -> None:
    marker = bridge_module.build_intermediate_thinking_marker(
        "I will inspect the relevant files."
    )

    assert marker.content == "I will inspect the relevant files."
    assert marker.tool_output_full is None
    assert marker.activity_presentation == ConsoleActivityPresentation(
        "thinking", "Thinking", "done"
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
