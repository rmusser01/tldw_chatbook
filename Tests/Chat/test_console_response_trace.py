"""Complete, displayable Console response traces for clipboard review."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_turn_grouping import ConsoleAssistantTurn
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    ThinkingEnvelopeValidationError,
)


def thinking(text: str, ordinal: int = 0) -> DisplayableThinkingBlock:
    return DisplayableThinkingBlock(
        block_id=f"thinking-{ordinal}",
        round_ordinal=ordinal,
        provider="local",
        model="test",
        protocol="openai_chat",
        source_format="summary",
        status="complete",
        text=text,
    )


def tool(identifier: str, result: str, ordinal: int = 0) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        ConsoleMessageRole.TOOL,
        "preview",
        id=identifier,
        activity_presentation=ConsoleActivityPresentation(
            "tool",
            "fs_read",
            "success",
            call_id=identifier,
            arguments='{"path": "' + identifier + '"}',
            result_preview="preview",
        ),
        activity_round_ordinal=ordinal,
        tool_output_full=result,
    )


def test_response_trace_preserves_all_sections():
    from tldw_chatbook.Chat.console_response_trace import (
        find_response_turn,
        render_response_trace_markdown,
    )

    answer = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "**Final answer**", id="a"
    )
    answer.thinking = ThinkingEnvelope((thinking("first"), thinking("second", 1)))
    messages = [
        answer,
        tool("one", "  full result one  \n"),
        tool("two", "second result", 1),
    ]
    turn = find_response_turn(messages, "two")
    assert turn is not None and turn.assistant is answer
    rendered = render_response_trace_markdown(turn)
    for text in (
        "first",
        "second",
        '{"path": "one"}',
        '{"path": "two"}',
        "  full result one  \n",
        "second result",
        "**Final answer**",
    ):
        assert text in rendered
    assert "preview" not in rendered
    assert (
        rendered.index("first") < rendered.index('"one"') < rendered.index("second\n")
    )
    assert rendered.index('"two"') < rendered.index("## Final answer")


@pytest.mark.parametrize("owned", [False, True])
def test_raw_command_copy_preserves_execution_metadata_and_truncation(owned):
    from tldw_chatbook.Chat.console_chat_models import RawCliPresentation
    from tldw_chatbook.Chat.console_response_trace import render_chat_trace_markdown

    command = tool("command", "stdout: full output\nstderr: retained error")
    command.raw_cli_presentation = RawCliPresentation(
        command="echo command",
        cwd="/work",
        shell="sh",
        caller="model" if owned else "user",
        invocation_id="invocation",
        lifecycle_state="exited",
        exit_code=0,
        started_at_monotonic=None,
        elapsed_seconds=1.5,
        truncated=True,
        cleanup_proven=True,
    )
    messages = (
        [ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer"), command]
        if owned
        else [command]
    )
    rendered = render_chat_trace_markdown(messages, title="Commands")
    for fact in (
        "echo command",
        "/work",
        "sh",
        "0",
        "1.5",
        "Truncated: Yes",
        "Cleanup: Proven",
        command.tool_output_full,
    ):
        assert fact in rendered
    assert "preview" not in rendered
    assert command.activity_presentation.arguments in rendered


def test_chat_trace_contains_only_supplied_active_branch():
    from tldw_chatbook.Chat.console_response_trace import render_chat_trace_markdown

    messages = [
        ConsoleChatMessage(ConsoleMessageRole.SYSTEM, "PRIVATE_SYSTEM_SENTINEL"),
        ConsoleChatMessage(ConsoleMessageRole.USER, "earlier user message"),
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "earlier answer"),
        tool("old", "old full result"),
        ConsoleChatMessage(ConsoleMessageRole.USER, "latest question"),
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "latest answer"),
    ]
    rendered = render_chat_trace_markdown(messages, title="Review <chat>")
    assert rendered is not None
    assert "PRIVATE_SYSTEM_SENTINEL" not in rendered
    assert rendered.index("earlier user message") < rendered.index("old full result")
    assert rendered.index("earlier answer") < rendered.index("latest question")
    assert rendered.count("old full result") == 1
    assert "latest answer" in rendered
    assert render_chat_trace_markdown([], title="empty") is None


def test_same_round_thinking_is_rejected_by_existing_envelope_contract():
    with pytest.raises(ThinkingEnvelopeValidationError, match="block ordering"):
        ThinkingEnvelope(
            (thinking("first"), replace(thinking("second"), block_id="other"))
        )


def test_embedded_fences_unicode_and_whitespace_survive():
    from tldw_chatbook.Chat.console_response_trace import render_response_trace_markdown

    result = "  café 漢字\n```python\nprint('hi')\n```\n`````\n  tail  "
    activity = tool("read", result)
    activity.tool_diff = ("example.py", "before\n", "after\n")
    answer = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "## Markdown answer\n", id="a"
    )
    rendered = render_response_trace_markdown(ConsoleAssistantTurn(answer, (activity,)))
    assert result in rendered
    assert "``````text\n" in rendered
    assert "before\n" in rendered and "after\n" in rendered
    assert rendered.count("example.py") == 1
    assert "## Markdown answer\n" in rendered


@pytest.mark.parametrize("status", ["streaming", "pending", "stopped", "failed"])
def test_partial_responses_do_not_claim_to_be_final(status):
    from tldw_chatbook.Chat.console_response_trace import render_response_trace_markdown

    answer = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "partial text", status=status
    )
    rendered = render_response_trace_markdown(ConsoleAssistantTurn(answer))
    assert status in rendered
    assert "## Final answer" not in rendered
    assert "partial text" in rendered


def test_unavailable_data_is_labelled_without_private_payloads():
    from tldw_chatbook.Chat.console_response_trace import render_response_trace_markdown

    answer = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer")
    answer.thinking = ThinkingEnvelope(
        (
            ProprietaryThinkingBlock(
                block_id="private",
                round_ordinal=0,
                provider="local",
                model="test",
                protocol="opaque",
                source_format="encrypted",
                status="complete",
            ),
        )
    )
    answer.opaque_thinking_json = "SECRET_OPAQUE"
    answer.provider_continuation = "SECRET_CONTINUATION"
    answer.exchanges = ("SECRET_RAW_EXCHANGE",)
    activity = ConsoleChatMessage(ConsoleMessageRole.TOOL, "legacy retained result")
    rendered = render_response_trace_markdown(ConsoleAssistantTurn(answer, (activity,)))
    assert "unavailable" in rendered.lower()
    assert "legacy retained result" in rendered
    assert "SECRET" not in rendered
    assert "Arguments unavailable" in rendered


def test_chat_trace_does_not_duplicate_owned_or_standalone_tools():
    from tldw_chatbook.Chat.console_response_trace import (
        find_response_turn,
        render_chat_trace_markdown,
    )

    orphan = tool("orphan", "standalone-result")
    answer = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer", id="a")
    owned = tool("owned", "owned-result")
    rendered = render_chat_trace_markdown([orphan, answer, owned], title="chat")
    assert rendered.count("standalone-result") == rendered.count("owned-result") == 1
    assert find_response_turn([orphan, answer, owned], "orphan") is None
    assert find_response_turn([orphan, answer, owned], "missing") is None


def test_full_results_have_no_clipboard_preview_cap():
    from tldw_chatbook.Chat.console_response_trace import render_response_trace_markdown

    full = "x" * 100_000 + "END OF RESULT"
    answer = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer")
    assert full in render_response_trace_markdown(
        ConsoleAssistantTurn(answer, (tool("read", full),))
    )
