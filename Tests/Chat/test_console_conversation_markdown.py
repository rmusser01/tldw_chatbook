"""Golden-string tests for the conversation markdown renderer (TASK-25714)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_conversation_markdown import (
    MarkdownMessage,
    render_conversation_markdown,
)


def _msg(role: str, content: str, **extra) -> MarkdownMessage:
    return MarkdownMessage(role=role, content=content, **extra)


def test_clean_rendering_is_user_and_assistant_text_only() -> None:
    markdown = render_conversation_markdown(
        title="Research thread",
        rendered_at="2026-08-31",
        messages=[
            _msg("user", "Summarize the plan"),
            _msg("tool", "fs_read output...", tool_label="fs_read"),
            _msg("assistant", "Here is the plan.", thinking="should I list steps?"),
        ],
        fidelity="clean",
    )
    assert markdown == (
        "# Research thread\n"
        "\n"
        "_2026-08-31 · 2 messages_\n"
        "\n"
        "## User\n"
        "\n"
        "Summarize the plan\n"
        "\n"
        "## Assistant\n"
        "\n"
        "Here is the plan.\n"
    )


def test_full_rendering_includes_tool_and_thinking_blocks() -> None:
    markdown = render_conversation_markdown(
        title="Research thread",
        rendered_at="2026-08-31",
        messages=[
            _msg("user", "Summarize the plan"),
            _msg("tool", "file bytes...", tool_label="fs_read"),
            _msg(
                "assistant",
                "Here is the plan.",
                thinking="consider steps",
                citations=(("Spec", "https://example.com/spec"),),
            ),
        ],
        fidelity="full",
    )
    assert markdown == (
        "# Research thread\n"
        "\n"
        "_2026-08-31 · 3 messages_\n"
        "\n"
        "## User\n"
        "\n"
        "Summarize the plan\n"
        "\n"
        "<details>\n"
        "<summary>Tool: fs_read</summary>\n"
        "\n"
        "file bytes...\n"
        "\n"
        "</details>\n"
        "\n"
        "## Assistant\n"
        "\n"
        "Here is the plan.\n"
        "\n"
        "<details>\n"
        "<summary>Thinking</summary>\n"
        "\n"
        "consider steps\n"
        "\n"
        "</details>\n"
        "\n"
        "**Sources**\n"
        "\n"
        "- [Spec](https://example.com/spec)\n"
    )


def test_clean_rendering_drops_thinking_keeps_citations() -> None:
    markdown = render_conversation_markdown(
        title="T",
        rendered_at="2026-08-31",
        messages=[
            _msg(
                "assistant",
                "Answer.",
                thinking="hmm",
                citations=(("A", "https://a"),),
            ),
        ],
        fidelity="clean",
    )
    assert "hmm" not in markdown
    assert "- [A](https://a)" in markdown
    assert "Answer." in markdown


def test_image_messages_render_as_placeholders() -> None:
    markdown = render_conversation_markdown(
        title="T",
        rendered_at="2026-08-31",
        messages=[_msg("user", "", image_label="screenshot.png")],
        fidelity="clean",
    )
    assert "![image](screenshot.png)" in markdown


def test_system_rows_render_only_in_full() -> None:
    messages = [_msg("system", "You are concise.")]
    clean = render_conversation_markdown(
        title="T", rendered_at="2026-08-31", messages=messages, fidelity="clean"
    )
    full = render_conversation_markdown(
        title="T", rendered_at="2026-08-31", messages=messages, fidelity="full"
    )
    assert clean is None
    assert "<summary>System</summary>" in full


def test_empty_conversation_renders_none() -> None:
    assert (
        render_conversation_markdown(
            title="T", rendered_at="2026-08-31", messages=[], fidelity="clean"
        )
        is None
    )


def test_unknown_fidelity_raises() -> None:
    with pytest.raises(ValueError):
        render_conversation_markdown(
            title="T",
            rendered_at="2026-08-31",
            messages=[_msg("user", "hi")],
            fidelity="yaml",
        )
