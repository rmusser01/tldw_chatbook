"""Golden-string tests for the conversation markdown renderer (TASK-25836)."""

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
    )


def test_clean_rendering_drops_thinking_and_sanitizes_title() -> None:
    markdown = render_conversation_markdown(
        title="Evil <script>alert(1)</script> [link](https://x)",
        rendered_at="2026-08-31",
        messages=[
            _msg("assistant", "Answer.", thinking="hmm"),
        ],
        fidelity="clean",
    )
    assert "hmm" not in markdown
    assert "<script>" not in markdown
    assert "[link]" not in markdown
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


# ---- Review-hardening (PR #2262) -----------------------------------------


def test_store_adapter_reads_enum_roles_by_value() -> None:
    """str(enum) yields the class name; exports must read .value."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_conversation_markdown import (
        markdown_messages_from_store,
    )

    class _Stub:
        role = ConsoleMessageRole.USER
        content = "enum question"

    normalized = markdown_messages_from_store([_Stub()])
    assert normalized[0].role == "user"
    markdown = render_conversation_markdown(
        title="T", rendered_at="2026-08-31", messages=normalized, fidelity="clean"
    )
    assert "enum question" in markdown


def test_store_adapter_joins_thinking_blocks_not_reprs() -> None:
    from tldw_chatbook.Chat.console_conversation_markdown import (
        markdown_messages_from_store,
    )

    class _Block:
        text = "step one"

    class _Envelope:
        blocks = (_Block(),)

    class _Stub:
        role = "assistant"
        content = "answer"
        thinking = _Envelope()

    normalized = markdown_messages_from_store([_Stub()])
    assert normalized[0].thinking == "step one"
    assert "Envelope" not in normalized[0].thinking


def test_store_adapter_never_stringifies_attachment_payloads() -> None:
    from tldw_chatbook.Chat.console_conversation_markdown import (
        markdown_messages_from_store,
    )

    class _Attachment:
        display_name = "notes.txt"
        data = b"\x00-binary-bytes"

    class _Stub:
        role = "user"
        content = ""
        attachments = (_Attachment(),)

    markdown = render_conversation_markdown(
        title="T",
        rendered_at="2026-08-31",
        messages=markdown_messages_from_store([_Stub()]),
        fidelity="clean",
    )
    assert "notes.txt" in markdown
    assert "binary-bytes" not in markdown


def test_store_adapter_prefers_full_tool_output() -> None:
    from tldw_chatbook.Chat.console_conversation_markdown import (
        markdown_messages_from_store,
    )

    class _Stub:
        role = "tool"
        content = "preview…"
        tool_output_full = "the complete result"

    normalized = markdown_messages_from_store([_Stub()])
    assert normalized[0].content == "the complete result"


def test_db_adapter_marks_images_by_mime_type() -> None:
    from tldw_chatbook.Chat.console_conversation_markdown import (
        markdown_messages_from_db_rows,
    )

    normalized = markdown_messages_from_db_rows(
        [{"sender": "user", "content": "", "image_mime_type": "image/png"}]
    )
    assert normalized[0].image_label == "image"
    markdown = render_conversation_markdown(
        title="T", rendered_at="2026-08-31", messages=normalized, fidelity="clean"
    )
    assert "![image](image)" in markdown
    assert "_(empty message)_" not in markdown
