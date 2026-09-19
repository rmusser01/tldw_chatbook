from dataclasses import replace

from Tests.UI.test_console_conversation_inspector import _row, _turn
from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Widgets.Console.console_inspector_presentation import (
    context_detail,
    context_sections,
    usage_items,
)


def test_section_labels_do_not_include_prompt_bodies():
    snapshot = ConsoleContextSnapshot(
        current_messages=[],
        next_send_payload={"system": "private body", "messages": [], "tools": []},
    )
    rows = context_sections(snapshot)
    assert "preview:system" in {row.key for row in rows}
    assert all("private body" not in row.label for row in rows)
    assert "private body" in context_detail(snapshot, "preview:system").text


def test_usage_selection_uses_message_identity_when_turns_reorder():
    rows = [_row(), _row(index=1)]
    turns = [_turn(native_message_id="a"), _turn(index=1, native_message_id="b")]
    assert [item.message_key for item in usage_items(rows, turns)] == ["a", "b"]
    assert usage_items(rows, turns) == usage_items(rows, list(reversed(turns)))
    assert (
        usage_items(rows, [turns[0], replace(turns[1], index=0)])[0].message_key == ""
    )
