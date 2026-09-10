"""Conversation recovery owns IDs, bounded text and honest lifecycle copy."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_conversations_state import (
    build_library_conversations_state,
)


def test_archived_row_has_workspace_and_date_and_restore_resume_state():
    state = build_library_conversations_state(
        [
            {
                "id": "c",
                "title": "[red]Literal",
                "archived": True,
                "version": 4,
                "workspace_name": "Project",
                "workspace_archived": True,
                "updated_at": "2026-09-10T12:00:00",
                "message_count": 3,
            }
        ],
        total_count=1,
        archive_scope="archived",
    )
    assert state.archive_scope == "archived"
    assert state.selected_archived
    assert "Project" in state.rows[0].secondary
    assert "archived" in state.rows[0].secondary
    assert "2026-09-10" in " ".join(state.preview_lines)


@pytest.mark.asyncio
async def test_undo_uses_only_successful_ids_and_returned_versions(monkeypatch):
    from tldw_chatbook.UI.Library_Modules import library_conversation_recovery as module

    calls = []

    async def change(app, ids, *, archived, expected_versions):
        calls.append((tuple(ids), archived, dict(expected_versions)))
        return (
            {"changed": {"a": 8}, "failures": {"b": "Changed; refresh first."}}
            if archived
            else {"changed": {"a": 9}, "failures": {}}
        )

    monkeypatch.setattr(module, "change_conversation_archive", change)
    screen = SimpleNamespace(
        app_instance=object(),
        _sync_library_conversation_canvas=lambda **kw: None,
        _library_conversation_requested_page=1,
        _library_conversation_requested_query="needle",
        _start_library_conversation_page_request=lambda *a, **kw: None,
    )
    controller = module.LibraryConversationRecovery(screen)
    await controller.change(
        ("a", "b"), archived=True, expected_versions={"a": 7, "b": 4}
    )
    assert "b" in controller.receipt_copy
    await controller.undo()
    assert calls[-1] == (("a",), False, {"a": 8})
