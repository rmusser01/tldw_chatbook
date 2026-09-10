"""Conversation recovery owns IDs, bounded text and honest lifecycle copy."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_conversations_state import (
    build_library_conversations_state,
)


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("workspace_archived", [False, True])
def test_row_distinguishes_conversation_and_workspace_archive_state(
    archived, workspace_archived
):
    state = build_library_conversations_state(
        [
            {
                "id": "c",
                "title": "[red]Literal",
                "archived": archived,
                "version": 4,
                "workspace_name": "Project",
                "workspace_archived": workspace_archived,
                "updated_at": "2026-09-10T12:00:00",
                "message_count": 3,
            }
        ],
        total_count=1,
        archive_scope="archived",
    )
    assert state.archive_scope == "archived"
    assert state.selected_archived == (archived or workspace_archived)
    segments = state.rows[0].secondary.split(" · ")
    assert ("Archived" if archived else "Active") in segments
    assert ("Active" if archived else "Archived") not in segments
    assert (
        "Project (workspace archived)" if workspace_archived else "Project"
    ) in segments
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


@pytest.mark.parametrize(
    "scope", ["active", "archived", "all", "invalid", "", None, []]
)
def test_recovery_scope_boundary_preserves_query_and_rejects_invalid_values(scope):
    from unittest.mock import Mock

    from tldw_chatbook.UI.Library_Modules.library_conversation_recovery import (
        LibraryConversationRecovery,
    )

    request = Mock()
    screen = SimpleNamespace(
        _library_conversation_requested_query="needle",
        _start_library_conversation_page_request=request,
    )
    recovery = LibraryConversationRecovery(screen)
    recovery.set_scope(scope)
    if isinstance(scope, str) and scope in ("active", "archived", "all"):
        request.assert_called_once_with(
            1, "needle", focus_after_apply=f"#library-conversations-scope-{scope}"
        )
        assert recovery.scope == scope
    else:
        request.assert_not_called()
        assert recovery.scope == "active"


@pytest.mark.parametrize(
    "button_id",
    [
        "library-conversations-scope-active",
        "library-conversations-scope-archived",
        "library-conversations-scope-all",
        "library-conversations-scope-deleted",
        None,
    ],
)
def test_scope_event_validates_before_changing_recovery(button_id):
    from unittest.mock import Mock

    from tldw_chatbook.UI.Library_Modules.library_conversations_controller import (
        LibraryConversationsController,
    )

    recovery = Mock()
    controller = SimpleNamespace(_conversation_recovery=lambda: recovery)
    event = SimpleNamespace(button=SimpleNamespace(id=button_id), stop=Mock())
    LibraryConversationsController.handle_library_conversation_scope(controller, event)
    event.stop.assert_called_once()
    if button_id and button_id.rsplit("-", 1)[-1] in ("active", "archived", "all"):
        recovery.set_scope.assert_called_once_with(button_id.rsplit("-", 1)[-1])
    else:
        recovery.set_scope.assert_not_called()
