"""TASK-34000.24: the one link-on-use seam Console hand-offs share.

``link_item_to_active_workspace`` is the notes hand-off's algorithm
(``library_notes_controller._link_open_note_to_active_workspace``)
generalised: read memberships first, write only when absent, name the
workspace by its display name, never raise. ``unlink_item_from_workspace`` is
its inverse. Pinned against a REAL ``LocalWorkspaceRegistryService`` over a
temp ``WorkspaceDB`` (the ``test_workspace_cold_provisioning.py`` shape), not
a fake registry -- ``link_membership`` is INSERT OR IGNORE and returns the
EXISTING row, which is exactly the contract a fake would get wrong.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.display_state import LibraryWorkspaceSourceRow
from tldw_chatbook.Workspaces.link_on_use import (
    NO_ACTIVE_WORKSPACE_COPY,
    LinkOnUseResult,
    link_item_to_active_workspace,
    resolve_workspace_item_id,
    unlink_item_from_workspace,
)
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.fixture
def registry(tmp_path) -> LocalWorkspaceRegistryService:
    service = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspace.sqlite", client_id="link-on-use")
    )
    service.ensure_default_workspace()
    service.set_active_workspace(DEFAULT_WORKSPACE_ID)
    return service


def _memberships(registry, item_id: str = "media-1"):
    return registry.get_item_memberships(item_type="media", item_id=item_id)


def test_first_use_writes_one_membership_and_names_local_default(registry) -> None:
    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result == LinkOnUseResult(
        workspace_id=DEFAULT_WORKSPACE_ID,
        workspace_name="Local Default",
        written=True,
        blocked_reason="",
    )
    rows = _memberships(registry)
    assert [(row.workspace_id, row.item_type, row.item_id, row.title) for row in rows] == [
        (DEFAULT_WORKSPACE_ID, "media", "media-1", "Interview Recording")
    ]


def test_second_use_is_already_linked_and_writes_nothing(registry) -> None:
    link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )

    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result.written is False
    assert result.blocked_reason == ""
    assert result.workspace_name == "Local Default"
    assert len(_memberships(registry)) == 1


def test_a_named_workspace_is_reported_by_its_name_never_its_id(registry) -> None:
    created = registry.create_workspace(workspace_id="ws-thesis-9f2c", name="Thesis")
    registry.set_active_workspace(created.workspace_id)

    result = link_item_to_active_workspace(
        registry, item_type="note", item_id="n-1", title="Reading notes"
    )

    assert result.workspace_id == "ws-thesis-9f2c"
    assert result.workspace_name == "Thesis"
    assert result.written is True


def test_no_active_workspace_is_blocked_with_the_existing_sentence(registry) -> None:
    registry.clear_active_workspace()

    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result.blocked_reason == NO_ACTIVE_WORKSPACE_COPY
    assert result.blocked_reason == (
        "Select an active workspace before using this item in Console."
    )
    assert result.written is False
    assert _memberships(registry) == ()


def test_no_registry_is_blocked_and_never_raises() -> None:
    result = link_item_to_active_workspace(
        None, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result.blocked_reason == NO_ACTIVE_WORKSPACE_COPY
    assert result.written is False


def test_a_failing_write_is_blocked_with_the_workspace_named(registry) -> None:
    def failing_link(*args, **kwargs):
        raise WorkspaceRegistryServiceError("disk full")

    registry.link_membership = failing_link

    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result.blocked_reason == (
        "Can't use this media in Console — it could not be linked to "
        "Local Default. Next: try again."
    )
    assert result.written is False
    assert "workspace-default" not in result.blocked_reason


def test_unlink_is_the_exact_inverse_of_a_fresh_link(registry) -> None:
    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )
    assert result.written is True

    assert (
        unlink_item_from_workspace(
            registry, result.workspace_id, item_type="media", item_id="media-1"
        )
        is True
    )
    assert _memberships(registry) == ()
    # Idempotent and silent on a second press.
    assert (
        unlink_item_from_workspace(
            registry, result.workspace_id, item_type="media", item_id="media-1"
        )
        is False
    )
    assert unlink_item_from_workspace(None, "x", item_type="media", item_id="media-1") is False


def _row(item_type: str, item_id: str) -> LibraryWorkspaceSourceRow:
    return LibraryWorkspaceSourceRow(
        item_type=item_type,
        item_id=item_id,
        title=item_id,
        workspace_ids=(),
        workspace_label="Unassigned",
        visible=True,
        active_context_eligible=False,
        authority_label="",
        context_label="",
        recovery_copy="",
        reason_code="not_in_active_workspace",
    )


def test_resolve_workspace_item_id_returns_the_id_the_row_model_reads() -> None:
    """Search `u` must write the SAME id the reader's row model reads, or the
    gate would stay closed after linking. The row model reads the media
    record's own ``id`` (``"media-1"`` in the Library fixtures, ``"3"`` for a
    real record); retrieval may hand the same item back wrapped
    (``local:media:3``, ``media_3``)."""
    rows = (_row("media", "media-1"), _row("media", "3"), _row("note", "n-1"))

    assert resolve_workspace_item_id(rows, "media", "media-1") == "media-1"
    assert resolve_workspace_item_id(rows, "media", "local:media:3") == "3"
    assert resolve_workspace_item_id(rows, "media", "media_3") == "3"
    assert resolve_workspace_item_id(rows, "media", "3") == "3"
    assert resolve_workspace_item_id(rows, "note", "n-1") == "n-1"
    # Unknown to the row model: the raw id, unchanged (never a guess).
    assert resolve_workspace_item_id(rows, "media", "local:media:9") == "local:media:9"
    assert resolve_workspace_item_id(rows, "conversation", "chat-9") == "chat-9"
