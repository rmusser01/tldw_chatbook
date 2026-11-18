"""TASK-34000.24: the one link-on-use seam Console hand-offs share.

``link_item_to_active_workspace`` is the notes hand-off's algorithm
(``library_notes_controller._link_open_note_to_active_workspace``)
generalised: read memberships first, write only when absent, name the
workspace by its display name, never raise. ``unlink_item_from_workspace`` is
its inverse and says whether it removed, found nothing, or failed (fix round
1, M-4). ``decide_link_on_use`` is the ONE gate-to-link decision both entry
points (the Media reader's ``c`` and Search ``u``) call (fix round 1, I-2).
Pinned against a REAL ``LocalWorkspaceRegistryService`` over a temp
``WorkspaceDB`` (the ``test_workspace_cold_provisioning.py`` shape), not a
fake registry -- ``link_membership`` is INSERT OR IGNORE and returns the
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
    UnlinkOutcome,
    decide_link_on_use,
    link_item_to_active_workspace,
    registry_unavailable_copy,
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


def test_no_registry_says_the_registry_is_unavailable_and_never_raises() -> None:
    """Fix round 1, M-6: the notes precedent's sentence, not a workspace
    selection the user cannot make."""
    result = link_item_to_active_workspace(
        None, item_type="media", item_id="media-1", title="Interview Recording"
    )

    assert result.blocked_reason == registry_unavailable_copy("media") == (
        "Can't use this media in Console — the workspace registry is "
        "unavailable. Next: restart Chatbook, then try again."
    )
    assert "Select an active workspace" not in result.blocked_reason
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


def test_unlink_distinguishes_removed_absent_and_failed(registry) -> None:
    """Fix round 1, M-4: "already removed elsewhere" is the outcome the user
    wanted and must not read as a failure to retry."""
    result = link_item_to_active_workspace(
        registry, item_type="media", item_id="media-1", title="Interview Recording"
    )
    assert result.written is True

    assert (
        unlink_item_from_workspace(
            registry, result.workspace_id, item_type="media", item_id="media-1"
        )
        is UnlinkOutcome.REMOVED
    )
    assert _memberships(registry) == ()
    assert (
        unlink_item_from_workspace(
            registry, result.workspace_id, item_type="media", item_id="media-1"
        )
        is UnlinkOutcome.ABSENT
    )
    assert (
        unlink_item_from_workspace(None, "x", item_type="media", item_id="media-1")
        is UnlinkOutcome.FAILED
    )

    def failing_unlink(*args, **kwargs):
        raise WorkspaceRegistryServiceError("disk full")

    registry.unlink_membership = failing_unlink
    assert (
        unlink_item_from_workspace(
            registry, result.workspace_id, item_type="media", item_id="media-1"
        )
        is UnlinkOutcome.FAILED
    )


# --- fix round 1, I-2: one gate-to-link decision for both entry points -------


def _row(item_type: str, item_id: str, *, eligible: bool, reason: str) -> LibraryWorkspaceSourceRow:
    return LibraryWorkspaceSourceRow(
        item_type=item_type,
        item_id=item_id,
        title=item_id,
        workspace_ids=() if not eligible else (DEFAULT_WORKSPACE_ID,),
        workspace_label="Unassigned",
        visible=True,
        active_context_eligible=eligible,
        authority_label="",
        context_label="",
        recovery_copy="" if eligible else f"Copy or link this {item_type} into the Local Default workspace before using it in Console.",
        reason_code=reason,
    )


def test_a_linkable_row_links_and_reports_it(registry) -> None:
    rows = (_row("media", "7", eligible=False, reason="not_in_active_workspace"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="7", title="Seven"
    )

    assert decision.kind == "linked"
    assert decision.result is not None and decision.result.written is True
    assert decision.result.workspace_name == "Local Default"
    assert [row.item_id for row in _memberships(registry, "7")] == ["7"]


def test_an_item_outside_the_snapshot_links_too(registry) -> None:
    """I-1: the depth state samples the first page of local media; an item
    past it has no row and must still link -- the seam reads memberships
    itself and needs no row."""
    rows = (_row("media", "1", eligible=False, reason="not_in_active_workspace"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="51", title="Fifty-one"
    )

    assert decision.kind == "linked"
    assert decision.result is not None and decision.result.written is True
    assert [row.item_id for row in _memberships(registry, "51")] == ["51"]
    assert _memberships(registry, "1") == ()


def test_only_an_existing_row_with_a_non_linkable_block_refuses(registry) -> None:
    rows = (_row("media", "7", eligible=False, reason="generic_block"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="7", title="Seven"
    )

    assert decision.kind == "refuse"
    assert decision.copy == rows[0].recovery_copy
    assert _memberships(registry, "7") == ()


def test_an_already_linked_row_reports_already_linked(registry) -> None:
    link_item_to_active_workspace(registry, item_type="media", item_id="7", title="Seven")
    rows = (_row("media", "7", eligible=True, reason="active_workspace_match"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="7", title="Seven"
    )

    assert decision.kind == "linked"
    assert decision.result is not None and decision.result.written is False
    assert len(_memberships(registry, "7")) == 1


def test_no_active_workspace_is_the_global_scope_and_stages_as_is(registry) -> None:
    registry.clear_active_workspace()
    rows = (_row("media", "7", eligible=True, reason="visible"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="7", title="Seven"
    )

    assert decision.kind == "stage"
    assert _memberships(registry, "7") == ()
    # ...and so does an item the snapshot does not carry.
    assert decide_link_on_use((), registry, item_type="media", item_id="9", title="").kind == "stage"


def test_a_non_local_item_stages_as_is(registry) -> None:
    decision = decide_link_on_use(
        (), registry, item_type="media", item_id="server:media:3", title="", local=False
    )

    assert decision.kind == "stage"
    assert _memberships(registry, "server:media:3") == ()


def test_a_failed_link_refuses_with_the_seams_sentence(registry) -> None:
    def failing_link(*args, **kwargs):
        raise WorkspaceRegistryServiceError("disk full")

    registry.link_membership = failing_link
    rows = (_row("media", "7", eligible=False, reason="not_in_active_workspace"),)

    decision = decide_link_on_use(
        rows, registry, item_type="media", item_id="7", title="Seven"
    )

    assert decision.kind == "refuse"
    assert decision.copy == (
        "Can't use this media in Console — it could not be linked to "
        "Local Default. Next: try again."
    )
