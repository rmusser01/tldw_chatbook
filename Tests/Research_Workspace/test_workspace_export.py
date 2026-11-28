"""Workspace exports follow Copy policy and use only exact Local owner content."""

from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    BoundedPageResult,
    QualifiedWorkspaceRef,
)
from tldw_chatbook.Research_Workspace.quick_notes import ResearchQuickNote
from tldw_chatbook.Research_Workspace.workspace_export import prepare_workspace_export
from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy


class _LocalOwner:
    ref = QualifiedWorkspaceRef("local", "reading-group")

    def __init__(self):
        self.notes = tuple(
            ResearchQuickNote(self.ref, note_id, f"Title {note_id}", content, 2)
            for note_id, content in (
                ("reviewed", "Reviewed notes"),
                ("metadata", "Metadata-only body"),
                ("private", "Device-only body"),
            )
        )
        self.reads = []
        self.policy = {
            "reviewed": WorkspaceTransferPolicy.COPY,
            "metadata": WorkspaceTransferPolicy.METADATA_ONLY,
            "private": WorkspaceTransferPolicy.LOCAL_ONLY,
        }

    async def list_sources(self, *args, **kwargs):
        return BoundedPageResult((), 100, 0, 0, False)

    async def list_notes(self, ref, _request):
        assert ref == self.ref
        return BoundedPageResult(self.notes, 100, 0, 3, False)

    async def list_outputs(self, ref, **kwargs):
        assert ref == self.ref
        return BoundedPageResult((), 100, 0, 0, False)

    async def get_note(self, ref, note_id):
        assert ref == self.ref
        self.reads.append(note_id)
        return next(note for note in self.notes if note.note_id == note_id)

    def get_item_memberships(self, item_type, owner_id):
        assert item_type == "note"
        return [
            SimpleNamespace(
                workspace_id=self.ref.workspace_id,
                role="note",
                transfer_policy=self.policy[owner_id],
            )
        ]


@pytest.mark.asyncio
async def test_export_freezes_publication_and_excludes_device_only_records():
    owner = _LocalOwner()
    prepared = await prepare_workspace_export(owner.ref, owner, owner, chatbooks=None)
    assert owner.reads == ["reviewed"]
    assert prepared.content_count == 1
    with ZipFile(BytesIO(prepared.bundle_bytes())) as archive:
        assert archive.namelist() == ["manifest.md", "items/1-note.md"]
        assert b"Reviewed notes" in archive.read("items/1-note.md")
        payload = archive.read("manifest.md")
        assert b"Metadata only" in payload
        assert b"Omit" in payload
        assert b"Device-only body" not in payload
        assert b"Title private" not in payload and b"private" not in payload

    owner.policy["reviewed"] = WorkspaceTransferPolicy.LOCAL_ONLY
    with pytest.raises(ValueError, match="policy changed"):
        await prepared.confirm_current_policy(owner)
