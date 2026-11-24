"""Copy only recognized canonical owner fields into real destination stores."""

from types import SimpleNamespace

import pytest

from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER
from Tests.Research_Workspace.test_local_studio import local_studio as local_studio
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Research_Workspace.copy_adapters import (
    LocalCopyNoteOwner,
    TextArtifactCopyOwner,
)
from tldw_chatbook.Research_Workspace.copy_models import CopyOwnerRef, ResearchCopyItem
from tldw_chatbook.Research_Workspace.copy_service import CopyOwnerActionError
from tldw_chatbook.Research_Workspace.local_adapter import LocalResearchWorkspaceAdapter
from tldw_chatbook.Research_Workspace.quick_notes import ResearchQuickNote

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class ArtifactServer:
    def __init__(self):
        self.records = {}
        self.writes = 0

    async def list_workspace_artifacts(self, workspace_id):
        return [
            record
            for record in self.records.values()
            if record["workspace_id"] == workspace_id
        ]

    async def create_workspace_artifact(self, workspace_id, request):
        self.writes += 1
        assert request.id not in self.records
        row = {**request.model_dump(), "workspace_id": workspace_id, "version": 1}
        self.records[request.id] = row
        return row


async def test_text_artifacts_copy_both_directions_and_retry_without_new_owner(
    local_studio,
):
    local_studio.registry.create_workspace(
        workspace_id=LOCAL.workspace_id, name="Local reading group"
    )
    server = ArtifactServer()
    adapter = TextArtifactCopyOwner(
        local_studio.registry,
        local_studio.artifacts,
        lambda ref: server,
    )
    source = await local_studio.artifacts.create_chatbook(
        name="Canonical report",
        metadata={
            "artifact_source": "research",
            "artifact_kind": "report",
            "content": "Exact body",
            "private_device_path": "/private/local-device-only",
        },
    )
    local_studio.registry.link_membership(
        LOCAL.workspace_id,
        item_type="chatbook",
        item_id=source["id"],
        role="output",
    )
    item = ResearchCopyItem(
        CopyOwnerRef(LOCAL, "artifact", source["id"], version=1), "copy_content"
    )
    captured = await adapter.inspect_source(item.source)
    saved = await adapter.create_destination(item, captured, SERVER, "a" * 64)
    assert await adapter.create_destination(item, captured, SERVER, "a" * 64) == saved
    assert server.writes == 1
    assert "private_device_path" not in repr(server.records)
    await adapter.associate_destination(item, saved)
    assert await adapter.verify_destination(item, saved)

    returned = ResearchCopyItem(saved, "copy_content")
    remote_snapshot = await adapter.inspect_source(saved)
    copied_back = await adapter.create_destination(
        returned, remote_snapshot, LOCAL, "b" * 64
    )
    assert (
        await adapter.create_destination(returned, remote_snapshot, LOCAL, "b" * 64)
        == copied_back
    )
    assert copied_back.owner_id != source["id"]
    assert not await adapter.verify_destination(returned, copied_back)
    await adapter.associate_destination(returned, copied_back)
    assert await adapter.verify_destination(returned, copied_back)
    copied_record = await local_studio.artifacts.get_chatbook(copied_back.owner_id)
    assert copied_record["metadata"]["content"] == "Exact body"
    assert not copied_record.get("file_path")
    assert (await local_studio.artifacts.get_chatbook(source["id"])) == source


async def test_server_note_copies_to_canonical_local_notes_and_recovers_its_id(
    local_studio, tmp_path
):
    db = local_studio.study.db
    local_studio.registry.create_workspace(
        workspace_id=LOCAL.workspace_id, name="Local reading group"
    )
    notes = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="copy-test", global_db_to_use=db
    )
    local = LocalResearchWorkspaceAdapter(
        local_studio.registry,
        notes_scope_service=NotesScopeService(
            local_notes_service=notes, server_service=None
        ),
        notes_user_id="copy-test",
    )
    server_note = ResearchQuickNote(
        SERVER, "1", "Server note", "Exact note text", 3, source_ids=("server-source",)
    )

    async def get_server_note(ref, note_id):
        assert (ref, note_id) == (SERVER, "1")
        return server_note

    adapter = LocalCopyNoteOwner(SimpleNamespace(get_note=get_server_note), local)
    item = ResearchCopyItem(
        CopyOwnerRef(SERVER, "note", "1", version=3), "copy_content"
    )
    captured = await adapter.inspect_source(item.source)
    owner = await adapter.create_destination(item, captured, LOCAL, "a" * 64)
    assert await adapter.create_destination(item, captured, LOCAL, "a" * 64) == owner
    note = await local.get_note(LOCAL, owner.owner_id)
    assert note.content == "Exact note text" and note.source_ids == ()
    assert await adapter.verify_destination(item, owner)


async def test_local_only_link_cannot_be_read_for_outbound_copy(local_studio):
    local_studio.registry.create_workspace(
        workspace_id=LOCAL.workspace_id, name="Local reading group"
    )
    row = await local_studio.artifacts.create_chatbook(
        name="Private report",
        metadata={
            "artifact_source": "research",
            "artifact_kind": "report",
            "content": "Local only",
        },
    )
    local_studio.registry.link_membership(
        LOCAL.workspace_id,
        item_type="chatbook",
        item_id=row["id"],
        role="output",
        transfer_policy="local-only",
    )
    adapter = TextArtifactCopyOwner(
        local_studio.registry, local_studio.artifacts, lambda _ref: None
    )
    with pytest.raises(CopyOwnerActionError, match="source_changed"):
        await adapter.inspect_source(
            CopyOwnerRef(LOCAL, "artifact", row["id"], version=1)
        )
