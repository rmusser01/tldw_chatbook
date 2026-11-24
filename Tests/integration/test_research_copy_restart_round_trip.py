"""Copy recovers real owner and SQLite checkpoints after a process interruption."""

from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_copy_adapters import ArtifactServer
from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER, TIME
from Tests.Research_Workspace.test_server_migration_receipts import RecordingServer
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.copy_adapters import TextArtifactCopyOwner
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyOwnerRef,
    ResearchCopyConfirmation,
    ResearchCopyItem,
    ResearchCopyManifest,
    ResearchCopyStatus,
)
from tldw_chatbook.Research_Workspace.copy_receipts import (
    CopyItemStage,
    ResearchCopyReceiptStore,
)
from tldw_chatbook.Research_Workspace.copy_service import ResearchCopyService
from tldw_chatbook.Research_Workspace.server_migration_receipts import (
    ServerMigrationReceiptService,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _Interrupted(BaseException):
    """Bypass normal failure recovery to model process loss at a checkpoint."""


class _CopyServer(RecordingServer, ArtifactServer):
    def __init__(self):
        RecordingServer.__init__(self)
        ArtifactServer.__init__(self)


def _connect(tmp_path, server, clock):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    registry = LocalWorkspaceRegistryService(db)
    chatbooks = LocalChatbookService(registry_path=tmp_path / "chatbooks.json")
    owner = TextArtifactCopyOwner(registry, chatbooks, lambda _ref: server)
    store = ResearchCopyReceiptStore(db, clock=lambda: clock[0])
    migration = ServerMigrationReceiptService(lambda _ref: server)
    service = ResearchCopyService(
        store,
        lambda _item, _ref: owner,
        assert_current=lambda _manifest: None,
        migrations=migration,
    )
    return db, registry, chatbooks, owner, service


@pytest.mark.parametrize(
    "checkpoint", ("canonical write", "association", "local receipt")
)
async def test_server_to_local_copy_recovers_the_same_canonical_owner(
    tmp_path, monkeypatch, checkpoint
):
    tick = [100]
    remote = ArtifactServer()
    remote.records["report"] = {
        "id": "report",
        "workspace_id": SERVER.workspace_id,
        "version": 2,
        "status": "complete",
        "artifact_type": "report",
        "title": "Saved report",
        "content": "Native report body",
    }
    db, registry, chatbooks, owner, service = _connect(tmp_path, remote, tick)
    registry.create_workspace(workspace_id=LOCAL.workspace_id, name="Local destination")
    item = ResearchCopyItem(
        CopyOwnerRef(SERVER, "artifact", "report", version=2), "copy_content"
    )
    manifest = ResearchCopyManifest(SERVER, LOCAL, (item,), TIME)

    if checkpoint == "canonical write":
        create = owner.create_destination

        async def stop_after_create(*args):
            await create(*args)
            raise _Interrupted

        monkeypatch.setattr(owner, "create_destination", stop_after_create)
    elif checkpoint == "association":
        associate = owner.associate_destination

        async def stop_after_associate(*args):
            await associate(*args)
            raise _Interrupted

        monkeypatch.setattr(owner, "associate_destination", stop_after_associate)
    else:
        record = service.receipts.record_item_stage

        def stop_after_receipt(*args, **kwargs):
            saved = record(*args, **kwargs)
            if saved.items[0].stage is CopyItemStage.VERIFIED:
                raise _Interrupted
            return saved

        monkeypatch.setattr(service.receipts, "record_item_stage", stop_after_receipt)

    try:
        with pytest.raises(_Interrupted):
            await service.execute_copy(
                manifest, ResearchCopyConfirmation(manifest.manifest_hash, True)
            )
        original = (await chatbooks.list_chatbooks())[0]["id"]
        assert (
            service.receipts.get(manifest.transfer_id).status
            is ResearchCopyStatus.IN_PROGRESS
        )
    finally:
        db.close()

    tick[0] += 301
    reopened, registry, chatbooks, _, service = _connect(tmp_path, remote, tick)
    try:
        completed = await service.resume_copy(manifest.transfer_id, workspace=LOCAL)
        assert completed.status is ResearchCopyStatus.COMPLETED
        assert completed.items[0].destination.owner_id == original
        assert service.receipts.list_history(LOCAL) == (completed,)
        assert service.receipts.list_recoverable(LOCAL) == ()
        assert [row["id"] for row in await chatbooks.list_chatbooks()] == [original]
        links = registry.get_item_memberships("chatbook", original)
        assert [(row.workspace_id, row.role) for row in links] == [
            (LOCAL.workspace_id, "output")
        ]
        assert remote.records["report"]["version"] == 2
    finally:
        reopened.close()


async def test_copy_recovers_a_server_chunk_without_replaying_the_owner_write(
    tmp_path, monkeypatch
):
    tick = [100]
    remote = _CopyServer()
    db, registry, chatbooks, _, service = _connect(tmp_path, remote, tick)
    registry.create_workspace(workspace_id=LOCAL.workspace_id, name="Local source")
    source = await chatbooks.create_chatbook(
        name="Local report",
        metadata={
            "artifact_source": "research",
            "artifact_kind": "report",
            "content": "Body",
        },
    )
    registry.link_membership(
        LOCAL.workspace_id,
        item_type="chatbook",
        item_id=source["id"],
        role="output",
        transfer_policy="copy",
    )
    item = ResearchCopyItem(
        CopyOwnerRef(LOCAL, "artifact", source["id"], version=1), "copy_content"
    )
    target = replace(SERVER, workspace_id="b5abc8e3-b12e-4caa-9238-280394803454")
    manifest = ResearchCopyManifest(LOCAL, target, (item,), TIME)
    acknowledge = service.migrations.ack_item

    async def stop_after_chunk(*args):
        await acknowledge(*args)
        raise _Interrupted

    monkeypatch.setattr(service.migrations, "ack_item", stop_after_chunk)
    try:
        with pytest.raises(_Interrupted):
            await service.execute_copy(
                manifest, ResearchCopyConfirmation(manifest.manifest_hash, True)
            )
        assert remote.writes == 1 and len(remote.accepted) == 1
    finally:
        db.close()

    tick[0] += 301
    reopened, _, _, _, service = _connect(tmp_path, remote, tick)
    try:
        completed = await service.resume_copy(manifest.transfer_id, workspace=LOCAL)
        assert (
            completed.status is ResearchCopyStatus.COMPLETED
            and completed.server_migration_finalized
        )
        assert remote.writes == 1
        assert sum(method == "PUT" for method, _ in remote.requests) == 1
        assert all("client-delete-ack" not in path for _, path in remote.requests)
    finally:
        reopened.close()
