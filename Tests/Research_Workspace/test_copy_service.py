"""Restart Copy at a durable owner checkpoint and retain the source."""

from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER, TIME
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyOwnerRef,
    ResearchCopyConfirmation,
    ResearchCopyItem,
    ResearchCopyManifest,
    ResearchCopyStatus,
)
from tldw_chatbook.Research_Workspace.copy_receipts import (
    CopyItemStage,
    CopyReceiptFailure,
    ResearchCopyReceiptStore,
)
from tldw_chatbook.Research_Workspace.copy_service import (
    CopyOwnerActionError,
    CopySourceSnapshot,
    ResearchCopyService,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class ItemOwner:
    def __init__(self, ref, *, fail_once=False):
        self.ref = ref
        self.saved = {}
        self.calls = []
        self.fail_once = fail_once
        self.bump_source_revision = False

    async def inspect_source(self, ref):
        self.calls.append("read-source")
        captured = (
            replace(ref, version=ref.version + 1) if self.bump_source_revision else ref
        )
        return CopySourceSnapshot(captured, "synthetic body")

    async def create_destination(self, item, source, workspace, item_key):
        self.calls.append("create")
        assert workspace == self.ref and source.body == "synthetic body"
        self.saved.setdefault(
            item_key,
            CopyOwnerRef(
                workspace,
                item.source.kind,
                "created-" + item.source.owner_id,
                version=1,
            ),
        )
        return self.saved[item_key]

    async def associate_destination(self, item, destination):
        self.calls.append("associate")
        assert destination in self.saved.values()
        if self.fail_once and item.source.owner_id == "42":
            self.fail_once = False
            raise CopyOwnerActionError(
                CopyReceiptFailure.ASSOCIATION_FAILED, retryable=True
            )

    async def verify_destination(self, item, destination):
        self.calls.append("verify")
        return destination in self.saved.values()


def scoped_copy_manifest(source=SERVER, destination=LOCAL):
    items = tuple(
        ResearchCopyItem(
            CopyOwnerRef(source, "artifact", owner_id, version=2), "copy_content"
        )
        for owner_id in ("41", "42")
    )
    return ResearchCopyManifest(source, destination, items, TIME)


async def test_copy_resumes_after_saved_owner_without_creating_a_duplicate(tmp_path):
    path = tmp_path / "workspaces.sqlite"
    manifest = scoped_copy_manifest()
    owner = ItemOwner(LOCAL, fail_once=True)
    db = WorkspaceDB(path)
    try:
        store = ResearchCopyReceiptStore(db)
        coordinator = ResearchCopyService(
            store, lambda _item, _target: owner, assert_current=lambda _manifest: None
        )
        confirmation = ResearchCopyConfirmation(manifest.manifest_hash, confirmed=True)
        with pytest.raises(ValueError, match="Confirm"):
            await coordinator.execute_copy(
                manifest, replace(confirmation, confirmed=False)
            )
        assert store.get(manifest.transfer_id) is None
        partial = await coordinator.execute_copy(manifest, confirmation)
        assert partial.status is ResearchCopyStatus.PARTIALLY_COMPLETED
        assert [row.stage for row in partial.items] == [
            CopyItemStage.VERIFIED,
            CopyItemStage.DESTINATION_SAVED,
        ]
        assert owner.calls.count("create") == 2
    finally:
        db.close()

    reopened = WorkspaceDB(path)
    try:
        store = ResearchCopyReceiptStore(reopened)
        restarted = ResearchCopyService(
            store, lambda _item, _target: owner, assert_current=lambda _manifest: None
        )
        completed = await restarted.resume_copy(manifest.transfer_id, workspace=LOCAL)
        assert completed.status is ResearchCopyStatus.COMPLETED
        assert owner.calls.count("create") == 2
        assert len(owner.saved) == 2
        assert all(row.stage is CopyItemStage.VERIFIED for row in completed.items)
    finally:
        reopened.close()


async def test_copy_blocks_a_changed_source_before_any_destination_write(tmp_path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    manifest = scoped_copy_manifest()
    owner = ItemOwner(LOCAL)
    owner.bump_source_revision = True
    try:
        service = ResearchCopyService(
            ResearchCopyReceiptStore(db),
            lambda _item, _target: owner,
            assert_current=lambda _manifest: None,
        )
        result = await service.execute_copy(
            manifest, ResearchCopyConfirmation(manifest.manifest_hash, confirmed=True)
        )
        assert result.status is ResearchCopyStatus.FAILED_TERMINAL
        assert all(
            row.failure is CopyReceiptFailure.SOURCE_CHANGED for row in result.items
        )
        assert owner.saved == {}
    finally:
        db.close()
