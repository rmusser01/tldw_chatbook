"""Copy acknowledges only exact, verified item receipts on the captured server."""

from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER, TIME, item
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyOwnerRef,
    ResearchCopyManifest,
)
from tldw_chatbook.Research_Workspace.copy_receipts import (
    CopyItemReceipt,
    CopyItemStage,
    CopyReceiptFailure,
)
from tldw_chatbook.Research_Workspace.server_migration_receipts import (
    ServerMigrationReceiptService,
)
from tldw_chatbook.tldw_api.exceptions import APIResponseError

pytestmark = pytest.mark.asyncio


class RecordingServer:
    def __init__(self):
        self.workspace = None
        self.session = None
        self.declared = {}
        self.accepted = {}
        self.requests = []

    async def get_workspace(self, workspace_id):
        if self.workspace is None:
            raise APIResponseError(404, "workspace absent")
        return self.workspace

    def reply(self):
        return {
            **self.session,
            "declared_chunk_count": len(self.declared),
            "accepted_chunk_count": len(self.accepted),
            "missing_chunk_ids": [
                key for key in self.declared if key not in self.accepted
            ],
            "chunks": list(self.accepted.values()),
        }

    async def _request(self, method, path, *, json_data=None):
        self.requests.append((method, path))
        root = "/api/v1/workspaces/migrations"
        if method == "GET":
            if self.session is None:
                raise APIResponseError(404, "migration absent")
            return self.reply()
        if path == root and method == "POST":
            self.declared = {row["id"]: row for row in json_data["declared_chunks"]}
            self.session = {
                key: json_data[key]
                for key in (
                    "id",
                    "idempotency_key",
                    "target_workspace_id",
                    "target_workspace_name",
                    "source_product",
                    "manifest_hash",
                )
            }
            self.session["status"] = "created"
            return self.reply()
        if method == "PUT":
            key = path.rsplit("/", 1)[-1]
            assert json_data["sha256"] == self.declared[key]["sha256"]
            saved = {
                "id": key,
                "migration_id": self.session["id"],
                "status": "accepted",
                **json_data,
            }
            self.accepted[key] = saved
            return saved
        assert method == "POST" and path.endswith("/finalize")
        self.session["status"] = "finalized"
        return self.reply()


def prepared_manifest():
    new_server = replace(SERVER, workspace_id="b5abc8e3-b12e-4caa-9238-280394803454")
    return ResearchCopyManifest(LOCAL, new_server, (item(),), TIME)


async def test_verified_item_receipts_replay_and_finalize_once_on_the_exact_server():
    manifest = prepared_manifest()
    server = RecordingServer()

    def client_for(ref):
        assert ref == manifest.destination
        return server

    service = ServerMigrationReceiptService(client_for)
    state = await service.begin(manifest, target_name="New Research copy")
    assert state.missing_chunk_ids == (
        manifest.item_idempotency_key(manifest.items[0]),
    )
    item_ref = manifest.items[0]
    destination = CopyOwnerRef(manifest.destination, "media", "57", version=3)
    receipt = CopyItemReceipt(
        manifest.item_idempotency_key(item_ref),
        item_ref,
        CopyItemStage.VERIFIED,
        destination,
        CopyReceiptFailure.NONE,
        False,
        False,
    )
    with pytest.raises(ValueError, match="verified"):
        await service.ack_item(
            manifest, replace(receipt, stage=CopyItemStage.ASSOCIATED)
        )
    with pytest.raises(ValueError, match="missing"):
        await service.finalize(manifest)
    saved = await service.ack_item(manifest, receipt)
    assert saved.metadata["destination_owner_id"] == "57"
    assert await service.ack_item(manifest, receipt) == saved
    assert (await service.finalize(manifest)).status == "finalized"
    assert (
        await service.begin(manifest, target_name="New Research copy")
    ).status == "finalized"
    assert sum(method == "PUT" for method, _ in server.requests) == 1
    assert all("client-delete-ack" not in path for _, path in server.requests)


async def test_server_migration_does_not_modify_an_existing_destination():
    manifest = prepared_manifest()
    server = RecordingServer()
    server.workspace = {
        "id": manifest.destination.workspace_id,
        "study_materials_policy": "general",
    }
    service = ServerMigrationReceiptService(lambda _ref: server)
    with pytest.raises(ValueError, match="new Server workspace"):
        await service.begin(manifest, target_name="Existing research")
    assert all(method == "GET" for method, _ in server.requests)
