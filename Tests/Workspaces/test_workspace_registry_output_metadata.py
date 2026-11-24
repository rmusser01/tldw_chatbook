from dataclasses import replace

import pytest

from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.registry_service import WorkspaceMembershipConflict

pytestmark = pytest.mark.bootstrap_profile


def provenance(**updates):
    return {
        "research_generation": {
            "operation_id": "operation-1",
            "output_kind": "report",
            "source_snapshot": [
                {
                    "source_id": "source-1",
                    "catalog_item_id": "42",
                    "source_type": "document",
                    "association_version": 2,
                    "catalog_item_version": 3,
                    "ready_modes": ["fts"],
                }
            ],
            "settings_fingerprint": "a" * 64,
            "processing_route_fingerprint": "b" * 64,
            "provider": "fixture",
            "model": "model",
            "status": "complete",
            "owner_version": 1,
            **updates,
        }
    }


@pytest.fixture
def registry(tmp_path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    service = LocalWorkspaceRegistryService(
        db, now_factory=lambda: "2026-11-24T09:00:00+00:00"
    )
    service.create_workspace(workspace_id="reading", name="Reading")
    try:
        yield service
    finally:
        db.close()


def test_idempotent_metadata_link_and_versioned_update(registry):
    link = registry.link_membership(
        "reading", item_type="chatbook", item_id="5", role="output"
    )
    payload = replace(link, metadata=provenance())
    saved = registry.upsert_membership_metadata(payload)
    assert saved.updated_at is not None
    assert saved.metadata == provenance()
    assert registry.upsert_membership_metadata(payload) == saved
    assert (
        registry.link_membership(
            "reading", item_type="chatbook", item_id="5", role="output"
        )
        == saved
    )

    updated = registry.upsert_membership_metadata(
        replace(link, metadata=provenance(owner_version=2)),
        expected_updated_at=saved.updated_at,
    )
    assert updated.updated_at != saved.updated_at
    with pytest.raises(WorkspaceMembershipConflict):
        registry.upsert_membership_metadata(
            payload, expected_updated_at=saved.updated_at
        )
    assert registry.list_workspace_memberships("reading") == (updated,)


def test_output_metadata_rejects_private_fields_non_outputs_and_oversized_records(
    registry,
):
    source = registry.link_membership("reading", item_type="media", item_id="42")
    with pytest.raises(ValueError, match="output"):
        replace(source, metadata=provenance())

    link = registry.link_membership(
        "reading", item_type="chatbook", item_id="5", role="output"
    )
    for metadata in (
        provenance(content="private body"),
        provenance(provider="token=private"),
        {"unknown": {}},
    ):
        with pytest.raises(ValueError, match="provenance"):
            replace(link, metadata=metadata)

    large = [
        {
            "source_id": f"source-{index}-" + "a" * 230,
            "catalog_item_id": f"media-{index}-" + "b" * 230,
            "source_type": "document",
        }
        for index in range(100)
    ]
    with pytest.raises(ValueError, match="32 KiB"):
        replace(link, metadata=provenance(source_snapshot=large))

    candidate = replace(link, metadata=provenance())
    candidate.metadata["research_generation"]["unexpected"] = "private body"
    with pytest.raises(ValueError, match="provenance"):
        registry.upsert_membership_metadata(candidate)
    assert all(
        row.metadata == {} for row in registry.list_workspace_memberships("reading")
    )


@pytest.mark.asyncio
async def test_unlink_keeps_the_canonical_chatbook(registry, tmp_path):
    artifacts = LocalChatbookService(registry_path=tmp_path / "chatbooks.json")
    owner = await artifacts.create_chatbook(name="Saved report")
    item_id = str(owner["chatbook_id"])
    link = registry.link_membership(
        "reading", item_type="chatbook", item_id=item_id, role="output"
    )
    registry.upsert_membership_metadata(replace(link, metadata=provenance()))
    registry.unlink_membership(
        "reading", item_type="chatbook", item_id=item_id, role="output"
    )
    assert registry.list_workspace_memberships("reading") == ()
    assert (await artifacts.get_chatbook(item_id))["name"] == "Saved report"
