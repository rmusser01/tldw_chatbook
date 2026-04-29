from __future__ import annotations

from tldw_chatbook.Sync_Interop.sync_mirror_report import build_sync_mirror_report
from tldw_chatbook.runtime_policy.server_parity_models import SyncIdentityMapEntry


class RemoteServiceShouldNotBeCalled:
    def create(self, *args, **kwargs):
        raise AssertionError("dry-run mirror report must not create remote records")

    def update(self, *args, **kwargs):
        raise AssertionError("dry-run mirror report must not update remote records")

    def delete(self, *args, **kwargs):
        raise AssertionError("dry-run mirror report must not delete remote records")


def test_mirror_report_is_dry_run_and_uses_identity_map_entries() -> None:
    identity = SyncIdentityMapEntry(
        domain="notes",
        source_authority="server",
        source_scope="workspace",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-1",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        remote_version="v1",
    )

    report = build_sync_mirror_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        identity_map=[identity],
        local_records=[{"id": "local-note-1", "version": "local-v2"}],
        remote_records=[{"id": "remote-note-1", "version": "v1"}],
        remote_service=RemoteServiceShouldNotBeCalled(),
    )

    assert report["dry_run"] is True
    assert report["write_enabled"] is False
    assert report["server_profile_id"] == "server-a"
    assert report["workspace_id"] == "workspace-1"
    assert report["mapped_count"] == 1
    assert report["actions"][0]["action"] == "would_compare"
    assert report["actions"][0]["identity"]["local_entity_id"] == "local-note-1"


def test_mirror_report_preserves_workspace_boundaries_when_filtering_identity_map() -> None:
    matching = SyncIdentityMapEntry(
        domain="notes",
        source_authority="server",
        source_scope="workspace",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-1",
        server_profile_id="server-a",
        workspace_id="workspace-a",
    )
    other_workspace = SyncIdentityMapEntry(
        domain="notes",
        source_authority="server",
        source_scope="workspace",
        local_entity_id="local-note-2",
        remote_entity_id="remote-note-2",
        server_profile_id="server-a",
        workspace_id="workspace-b",
    )

    report = build_sync_mirror_report(
        domain="notes",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        identity_map=[matching, other_workspace],
    )

    assert report["mapped_count"] == 1
    assert report["actions"][0]["identity"]["workspace_id"] == "workspace-a"

