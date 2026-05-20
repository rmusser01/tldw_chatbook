"""Workspace operating-context model tests."""

from __future__ import annotations

import pytest

from tldw_chatbook.Workspaces import (
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceMembership,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
)


def test_workspace_record_requires_non_empty_identity() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        WorkspaceRecord(workspace_id="", name="Research")

    with pytest.raises(ValueError, match="name"):
        WorkspaceRecord(workspace_id="ws-a", name="")


def test_workspace_record_validates_authority_and_sync_status() -> None:
    record = WorkspaceRecord(
        workspace_id="ws-a",
        name="Research",
        authority=WorkspaceAuthority.LOCAL_ONLY,
        sync_status=WorkspaceSyncStatus.NOT_CONFIGURED,
    )

    assert record.workspace_id == "ws-a"
    assert record.authority is WorkspaceAuthority.LOCAL_ONLY
    assert record.sync_status is WorkspaceSyncStatus.NOT_CONFIGURED


def test_workspace_membership_supports_multi_workspace_visibility() -> None:
    first = WorkspaceMembership(
        workspace_id="ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
    )
    second = WorkspaceMembership(
        workspace_id="ws-b",
        item_type="note",
        item_id="note-1",
        role="reference",
    )

    assert {first.workspace_id, second.workspace_id} == {"ws-a", "ws-b"}
    assert first.transfer_policy is WorkspaceTransferPolicy.REFERENCE


def test_workspace_runtime_binding_strips_secret_metadata() -> None:
    binding = WorkspaceRuntimeBinding(
        workspace_id="ws-a",
        binding_id="binding-1",
        binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
        label="Local project",
        locator="/tmp/project",
        status=RuntimeBindingStatus.READY,
        metadata={
            "branch": "dev",
            "api_key": "should-not-persist",
            "api-key": "should-not-persist",
            "privateKey": "should-not-persist",
            "nested": {"token": "also-secret", "safe": "value"},
        },
    )

    assert binding.metadata == {"branch": "dev", "nested": {"safe": "value"}}


def test_workspace_membership_requires_required_fields() -> None:
    with pytest.raises(ValueError, match="item_id"):
        WorkspaceMembership(workspace_id="ws-a", item_type="note", item_id="")

    with pytest.raises(ValueError, match="workspace_id"):
        WorkspaceRuntimeBinding(
            workspace_id="",
            binding_id="binding-1",
            binding_kind=RuntimeBindingKind.ACP_SESSION,
            label="Run",
            locator="acp://run/1",
        )
