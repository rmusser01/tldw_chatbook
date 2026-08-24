from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchSourceSummary,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
    require_capability,
)


@pytest.mark.parametrize("workspace_id", ["", "   "])
def test_qualified_workspace_ref_rejects_blank_workspace_ids(workspace_id: str) -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, workspace_id)


def test_server_ref_requires_profile_identity() -> None:
    with pytest.raises(ValueError, match="server_profile_id"):
        QualifiedWorkspaceRef(WorkspaceDataSource.SERVER, "workspace-1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("server_profile_id", "Bearer secret-value"),
        ("principal_id", "sk-secret-value"),
        ("principal_id", "api_key=secret-value"),
    ],
)
def test_qualified_workspace_ref_rejects_secret_looking_identity_metadata(
    field: str, value: str
) -> None:
    values = {
        "data_source": WorkspaceDataSource.SERVER,
        "workspace_id": "workspace-1",
        "server_profile_id": "profile-1",
        "principal_id": "principal-1",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        QualifiedWorkspaceRef(**values)


def test_local_ref_rejects_server_identity_metadata() -> None:
    with pytest.raises(ValueError, match="Local workspace refs"):
        QualifiedWorkspaceRef(
            WorkspaceDataSource.LOCAL,
            "workspace-1",
            server_profile_id="profile-1",
        )


def test_normalized_rows_are_frozen_and_authority_qualified() -> None:
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")
    workspace = ResearchWorkspaceSummary(ref=ref, name="Research")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="source-1",
        title="Paper",
        source_type="pdf",
    )

    assert workspace.ref == ref
    assert source.ref == ref
    with pytest.raises(FrozenInstanceError):
        workspace.name = "Changed"  # type: ignore[misc]


def test_bounded_page_rejects_unbounded_limits() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        BoundedPageResult(items=(), limit=101)


def test_bounded_page_rejects_more_rows_than_its_limit() -> None:
    with pytest.raises(ValueError, match="more items than limit"):
        BoundedPageResult(items=("one", "two"), limit=1)


def test_unknown_capability_fails_closed_with_typed_exact_capability() -> None:
    with pytest.raises(CapabilityUnavailableError) as exc_info:
        require_capability({}, "workspace.launch")

    assert exc_info.value.capability == ResearchCapability(
        available=False,
        reason_code="unknown_capability",
        user_message="This action is unavailable because its capability is unknown.",
        owner="research_workspace",
        recovery_action="Refresh capabilities or choose another action.",
    )
