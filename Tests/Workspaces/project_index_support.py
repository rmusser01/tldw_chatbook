"""Synthetic, private folder-index fixtures shared by the delivery tests."""

from hashlib import sha256
from pathlib import Path

from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    WorkspaceProjectIndexDB,
    prepare_scan_page,
)
from tldw_chatbook.Tools.remote_root_types import LocalRoot
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexKey,
    ProjectIndexSearchAuthority,
    FileCapture,
    ScanEntry,
    ScanPage,
    locator_fingerprint,
    root_identity_fingerprint,
)


def make_index_key(tmp_path: Path) -> ProjectIndexKey:
    chain = capture_directory_chain(tmp_path)
    return ProjectIndexKey(
        "authority-1",
        "workspace-1",
        "binding-1",
        locator_fingerprint(tmp_path),
        root_identity_fingerprint(chain),
    )


def open_index_store(tmp_path: Path) -> WorkspaceProjectIndexDB:
    return WorkspaceProjectIndexDB(tmp_path / "workspace_project_index.db")


def page(text="uniqueword evidence", path="src/example.py", revision=1):
    data = text.encode()
    capture = FileCapture(
        path, "identity-1", 123, len(data), sha256(data).hexdigest(), text
    )
    return prepare_scan_page(
        ScanPage(
            "src",
            "directory-1",
            (ScanEntry(path, "file", "identity-1", 123, len(data)),),
            (capture,),
            True,
            revision,
        )
    )


def search_authority(tmp_path, scope, exclusions=()):
    return ProjectIndexSearchAuthority(
        scope,
        LocalRoot(tmp_path),
        capture_directory_chain(tmp_path),
        exclusions,
        1,
        lambda: True,
        None,
        lambda candidate: None,
    )


def open_semantic_store(tmp_path: Path) -> WorkspaceProjectIndexDB:
    """Use the real binding/namespace shape for semantic allocation tests."""
    key = make_index_key(tmp_path)
    directory = tmp_path / key.binding_namespace / key.namespace
    directory.mkdir(parents=True, exist_ok=True)
    return WorkspaceProjectIndexDB(directory / "workspace_project_index.db")
