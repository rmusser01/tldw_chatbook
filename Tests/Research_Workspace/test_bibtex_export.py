from dataclasses import replace

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Research_Workspace.bibtex_export import (
    prepare_workspace_bibtex,
    render_bibtex,
)
from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    ResearchSourceSummary,
)
from tldw_chatbook.Research_Workspace.local_adapter import LocalResearchWorkspaceAdapter
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


def test_bibtex_escapes_text_preserves_published_date_and_deduplicates_native_item():
    ref = QualifiedWorkspaceRef("local", "field")
    source = ResearchSourceSummary(
        ref, "link-1", "Estuary {samples} & 90%", "article", "42"
    )
    good = {
        "author": "A. Reader",
        "published_at": "2022-05-06",
        "url": "https://example.invalid/reports/42",
    }
    bibliography = render_bibtex(
        ref, [(source, good), (replace(source, source_id="link-2"), {})]
    )
    assert bibliography.count("@misc{") == 1
    assert r"Estuary \{samples\} \& 90\%" in bibliography
    assert "author = {A. Reader}" in bibliography
    assert "year = {2022}" in bibliography
    assert "url = {https://example.invalid/reports/42}" in bibliography
    assert "Incomplete" not in bibliography
    assert (
        bibliography.splitlines()[0]
        == render_bibtex(
            ref, [(replace(source, title="New title"), good)]
        ).splitlines()[0]
    )


def test_bibtex_omits_private_urls_and_labels_missing_metadata():
    ref = QualifiedWorkspaceRef("server", "field", "profile", "operator")
    source = ResearchSourceSummary(ref, "link-1", "Source title", "document", "42")
    bibliography = render_bibtex(
        ref, [(source, {"url": "https://example.invalid/?access_key=dummy"})]
    )
    assert "access_key" not in bibliography
    assert "Incomplete metadata" in bibliography
    with pytest.raises(ValueError, match="different workspace"):
        render_bibtex(replace(ref, principal_id="other"), [(source, {})])


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_local_bibliography_reads_real_media_and_omits_device_only_items(
    tmp_path,
):
    workspace_db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    media = MediaDatabase(tmp_path / "media.sqlite", client_id="bibliography-test")
    registry = LocalWorkspaceRegistryService(workspace_db)
    ref = QualifiedWorkspaceRef("local", "field")
    registry.create_workspace(workspace_id=ref.workspace_id, name="Study group")
    try:
        for title, policy in (
            ("Published field notes", "metadata-only"),
            ("Private observations", "local-only"),
        ):
            owner_id, _, _ = media.add_media_with_keywords(
                title=title,
                media_type="document",
                content=title,
                chunks=[{"text": title, "start_char": 0, "end_char": len(title)}],
            )
            assert owner_id is not None
            registry.link_membership(
                ref.workspace_id,
                item_type="media",
                item_id=str(owner_id),
                transfer_policy=policy,
            )
        port = LocalResearchWorkspaceAdapter(
            registry,
            media_scope_service=MediaReadingScopeService(
                LocalMediaReadingService(media), None
            ),
        )
        bibliography = await prepare_workspace_bibtex(ref, port, registry)
        await bibliography.confirm_current_policy(registry)
        assert "Published field notes" in bibliography.text
        assert "Private observations" not in bibliography.text
    finally:
        media.close()
        workspace_db.close()
