from types import SimpleNamespace

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    ResearchWorkspaceSummary,
)
from tldw_chatbook.Research_Workspace.controller import ResearchWorkspaceCatalogState
from tldw_chatbook.Research_Workspace.workspace_search import search_workspace


@pytest.mark.asyncio
async def test_search_reads_current_authority_owner_pages_and_keeps_exact_ids():
    ref = QualifiedWorkspaceRef("server", "field", "test-server", "reader")
    item = ResearchWorkspaceSummary(ref, "Estuary fieldwork")
    calls = []

    class Port:
        async def list_sources(self, workspace, *, limit, offset):
            assert workspace == ref
            calls.append(offset)
            if offset == 0:
                return SimpleNamespace(
                    items=(
                        SimpleNamespace(ref=ref, title="Other source", source_id="12"),
                    ),
                    has_more=True,
                )
            return SimpleNamespace(
                items=(
                    SimpleNamespace(ref=ref, title="Estuary source", source_id="42"),
                ),
                has_more=False,
            )

        async def list_notes(self, workspace, page):
            assert workspace == ref
            return SimpleNamespace(
                items=(SimpleNamespace(ref=ref, title="Estuary notes", note_id="3"),),
                has_more=False,
            )

        async def list_outputs(self, workspace, *, limit, offset):
            assert workspace == ref
            return SimpleNamespace(
                items=(SimpleNamespace(workspace=ref, owner_id="report-1"),),
                has_more=False,
            )

        async def resolve_output(self, output):
            return {"title": "Estuary report"}

    catalog = ResearchWorkspaceCatalogState(ref.data_source, 1, 1, workspaces=(item,))
    matches = [hit async for hit in search_workspace(Port(), catalog, "estuary")]
    assert [hit.kind for hit in matches] == ["workspace", "source", "note", "output"]
    assert all(hit.workspace == ref for hit in matches)
    assert matches[1].item_id == "42" and matches[1].owner_offset == calls[-1]
    assert matches[2].item_id == "3" and matches[3].item_id == "report-1"

    calls.clear()
    notes = [hit async for hit in search_workspace(Port(), catalog, "note: estuary")]
    assert [hit.kind for hit in notes] == ["note"] and calls == []
