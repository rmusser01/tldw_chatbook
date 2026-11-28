from dataclasses import replace

import pytest

from tldw_chatbook.Research_Workspace.chat_models import (
    ResearchRetrievalSettings,
    ResearchSourceVersionRef,
)
from tldw_chatbook.Research_Workspace.contracts import (
    BoundedPageResult,
    ProcessingRoute,
    QualifiedWorkspaceRef,
)
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchOutputKind,
    ResearchStudioRequest,
    StudioOwnerKind,
    WorkspaceOutputRef,
    validate_output_page,
)

LOCAL = QualifiedWorkspaceRef("local", "study")
SERVER = QualifiedWorkspaceRef("server", "study", "server-a", "reader")
ROUTE = ProcessingRoute(
    "local", "local-chat", "fixture", "model", endpoint_class="on_device"
)
SOURCE = ResearchSourceVersionRef(
    LOCAL, "source-1", "42", "document", ready_modes=("fts",)
)
SECOND = replace(SOURCE, source_id="source-2", catalog_item_id="43")


def test_compare_requires_two_distinct_ready_sources_in_the_effective_retrieval_mode():
    config = ResearchGenerationConfig(
        retrieval=ResearchRetrievalSettings(mode="hybrid")
    )
    with pytest.raises(ValueError, match="at least 2 selected ready sources"):
        ResearchStudioRequest(
            "operation", LOCAL, "compare_sources", (SOURCE,), config, ROUTE, 4
        )

    request = ResearchStudioRequest(
        "operation",
        LOCAL,
        "compare_sources",
        (SOURCE, SECOND),
        config,
        ROUTE,
        4,
    )
    assert request.effective_retrieval == "fts"
    assert request.retrieval_sources == (SOURCE, SECOND)
    assert request.is_current(workspace=LOCAL, context_revision=4)
    assert not request.is_current(workspace=LOCAL, context_revision=5)
    assert not request.is_current(workspace=SERVER, context_revision=4)
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            request, source_snapshot=(SOURCE, replace(SECOND, catalog_item_id="42"))
        )
    with pytest.raises(ValueError, match="workspace"):
        replace(request, source_snapshot=(SOURCE, replace(SECOND, ref=SERVER)))
    with pytest.raises(ValueError, match="processing route"):
        replace(request, processing_route=ProcessingRoute("server", "server-chat"))


def test_generation_options_are_bounded_and_fingerprints_change_only_with_the_configuration():
    base = ResearchGenerationConfig(title="Reading notes")
    assert (
        base.fingerprint == ResearchGenerationConfig(title="Reading notes").fingerprint
    )
    assert len(base.fingerprint) == 64
    assert replace(base, item_count=4).fingerprint != base.fingerprint
    for invalid in (
        {"temperature": float("nan")},
        {"top_p": 1.2},
        {"item_count": True},
        {"max_tokens": 0},
        {"title": "a" * 121},
        {"title": "token=private"},
    ):
        with pytest.raises(ValueError):
            replace(base, **invalid)


def test_output_page_refuses_a_duplicate_owner_or_a_different_workspace():
    output = WorkspaceOutputRef(
        LOCAL,
        ResearchOutputKind.SUMMARY,
        StudioOwnerKind.LOCAL_CHATBOOK,
        "chatbook-id",
        2,
        (SOURCE,),
        "a" * 64,
        "complete",
    )
    page = BoundedPageResult((output,), limit=20)
    assert validate_output_page(LOCAL, page) is page
    with pytest.raises(ValueError, match="duplicate canonical owner"):
        validate_output_page(LOCAL, BoundedPageResult((output, output), limit=20))
    with pytest.raises(ValueError, match="workspace"):
        validate_output_page(SERVER, page)
    with pytest.raises(ValueError, match="owner"):
        replace(output, owner_kind=StudioOwnerKind.SERVER_ARTIFACT)
    with pytest.raises(ValueError, match="owner revision"):
        replace(output, owner_version=True)
    with pytest.raises(ValueError, match="fingerprint"):
        replace(output, generation_fingerprint="private source text")
