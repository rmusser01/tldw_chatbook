from dataclasses import replace

import pytest

from tldw_chatbook.Research_Workspace.chat_models import (
    ResearchChatMode,
    ResearchChatTurnOutcome,
    ResearchChatTurnRequest,
    ResearchRetrievalSettings,
    ResearchSourceVersionRef,
)
from tldw_chatbook.Research_Workspace.contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    RetrievalMode,
    WorkspaceDataSource,
)


LOCAL = QualifiedWorkspaceRef("local", "workspace-a")
SOURCE = ResearchSourceVersionRef(
    LOCAL,
    "association-a",
    "42",
    "article",
    catalog_item_version=5,
    ready_modes=(RetrievalMode.FTS,),
)


def turn(**changes):
    request = ResearchChatTurnRequest(
        workspace=LOCAL,
        conversation_id="conversation-a",
        user_text="What does the study show?",
        mode=ResearchChatMode.AUTO,
        source_snapshot=(SOURCE,),
        retrieval=ResearchRetrievalSettings(mode=RetrievalMode.HYBRID),
        processing_route=ProcessingRoute(WorkspaceDataSource.LOCAL, "local-model"),
        context_revision=3,
    )
    return replace(request, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(mode="unknown"),
        dict(top_k=0),
        dict(threshold=float("nan")),
        dict(history_limit=101),
        dict(rerank="true"),
    ],
)
def test_invalid_retrieval_settings_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        ResearchRetrievalSettings(**changes)


def test_turn_fences_workspace_and_selection_revision():
    other = QualifiedWorkspaceRef("local", "workspace-b")
    with pytest.raises(ValueError, match="workspace"):
        turn(source_snapshot=(replace(SOURCE, ref=other),))
    with pytest.raises(TypeError, match="source"):
        turn(source_snapshot=("42",))
    with pytest.raises(ValueError, match="mode"):
        turn(mode="unknown")
    with pytest.raises(ValueError, match="draft"):
        turn(user_text="é" * 33000)
    request = turn()
    assert request.is_current(workspace=LOCAL, context_revision=3)
    assert not request.is_current(workspace=LOCAL, context_revision=4)
    assert not request.is_current(workspace=other, context_revision=3)


def test_general_does_not_retrieve_and_grounded_requires_ready_sources():
    assert turn(mode=ResearchChatMode.GENERAL).retrieval_sources == ()
    with pytest.raises(ValueError, match="ready"):
        turn(
            mode=ResearchChatMode.RAG,
            source_snapshot=(replace(SOURCE, ready_modes=()),),
        )
    assert turn(mode=ResearchChatMode.RAG).effective_retrieval is RetrievalMode.FTS
    assert turn(mode=ResearchChatMode.RAG).retrieval_sources == (SOURCE,)


@pytest.mark.parametrize("retrieved", [True, False])
def test_auto_outcome_discloses_actual_retrieval(retrieved):
    request = turn()
    outcome = ResearchChatTurnOutcome.from_request(
        request,
        status="complete",
        retrieved=retrieved,
    )
    assert outcome.retrieval_disclosure == (
        "Retrieved" if retrieved else "Did not retrieve"
    )
    assert outcome.requested_retrieval is RetrievalMode.HYBRID
    assert outcome.effective_retrieval is RetrievalMode.FTS
    assert outcome.source_snapshot == (SOURCE,)
    if retrieved:
        with pytest.raises(ValueError, match="retrieval"):
            ResearchChatTurnOutcome.from_request(
                turn(mode=ResearchChatMode.GENERAL), status="complete", retrieved=True
            )
