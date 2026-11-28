import pytest

from Tests.Research_Workspace.test_studio_models import LOCAL, SERVER, SOURCE
from tldw_chatbook.Research_Workspace.studio_models import WorkspaceOutputRef
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffValueError,
    PendingHandoffStore,
    ResearchStudyOutputTarget,
)
from tldw_chatbook.UI.Navigation.research_output_handoffs import output_handoff


def test_chatbook_reuses_exact_artifacts_channel():
    output = WorkspaceOutputRef(
        LOCAL, "report", "local_chatbook", "42", 1, (SOURCE,), "a" * 64, "complete"
    )
    channel, target = output_handoff(output)
    assert channel is HandoffChannel.ARTIFACT_CHATBOOK_TARGET
    store = PendingHandoffStore()
    store.stage(channel, target)
    assert store.claim(channel).value == "local:chatbook:42"


@pytest.mark.parametrize(
    ("kind", "owner", "channel"),
    [
        ("flashcards", "server_study_deck", "research_study_deck_target"),
        ("quiz", "server_quiz", "research_quiz_target"),
    ],
)
def test_study_handoff_preserves_the_qualified_server_and_uses_one_exact_id(
    kind, owner, channel
):
    output = WorkspaceOutputRef(
        SERVER, kind, owner, "study-item-45", 3, (), "a" * 64, "complete"
    )
    selected, target = output_handoff(output)
    assert selected.value == channel
    assert target == ResearchStudyOutputTarget(SERVER, "study-item-45")
    store = PendingHandoffStore()
    with pytest.raises(HandoffValueError):
        store.stage(selected, {"owner_id": "study-item-45"})
    store.stage(selected, target)
    claim = store.claim(selected)
    assert claim.value.workspace == SERVER
    assert claim.value.owner_id == "study-item-45"
    assert store.acknowledge(claim)
    assert store.claim(selected) is None
