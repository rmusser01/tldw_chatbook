"""Route one Studio output to its exact canonical artifact or Study owner."""

from ...Research_Workspace.studio_models import StudioOwnerKind, WorkspaceOutputRef
from .pending_handoff_store import (
    ARTIFACT_CHATBOOK_RECORD_PREFIX,
    HandoffChannel,
    ResearchStudyOutputTarget,
)


def output_handoff(
    output: WorkspaceOutputRef,
) -> tuple[HandoffChannel, str | ResearchStudyOutputTarget]:
    if output.status != "complete" or not output.owner_id:
        raise ValueError("Open a saved Studio output first")
    if output.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK:
        return (
            HandoffChannel.RESEARCH_CHATBOOK_TARGET,
            ARTIFACT_CHATBOOK_RECORD_PREFIX + output.owner_id,
        )
    targets = {
        StudioOwnerKind.LOCAL_STUDY_DECK: HandoffChannel.RESEARCH_STUDY_DECK_TARGET,
        StudioOwnerKind.SERVER_STUDY_DECK: HandoffChannel.RESEARCH_STUDY_DECK_TARGET,
        StudioOwnerKind.LOCAL_QUIZ: HandoffChannel.RESEARCH_QUIZ_TARGET,
        StudioOwnerKind.SERVER_QUIZ: HandoffChannel.RESEARCH_QUIZ_TARGET,
    }
    channel = targets.get(output.owner_kind)
    if channel is None:
        raise ValueError("Open this Server artifact in its Research workspace")
    return channel, ResearchStudyOutputTarget(output.workspace, output.owner_id)
