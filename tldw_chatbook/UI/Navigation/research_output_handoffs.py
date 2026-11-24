"""Route one Studio output to its exact canonical artifact or Study owner."""

from ...Research_Workspace.contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from ...runtime_policy.server_context import ServerContextError
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
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


def study_target_authority_current(app, ref: QualifiedWorkspaceRef) -> bool:
    """Check the live Study backend and exact Server profile and account."""
    if app.get_authoritative_runtime_source() != ref.data_source.value:
        return False
    if ref.data_source is WorkspaceDataSource.LOCAL:
        return True
    try:
        context = app.server_context_provider.get_active_context()
    except ServerContextError:
        return False
    return (
        context.active_server_id == ref.server_profile_id
        and event_principal_id_from_active_context(context) == ref.principal_id
    )
