"""Canonical destinations and readiness for Research Studio outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from .chat_models import ResearchSourceVersionRef
from .contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCapability,
    RetrievalMode,
    WorkspaceDataSource,
)
from .studio_models import (
    PRIMARY_RESEARCH_OUTPUT_KINDS,
    ResearchOutputKind,
    ResearchWorkProductKind,
    StudioOutputKind,
    StudioOwnerKind,
    studio_retrieval,
    validate_studio_sources,
)
from .work_product_templates import TEXT_WORK_PRODUCT_TEMPLATES


@dataclass(frozen=True, slots=True)
class StudioOwnerMapping:
    owner_kind: StudioOwnerKind
    owner_label: str
    id_field: str
    version_field: str
    association: str
    create_capability: str
    replace_capability: str | None
    delete_capability: str | None
    reopen_destination: str
    reopen_channel: str
    membership_item_type: str | None = None
    artifact_type: str | None = None

    @property
    def supported_actions(self) -> frozenset[str]:
        actions = {"view", "reopen", "new_version"}
        if self.replace_capability:
            actions.update({"edit", "replace"})
        if self.delete_capability:
            actions.add("delete")
        return frozenset(actions)


_LOCAL_TEXT = StudioOwnerMapping(
    StudioOwnerKind.LOCAL_CHATBOOK,
    "Local Artifacts",
    "chatbook_id",
    "artifact_revision",
    "workspace_membership",
    "create_chatbook",
    "update_chatbook",
    "delete_chatbook",
    "artifacts",
    "artifact_chatbook_target",
    membership_item_type="chatbook",
)
_SERVER_TEXT = StudioOwnerMapping(
    StudioOwnerKind.SERVER_ARTIFACT,
    "Server workspace artifacts",
    "id",
    "version",
    "workspace_id",
    "save_workspace_artifact",
    "update_workspace_artifact",
    None,
    "research_workspace",
    "research_workspace_artifact_target",
)

PRIMARY_OUTPUT_MAPPINGS: Mapping[
    tuple[WorkspaceDataSource, ResearchOutputKind], StudioOwnerMapping
] = MappingProxyType(
    {
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.SUMMARY): _LOCAL_TEXT,
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.FLASHCARDS): StudioOwnerMapping(
            StudioOwnerKind.LOCAL_STUDY_DECK,
            "Local Study decks",
            "id",
            "version",
            "workspace_membership",
            "create_deck",
            None,
            "delete_deck",
            "study",
            "research_study_deck_target",
            membership_item_type="study_deck",
        ),
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.QUIZ): StudioOwnerMapping(
            StudioOwnerKind.LOCAL_QUIZ,
            "Local Study quizzes",
            "id",
            "version",
            "workspace_membership",
            "create_quiz",
            None,
            "delete_quiz",
            "study",
            "research_quiz_target",
            membership_item_type="study_quiz",
        ),
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.REPORT): _LOCAL_TEXT,
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.COMPARE_SOURCES): _LOCAL_TEXT,
        (WorkspaceDataSource.SERVER, ResearchOutputKind.SUMMARY): replace(
            _SERVER_TEXT, artifact_type="summary"
        ),
        (WorkspaceDataSource.SERVER, ResearchOutputKind.FLASHCARDS): StudioOwnerMapping(
            StudioOwnerKind.SERVER_STUDY_DECK,
            "Server Study decks",
            "id",
            "version",
            "workspace_id",
            "create_deck",
            None,
            None,
            "study",
            "research_study_deck_target",
        ),
        (WorkspaceDataSource.SERVER, ResearchOutputKind.QUIZ): StudioOwnerMapping(
            StudioOwnerKind.SERVER_QUIZ,
            "Server Study quizzes",
            "id",
            "version",
            "workspace_id",
            "create_quiz",
            None,
            None,
            "study",
            "research_quiz_target",
        ),
        (WorkspaceDataSource.SERVER, ResearchOutputKind.REPORT): replace(
            _SERVER_TEXT, artifact_type="report"
        ),
        (WorkspaceDataSource.SERVER, ResearchOutputKind.COMPARE_SOURCES): replace(
            _SERVER_TEXT, artifact_type="compare_sources"
        ),
    }
)

SERVER_DIAGRAM_TYPES = MappingProxyType(
    {ResearchOutputKind.MIND_MAP: "mindmap", ResearchOutputKind.TIMELINE: "timeline"}
)

SERVER_TABLE_KINDS = (
    ResearchOutputKind.DATA_TABLE,
    ResearchWorkProductKind.LITERATURE_MATRIX,
    ResearchWorkProductKind.CORPUS_GAP_FINDER,
)

SERVER_NATIVE_OWNERS = frozenset({
    StudioOwnerKind.SERVER_DATA_TABLE, StudioOwnerKind.SERVER_SLIDES, StudioOwnerKind.SERVER_AUDIO,
})

STUDIO_OUTPUT_MAPPINGS: Mapping[
    tuple[WorkspaceDataSource, StudioOutputKind], StudioOwnerMapping
] = MappingProxyType(
    {
        **PRIMARY_OUTPUT_MAPPINGS,
        (WorkspaceDataSource.SERVER, ResearchOutputKind.SLIDES): StudioOwnerMapping(
            StudioOwnerKind.SERVER_SLIDES, "Server Slides", "id", "version", "workspace_id",
            "generate_presentation_from_media", None, None,
            "research_workspace", "research_workspace_artifact_target", artifact_type="slides",
        ),
        (WorkspaceDataSource.SERVER, ResearchOutputKind.AUDIO_SUMMARY): StudioOwnerMapping(
            StudioOwnerKind.SERVER_AUDIO, "Server Speech outputs", "output_id", "version", "workspace_id",
            "create_audio_speech_job", None, None,
            "research_workspace", "research_workspace_artifact_target", artifact_type="audio_overview",
        ),
        (WorkspaceDataSource.LOCAL, ResearchOutputKind.AUDIO_SUMMARY): StudioOwnerMapping(
            StudioOwnerKind.LOCAL_AUDIO, "Local TTS history", "id", "version", "workspace_membership",
            "create_audio_speech", None, "delete_tts_history_entry",
            "research_workspace", "research_workspace_artifact_target", membership_item_type="tts_audio",
        ),
        **{
            (WorkspaceDataSource.SERVER, kind): StudioOwnerMapping(
                StudioOwnerKind.SERVER_DATA_TABLE,
                "Server Data Tables",
                "uuid",
                "version",
                "workspace_id",
                f"generate_{kind.value}",
                None,
                None,
                "research_workspace",
                "research_workspace_artifact_target",
                artifact_type="data_table",
            )
            for kind in SERVER_TABLE_KINDS
        },
        **{
            (WorkspaceDataSource.SERVER, kind): replace(
                _SERVER_TEXT,
                artifact_type=artifact_type,
                create_capability=f"generate_{kind.value}",
            )
            for kind, artifact_type in SERVER_DIAGRAM_TYPES.items()
        },
        **{
            (authority, kind): replace(
                owner, create_capability=f"generate_{kind.value}"
            )
            for authority, owner in (
                (WorkspaceDataSource.LOCAL, _LOCAL_TEXT),
                (
                    WorkspaceDataSource.SERVER,
                    replace(_SERVER_TEXT, artifact_type="report"),
                ),
            )
            for kind in TEXT_WORK_PRODUCT_TEMPLATES
        },
    }
)


@dataclass(frozen=True, slots=True)
class StudioAvailability:
    available: bool
    owner_label: str
    processing_route: ProcessingRoute | None
    reason_code: str
    user_message: str
    recovery: str
    capability_revision: str


def project_studio_availability(
    workspace: QualifiedWorkspaceRef,
    source_snapshot: tuple[ResearchSourceVersionRef, ...],
    route: ProcessingRoute | None,
    capabilities: Mapping[str, ResearchCapability],
    *,
    retrieval: RetrievalMode = RetrievalMode.FTS,
) -> dict[ResearchOutputKind, StudioAvailability]:
    """Project readiness using the actual selected owner and captured sources."""
    validate_studio_sources(workspace, source_snapshot)
    result = {}
    if route is not None and route.data_source is not workspace.data_source:
        route = None
    for kind in PRIMARY_RESEARCH_OUTPUT_KINDS:
        owner = PRIMARY_OUTPUT_MAPPINGS[workspace.data_source, kind]
        capability = capabilities.get(owner.create_capability)
        if capability is None:
            capability = ResearchCapability(
                False,
                "studio_capability_unknown",
                "The output owner is unavailable.",
                owner.owner_kind,
                recovery_action="Refresh workspace capabilities.",
            )
        if capability.available:
            mode, _ = studio_retrieval(kind, source_snapshot, retrieval)
            if route is None:
                capability = ResearchCapability(
                    False,
                    "processing_route_unavailable",
                    "No processing route is ready.",
                    owner.owner_kind,
                    "Choose an available provider and model.",
                    capability.capability_revision,
                )
            elif mode is None:
                recovery = (
                    "Select two ready sources."
                    if kind.minimum_sources == 2
                    else "Select a ready source."
                )
                capability = ResearchCapability(
                    False,
                    "insufficient_ready_sources",
                    "More ready sources are required.",
                    owner.owner_kind,
                    recovery,
                    capability.capability_revision,
                )
        result[kind] = StudioAvailability(
            capability.available,
            owner.owner_label,
            route,
            capability.reason_code,
            capability.user_message,
            capability.recovery_action,
            capability.capability_revision,
        )
    return result
