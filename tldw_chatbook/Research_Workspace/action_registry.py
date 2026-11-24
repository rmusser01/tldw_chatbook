"""Definitions and exact readiness for the audited Research Studio actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from .chat_models import ResearchSourceVersionRef
from .contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCapability,
    WorkspaceDataSource,
)
from .studio_mapping import PRIMARY_OUTPUT_MAPPINGS
from .studio_models import (
    PRIMARY_RESEARCH_OUTPUT_KINDS,
    ResearchOutputKind,
    ResearchWorkProductKind,
    validate_studio_sources,
)


ResearchStudioAction = ResearchOutputKind | ResearchWorkProductKind
ResearchActionGroup = Literal["Learn", "Analyze", "Present", "Work products"]
LOCAL = WorkspaceDataSource.LOCAL
SERVER = WorkspaceDataSource.SERVER


@dataclass(frozen=True, slots=True)
class ResearchActionOwner:
    label: str
    required_capabilities: tuple[str, ...]
    reopen_target: str
    recovery: str
    requires_media: bool = False


@dataclass(frozen=True, slots=True)
class ResearchActionDefinition:
    kind: ResearchStudioAction
    label: str
    group: ResearchActionGroup
    owner_by_authority: Mapping[WorkspaceDataSource, ResearchActionOwner]
    option_schema_id: str

    @property
    def action_id(self) -> str:
        return self.kind.value


@dataclass(frozen=True, slots=True)
class PlannedOutputLabel:
    label: str


PLANNED_OUTPUTS = (
    PlannedOutputLabel("Research Dossier"),
    PlannedOutputLabel("Competitive Market Memo"),
    PlannedOutputLabel("Technical Project Spec"),
)


def _primary(kind: ResearchOutputKind) -> ResearchActionDefinition:
    owners = {}
    for authority in WorkspaceDataSource:
        owner = PRIMARY_OUTPUT_MAPPINGS[authority, kind]
        owners[authority] = ResearchActionOwner(
            owner.owner_label,
            (owner.create_capability,),
            owner.reopen_destination,
            "Refresh Studio capabilities and verify access to this owner.",
        )
    group: ResearchActionGroup = (
        "Learn" if kind in PRIMARY_RESEARCH_OUTPUT_KINDS[:3] else "Analyze"
    )
    return ResearchActionDefinition(
        kind,
        kind.value.replace("_", " ").title(),
        group,
        MappingProxyType(owners),
        kind.value,
    )


def _extension(
    kind: ResearchStudioAction,
    label: str,
    group: ResearchActionGroup,
    local: ResearchActionOwner | None,
    server: ResearchActionOwner,
    *,
    options: str,
) -> ResearchActionDefinition:
    unavailable = ResearchActionOwner(
        f"Local {label}",
        (),
        "",
        "Choose a Server workspace with a supported owner for this output.",
    )
    return ResearchActionDefinition(
        kind,
        label,
        group,
        MappingProxyType({LOCAL: local or unavailable, SERVER: server}),
        options,
    )


def _grounded(
    kind: ResearchStudioAction, owner: ResearchActionOwner
) -> ResearchActionOwner:
    capabilities = (f"generate_{kind.value}", *owner.required_capabilities)
    return replace(owner, required_capabilities=capabilities)


_SERVER_TEXT = ResearchActionOwner(
    "Server workspace artifacts",
    ("save_workspace_artifact",),
    "research_workspace",
    "Check the Server provider and workspace artifact permissions.",
)
_LOCAL_TEXT = ResearchActionOwner(
    "Local Artifacts",
    ("create_chatbook",),
    "artifacts",
    "Choose a Local processing route and verify Artifacts storage.",
)
_SERVER_TABLE = ResearchActionOwner(
    "Server Data Tables",
    ("generate_data_table", "get_data_table", "save_workspace_artifact"),
    "data_tables",
    "Check Data Tables access and select ready Server Media.",
    requires_media=True,
)
_SERVER_SLIDES = ResearchActionOwner(
    "Server Slides",
    ("generate_presentation_from_media", "get_presentation", "save_workspace_artifact"),
    "slides",
    "Check Slides access and select ready Server Media.",
    requires_media=True,
)
_LOCAL_AUDIO = ResearchActionOwner(
    "Local TTS history",
    ("generate_grounded_artifact", "create_audio_speech", "get_tts_history_entry"),
    "tts",
    "Select a Local TTS provider and voice with writable history.",
)
_SERVER_AUDIO = ResearchActionOwner(
    "Server TTS history",
    (
        "generate_grounded_artifact",
        "create_audio_speech_job",
        "get_tts_history_entry",
        "save_workspace_artifact",
    ),
    "tts",
    "Select a Server TTS provider and voice with an inspectable saved result.",
)


def _work_product(
    kind: ResearchWorkProductKind, label: str, *, table=False
) -> ResearchActionDefinition:
    return _extension(
        kind,
        label,
        "Work products",
        None if table else _grounded(kind, _LOCAL_TEXT),
        _grounded(kind, _SERVER_TABLE if table else _SERVER_TEXT),
        options="data_table" if table else "report",
    )


_ACTIONS = (
    *(_primary(kind) for kind in PRIMARY_RESEARCH_OUTPUT_KINDS),
    _extension(
        ResearchOutputKind.MIND_MAP,
        "Mind Map",
        "Analyze",
        None,
        _grounded(ResearchOutputKind.MIND_MAP, _SERVER_TEXT),
        options="diagram",
    ),
    _extension(
        ResearchOutputKind.TIMELINE,
        "Timeline",
        "Analyze",
        None,
        _grounded(ResearchOutputKind.TIMELINE, _SERVER_TEXT),
        options="diagram",
    ),
    _extension(
        ResearchOutputKind.DATA_TABLE,
        "Data Table",
        "Analyze",
        None,
        _SERVER_TABLE,
        options="data_table",
    ),
    _extension(
        ResearchOutputKind.SLIDES,
        "Slides",
        "Present",
        None,
        _SERVER_SLIDES,
        options="slides",
    ),
    _extension(
        ResearchOutputKind.AUDIO_SUMMARY,
        "Audio Summary",
        "Present",
        _LOCAL_AUDIO,
        _SERVER_AUDIO,
        options="audio",
    ),
    _work_product(ResearchWorkProductKind.EXECUTIVE_BRIEF, "Executive Brief"),
    _work_product(
        ResearchWorkProductKind.LITERATURE_MATRIX, "Literature Matrix", table=True
    ),
    _work_product(
        ResearchWorkProductKind.CORPUS_GAP_FINDER, "Corpus Gap Finder", table=True
    ),
    _work_product(
        ResearchWorkProductKind.EVIDENCE_BOUND_HYPOTHESES, "Evidence-Bound Hypotheses"
    ),
    _work_product(
        ResearchWorkProductKind.RESEARCH_PROPOSAL_PACK, "Research Proposal Pack"
    ),
)

RESEARCH_ACTIONS: Mapping[str, ResearchActionDefinition] = MappingProxyType(
    {row.action_id: row for row in _ACTIONS}
)


@dataclass(frozen=True, slots=True)
class ResearchActionContext:
    workspace: QualifiedWorkspaceRef
    capability_revision: str
    processing_route: ProcessingRoute | None
    source_snapshot: tuple[ResearchSourceVersionRef, ...]

    def __post_init__(self) -> None:
        validate_studio_sources(self.workspace, self.source_snapshot)
        if (
            self.processing_route
            and self.processing_route.data_source != self.workspace.data_source
        ):
            raise ValueError(
                "Studio capability probes require the selected authority's route"
            )


@dataclass(frozen=True, slots=True)
class ResearchActionProbes:
    context: ResearchActionContext
    capabilities: Mapping[str, ResearchCapability]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )


def project_research_actions(
    context: ResearchActionContext, probes: ResearchActionProbes | None
) -> dict[str, ResearchCapability]:
    """Project only capabilities captured with the same authority, route, and source versions."""
    ready = tuple(source for source in context.source_snapshot if source.ready_modes)
    return {
        action_id: _project(action, context, probes, ready)
        for action_id, action in RESEARCH_ACTIONS.items()
    }


def _project(action, context, probes, ready) -> ResearchCapability:
    owner = action.owner_by_authority[context.workspace.data_source]

    def unavailable(reason, message, recovery=owner.recovery):
        return ResearchCapability(
            False, reason, message, owner.label, recovery, context.capability_revision
        )

    if not owner.required_capabilities:
        return unavailable(
            "studio_owner_unavailable",
            "This authority has no canonical owner for the output.",
        )
    if probes is None or probes.context != context:
        return unavailable(
            "studio_capability_stale",
            "Refresh the capability check for this workspace and source selection.",
        )
    for name in owner.required_capabilities:
        capability = probes.capabilities.get(name)
        if capability is None:
            return unavailable(
                "studio_capability_unknown",
                "The exact generator or owner capability is unavailable.",
            )
        if (
            capability.capability_revision
            and capability.capability_revision != context.capability_revision
        ):
            return unavailable(
                "studio_capability_stale",
                "A required owner capability changed. Refresh it before continuing.",
            )
        if not capability.available:
            return unavailable(
                capability.reason_code,
                capability.user_message,
                capability.recovery_action or owner.recovery,
            )
    if context.processing_route is None:
        return unavailable(
            "processing_route_unavailable",
            "No processing route is ready.",
            "Choose an available provider and model.",
        )
    if len(ready) < action.kind.minimum_sources:
        return unavailable(
            "insufficient_ready_sources",
            "More ready sources are required.",
            f"Select at least {action.kind.minimum_sources} ready sources.",
        )
    if (
        owner.requires_media
        and len(tuple(source for source in ready if _media_id(source.catalog_item_id)))
        < action.kind.minimum_sources
    ):
        return unavailable(
            "studio_source_owner_unavailable",
            "Select enough ready sources with native Media IDs.",
        )
    return ResearchCapability(
        True,
        "available",
        f"{owner.label} is ready.",
        owner.label,
        capability_revision=context.capability_revision,
    )


def _media_id(value: str) -> bool:
    return (
        value.isascii()
        and value.isdecimal()
        and len(value) <= 19
        and 0 < int(value) < 2**63
    )
