"""Authority, source versions, and options captured for Studio generation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from .chat_models import (
    MAX_CHAT_SOURCES,
    ResearchRetrievalSettings,
    ResearchSourceVersionRef,
    ResearchTurnStatus,
    validate_chat_draft,
    validate_chat_id,
)
from .contracts import (
    BoundedPageResult,
    ProcessingRoute,
    QualifiedWorkspaceRef,
    RetrievalMode,
    _safe_identity,
)
from .deep_research_context import ResearchWorkspaceRunRef


class ResearchOutputKind(StrEnum):
    SUMMARY = "summary"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    REPORT = "report"
    COMPARE_SOURCES = "compare_sources"
    MIND_MAP = "mind_map"
    TIMELINE = "timeline"
    DATA_TABLE = "data_table"
    SLIDES = "slides"
    AUDIO_SUMMARY = "audio_summary"

    @property
    def minimum_sources(self) -> int:
        return 2 if self is self.COMPARE_SOURCES else 1


PRIMARY_RESEARCH_OUTPUT_KINDS = (
    ResearchOutputKind.SUMMARY,
    ResearchOutputKind.FLASHCARDS,
    ResearchOutputKind.QUIZ,
    ResearchOutputKind.REPORT,
    ResearchOutputKind.COMPARE_SOURCES,
)


class ResearchWorkProductKind(StrEnum):
    EXECUTIVE_BRIEF = "executive_brief"
    LITERATURE_MATRIX = "literature_matrix"
    CORPUS_GAP_FINDER = "corpus_gap_finder"
    EVIDENCE_BOUND_HYPOTHESES = "evidence_bound_hypotheses"
    RESEARCH_PROPOSAL_PACK = "research_proposal_pack"

    @property
    def minimum_sources(self) -> int:
        return 1 if self is self.EXECUTIVE_BRIEF else 2


StudioOutputKind = ResearchOutputKind | ResearchWorkProductKind


def studio_output_kind(value: str) -> StudioOutputKind:
    try:
        return ResearchOutputKind(value)
    except ValueError:
        return ResearchWorkProductKind(value)


class StudioOwnerKind(StrEnum):
    LOCAL_CHATBOOK = "local_chatbook"
    LOCAL_STUDY_DECK = "local_study_deck"
    LOCAL_QUIZ = "local_quiz"
    SERVER_ARTIFACT = "server_artifact"
    SERVER_STUDY_DECK = "server_study_deck"
    SERVER_QUIZ = "server_quiz"


@dataclass(frozen=True, slots=True)
class ResearchGenerationConfig:
    title: str = ""
    focus: str = field(default="", repr=False)
    detail: Literal["brief", "standard", "detailed"] = "standard"
    item_count: int = 8
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 4096
    retrieval: ResearchRetrievalSettings = field(
        default_factory=lambda: ResearchRetrievalSettings(history_limit=0)
    )

    def __post_init__(self) -> None:
        title = _safe_identity(self.title, "output title")
        if len(title) > 120 or any(char in title for char in "\n\r\t"):
            raise ValueError(
                "Output title must fit on one line of at most 120 characters"
            )
        object.__setattr__(self, "title", title)
        focus = validate_chat_draft(self.focus).strip()
        if len(focus) > 512:
            raise ValueError("The retrieval focus must be at most 512 characters")
        object.__setattr__(self, "focus", focus)
        if self.detail not in {"brief", "standard", "detailed"}:
            raise ValueError("Output detail is invalid")
        if type(self.item_count) is not int or not 1 <= self.item_count <= 50:
            raise ValueError("Choose between 1 and 50 study items")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 16384:
            raise ValueError("Choose between 1 and 16384 output tokens")
        if type(self.temperature) not in {float, int} or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if type(self.top_p) not in {float, int} or not 0 < self.top_p <= 1:
            raise ValueError("Top P must be greater than 0 and at most 1")
        if not isinstance(self.retrieval, ResearchRetrievalSettings):
            raise TypeError("Studio retrieval options must be validated")

    @property
    def fingerprint(self) -> str:
        """Identify generation options without persisting user-supplied labels."""
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def validate_studio_sources(
    workspace: QualifiedWorkspaceRef, sources: tuple[ResearchSourceVersionRef, ...]
) -> None:
    if not isinstance(workspace, QualifiedWorkspaceRef):
        raise TypeError("Studio requires a qualified workspace")
    if not isinstance(sources, tuple) or len(sources) > MAX_CHAT_SOURCES:
        raise ValueError(f"Select at most {MAX_CHAT_SOURCES} sources for this output")
    if any(not isinstance(row, ResearchSourceVersionRef) for row in sources):
        raise TypeError("Studio requires qualified source versions")
    if any(row.ref != workspace for row in sources):
        raise ValueError("A selected source belongs to a different workspace")
    if len({row.source_id for row in sources}) != len(sources) or len(
        {row.catalog_item_id for row in sources}
    ) != len(sources):
        raise ValueError(
            "Selected sources contain a duplicate association or Media item"
        )


def studio_retrieval(
    kind: StudioOutputKind,
    sources: tuple[ResearchSourceVersionRef, ...],
    requested: RetrievalMode,
) -> tuple[RetrievalMode | None, tuple[ResearchSourceVersionRef, ...]]:
    """Require enough distinct sources for one retrieval mode."""
    for mode in dict.fromkeys((requested, RetrievalMode.FTS)):
        ready = tuple(row for row in sources if mode in row.ready_modes)
        if len(ready) >= kind.minimum_sources:
            return mode, ready
    return None, ()


@dataclass(frozen=True, slots=True)
class ResearchStudioRequest:
    operation_id: str
    workspace: QualifiedWorkspaceRef
    output_kind: StudioOutputKind
    source_snapshot: tuple[ResearchSourceVersionRef, ...]
    generation: ResearchGenerationConfig
    processing_route: ProcessingRoute
    context_revision: int
    research_run_ref: ResearchWorkspaceRunRef | None = None

    def __post_init__(self) -> None:
        from .studio_mapping import STUDIO_OUTPUT_MAPPINGS
        from .work_product_templates import TEXT_WORK_PRODUCT_TEMPLATES

        validate_studio_sources(self.workspace, self.source_snapshot)
        object.__setattr__(
            self, "operation_id", validate_chat_id(self.operation_id, "operation ID")
        )
        kind = studio_output_kind(self.output_kind)
        if (self.workspace.data_source, kind) not in STUDIO_OUTPUT_MAPPINGS:
            raise ValueError("Choose an available Studio output owner")
        object.__setattr__(self, "output_kind", kind)
        if self.research_run_ref is not None and (
            not isinstance(self.research_run_ref, ResearchWorkspaceRunRef)
            or self.research_run_ref.origin != self.workspace
            or kind is not ResearchOutputKind.REPORT
        ):
            raise ValueError(
                "A Research run can only draft a Report in its origin workspace"
            )
        if not isinstance(self.generation, ResearchGenerationConfig):
            raise TypeError("Studio generation options must be validated")
        if (
            kind in TEXT_WORK_PRODUCT_TEMPLATES
            and not self.generation.retrieval.citations
        ):
            raise ValueError(
                "Work products require citations to their selected sources"
            )
        if (
            not isinstance(self.processing_route, ProcessingRoute)
            or self.processing_route.data_source is not self.workspace.data_source
        ):
            raise ValueError(
                "The processing route belongs to a different workspace authority"
            )
        if type(self.context_revision) is not int or self.context_revision < 0:
            raise ValueError("Context revision must be non-negative")
        if self.generation.retrieval.top_k < kind.minimum_sources:
            raise ValueError(
                f"Choose at least {kind.minimum_sources} retrieval results for this output"
            )
        if self.effective_retrieval is None:
            minimum = kind.minimum_sources
            noun = "source" if minimum == 1 else "sources"
            raise ValueError(
                f"This output requires at least {minimum} selected ready {noun}"
            )

    @property
    def effective_retrieval(self) -> RetrievalMode | None:
        return studio_retrieval(
            self.output_kind, self.source_snapshot, self.generation.retrieval.mode
        )[0]

    @property
    def retrieval_sources(self) -> tuple[ResearchSourceVersionRef, ...]:
        return studio_retrieval(
            self.output_kind, self.source_snapshot, self.generation.retrieval.mode
        )[1]

    def is_current(
        self, *, workspace: QualifiedWorkspaceRef, context_revision: int
    ) -> bool:
        return workspace == self.workspace and context_revision == self.context_revision


@dataclass(frozen=True, slots=True)
class WorkspaceOutputRef:
    workspace: QualifiedWorkspaceRef
    output_kind: StudioOutputKind
    owner_kind: StudioOwnerKind
    owner_id: str
    owner_version: int | None
    source_snapshot: tuple[ResearchSourceVersionRef, ...]
    generation_fingerprint: str
    status: ResearchTurnStatus

    def __post_init__(self) -> None:
        from .studio_mapping import STUDIO_OUTPUT_MAPPINGS

        validate_studio_sources(self.workspace, self.source_snapshot)
        object.__setattr__(self, "output_kind", studio_output_kind(self.output_kind))
        owner = StudioOwnerKind(self.owner_kind)
        object.__setattr__(self, "owner_kind", owner)
        mapping = STUDIO_OUTPUT_MAPPINGS.get(
            (self.workspace.data_source, self.output_kind)
        )
        if mapping is None or owner is not mapping.owner_kind:
            raise ValueError(
                "Output owner does not match this workspace and output kind"
            )
        if self.status not in {"complete", "cancelled", "failed"}:
            raise ValueError("Output status is invalid")
        object.__setattr__(
            self,
            "owner_id",
            validate_chat_id(
                self.owner_id, "owner ID", allow_empty=self.status != "complete"
            ),
        )
        if self.owner_version is not None and (
            type(self.owner_version) is not int or self.owner_version < 0
        ):
            raise ValueError("Output owner revision must be non-negative")
        if not isinstance(self.generation_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.generation_fingerprint
        ):
            raise ValueError("Generation fingerprint must be a SHA-256 digest")


def validate_output_page(
    workspace: QualifiedWorkspaceRef, page: BoundedPageResult[WorkspaceOutputRef]
) -> BoundedPageResult[WorkspaceOutputRef]:
    """Validate references from a canonical history page before publication."""
    if any(row.workspace != workspace for row in page.items):
        raise ValueError("An output belongs to a different workspace")
    owners = {(row.owner_kind, row.owner_id) for row in page.items}
    if len(owners) != len(page.items):
        raise ValueError("Output history contains a duplicate canonical owner")
    return page
