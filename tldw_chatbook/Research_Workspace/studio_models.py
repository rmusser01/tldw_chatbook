"""Authority, source versions, and options captured for Studio generation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from tldw_chatbook.Workspaces.output_membership import OutputCopyProvenance

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

COPYABLE_STUDIO_TEXT_KINDS = frozenset(
    {
        ResearchOutputKind.SUMMARY,
        ResearchOutputKind.REPORT,
        ResearchOutputKind.COMPARE_SOURCES,
    }
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
    SERVER_DATA_TABLE = "server_data_table"
    SERVER_SLIDES = "server_slides"
    SERVER_AUDIO = "server_audio"
    LOCAL_AUDIO = "local_audio"


STUDIO_SPEECH_FORMATS = frozenset({"aac", "flac", "mp3", "ogg", "opus", "wav"})


@dataclass(frozen=True, slots=True)
class StudioSpeechChoice:
    provider: str
    model: str
    voice: str
    format: str
    supports_speed: bool = False
    max_characters: int = 5000

    def __post_init__(self) -> None:
        for label in (self.provider, self.model, self.voice):
            if not isinstance(label, str) or not 1 <= len(label) <= 256 or any(
                ord(char) < 32 for char in label
            ):
                raise ValueError("Choose a bounded speech provider, model, and voice")
        if self.format not in STUDIO_SPEECH_FORMATS or type(self.supports_speed) is not bool:
            raise ValueError("Choose a supported speech format")
        if type(self.max_characters) is not int or not 1 <= self.max_characters <= 5000:
            raise ValueError("The speech input limit is invalid")


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

    table_columns: tuple[str, ...] = ()
    table_max_rows: int | None = None
    slides_visual_style: tuple[str, str] | None = None
    speech: StudioSpeechChoice | None = None
    speech_speed: float = 1.0

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
        if (
            not isinstance(self.table_columns, tuple) or len(self.table_columns) > 24
            or any(
                not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
                or any(ord(char) < 32 for char in name)
                for name in self.table_columns
            )
        ):
            raise ValueError("Enter up to 24 column names of 80 characters each")
        columns = tuple(name.strip() for name in self.table_columns)
        if len({name.casefold() for name in columns}) != len(columns):
            raise ValueError("Column names must be distinct")
        object.__setattr__(self, "table_columns", columns)
        if self.table_max_rows is not None and (
            type(self.table_max_rows) is not int or not 1 <= self.table_max_rows <= 2000
        ):
            raise ValueError("Choose a maximum of 1 to 2,000 table rows")


        if self.slides_visual_style is not None and (
            not isinstance(self.slides_visual_style, tuple)
            or len(self.slides_visual_style) != 2
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.slides_visual_style[0])
            or self.slides_visual_style[1] not in {"builtin", "user"}
        ):
            raise ValueError("Choose a Server Slides visual style")

        if self.speech is not None and not isinstance(self.speech, StudioSpeechChoice):
            raise ValueError("Choose an available speech route")
        if (
            type(self.speech_speed) not in {float, int} or not 0.25 <= self.speech_speed <= 4
            or (self.speech_speed != 1 and (self.speech is None or not self.speech.supports_speed))
        ):
            raise ValueError("Choose a speed supported by this speech model")

    @property
    def fingerprint(self) -> str:
        """Identify generation options without persisting user-supplied labels."""
        options = asdict(self)
        for key in ("table_columns", "table_max_rows", "slides_visual_style", "speech"):
            if not options[key]:
                del options[key]
        if self.speech is None:
            del options["speech_speed"]
        return sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()


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
        if kind is ResearchOutputKind.AUDIO_SUMMARY and (
            self.generation.speech is None or not self.generation.retrieval.citations
        ):
            raise ValueError("Audio Summary requires a speech route and source citations")
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
    generation_fingerprint: str | None
    status: ResearchTurnStatus
    copy_provenance: OutputCopyProvenance | None = None

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
        if self.copy_provenance is not None:
            if (
                not isinstance(self.copy_provenance, OutputCopyProvenance)
                or self.output_kind not in COPYABLE_STUDIO_TEXT_KINDS
                or self.status != "complete"
                or not self.owner_version
                or self.source_snapshot
                or self.generation_fingerprint is not None
            ):
                raise ValueError("A copied output requires an exact text artifact revision")
        elif not isinstance(self.generation_fingerprint, str) or not re.fullmatch(
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
