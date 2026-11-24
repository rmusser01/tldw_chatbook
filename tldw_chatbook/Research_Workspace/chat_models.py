"""Immutable, authority-qualified input and result for Research Chat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from .contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchSourceSummary,
    RetrievalMode,
    SourceReadiness,
    SourceReadinessState,
    _required_text,
    _safe_identity,
)

MAX_CHAT_DRAFT_BYTES = 64 * 1024
MAX_CHAT_APPEND_RECEIPTS = 128
MAX_CHAT_SOURCES = 100

ResearchTurnStatus = Literal["complete", "cancelled", "failed"]
AppendStage = Literal["pending", "user_committed", "assistant_committed", "complete"]


class ResearchChatMode(StrEnum):
    GENERAL = "general"
    RAG = "rag"
    AUTO = "auto"


def validate_chat_id(value: str, name: str, *, allow_empty: bool = False) -> str:
    value = _safe_identity(value, name)
    if len(value) > 256 or "://" in value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be a bounded identifier")
    if not value and not allow_empty:
        raise ValueError(f"{name} is required")
    return value


def validate_chat_draft(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("chat draft must be text")
    if "\x00" in value or len(value.encode("utf-8")) > MAX_CHAT_DRAFT_BYTES:
        raise ValueError("chat draft contains invalid text or exceeds 64 KiB")
    return value


@dataclass(frozen=True, slots=True)
class ResearchRetrievalSettings:
    mode: RetrievalMode = RetrievalMode.FTS
    top_k: int = 5
    threshold: float | None = None
    rerank: bool = False
    citations: bool = True
    history_limit: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RetrievalMode(self.mode))
        if type(self.top_k) is not int or not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if self.threshold is not None and (
            type(self.threshold) not in {int, float} or not 0 <= self.threshold <= 1
        ):
            raise ValueError("threshold must be between 0 and 1")
        if type(self.rerank) is not bool or type(self.citations) is not bool:
            raise TypeError("rerank and citations must be bool")
        if type(self.history_limit) is not int or not 0 <= self.history_limit <= 100:
            raise ValueError("history_limit must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class ResearchSourceVersionRef:
    ref: QualifiedWorkspaceRef
    source_id: str
    catalog_item_id: str
    source_type: str
    catalog_item_version: int | None = None
    association_version: int | None = None
    ready_modes: tuple[RetrievalMode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, QualifiedWorkspaceRef):
            raise TypeError("source must have a qualified workspace")
        for name in ("source_id", "catalog_item_id", "source_type"):
            object.__setattr__(self, name, validate_chat_id(getattr(self, name), name))
        for version in (self.catalog_item_version, self.association_version):
            if version is not None and (type(version) is not int or version < 0):
                raise ValueError("source version must be a non-negative integer")
        modes = tuple(RetrievalMode(mode) for mode in self.ready_modes)
        if len(set(modes)) != len(modes):
            raise ValueError("source readiness modes must be unique")
        object.__setattr__(self, "ready_modes", modes)

    @classmethod
    def capture(
        cls, source: ResearchSourceSummary, readiness: SourceReadiness | None
    ) -> ResearchSourceVersionRef:
        """Join the attached source's version to its current readiness."""
        modes = []
        if readiness is not None:
            if (readiness.ref, readiness.source_id) != (source.ref, source.source_id):
                raise ValueError(
                    "source readiness has a different workspace or association"
                )
            if (
                readiness.catalog_item_id == source.catalog_item_id
                and not readiness.stale
                and readiness.state
                not in {
                    SourceReadinessState.FAILED,
                    SourceReadinessState.UNAVAILABLE,
                    SourceReadinessState.STALE,
                }
            ):
                if readiness.fts_ready:
                    modes.append(RetrievalMode.FTS)
                if readiness.vector_ready:
                    modes.append(RetrievalMode.SEMANTIC)
                if readiness.fts_ready and readiness.vector_ready:
                    modes.append(RetrievalMode.HYBRID)
        return cls(
            source.ref,
            source.source_id,
            source.catalog_item_id,
            source.source_type,
            source.catalog_item_version,
            source.version,
            tuple(modes),
        )


@dataclass(frozen=True, slots=True)
class ResearchCitationRef:
    citation_id: str
    source: ResearchSourceVersionRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "citation_id", validate_chat_id(self.citation_id, "citation_id")
        )
        if not isinstance(self.source, ResearchSourceVersionRef):
            raise TypeError("citation source must be qualified")


@dataclass(frozen=True, slots=True)
class ResearchChatSessionRequest:
    title: str = "Research Chat"

    def __post_init__(self) -> None:
        title = _required_text(self.title, "title")
        if len(title) > 300:
            raise ValueError("title exceeds 300 characters")
        object.__setattr__(self, "title", title)


@dataclass(frozen=True, slots=True)
class ResearchChatSession:
    ref: QualifiedWorkspaceRef
    conversation_id: str
    title: str


@dataclass(frozen=True, slots=True)
class ResearchChatTurnRequest:
    workspace: QualifiedWorkspaceRef
    conversation_id: str
    user_text: str = field(repr=False)
    mode: ResearchChatMode
    source_snapshot: tuple[ResearchSourceVersionRef, ...]
    retrieval: ResearchRetrievalSettings
    processing_route: ProcessingRoute
    context_revision: int
    turn_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, QualifiedWorkspaceRef):
            raise TypeError("workspace must be qualified")
        if self.processing_route.data_source is not self.workspace.data_source:
            raise ValueError("processing route does not match the workspace authority")
        for name in ("conversation_id", "turn_id"):
            object.__setattr__(self, name, validate_chat_id(getattr(self, name), name))
        if not validate_chat_draft(self.user_text).strip():
            raise ValueError("chat draft must contain a question")
        try:
            object.__setattr__(self, "mode", ResearchChatMode(self.mode))
        except (TypeError, ValueError):
            raise ValueError("chat mode is invalid") from None
        if type(self.context_revision) is not int or self.context_revision < 0:
            raise ValueError("context revision must be non-negative")
        if not isinstance(self.retrieval, ResearchRetrievalSettings):
            raise TypeError("retrieval settings must be validated")
        if (
            not isinstance(self.source_snapshot, tuple)
            or len(self.source_snapshot) > MAX_CHAT_SOURCES
        ):
            raise ValueError(
                f"source snapshot must contain at most {MAX_CHAT_SOURCES} sources"
            )
        if any(
            not isinstance(source, ResearchSourceVersionRef)
            for source in self.source_snapshot
        ):
            raise TypeError("source snapshot must contain qualified source versions")
        if any(source.ref != self.workspace for source in self.source_snapshot):
            raise ValueError("source snapshot contains another workspace")
        if len({source.source_id for source in self.source_snapshot}) != len(
            self.source_snapshot
        ):
            raise ValueError("source snapshot contains duplicate associations")
        if self.mode is ResearchChatMode.RAG and not self.retrieval_sources:
            raise ValueError("Grounded requires a selected ready source")

    def is_current(
        self, *, workspace: QualifiedWorkspaceRef, context_revision: int
    ) -> bool:
        return workspace == self.workspace and context_revision == self.context_revision

    @property
    def effective_retrieval(self) -> RetrievalMode | None:
        if self.mode is ResearchChatMode.GENERAL:
            return None
        if any(
            self.retrieval.mode in source.ready_modes for source in self.source_snapshot
        ):
            return self.retrieval.mode
        if any(
            RetrievalMode.FTS in source.ready_modes for source in self.source_snapshot
        ):
            return RetrievalMode.FTS
        return None

    @property
    def retrieval_sources(self) -> tuple[ResearchSourceVersionRef, ...]:
        mode = self.effective_retrieval
        return tuple(
            source for source in self.source_snapshot if mode in source.ready_modes
        )


@dataclass(frozen=True, slots=True)
class ResearchChatTurnOutcome:
    status: ResearchTurnStatus
    mode: ResearchChatMode
    requested_retrieval: RetrievalMode
    effective_retrieval: RetrievalMode | None
    retrieved: bool
    source_snapshot: tuple[ResearchSourceVersionRef, ...]
    citations: tuple[ResearchCitationRef, ...]
    provider: str
    model: str
    processing_route_fingerprint: str
    user_message_id: str = ""
    assistant_message_id: str = ""

    @classmethod
    def from_request(
        cls,
        request: ResearchChatTurnRequest,
        *,
        status: ResearchTurnStatus,
        retrieved: bool,
        citations: tuple[ResearchCitationRef, ...] = (),
        user_message_id: str = "",
        assistant_message_id: str = "",
    ) -> ResearchChatTurnOutcome:
        if status not in {"complete", "cancelled", "failed"}:
            raise ValueError("generation status is invalid")
        if type(retrieved) is not bool or (
            retrieved and request.effective_retrieval is None
        ):
            raise ValueError("retrieval result is inconsistent with the request")
        if citations and (
            not retrieved
            or any(c.source not in request.retrieval_sources for c in citations)
        ):
            raise ValueError("citation source is outside the retrieval snapshot")
        if len(citations) > 50:
            raise ValueError("citation count exceeds 50")
        route = request.processing_route
        route_fields = (
            route.data_source.value,
            route.processor,
            route.provider,
            route.model,
            route.remote,
            route.endpoint_class,
            route.endpoint_fingerprint,
            route.redaction_policy,
            route.source_body_mode,
        )
        return cls(
            status,
            request.mode,
            request.retrieval.mode,
            request.effective_retrieval,
            retrieved,
            request.source_snapshot,
            citations,
            route.provider,
            route.model,
            sha256(json.dumps(route_fields).encode()).hexdigest(),
            validate_chat_id(user_message_id, "user_message_id", allow_empty=True),
            validate_chat_id(
                assistant_message_id, "assistant_message_id", allow_empty=True
            ),
        )

    @property
    def retrieval_disclosure(self) -> str:
        return "Retrieved" if self.retrieved else "Did not retrieve"


@dataclass(frozen=True, slots=True)
class ResearchChatAppendReceipt:
    turn_id: str
    conversation_id: str
    stage: AppendStage
    user_message_id: str = ""
    assistant_message_id: str = ""

    def __post_init__(self) -> None:
        if self.stage not in {
            "pending",
            "user_committed",
            "assistant_committed",
            "complete",
        }:
            raise ValueError("chat receipt stage is invalid")
        for name in (
            "turn_id",
            "conversation_id",
            "user_message_id",
            "assistant_message_id",
        ):
            value = validate_chat_id(
                getattr(self, name), name, allow_empty=name.endswith("message_id")
            )
            object.__setattr__(self, name, value)
        if bool(self.user_message_id) != (self.stage != "pending"):
            raise ValueError("chat receipt user message does not match its stage")
        if bool(self.assistant_message_id) != (
            self.stage in {"assistant_committed", "complete"}
        ):
            raise ValueError("chat receipt assistant message does not match its stage")
