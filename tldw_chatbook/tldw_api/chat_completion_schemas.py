"""Closed request subsets for server-owned Research Workspace Q&A."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
    ),
]
RagSearchMode = Literal["fts", "vector", "hybrid"]


class _ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ResearchChatHistoryMessage(_ResearchRequest):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=256 * 1024, repr=False)


class ResearchChatCompletionRequest(_ResearchRequest):
    conversation_id: Identifier | None = None
    messages: tuple[ResearchChatHistoryMessage, ...] = Field(
        min_length=1, max_length=102, repr=False
    )
    model: str = Field(min_length=1, max_length=256)
    api_provider: str | None = Field(default=None, min_length=1, max_length=128)
    stream: bool
    save_to_db: bool
    max_tokens: int = Field(default=4096, ge=1, le=16384)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    tool_choice: Literal["none"] = "none"

    @model_validator(mode="after")
    def _validate_send(self) -> Self:
        if self.save_to_db and self.conversation_id is None:
            raise ValueError("Saved Research chat requires an exact conversation ID")
        if self.stream and self.save_to_db:
            raise ValueError("Streaming Research answers are appended explicitly")
        if (
            sum(len(message.content.encode("utf-8")) for message in self.messages)
            > 512 * 1024
        ):
            raise ValueError("Research completion exceeds its prompt limit")
        return self


class ResearchChatStartup(_ResearchRequest):
    scope_type: Literal["workspace"] = "workspace"
    workspace_id: Identifier
    workspace_assistant_selection: Literal["none"] = "none"
    title: str = Field(min_length=1, max_length=300)
    topic_label: Literal["research-chat"] = "research-chat"


class ResearchChatRagSearch(_ResearchRequest):
    query: str = Field(min_length=1, max_length=20_000, repr=False)
    workspace_id: Identifier
    include_media_ids: tuple[Annotated[int, Field(strict=True, gt=0)], ...] = Field(
        min_length=1, max_length=100
    )
    search_mode: RagSearchMode
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0, le=1)
    enable_reranking: bool = False
    sources: tuple[Literal["media_db"]] = ("media_db",)
    enable_generation: Literal[False] = False
    enable_cache: Literal[False] = False


class ResearchChatRagDocument(_ResearchRequest):
    id: Identifier
    source_type: Literal["media_db"] = "media_db"
    title: str = Field(max_length=300)
    excerpt: str = Field(max_length=6000, repr=False)
    score: float | None = Field(default=None, allow_inf_nan=False)


class ResearchChatRagCitation(_ResearchRequest):
    id: Identifier
    media_id: Identifier
    marker: str = Field(max_length=12)
    title: str = Field(max_length=300)


class ResearchChatRagSource(_ResearchRequest):
    source_id: Identifier
    catalog_item_id: Identifier
    association_version: int | None = Field(default=None, ge=0)
    catalog_item_version: int | None = Field(default=None, ge=0)


class ResearchChatRagSettings(_ResearchRequest):
    requested_mode: Literal["general", "rag", "auto"]
    requested_retrieval: RagSearchMode
    effective_retrieval: RagSearchMode | None
    retrieved: bool
    generation_state: Literal["complete", "cancelled", "failed"]
    provider: str = Field(max_length=256)
    model: str = Field(max_length=256)
    top_k: int = Field(ge=1, le=50)
    threshold: float | None = Field(default=None, ge=0, le=1)
    rerank: bool
    sources: tuple[ResearchChatRagSource, ...] = Field(default=(), max_length=100)
    processing_route_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchChatRagContext(_ResearchRequest):
    search_query: str = Field(max_length=64 * 1024, repr=False)
    search_mode: RagSearchMode | Literal["general"]
    settings_snapshot: ResearchChatRagSettings | None = None
    retrieved_documents: tuple[ResearchChatRagDocument, ...] = Field(
        default=(), max_length=50, repr=False
    )
    generated_answer: str | None = Field(
        default=None, max_length=256 * 1024, repr=False
    )
    citations: tuple[ResearchChatRagCitation, ...] = Field(default=(), max_length=50)
