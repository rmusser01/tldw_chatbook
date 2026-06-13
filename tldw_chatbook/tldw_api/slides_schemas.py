from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SlideLayout = Literal["title", "content", "two_column", "quote", "section", "blank"]
SlidesExportFormat = Literal["revealjs", "markdown", "json", "pdf"]
PresentationRenderFormat = Literal["mp4", "webm"]


class Slide(BaseModel):
    order: int
    layout: SlideLayout
    title: str | None = None
    content: str = ""
    speaker_notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _validate_visual_style_selection_pair(
    *,
    visual_style_id: str | None,
    visual_style_scope: str | None,
) -> None:
    if (visual_style_id is None) != (visual_style_scope is None):
        raise ValueError("visual_style_id and visual_style_scope must be provided together")


class VisualStyleSelectionMixin(BaseModel):
    visual_style_id: str | None = None
    visual_style_scope: str | None = None

    @model_validator(mode="after")
    def _validate_visual_style_selection(self) -> VisualStyleSelectionMixin:
        _validate_visual_style_selection_pair(
            visual_style_id=self.visual_style_id,
            visual_style_scope=self.visual_style_scope,
        )
        return self


class PresentationBase(VisualStyleSelectionMixin):
    title: str
    description: str | None = None
    theme: str = "black"
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    studio_data: dict[str, Any] | None = None
    slides: list[Slide] = Field(default_factory=list)
    custom_css: str | None = None


class PresentationCreateRequest(PresentationBase):
    pass


class PresentationUpdateRequest(PresentationBase):
    pass


class PresentationPatchRequest(VisualStyleSelectionMixin):
    title: str | None = None
    description: str | None = None
    theme: str | None = None
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    studio_data: dict[str, Any] | None = None
    slides: list[Slide] | None = None
    custom_css: str | None = None


class PresentationReorderRequest(BaseModel):
    order: list[int] = Field(..., min_length=1)


class PresentationResponse(PresentationBase):
    id: str
    visual_style_name: str | None = None
    visual_style_version: int | None = None
    visual_style_snapshot: dict[str, Any] | None = None
    source_type: str | None = None
    source_ref: Any | None = None
    source_query: str | None = None
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


class PresentationVersionSummary(BaseModel):
    presentation_id: str
    version: int
    created_at: datetime
    title: str | None = None
    deleted: bool | None = None


class PresentationVersionListResponse(BaseModel):
    versions: list[PresentationVersionSummary]
    total: int
    limit: int
    offset: int


class SlidesTemplateResponse(BaseModel):
    id: str
    name: str
    theme: str
    marp_theme: str | None = None
    settings: dict[str, Any] | None = None
    default_slides: list[Slide] | None = None
    custom_css: str | None = None


class SlidesTemplateListResponse(BaseModel):
    templates: list[SlidesTemplateResponse]


class VisualStyleBase(BaseModel):
    name: str
    description: str | None = None
    generation_rules: dict[str, Any] = Field(default_factory=dict)
    artifact_preferences: list[str] = Field(default_factory=list)
    appearance_defaults: dict[str, Any] = Field(default_factory=dict)
    fallback_policy: dict[str, Any] = Field(default_factory=dict)


class VisualStyleCreateRequest(VisualStyleBase):
    pass


class VisualStylePatchRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    generation_rules: dict[str, Any] | None = None
    artifact_preferences: list[str] | None = None
    appearance_defaults: dict[str, Any] | None = None
    fallback_policy: dict[str, Any] | None = None


class VisualStyleResponse(VisualStyleBase):
    id: str
    scope: str
    category: str | None = None
    guide_number: int | None = None
    tags: list[str] | None = None
    best_for: list[str] | None = None
    version: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class VisualStyleListResponse(BaseModel):
    styles: list[VisualStyleResponse]
    total_count: int
    limit: int
    offset: int


class PresentationSummary(BaseModel):
    id: str
    title: str
    description: str | None = None
    theme: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    version: int


class PresentationListResponse(BaseModel):
    presentations: list[PresentationSummary]
    total: int
    limit: int
    offset: int


class PresentationSearchResponse(BaseModel):
    presentations: list[PresentationSummary]
    total: int
    limit: int
    offset: int


class SlideGenerationBase(VisualStyleSelectionMixin):
    title_hint: str | None = None
    theme: str | None = None
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    custom_css: str | None = None
    max_source_tokens: int | None = Field(default=None, ge=1)
    max_source_chars: int | None = Field(default=None, ge=1)
    enable_chunking: bool = False
    chunk_size_tokens: int | None = Field(default=None, ge=1)
    summary_tokens: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateFromPromptRequest(SlideGenerationBase):
    prompt: str


class GenerateFromChatRequest(SlideGenerationBase):
    conversation_id: str


class GenerateFromNotesRequest(SlideGenerationBase):
    note_ids: list[str]


class GenerateFromMediaRequest(SlideGenerationBase):
    media_id: int = Field(..., ge=1)


class GenerateFromRagRequest(SlideGenerationBase):
    query: str
    top_k: int | None = Field(default=8, ge=1)


class PresentationRenderRequest(BaseModel):
    format: PresentationRenderFormat


class PresentationRenderJobResponse(BaseModel):
    job_id: int
    status: str
    job_type: str
    presentation_id: str
    presentation_version: int
    format: PresentationRenderFormat


class PresentationRenderJobStatusResponse(BaseModel):
    job_id: int
    status: str
    job_type: str
    presentation_id: str | None = None
    presentation_version: int | None = None
    format: PresentationRenderFormat | None = None
    output_id: int | None = None
    download_url: str | None = None
    error: str | None = None


class PresentationRenderArtifactInfo(BaseModel):
    output_id: int
    format: PresentationRenderFormat
    title: str | None = None
    download_url: str
    presentation_version: int | None = None
    created_at: datetime | None = None


class PresentationRenderArtifactListResponse(BaseModel):
    presentation_id: str
    artifacts: list[PresentationRenderArtifactInfo]


class SlidesHealthResponse(BaseModel):
    service: str
    status: str
    detail: str | None = None


__all__ = [
    "GenerateFromChatRequest",
    "GenerateFromMediaRequest",
    "GenerateFromNotesRequest",
    "GenerateFromPromptRequest",
    "GenerateFromRagRequest",
    "PresentationCreateRequest",
    "PresentationListResponse",
    "PresentationPatchRequest",
    "PresentationRenderArtifactInfo",
    "PresentationRenderArtifactListResponse",
    "PresentationRenderFormat",
    "PresentationRenderJobResponse",
    "PresentationRenderJobStatusResponse",
    "PresentationRenderRequest",
    "PresentationReorderRequest",
    "PresentationResponse",
    "PresentationSearchResponse",
    "PresentationSummary",
    "PresentationUpdateRequest",
    "PresentationVersionListResponse",
    "PresentationVersionSummary",
    "Slide",
    "SlideGenerationBase",
    "SlideLayout",
    "SlidesExportFormat",
    "SlidesHealthResponse",
    "SlidesTemplateListResponse",
    "SlidesTemplateResponse",
    "VisualStyleBase",
    "VisualStyleCreateRequest",
    "VisualStyleListResponse",
    "VisualStylePatchRequest",
    "VisualStyleResponse",
    "VisualStyleSelectionMixin",
]
