from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


OutputTemplateType = Literal[
    "newsletter_markdown",
    "briefing_markdown",
    "mece_markdown",
    "newsletter_html",
    "tts_audio",
]

OutputFormat = Literal["md", "html", "mp3"]
PreviewFormat = Literal["md", "html"]


class OutputTemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: OutputTemplateType
    format: OutputFormat
    body: str
    description: str | None = Field(default=None, max_length=500)
    is_default: bool = False
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_format_matches_type(self) -> "OutputTemplateCreateRequest":
        if self.type in {"newsletter_markdown", "briefing_markdown", "mece_markdown"} and self.format != "md":
            raise ValueError("Markdown templates must use format 'md'.")
        if self.type == "newsletter_html" and self.format != "html":
            raise ValueError("HTML templates must use format 'html'.")
        if self.type == "tts_audio" and self.format != "mp3":
            raise ValueError("TTS templates must use format 'mp3'.")
        return self


class OutputTemplateUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    type: OutputTemplateType | None = None
    format: OutputFormat | None = None
    body: str | None = None
    description: str | None = Field(default=None, max_length=500)
    is_default: bool | None = None
    metadata: dict[str, Any] | None = None


class OutputTemplateResponse(BaseModel):
    id: int
    user_id: str | None = None
    name: str
    type: OutputTemplateType
    format: OutputFormat
    body: str
    description: str | None = None
    is_default: bool = False
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] | None = None


class OutputTemplateListResponse(BaseModel):
    items: list[OutputTemplateResponse]
    total: int


class OutputTemplatePreviewRequest(BaseModel):
    template_id: int
    item_ids: list[int] | None = None
    run_id: int | None = None
    limit: int = Field(default=50, ge=1, le=1000)
    data: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> "OutputTemplatePreviewRequest":
        if not self.item_ids and self.run_id is None and self.data is None:
            raise ValueError("Provide item_ids, run_id, or data for preview.")
        return self


class OutputTemplatePreviewResponse(BaseModel):
    rendered: str
    format: PreviewFormat


class OutputCreateRequest(BaseModel):
    template_id: int
    item_ids: list[int] | None = None
    run_id: int | None = None
    title: str | None = None
    data: dict[str, Any] | None = None
    workspace_tag: str | None = None
    generate_mece: bool = False
    mece_template_id: int | None = None
    generate_tts: bool = False
    tts_template_id: int | None = None
    ingest_to_media_db: bool = False
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_speed: float | None = Field(default=None, ge=0.25, le=4.0)


class OutputArtifactResponse(BaseModel):
    id: int
    title: str
    type: str
    format: OutputFormat
    storage_path: str
    media_item_id: int | None = None
    created_at: datetime
    workspace_tag: str | None = None


class OutputListResponse(BaseModel):
    items: list[OutputArtifactResponse]
    total: int
    page: int
    size: int


class OutputUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    retention_until: str | None = None
    format: Literal["md", "html"] | None = None
