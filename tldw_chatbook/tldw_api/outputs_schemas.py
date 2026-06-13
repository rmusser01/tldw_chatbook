"""Server output templates and output artifact schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


TemplateType = Literal[
    "newsletter_markdown",
    "briefing_markdown",
    "mece_markdown",
    "newsletter_html",
    "tts_audio",
]
TemplateFormat = Literal["md", "html", "mp3"]
OutputFormat = Literal["md", "html", "mp3"]


class OutputTemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: TemplateType
    format: TemplateFormat
    body: str
    description: str | None = Field(default=None, max_length=500)
    is_default: bool = False
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_format_matches_type(self) -> "OutputTemplateCreate":
        if self.type in {"newsletter_markdown", "briefing_markdown", "mece_markdown"} and self.format != "md":
            raise ValueError("Markdown-type templates must use format 'md'.")
        if self.type == "newsletter_html" and self.format != "html":
            raise ValueError("newsletter_html templates must use format 'html'.")
        if self.type == "tts_audio" and self.format != "mp3":
            raise ValueError("tts_audio templates must use format 'mp3'.")
        return self


class OutputTemplateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    type: TemplateType | None = None
    format: TemplateFormat | None = None
    body: str | None = None
    description: str | None = Field(default=None, max_length=500)
    is_default: bool | None = None
    metadata: dict[str, Any] | None = None


class OutputTemplate(BaseModel):
    id: int
    user_id: str | None = None
    name: str
    type: TemplateType
    format: TemplateFormat
    body: str
    description: str | None = None
    is_default: bool = False
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] | None = None


class OutputTemplateList(BaseModel):
    items: list[OutputTemplate]
    total: int


class TemplatePreviewRequest(BaseModel):
    template_id: int
    item_ids: list[int] | None = None
    run_id: int | None = None
    limit: int = Field(default=50, ge=1, le=1000)
    data: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> "TemplatePreviewRequest":
        if not self.item_ids and not self.run_id and not self.data:
            raise ValueError("Provide item_ids, run_id, or inline data for preview.")
        return self


class TemplatePreviewResponse(BaseModel):
    rendered: str
    format: Literal["md", "html"]


# Backward-compatible names used by merged server-parity service wrappers.
OutputTemplateCreateRequest = OutputTemplateCreate
OutputTemplateUpdateRequest = OutputTemplateUpdate
OutputTemplatePreviewRequest = TemplatePreviewRequest


class OutputCreateRequest(BaseModel):
    template_id: int
    item_ids: list[int] | None = None
    run_id: int | None = None
    title: str | None = None
    data: dict[str, object] | None = None
    workspace_tag: str | None = None
    generate_mece: bool = False
    mece_template_id: int | None = None
    generate_tts: bool = False
    tts_template_id: int | None = None
    ingest_to_media_db: bool = False
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_speed: float | None = Field(default=None, ge=0.25, le=4.0)


class OutputArtifact(BaseModel):
    id: int
    title: str
    type: str
    format: OutputFormat
    storage_path: str
    media_item_id: int | None = None
    created_at: datetime
    workspace_tag: str | None = None


class OutputListResponse(BaseModel):
    items: list[OutputArtifact]
    total: int
    page: int
    size: int


class OutputUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    retention_until: str | None = None
    format: Literal["md", "html"] | None = None


class OutputDeleteResponse(BaseModel):
    success: bool
    file_deleted: bool = False


class OutputsPurgeRequest(BaseModel):
    delete_files: bool = False
    soft_deleted_grace_days: int = 30
    include_retention: bool = True


class OutputsPurgeResponse(BaseModel):
    removed: int
    files_deleted: int
