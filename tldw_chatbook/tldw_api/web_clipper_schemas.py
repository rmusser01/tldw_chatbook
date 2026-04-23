"""Typed client schemas for the server Web Clipper API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WebClipperDestination = Literal["note", "workspace", "both"]
WebClipperOutcomeState = Literal["saved", "saved_with_warnings", "partially_saved", "failed"]
WebClipperEnrichmentStatus = Literal["pending", "running", "complete", "failed"]
WebClipperEnrichmentType = Literal["ocr", "vlm"]


class WebClipperSaveRequest(BaseModel):
    """Typed save request for a browser clip."""

    class NotePayload(BaseModel):
        title: str | None = Field(default=None, min_length=1)
        comment: str | None = None
        folder_id: int | None = Field(default=None, ge=1)
        keywords: list[str] = Field(default_factory=list)

        model_config = ConfigDict(extra="forbid")

    class WorkspacePayload(BaseModel):
        workspace_id: str = Field(..., min_length=1)

        model_config = ConfigDict(extra="forbid")

    class ContentPayload(BaseModel):
        visible_body: str | None = None
        full_extract: str | None = None
        selected_text: str | None = None

        model_config = ConfigDict(extra="forbid")

    class AttachmentPayload(BaseModel):
        slot: str = Field(..., min_length=1)
        file_name: str | None = Field(default=None, min_length=1)
        media_type: str = Field(..., min_length=1)
        text_content: str | None = None
        content_base64: str | None = None
        source_url: str | None = None

        model_config = ConfigDict(extra="forbid")

    class EnhancementOptions(BaseModel):
        run_ocr: bool = False
        run_vlm: bool = False

        model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(..., min_length=1)
    clip_type: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    source_title: str = Field(..., min_length=1)
    destination_mode: WebClipperDestination = "note"
    note: NotePayload = Field(default_factory=NotePayload)
    workspace: WorkspacePayload | None = None
    content: ContentPayload = Field(default_factory=ContentPayload)
    attachments: list[AttachmentPayload] = Field(default_factory=list)
    enhancements: EnhancementOptions = Field(default_factory=EnhancementOptions)
    capture_metadata: dict[str, Any] = Field(default_factory=dict)
    source_note_version: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def validate_workspace_destination(self) -> "WebClipperSaveRequest":
        if self.destination_mode in {"workspace", "both"} and self.workspace is None:
            raise ValueError("workspace is required when destination_mode targets a workspace.")
        return self


class WebClipperEnrichmentPayload(BaseModel):
    """Structured payload for OCR/VLM follow-up work."""

    clip_id: str = Field(..., min_length=1)
    enrichment_type: WebClipperEnrichmentType
    status: WebClipperEnrichmentStatus = "pending"
    inline_summary: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    source_note_version: int = Field(..., ge=1)
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


class WebClipperSavedNote(BaseModel):
    """Canonical note information returned from clipper flows."""

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperWorkspacePlacement(BaseModel):
    """Read-only workspace placement summary for a canonical clip note."""

    workspace_id: str = Field(..., min_length=1)
    workspace_note_id: int = Field(..., ge=1)
    source_note_id: str = Field(..., min_length=1)
    source_note_version: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperAttachmentRecord(BaseModel):
    """Persisted attachment metadata returned by the clipper service."""

    slot: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    original_file_name: str = Field(..., min_length=1)
    content_type: str | None = None
    size_bytes: int = Field(..., ge=0)
    uploaded_at: datetime
    url: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperSaveResponse(BaseModel):
    """Detailed outcome payload for a clip save."""

    clip_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState
    note: WebClipperSavedNote | None = None
    workspace_placement: WebClipperWorkspacePlacement | None = None
    attachments: list[WebClipperAttachmentRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    note_id: str = Field(..., min_length=1)
    workspace_placement_saved: bool = False
    workspace_placement_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WebClipperSaveResult(BaseModel):
    """Backward-compatible minimal outcome payload for earlier server callers."""

    clip_id: str = Field(..., min_length=1)
    note_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState
    workspace_placement_saved: bool = False
    workspace_placement_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WebClipperStatusResponse(BaseModel):
    """Current canonical state for a saved clip."""

    clip_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState
    note: WebClipperSavedNote
    workspace_placements: list[WebClipperWorkspacePlacement] = Field(default_factory=list)
    attachments: list[WebClipperAttachmentRecord] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    content_budget: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WebClipperEnrichmentResponse(BaseModel):
    """Result of storing an OCR/VLM enrichment payload."""

    clip_id: str = Field(..., min_length=1)
    enrichment_type: WebClipperEnrichmentType
    status: WebClipperEnrichmentStatus
    source_note_version: int = Field(..., ge=1)
    inline_applied: bool = False
    inline_summary: str | None = None
    conflict_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
