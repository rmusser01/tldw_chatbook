from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SourceInputType = Literal["epub", "pdf", "txt", "md", "srt", "vtt", "ass"]
AudiobookAudioFormat = Literal["wav", "mp3", "flac", "opus", "m4b"]
SubtitleFormat = Literal["srt", "vtt", "ass"]
SubtitleMode = Literal["line", "sentence", "word_count", "highlight"]
SubtitleVariant = Literal["wide", "narrow", "centered"]
AudiobookJobStatus = Literal["queued", "processing", "completed", "failed", "canceled"]
AudiobookArtifactType = Literal["audio", "subtitle", "package", "alignment"]
AudiobookArtifactScope = Literal["chapter", "merged"]
AlignmentEngine = Literal["kokoro"]


class SourceRef(BaseModel):
    input_type: SourceInputType
    upload_id: str | None = None
    media_id: int | str | None = None
    raw_text: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> SourceRef:
        if not self.upload_id and not self.media_id and not (self.raw_text or "").strip():
            raise ValueError("source requires upload_id, media_id, or raw_text")
        return self


class ChapterSelection(BaseModel):
    chapter_id: str
    include: bool
    voice: str | None = None
    speed: float | None = Field(None, ge=0.25, le=4.0)


class ChapterVoiceOverride(BaseModel):
    chapter_id: str
    voice: str | None = None
    speed: float | None = Field(None, ge=0.25, le=4.0)


class OutputOptions(BaseModel):
    merge: bool = True
    per_chapter: bool = True
    formats: list[AudiobookAudioFormat]

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, value: list[AudiobookAudioFormat]) -> list[AudiobookAudioFormat]:
        if not value:
            raise ValueError("formats must include at least one audio format")
        if len(set(value)) != len(value):
            raise ValueError("formats must not contain duplicates")
        return value


class SubtitleOptions(BaseModel):
    formats: list[SubtitleFormat]
    mode: SubtitleMode
    variant: SubtitleVariant
    words_per_cue: int | None = Field(12, ge=1)
    max_chars: int | None = Field(None, ge=10)
    max_lines: int | None = Field(None, ge=1)

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, value: list[SubtitleFormat]) -> list[SubtitleFormat]:
        if not value:
            raise ValueError("formats must include at least one subtitle format")
        if len(set(value)) != len(value):
            raise ValueError("formats must not contain duplicates")
        return value


class QueueOptions(BaseModel):
    priority: int = Field(5, ge=1, le=10)
    batch_group: str | None = Field(None, max_length=100)


class AudiobookJobItem(BaseModel):
    source: SourceRef
    tts_provider: str | None = None
    tts_model: str | None = None
    voice_profile_id: str | None = None
    chapters: list[ChapterSelection] | None = None
    output: OutputOptions | None = None
    subtitles: SubtitleOptions | None = None
    metadata: dict[str, Any] | None = None


class AlignmentWord(BaseModel):
    word: str
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)


class AlignmentPayload(BaseModel):
    engine: AlignmentEngine
    sample_rate: int = Field(..., ge=8000)
    words: list[AlignmentWord]

    @field_validator("words")
    @classmethod
    def _validate_words(cls, value: list[AlignmentWord]) -> list[AlignmentWord]:
        if not value:
            raise ValueError("alignment words must not be empty")
        return value


class ChapterPreview(BaseModel):
    chapter_id: str
    title: str | None = None
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)


class AudiobookParseRequest(BaseModel):
    source: SourceRef
    detect_chapters: bool = True
    custom_chapter_pattern: str | None = None
    language: str | None = None
    max_chars: int | None = Field(None, ge=1)


class AudiobookParseResponse(BaseModel):
    project_id: str
    normalized_text: str
    chapters: list[ChapterPreview]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudiobookJobRequest(BaseModel):
    project_title: str = Field(..., min_length=1, max_length=200)
    source: SourceRef | None = None
    items: list[AudiobookJobItem] | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    voice_profile_id: str | None = None
    chapters: list[ChapterSelection] | None = None
    output: OutputOptions | None = None
    subtitles: SubtitleOptions | None = None
    queue: QueueOptions | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> AudiobookJobRequest:
        has_items = self.items is not None
        has_source = self.source is not None
        if has_items and has_source:
            raise ValueError("provide either items or source, not both")
        if not has_items and not has_source:
            raise ValueError("source is required when items are not provided")
        if has_items and not self.items:
            raise ValueError("items must include at least one entry")
        if has_source:
            if not self.chapters:
                raise ValueError("chapters must be provided for single-source jobs")
            if self.output is None:
                raise ValueError("output is required for single-source jobs")
        return self


class AudiobookJobCreateResponse(BaseModel):
    job_id: int
    project_id: str
    status: AudiobookJobStatus


class JobProgress(BaseModel):
    stage: str
    chapter_index: int | None = Field(None, ge=0)
    chapters_total: int | None = Field(None, ge=0)
    item_index: int | None = Field(None, ge=0)
    items_total: int | None = Field(None, ge=0)
    percent: int | None = Field(None, ge=0, le=100)


class AudiobookJobStatusResponse(BaseModel):
    job_id: int
    project_id: str
    status: AudiobookJobStatus
    progress: JobProgress | None = None
    errors: list[str] = Field(default_factory=list)


class AudiobookProjectInfo(BaseModel):
    project_db_id: int
    project_id: str | None = None
    title: str | None = None
    status: str | None = None
    source_ref: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class AudiobookProjectListResponse(BaseModel):
    projects: list[AudiobookProjectInfo]


class AudiobookProjectResponse(BaseModel):
    project: AudiobookProjectInfo


class AudiobookChapterInfo(BaseModel):
    id: int
    chapter_index: int
    title: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    voice_profile_id: str | None = None
    speed: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudiobookChapterListResponse(BaseModel):
    project_id: str
    chapters: list[AudiobookChapterInfo]


class ArtifactInfo(BaseModel):
    artifact_type: AudiobookArtifactType
    format: str
    scope: AudiobookArtifactScope | None = None
    chapter_id: str | None = None
    output_id: int
    download_url: str


class AudiobookArtifactsResponse(BaseModel):
    project_id: str
    artifacts: list[ArtifactInfo]


class VoiceProfileCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    default_voice: str
    default_speed: float = Field(..., ge=0.25, le=4.0)
    chapter_overrides: list[ChapterVoiceOverride] | None = Field(default_factory=list)


class VoiceProfileResponse(VoiceProfileCreateRequest):
    profile_id: str


class VoiceProfileListResponse(BaseModel):
    profiles: list[VoiceProfileResponse]


class VoiceProfileDeleteResponse(BaseModel):
    profile_id: str
    deleted: bool = True


class SubtitleExportRequest(BaseModel):
    format: SubtitleFormat
    mode: SubtitleMode
    variant: SubtitleVariant
    alignment: AlignmentPayload | None = None
    alignment_output_id: int | None = Field(None, ge=1)
    persist: bool | None = None
    cache_ttl_hours: int | None = Field(None, ge=1)
    project_id: str | None = None
    chapter_id: str | None = None
    chapter_index: int | None = Field(None, ge=0)
    item_index: int | None = Field(None, ge=0)
    metadata: dict[str, Any] | None = None
    words_per_cue: int | None = Field(12, ge=1)
    max_chars: int | None = Field(None, ge=10)
    max_lines: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _validate_alignment_source(self) -> SubtitleExportRequest:
        if self.alignment is None and self.alignment_output_id is None:
            raise ValueError("alignment or alignment_output_id is required")
        return self


__all__ = [
    "AlignmentPayload",
    "AlignmentWord",
    "ArtifactInfo",
    "AudiobookArtifactScope",
    "AudiobookArtifactType",
    "AudiobookArtifactsResponse",
    "AudiobookAudioFormat",
    "AudiobookChapterInfo",
    "AudiobookChapterListResponse",
    "AudiobookJobCreateResponse",
    "AudiobookJobItem",
    "AudiobookJobRequest",
    "AudiobookJobStatus",
    "AudiobookJobStatusResponse",
    "AudiobookParseRequest",
    "AudiobookParseResponse",
    "AudiobookProjectInfo",
    "AudiobookProjectListResponse",
    "AudiobookProjectResponse",
    "ChapterPreview",
    "ChapterSelection",
    "ChapterVoiceOverride",
    "JobProgress",
    "OutputOptions",
    "QueueOptions",
    "SourceInputType",
    "SourceRef",
    "SubtitleExportRequest",
    "SubtitleFormat",
    "SubtitleMode",
    "SubtitleOptions",
    "SubtitleVariant",
    "VoiceProfileCreateRequest",
    "VoiceProfileDeleteResponse",
    "VoiceProfileListResponse",
    "VoiceProfileResponse",
]
