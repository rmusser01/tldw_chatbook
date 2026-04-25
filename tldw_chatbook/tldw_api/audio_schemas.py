from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "ogg", "webm", "ulaw"]
TranscriptResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class NormalizationOptions(BaseModel):
    normalize: bool = True
    unit_normalization: bool = False
    url_normalization: bool = True
    email_normalization: bool = True
    optional_pluralization_normalization: bool = True
    phone_normalization: bool = True


class OpenAISpeechRequest(BaseModel):
    model: str = Field(..., min_length=1)
    input: str
    voice: str = "af_heart"
    response_format: AudioFormat = "mp3"
    download_format: AudioFormat | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = True
    target_sample_rate: int | None = Field(default=None, ge=1)
    return_download_link: bool = False
    lang_code: str | None = None
    normalization_options: NormalizationOptions | None = Field(default_factory=NormalizationOptions)
    voice_reference: str | None = None
    reference_duration_min: float | None = Field(default=None, ge=3.0, le=60.0)
    extra_params: dict[str, Any] | None = None


class TTSHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    providers: dict[str, Any] | None = None
    circuit_breakers: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    capabilities_envelope: list[dict[str, Any]] | None = None
    timestamp: str | None = None
    model: str | None = None
    provider: str | None = None
    warm: bool | None = None
    detail: str | None = None


class TTSProvidersResponse(BaseModel):
    providers: dict[str, Any]
    voices: dict[str, Any]
    timestamp: str | None = None


class TTSVoicesResponse(RootModel[dict[str, Any]]):
    pass


class AudioSpeechJobCreateResponse(BaseModel):
    job_id: int
    status: str


class AudioSpeechArtifact(BaseModel):
    output_id: int
    format: str
    type: str
    title: str | None = None
    download_url: str
    metadata: dict[str, Any] | None = None


class AudioSpeechJobArtifactsResponse(BaseModel):
    job_id: int
    artifacts: list[AudioSpeechArtifact]


class SubmitAudioJobRequest(BaseModel):
    url: str | None = None
    local_path: str | None = None
    model: str | None = None
    hotwords: str | list[str] | None = None
    perform_chunking: bool = True
    perform_analysis: bool = False
    api_name: str | None = None

    @model_validator(mode="after")
    def _validate_inputs(self) -> SubmitAudioJobRequest:
        has_url = bool((self.url or "").strip())
        has_local_path = bool((self.local_path or "").strip())
        if has_url == has_local_path:
            raise ValueError("provide exactly one of url or local_path")
        return self


class SubmitAudioJobResponse(BaseModel):
    id: int
    uuid: str | None = None
    domain: str
    queue: str
    job_type: str
    status: str


class AudioJobResponse(BaseModel):
    id: int
    uuid: str | None = None
    job_type: str
    status: str
    priority: int
    retry_count: int
    max_retries: int
    owner_user_id: str | None = None
    available_at: str | None = None
    started_at: str | None = None
    leased_until: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


class TTSHistoryListItem(BaseModel):
    id: int
    created_at: str
    has_text: bool
    text_preview: str | None = None
    provider: str | None = None
    model: str | None = None
    voice_id: str | None = None
    voice_name: str | None = None
    voice_info: dict[str, Any] | None = None
    duration_ms: int | None = None
    format: str | None = None
    status: str | None = None
    favorite: bool = False
    job_id: int | None = None
    output_id: int | None = None
    artifact_deleted_at: str | None = None


class TTSHistoryListResponse(BaseModel):
    items: list[TTSHistoryListItem]
    total: int | None = None
    limit: int
    offset: int
    next_cursor: str | None = None


class TTSHistoryDetailResponse(BaseModel):
    id: int
    created_at: str
    has_text: bool
    text: str | None = None
    text_length: int | None = None
    provider: str | None = None
    model: str | None = None
    voice_id: str | None = None
    voice_name: str | None = None
    voice_info: dict[str, Any] | None = None
    format: str | None = None
    duration_ms: int | None = None
    generation_time_ms: int | None = None
    params_json: dict[str, Any] | None = None
    status: str | None = None
    segments_json: dict[str, Any] | None = None
    favorite: bool = False
    job_id: int | None = None
    output_id: int | None = None
    artifact_ids: list[Any] | None = None
    artifact_deleted_at: str | None = None
    error_message: str | None = None


class TTSHistoryFavoriteUpdate(BaseModel):
    favorite: bool


class AudioTranscriptionRequest(BaseModel):
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    hotwords: str | None = None
    response_format: TranscriptResponseFormat = "json"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_granularities: list[Literal["word", "segment"]] = Field(default_factory=lambda: ["segment"])


class AudioTranslationRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    response_format: TranscriptResponseFormat = "json"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class AudioTranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    words: list[Any] | None = None
    segments: list[Any] | None = None


__all__ = [
    "AudioFormat",
    "AudioJobResponse",
    "AudioSpeechArtifact",
    "AudioSpeechJobArtifactsResponse",
    "AudioSpeechJobCreateResponse",
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "AudioTranslationRequest",
    "NormalizationOptions",
    "OpenAISpeechRequest",
    "SubmitAudioJobRequest",
    "SubmitAudioJobResponse",
    "TranscriptResponseFormat",
    "TTSHealthResponse",
    "TTSHistoryDetailResponse",
    "TTSHistoryFavoriteUpdate",
    "TTSHistoryListItem",
    "TTSHistoryListResponse",
    "TTSProvidersResponse",
    "TTSVoicesResponse",
]
