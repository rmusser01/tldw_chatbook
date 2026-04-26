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


class VoiceEncodeRequest(BaseModel):
    voice_id: str
    provider: str = "neutts"
    reference_text: str | None = None
    force: bool = False


class VoiceEncodeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    voice_id: str
    provider: str
    cached: bool = False
    ref_codes_len: int | None = None
    reference_text: str | None = None


class CustomVoiceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    voice_id: str | None = None
    id: str | None = None
    name: str | None = None
    provider: str | None = None
    status: str | None = None


class CustomVoiceListResponse(BaseModel):
    voices: list[CustomVoiceResponse] = Field(default_factory=list)
    count: int = 0


class CustomVoiceDeleteResponse(BaseModel):
    message: str
    voice_id: str


class AudioTokenizerEncodeRequest(BaseModel):
    audio_base64: str
    tokenizer_model: str | None = None
    sample_rate: int | None = None
    token_format: Literal["list", "base64"] = "list"


class AudioTokenizerEncodeResponse(BaseModel):
    tokens: Any
    token_format: Literal["list", "base64"]
    sample_rate: int
    frame_rate: float | None = None
    tokenizer_model: str
    duration_seconds: float


class AudioTokenizerDecodeRequest(BaseModel):
    tokens: Any
    tokenizer_model: str | None = None
    response_format: Literal["wav", "pcm"] = "wav"


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


class StreamingStatusFeatures(BaseModel):
    partial_results: bool = True
    multiple_languages: bool = True
    concurrent_streams: bool = True
    segment_metadata: bool = True
    live_insights: bool = True
    meeting_notes: bool = True
    speaker_diarization: bool = True
    audio_persistence: bool = True


class StreamingStatusResponse(BaseModel):
    status: Literal["available", "unavailable", "error"]
    available_models: list[str] = Field(default_factory=list)
    websocket_endpoint: str
    supported_features: StreamingStatusFeatures


class StreamingLimitsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int | str
    tier: str
    limits: dict[str, Any]
    used_today_minutes: float
    remaining_minutes: float | None = None
    active_streams: int
    can_start_stream: bool
    legacy_can_start_stream: bool = Field(..., alias="_can_start_stream")


class StreamingTestResponse(BaseModel):
    status: Literal["success", "error"]
    test_passed: bool
    message: str
    test_result: Any | None = None


class SpeechChatSTTConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    language: str | None = None
    extra_params: dict[str, Any] | None = None


class SpeechChatLLMConfig(BaseModel):
    api_provider: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    extra_params: dict[str, Any] | None = None


class SpeechChatTTSConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    voice: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    extra_params: dict[str, Any] | None = None


class SpeechChatRequest(BaseModel):
    session_id: str | None = None
    input_audio: str
    input_audio_format: str
    stt_config: SpeechChatSTTConfig | None = None
    llm_config: SpeechChatLLMConfig
    tts_config: SpeechChatTTSConfig | None = None
    store_audio: bool | None = False
    metadata: dict[str, Any] | None = None


class SpeechChatTiming(BaseModel):
    stt_ms: float
    llm_ms: float
    tts_ms: float


class SpeechChatTokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class SpeechChatResponse(BaseModel):
    session_id: str
    user_transcript: str
    assistant_text: str
    output_audio: str
    output_audio_mime_type: str
    timing: SpeechChatTiming
    token_usage: SpeechChatTokenUsage | None = None
    metadata: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None


__all__ = [
    "AudioFormat",
    "AudioJobResponse",
    "AudioSpeechArtifact",
    "AudioSpeechJobArtifactsResponse",
    "AudioSpeechJobCreateResponse",
    "AudioTokenizerDecodeRequest",
    "AudioTokenizerEncodeRequest",
    "AudioTokenizerEncodeResponse",
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "AudioTranslationRequest",
    "CustomVoiceDeleteResponse",
    "CustomVoiceListResponse",
    "CustomVoiceResponse",
    "NormalizationOptions",
    "OpenAISpeechRequest",
    "SpeechChatLLMConfig",
    "SpeechChatRequest",
    "SpeechChatResponse",
    "SpeechChatSTTConfig",
    "SpeechChatTTSConfig",
    "SpeechChatTiming",
    "SpeechChatTokenUsage",
    "StreamingLimitsResponse",
    "StreamingStatusFeatures",
    "StreamingStatusResponse",
    "StreamingTestResponse",
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
    "VoiceEncodeRequest",
    "VoiceEncodeResponse",
]
