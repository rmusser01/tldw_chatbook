"""
Character, persona, and chat-session API schemas for the shared TLDW client.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationInfo, field_validator, model_validator


PersonaMode = Literal["session_scoped", "persistent_scoped"]
PersonaSessionStatus = Literal["active", "paused", "closed", "archived"]
PersonaExemplarKind = Literal["style", "catchphrase", "boundary", "scenario_demo", "tool_behavior"]
PersonaExemplarSourceType = Literal["manual", "transcript_import", "character_seed", "generated_candidate"]
PersonaExemplarReviewAction = Literal["approve", "reject"]
PersonaConfirmationMode = Literal["always", "destructive_only", "never"]
PersonaSetupStatus = Literal["not_started", "in_progress", "completed"]
PersonaSetupStep = Literal["archetype", "persona", "voice", "commands", "safety", "test"]
PersonaSetupTestType = Literal["dry_run", "live_session"]
CharacterChatSessionState = Literal["in-progress", "resolved", "backlog", "non-viable"]
CharacterAssistantKind = Literal["character", "persona"]
PersonaMemoryMode = Literal["read_only", "read_write"]


def _strip_optional_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return stripped or None


def _parse_jsonish_collection(value: Any, *, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            if field_name in {"alternate_greetings", "tags"}:
                return []
            if field_name == "extensions":
                return {}
            return value
        parsed = json.loads(value)
        value = parsed
    return value


def _normalize_chat_session_state(value: str | None) -> CharacterChatSessionState | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("state cannot be empty")
    allowed = {"in-progress", "resolved", "backlog", "non-viable"}
    if normalized not in allowed:
        raise ValueError("state must be one of: in-progress, resolved, backlog, non-viable")
    return normalized  # type: ignore[return-value]


class CharacterBase(BaseModel):
    name: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=50_000)
    personality: Optional[str] = Field(None, max_length=50_000)
    scenario: Optional[str] = Field(None, max_length=50_000)
    system_prompt: Optional[str] = Field(None, max_length=100_000)
    post_history_instructions: Optional[str] = Field(None, max_length=100_000)
    first_message: Optional[str] = Field(None, max_length=50_000)
    message_example: Optional[str] = Field(None, max_length=100_000)
    creator_notes: Optional[str] = Field(None, max_length=50_000)
    alternate_greetings: list[str] | str | None = None
    tags: list[str] | str | None = None
    creator: Optional[str] = Field(None, max_length=500)
    character_version: Optional[str] = Field(None, max_length=100)
    extensions: dict[str, Any] | str | None = None
    image_base64: Optional[str] = Field(None, max_length=20_000_000)

    @field_validator("alternate_greetings", "tags", "extensions", mode="before")
    @classmethod
    def _parse_jsonish_fields(cls, value: Any, info: ValidationInfo) -> Any:
        value = _parse_jsonish_collection(value, field_name=info.field_name)
        return value


class CharacterCreateRequest(CharacterBase):
    name: str = Field(..., min_length=1, max_length=500)


class CharacterUpdateRequest(CharacterBase):
    pass


class CharacterResponse(CharacterBase):
    id: int = Field(..., gt=0)
    version: int = Field(default=1, ge=1)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_modified: datetime | None = None
    image_present: bool = False

    model_config = {"from_attributes": True}


class CharacterQueryRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    query: str | None = None
    tags: list[str] = Field(default_factory=list)
    match_all_tags: bool = False
    creator: str | None = None
    has_conversations: bool | None = None
    favorite_only: bool = False
    created_from: str | None = None
    created_to: str | None = None
    updated_from: str | None = None
    updated_to: str | None = None
    include_deleted: bool = False
    deleted_only: bool = False
    sort_by: Literal["name", "creator", "created_at", "updated_at", "last_used_at", "conversation_count"] = "name"
    sort_order: Literal["asc", "desc"] = "asc"
    include_image_base64: bool = False


class CharacterQueryResponse(BaseModel):
    items: list[CharacterResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 25
    has_more: bool = False


CharacterListResponse = list[CharacterResponse]


class CharacterDeleteResponse(BaseModel):
    message: str
    character_id: int


class CharacterRestoreRequest(BaseModel):
    expected_version: int = Field(..., ge=1)


class CharacterExemplarSource(BaseModel):
    type: Literal["audio_transcript", "video_transcript", "article", "other"] = "other"
    url_or_id: str | None = None
    date: str | None = None


class CharacterExemplarLabels(BaseModel):
    emotion: Literal["angry", "neutral", "happy", "other"] = "other"
    scenario: Literal["press_challenge", "fan_banter", "debate", "boardroom", "small_talk", "other"] = "other"
    rhetorical: list[str] = Field(default_factory=list)
    register_: str | None = Field(default=None, alias="register", serialization_alias="register")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class CharacterExemplarSafety(BaseModel):
    allowed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)


class CharacterExemplarRights(BaseModel):
    public_figure: bool = True
    notes: str | None = None


class CharacterExemplarCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    source: CharacterExemplarSource = Field(default_factory=CharacterExemplarSource)
    novelty_hint: Literal["post_cutoff", "unknown", "pre_cutoff"] = "unknown"
    labels: CharacterExemplarLabels = Field(default_factory=CharacterExemplarLabels)
    safety: CharacterExemplarSafety = Field(default_factory=CharacterExemplarSafety)
    rights: CharacterExemplarRights = Field(default_factory=CharacterExemplarRights)
    length_tokens: int | None = Field(default=None, ge=1, le=10_000)

    model_config = {"extra": "forbid"}


class CharacterExemplarUpdate(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=100_000)
    source: CharacterExemplarSource | None = None
    novelty_hint: Literal["post_cutoff", "unknown", "pre_cutoff"] | None = None
    labels: CharacterExemplarLabels | None = None
    safety: CharacterExemplarSafety | None = None
    rights: CharacterExemplarRights | None = None
    length_tokens: int | None = Field(default=None, ge=1, le=10_000)


class CharacterExemplarResponse(CharacterExemplarCreate):
    id: str
    character_id: int
    created_at: datetime
    updated_at: datetime | None = None


class CharacterExemplarSearchFilter(BaseModel):
    emotion: Literal["angry", "neutral", "happy", "other"] | None = None
    scenario: Literal["press_challenge", "fan_banter", "debate", "boardroom", "small_talk", "other"] | None = None
    rhetorical: list[str] = Field(default_factory=list)


class CharacterExemplarSearchRequest(BaseModel):
    query: str | None = None
    filter: CharacterExemplarSearchFilter = Field(default_factory=CharacterExemplarSearchFilter)
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    use_embedding_scores: bool = False
    embedding_model_id: str | None = None


class CharacterExemplarSearchResponse(BaseModel):
    items: list[CharacterExemplarResponse] = Field(default_factory=list)
    total: int = 0


class CharacterExemplarSelectionConfig(BaseModel):
    budget_tokens: int = Field(default=600, ge=1, le=20_000)
    max_exemplar_tokens: int = Field(default=120, ge=1, le=20_000)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    use_embedding_scores: bool = False
    embedding_model_id: str | None = None


class CharacterExemplarSelectionDebugRequest(BaseModel):
    user_turn: str = Field(..., min_length=1, max_length=100_000)
    selection_config: CharacterExemplarSelectionConfig = Field(default_factory=CharacterExemplarSelectionConfig)


class CharacterExemplarCoverage(BaseModel):
    openers: int = 0
    emphasis: int = 0
    enders: int = 0
    catchphrases_used: int = 0


class CharacterExemplarScore(BaseModel):
    id: str
    score: float


class CharacterExemplarSelectionDebug(BaseModel):
    selected: list[CharacterExemplarResponse] = Field(default_factory=list)
    budget_tokens: int = Field(..., ge=0)
    coverage: CharacterExemplarCoverage = Field(default_factory=CharacterExemplarCoverage)
    scores: list[CharacterExemplarScore] = Field(default_factory=list)


class CharacterExemplarDeletionResponse(BaseModel):
    message: str
    character_id: int
    exemplar_id: str


class PersonaBuddyVisualSummary(BaseModel):
    """Compact visual traits used to render a persona buddy preview."""

    species_id: StrictStr
    silhouette_id: StrictStr
    palette_id: StrictStr
    accessory_id: StrictStr | None = None
    eye_style: StrictStr | None = None
    expression_profile: StrictStr | None = None


class PersonaBuddySummary(BaseModel):
    """Small buddy summary embedded into persona profile and catalog responses."""

    has_buddy: bool = False
    persona_name: str
    role_summary: str | None = None
    visual: PersonaBuddyVisualSummary | None = None


class PersonaVoiceDefaults(BaseModel):
    stt_language: StrictStr | None = None
    stt_model: StrictStr | None = None
    tts_provider: StrictStr | None = None
    tts_voice: StrictStr | None = None
    confirmation_mode: PersonaConfirmationMode | None = None
    voice_chat_trigger_phrases: list[str] = Field(default_factory=list)
    auto_resume: bool | None = None
    barge_in: bool | None = None
    auto_commit_enabled: bool | None = None
    vad_threshold: float | None = None
    min_silence_ms: int | None = None
    turn_stop_secs: float | None = None
    min_utterance_secs: float | None = None

    @field_validator("stt_language", "stt_model", "tts_provider", "tts_voice", mode="before")
    @classmethod
    def _strip_optional_text_fields(cls, value: Any) -> Any:
        return _strip_optional_text(value)

    @field_validator("voice_chat_trigger_phrases", mode="before")
    @classmethod
    def _normalize_trigger_phrases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        seen: set[str] = set()
        normalized: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @field_validator("vad_threshold", "turn_stop_secs", "min_utterance_secs", mode="before")
    @classmethod
    def _normalize_turn_detection_floats(cls, value: Any, info: ValidationInfo) -> float | None:
        if value is None or value == "":
            return None
        bounds = {
            "vad_threshold": (0.0, 1.0),
            "turn_stop_secs": (0.05, 10.0),
            "min_utterance_secs": (0.0, 10.0),
        }
        min_value, max_value = bounds[info.field_name]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(min_value, min(max_value, numeric))

    @field_validator("min_silence_ms", mode="before")
    @classmethod
    def _normalize_min_silence_ms(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return max(50, min(10_000, numeric))


class PersonaSetupState(BaseModel):
    """Persisted wizard progress for one persona setup run."""

    status: PersonaSetupStatus = "not_started"
    version: int = Field(default=1, ge=1)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    current_step: PersonaSetupStep = "persona"
    completed_steps: list[PersonaSetupStep] = Field(default_factory=list)
    completed_at: str | None = None
    last_test_type: PersonaSetupTestType | None = None


class PersonaProfileCreate(BaseModel):
    id: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    archetype_key: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    mode: PersonaMode = "session_scoped"
    system_prompt: str | None = None
    is_active: bool = True
    use_persona_state_context_default: bool = True
    voice_defaults: PersonaVoiceDefaults = Field(default_factory=PersonaVoiceDefaults)
    setup: PersonaSetupState = Field(default_factory=PersonaSetupState)


class PersonaProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    mode: PersonaMode | None = None
    system_prompt: str | None = None
    is_active: bool | None = None
    use_persona_state_context_default: bool | None = None
    voice_defaults: PersonaVoiceDefaults | None = None
    setup: PersonaSetupState | None = None


class PersonaProfileResponse(BaseModel):
    id: StrictStr
    name: str
    archetype_key: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    origin_character_id: int | None = None
    origin_character_name: str | None = None
    origin_character_snapshot_at: str | None = None
    mode: PersonaMode
    system_prompt: str | None = None
    is_active: bool = True
    use_persona_state_context_default: bool = True
    voice_defaults: PersonaVoiceDefaults = Field(default_factory=PersonaVoiceDefaults)
    setup: PersonaSetupState = Field(default_factory=PersonaSetupState)
    created_at: str
    last_modified: str
    version: int = 1
    buddy_summary: PersonaBuddySummary | None = None


class PersonaProfileDeleteResponse(BaseModel):
    status: str
    persona_id: StrictStr


class PersonaInfo(BaseModel):
    id: StrictStr
    name: str
    description: str | None = None
    voice: str | None = None
    avatar_url: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    default_tools: list[str] = Field(default_factory=list)
    buddy_summary: PersonaBuddySummary | None = None


class PersonaSessionRequest(BaseModel):
    persona_id: StrictStr
    project_id: StrictStr | None = None
    resume_session_id: StrictStr | None = None
    surface: str | None = Field(default=None, max_length=120)

    @field_validator("project_id", "resume_session_id", "surface", mode="before")
    @classmethod
    def _strip_optional_text_fields(cls, value: Any) -> Any:
        return _strip_optional_text(value)


class PersonaSessionResponse(BaseModel):
    session_id: StrictStr
    persona: PersonaInfo
    scopes: list[str] = Field(default_factory=list)
    runtime_mode: PersonaMode | None = None
    scope_snapshot_id: str | None = None
    scope_audit: dict[str, Any] = Field(default_factory=dict)


class PersonaSessionSummary(BaseModel):
    session_id: StrictStr
    persona_id: StrictStr
    created_at: str
    updated_at: str
    turn_count: int = 0
    pending_plan_count: int = 0
    preferences: dict[str, Any] = Field(default_factory=dict)
    runtime_mode: PersonaMode | None = None
    status: PersonaSessionStatus | None = None
    reuse_allowed: bool | None = None
    scope_snapshot_id: str | None = None
    scope_audit: dict[str, Any] = Field(default_factory=dict)


class PersonaSessionDetail(PersonaSessionSummary):
    turns: list[dict[str, Any]] = Field(default_factory=list)


class CharacterChatSessionCreate(BaseModel):
    """Request body for creating a character/persona chat session."""

    character_id: int | None = Field(None, gt=0)
    assistant_kind: CharacterAssistantKind | None = None
    assistant_id: str | None = Field(None, min_length=1)
    persona_memory_mode: PersonaMemoryMode | None = None
    title: str | None = None
    parent_conversation_id: str | None = None
    forked_from_message_id: str | None = None
    state: CharacterChatSessionState | None = None
    topic_label: str | None = None
    cluster_id: str | None = None
    source: str | None = None
    external_ref: str | None = None
    scope_type: Literal["global", "workspace"] | None = None
    workspace_id: str | None = None

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: str | None) -> CharacterChatSessionState | None:
        return _normalize_chat_session_state(value)

    @field_validator(
        "assistant_id",
        "title",
        "parent_conversation_id",
        "forked_from_message_id",
        "topic_label",
        "cluster_id",
        "source",
        "external_ref",
        "workspace_id",
        mode="before",
    )
    @classmethod
    def _strip_optional_text_fields(cls, value: Any) -> Any:
        return _strip_optional_text(value)

    @model_validator(mode="after")
    def _normalize_assistant_identity(self) -> "CharacterChatSessionCreate":
        if self.assistant_kind is None:
            self.assistant_kind = "character" if self.character_id is not None else None
        if self.assistant_kind is None:
            raise ValueError("Provide either character_id or assistant_kind + assistant_id.")

        if self.assistant_kind == "character":
            if self.character_id is None:
                if not self.assistant_id:
                    raise ValueError("Character chats require character_id or a numeric assistant_id.")
                try:
                    self.character_id = int(self.assistant_id)
                except ValueError as exc:
                    raise ValueError("Character assistant_id must be numeric.") from exc
            self.assistant_id = str(self.character_id)
            if self.persona_memory_mode is not None:
                raise ValueError("persona_memory_mode is only valid for persona chats.")
            return self

        if not self.assistant_id:
            raise ValueError("Persona chats require assistant_id.")
        self.character_id = None
        return self


class CharacterChatSessionUpdate(BaseModel):
    """Request body for updating character/persona chat metadata."""

    title: str | None = None
    rating: int | None = Field(None, ge=1, le=5)
    state: CharacterChatSessionState | None = None
    topic_label: str | None = None
    cluster_id: str | None = None
    source: str | None = None
    external_ref: str | None = None

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: str | None) -> CharacterChatSessionState | None:
        return _normalize_chat_session_state(value)

    @field_validator("title", "topic_label", "cluster_id", "source", "external_ref", mode="before")
    @classmethod
    def _strip_optional_text_fields(cls, value: Any) -> Any:
        return _strip_optional_text(value)


class ChatSettingsUpdate(BaseModel):
    """Request body for replacing or merging server-side character-chat settings."""

    settings: dict[str, Any] = Field(default_factory=dict)


class PersonaExemplarCreate(BaseModel):
    id: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    kind: PersonaExemplarKind = "style"
    content: str = Field(..., min_length=1, max_length=20_000)
    tone: str | None = Field(default=None, min_length=1, max_length=200)
    scenario_tags: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    source_type: PersonaExemplarSourceType = "manual"
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)

    model_config = {"extra": "forbid"}


class PersonaExemplarUpdate(BaseModel):
    kind: PersonaExemplarKind | None = None
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    tone: str | None = Field(default=None, min_length=1, max_length=200)
    scenario_tags: list[str] | None = None
    capability_tags: list[str] | None = None
    priority: int | None = None
    enabled: bool | None = None
    source_type: PersonaExemplarSourceType | None = None
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)


class PersonaExemplarResponse(BaseModel):
    id: StrictStr
    persona_id: StrictStr
    user_id: StrictStr
    kind: PersonaExemplarKind
    content: str
    tone: str | None = None
    scenario_tags: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    source_type: PersonaExemplarSourceType
    source_ref: str | None = None
    notes: str | None = None
    created_at: str
    last_modified: str
    deleted: bool = False
    version: int = 1


class PersonaExemplarImportRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=100_000)
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)
    max_candidates: int = Field(default=5, ge=1, le=10)


class PersonaExemplarReviewRequest(BaseModel):
    action: PersonaExemplarReviewAction
    notes: str | None = Field(default=None, max_length=10_000)


class PersonaExemplarDeleteResponse(BaseModel):
    status: str
    persona_id: StrictStr
    exemplar_id: StrictStr


class GreetingItem(BaseModel):
    index: int
    text: str
    preview: str


class GreetingListResponse(BaseModel):
    chat_id: StrictStr
    character_id: StrictStr | None = None
    character_name: str | None = None
    greetings: list[GreetingItem] = Field(default_factory=list)
    current_selection: int | None = None
    staleness_warning: str | None = None


class GreetingSelectRequest(BaseModel):
    index: int


class GreetingSelectResponse(BaseModel):
    chat_id: StrictStr
    selected_index: int
    greeting_preview: str
    checksum_updated: bool


class PresetTokenInfo(BaseModel):
    token: str
    description: str


class PresetDetail(BaseModel):
    preset_id: StrictStr
    name: str
    builtin: bool = False
    section_order: list[str] = Field(default_factory=list)
    section_templates: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class PresetListResponse(BaseModel):
    presets: list[PresetDetail] = Field(default_factory=list)


class PresetCreate(BaseModel):
    preset_id: StrictStr = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=256)
    section_order: list[str]
    section_templates: dict[str, str]

    @field_validator("preset_id")
    @classmethod
    def _validate_preset_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized in ("default", "st_default"):
            raise ValueError("Cannot use a built-in preset ID")
        return normalized


class PresetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    section_order: list[str] | None = None
    section_templates: dict[str, str] | None = None


__all__ = [
    "CharacterBase",
    "CharacterAssistantKind",
    "CharacterChatSessionCreate",
    "CharacterChatSessionState",
    "CharacterChatSessionUpdate",
    "CharacterCreateRequest",
    "CharacterUpdateRequest",
    "CharacterResponse",
    "CharacterListResponse",
    "CharacterQueryRequest",
    "CharacterQueryResponse",
    "CharacterDeleteResponse",
    "CharacterRestoreRequest",
    "CharacterExemplarSource",
    "CharacterExemplarLabels",
    "CharacterExemplarSafety",
    "CharacterExemplarRights",
    "CharacterExemplarCreate",
    "CharacterExemplarUpdate",
    "CharacterExemplarResponse",
    "CharacterExemplarSearchFilter",
    "CharacterExemplarSearchRequest",
    "CharacterExemplarSearchResponse",
    "CharacterExemplarSelectionConfig",
    "CharacterExemplarSelectionDebugRequest",
    "CharacterExemplarCoverage",
    "CharacterExemplarScore",
    "CharacterExemplarSelectionDebug",
    "CharacterExemplarDeletionResponse",
    "PersonaMode",
    "PersonaSessionStatus",
    "PersonaProfileCreate",
    "PersonaProfileUpdate",
    "PersonaProfileResponse",
    "PersonaProfileDeleteResponse",
    "PersonaInfo",
    "PersonaSessionRequest",
    "PersonaSessionResponse",
    "PersonaSessionSummary",
    "PersonaSessionDetail",
    "PersonaExemplarKind",
    "PersonaExemplarSourceType",
    "PersonaExemplarReviewAction",
    "PersonaExemplarCreate",
    "PersonaExemplarUpdate",
    "PersonaExemplarResponse",
    "PersonaExemplarImportRequest",
    "PersonaExemplarReviewRequest",
    "PersonaExemplarDeleteResponse",
    "GreetingItem",
    "GreetingListResponse",
    "GreetingSelectRequest",
    "GreetingSelectResponse",
    "ChatSettingsUpdate",
    "PresetTokenInfo",
    "PresetDetail",
    "PresetListResponse",
    "PresetCreate",
    "PresetUpdate",
]
