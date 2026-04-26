"""Pydantic schemas for the server Voice Assistant REST surface."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class VoiceAssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ERROR = "error"


class VoiceActionType(str, Enum):
    MCP_TOOL = "mcp_tool"
    WORKFLOW = "workflow"
    CUSTOM = "custom"
    LLM_CHAT = "llm_chat"


class WSIntentMessage(BaseModel):
    type: Literal["intent"] = "intent"
    action_type: VoiceActionType
    command_name: Optional[str] = None
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    requires_confirmation: bool = False


class WSActionResultMessage(BaseModel):
    type: Literal["action_result"] = "action_result"
    success: bool
    action_type: VoiceActionType
    result_data: Optional[dict[str, Any]] = None
    response_text: str
    execution_time_ms: float = 0.0


class VoiceCommandRequest(BaseModel):
    text: str
    persona_id: Optional[str] = None
    session_id: Optional[str] = None
    include_tts: bool = True
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_format: Literal["mp3", "opus", "wav", "pcm"] = "mp3"


class VoiceCommandResponse(BaseModel):
    session_id: str
    success: bool
    transcription: str
    intent: WSIntentMessage
    action_result: WSActionResultMessage
    output_audio: Optional[str] = None
    output_audio_format: Optional[str] = None
    processing_time_ms: float


class VoiceCommandDefinition(BaseModel):
    persona_id: Optional[str] = None
    connection_id: Optional[str] = None
    name: str
    phrases: list[str] = Field(..., min_length=1)
    action_type: VoiceActionType
    action_config: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=100)
    enabled: bool = True
    requires_confirmation: bool = False
    description: Optional[str] = None


class VoiceCommandInfo(BaseModel):
    id: str
    user_id: int
    persona_id: Optional[str] = None
    connection_id: Optional[str] = None
    connection_status: Optional[Literal["ok", "missing"]] = None
    connection_name: Optional[str] = None
    name: str
    phrases: list[str]
    action_type: VoiceActionType
    action_config: dict[str, Any]
    priority: int
    enabled: bool
    requires_confirmation: bool
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class VoiceCommandListResponse(BaseModel):
    commands: list[VoiceCommandInfo]
    total: int


class VoiceCommandToggleRequest(BaseModel):
    enabled: bool


class VoiceCommandValidationStep(BaseModel):
    name: str
    passed: bool
    message: str
    details: Optional[dict[str, Any]] = None


class VoiceCommandValidationResponse(BaseModel):
    command_id: str
    command_name: str
    action_type: VoiceActionType
    valid: bool
    steps: list[VoiceCommandValidationStep]


class VoiceCommandUsage(BaseModel):
    command_id: str
    command_name: Optional[str] = None
    total_invocations: int
    success_count: int
    error_count: int
    avg_response_time_ms: float
    last_used: Optional[datetime] = None


class VoiceSessionInfo(BaseModel):
    session_id: str
    user_id: int
    state: VoiceAssistantState
    created_at: datetime
    last_activity: datetime
    turn_count: int


class VoiceSessionListResponse(BaseModel):
    sessions: list[VoiceSessionInfo]
    total: int


class VoiceAnalyticsTopCommand(BaseModel):
    command_id: str
    command_name: Optional[str] = None
    count: int


class VoiceAnalytics(BaseModel):
    date: str
    total_commands: int
    unique_users: int
    success_rate: float
    avg_response_time_ms: float
    top_commands: list[VoiceAnalyticsTopCommand] = Field(default_factory=list)


class VoiceAnalyticsSummary(BaseModel):
    total_commands_processed: int
    active_sessions: int
    total_voice_commands: int
    enabled_commands: int
    success_rate: float
    avg_response_time_ms: float
    top_commands: list[VoiceCommandUsage]
    usage_by_day: list[VoiceAnalytics]


class VoiceCommandDryRunRequest(BaseModel):
    phrase: str = Field(..., min_length=1, max_length=500)
    command_id: str | None = None


class VoiceCommandDryRunAlternative(BaseModel):
    action_type: str
    confidence: float | None = None
    raw_text: str | None = None


class VoiceCommandDryRunResponse(BaseModel):
    dry_run: bool = True
    phrase: str
    matched: bool
    match_method: str
    matched_phrase: str | None = None
    confidence: float | None = None
    action_type: str
    action_config: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float
    alternatives: list[VoiceCommandDryRunAlternative] = Field(default_factory=list)
