"""Schemas for the server chat-loop execution/control contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ChatLoopEventType = Literal[
    "run_started",
    "llm_chunk",
    "llm_complete",
    "tool_proposed",
    "approval_required",
    "approval_resolved",
    "tool_started",
    "tool_finished",
    "tool_failed",
    "assistant_message_committed",
    "run_complete",
    "run_error",
    "run_cancelled",
]
ChatLoopApprovalDecision = Literal["approve", "reject"]


class ChatLoopEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    seq: int = Field(..., ge=1)
    ts: datetime = Field(default_factory=datetime.utcnow)
    event: ChatLoopEventType
    data: dict[str, Any] = Field(default_factory=dict)


class ChatLoopStartRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]] = Field(...)

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class ChatLoopStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)


class ChatLoopApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(..., min_length=1)
    decision: ChatLoopApprovalDecision


class ChatLoopActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True


class ChatLoopEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    events: list[ChatLoopEvent] = Field(default_factory=list)
