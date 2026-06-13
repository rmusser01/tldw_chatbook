from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Text2SQLRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query or SQL text")
    target_id: str = Field(..., min_length=1, description="Approved SQL target identifier")
    max_rows: int = Field(default=100, ge=1, le=1000)
    timeout_ms: int = Field(default=5000, ge=100, le=30000)
    include_sql: bool = Field(default=True, description="Whether to include executed SQL in response")


class Text2SQLGuardrail(BaseModel):
    limit_injected: bool = False
    limit_clamped: bool = False


class Text2SQLResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    duration_ms: int
    target_id: str
    guardrail: Text2SQLGuardrail = Field(default_factory=Text2SQLGuardrail)
    truncated: bool = False
