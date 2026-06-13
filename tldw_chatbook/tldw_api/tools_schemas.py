from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolInfo(BaseModel):
    name: str
    description: str | None = None
    module: str | None = None
    inputSchema: dict[str, Any] | None = None
    canExecute: bool = Field(False, description="Whether current user can execute this tool")


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]


class ExecuteToolRequest(BaseModel):
    tool_name: str = Field(..., description="Tool name (registry id)")
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(
        default=None,
        description="Optional key for deduplicating write-capable tools",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, only checks permission and validates args when possible without executing",
    )


class ExecuteToolResult(BaseModel):
    ok: bool
    result: Any | None = None
    module: str | None = None
    error: str | None = None
