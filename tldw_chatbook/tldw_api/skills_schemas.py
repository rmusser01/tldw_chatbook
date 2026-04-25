from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
SUPPORTING_FILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
MAX_SUPPORTING_FILES_COUNT = 20
MAX_SUPPORTING_FILE_BYTES = 500000
MAX_SUPPORTING_FILES_TOTAL_BYTES = 5 * 1024 * 1024


def _normalize_skill_name(value: str) -> str:
    normalized = value.strip().lower()
    if not SKILL_NAME_PATTERN.match(normalized):
        raise ValueError(
            "Skill name must start with a letter, contain only lowercase letters, "
            "numbers, and hyphens, and be 1-64 characters long"
        )
    return normalized


def _validate_supporting_files(
    value: dict[str, str | None] | None,
    *,
    allow_deletes: bool,
) -> dict[str, str | None] | None:
    if value is None:
        return None
    non_null_count = 0
    total_bytes = 0
    for filename, content in value.items():
        if not SUPPORTING_FILE_NAME_PATTERN.match(filename):
            raise ValueError(f"Invalid supporting file name: {filename}")
        if filename.lower() == "skill.md":
            raise ValueError("SKILL.md cannot be a supporting file")
        if content is None:
            if allow_deletes:
                continue
            raise ValueError(f"Supporting file {filename} cannot be null")
        non_null_count += 1
        if non_null_count > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError(f"Too many supporting files ({non_null_count}); maximum is {MAX_SUPPORTING_FILES_COUNT}")
        file_bytes = len(content.encode("utf-8"))
        if file_bytes > MAX_SUPPORTING_FILE_BYTES:
            raise ValueError(f"Supporting file {filename} exceeds 500KB limit")
        total_bytes += file_bytes
    if total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
        raise ValueError("Total supporting files size exceeds 5MB limit")
    return value


class SkillBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(None, max_length=1000)
    argument_hint: str | None = Field(None, max_length=100)
    disable_model_invocation: bool = False
    user_invocable: bool = True
    allowed_tools: list[str] | None = None
    model: str | None = None
    context: Literal["inline", "fork"] = "inline"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _normalize_skill_name(value)


class SkillCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=500000)
    supporting_files: dict[str, str] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _normalize_skill_name(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        validated = _validate_supporting_files(value, allow_deletes=False)
        return validated  # type: ignore[return-value]


class SkillUpdate(BaseModel):
    content: str | None = Field(None, min_length=1, max_length=500000)
    supporting_files: dict[str, str | None] | None = None

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: dict[str, str | None] | None) -> dict[str, str | None] | None:
        return _validate_supporting_files(value, allow_deletes=True)


class SkillResponse(SkillBase):
    id: str
    content: str
    supporting_files: dict[str, str] | None = None
    directory_path: str
    created_at: datetime
    last_modified: datetime
    version: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SkillSummary(BaseModel):
    name: str
    description: str | None = None
    argument_hint: str | None = None
    user_invocable: bool
    disable_model_invocation: bool
    context: Literal["inline", "fork"]

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SkillsListResponse(BaseModel):
    skills: list[SkillSummary]
    count: int
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SkillExecuteRequest(BaseModel):
    args: str | None = Field(None, max_length=10000)


class SkillExecutionResult(BaseModel):
    skill_name: str
    rendered_prompt: str
    allowed_tools: list[str] | None = None
    model_override: str | None = None
    execution_mode: Literal["inline", "fork"]
    fork_output: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SkillImportRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=500000)
    supporting_files: dict[str, str] | None = None
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_skill_name(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        validated = _validate_supporting_files(value, allow_deletes=False)
        return validated  # type: ignore[return-value]


class SkillContextPayload(BaseModel):
    available_skills: list[SkillSummary]
    context_text: str

    model_config = ConfigDict(from_attributes=True, extra="allow")
