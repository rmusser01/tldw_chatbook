from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
SUPPORTING_FILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
SEGMENT_PATTERN = SUPPORTING_FILE_NAME_PATTERN  # each path segment obeys the same rule
MAX_SUPPORTING_FILES_COUNT = 500
MAX_SUPPORTING_FILE_BYTES = 5 * 1024 * 1024
MAX_SUPPORTING_FILES_TOTAL_BYTES = 25 * 1024 * 1024
MAX_SUPPORTING_FILE_PATH_DEPTH = 8
MAX_SUPPORTING_FILE_PATH_LEN = 255
_RESERVED_BODY_BASENAME = "skill.md"


def validate_supporting_file_path(path: str) -> str:
    """Validate a relative POSIX supporting-file subpath, returning it normalized.

    Args:
        path: Candidate relative POSIX path (e.g. ``scripts/build.sh``).

    Returns:
        The same path when valid.

    Raises:
        ValueError: On absolute paths, ``..``/``.``/empty segments, backslashes,
            a segment failing ``SEGMENT_PATTERN``, any-case ``skill.md`` basename,
            depth greater than ``MAX_SUPPORTING_FILE_PATH_DEPTH``, or a total
            length exceeding ``MAX_SUPPORTING_FILE_PATH_LEN``.
    """
    if not path or path != path.strip():
        raise ValueError(f"Invalid supporting file path: {path!r}")
    if "\\" in path or path.startswith("/"):
        raise ValueError(f"Invalid supporting file path: {path!r}")
    if len(path.encode("utf-8")) > MAX_SUPPORTING_FILE_PATH_LEN:
        raise ValueError(f"Supporting file path too long: {path!r}")
    segments = path.split("/")
    if len(segments) > MAX_SUPPORTING_FILE_PATH_DEPTH:
        raise ValueError(f"Supporting file path too deep: {path!r}")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ValueError(f"Invalid path segment in {path!r}")
        if not SEGMENT_PATTERN.fullmatch(segment):
            raise ValueError(f"Invalid path segment {segment!r} in {path!r}")
    if segments[-1].lower() == _RESERVED_BODY_BASENAME:
        raise ValueError("SKILL.md is the skill body, not a supporting file")
    return path


def _normalize_skill_name(value: str) -> str:
    normalized = value.strip().lower()
    if not SKILL_NAME_PATTERN.match(normalized) or "--" in normalized:
        raise ValueError(
            "Skill name must contain only lowercase letters, numbers, and hyphens, "
            "be 1-64 characters long, and not start or end with a hyphen"
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
        validate_supporting_file_path(filename)  # replaces pattern + skill.md checks
        if content is None:
            if allow_deletes:
                continue
            raise ValueError(f"Supporting file {filename} cannot be null")
        non_null_count += 1
        if non_null_count > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError(
                f"Too many supporting files ({non_null_count}); maximum is {MAX_SUPPORTING_FILES_COUNT}"
            )
        file_bytes = len(content.encode("utf-8"))
        if file_bytes > MAX_SUPPORTING_FILE_BYTES:
            raise ValueError(f"Supporting file {filename} exceeds file size limit")
        total_bytes += file_bytes
    if total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
        raise ValueError("Total supporting files size exceeds bundle limit")
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
    def validate_supporting_files(
        cls, value: dict[str, str] | None
    ) -> dict[str, str] | None:
        validated = _validate_supporting_files(value, allow_deletes=False)
        return validated  # type: ignore[return-value]


class SkillUpdate(BaseModel):
    content: str | None = Field(None, min_length=1, max_length=500000)
    supporting_files: dict[str, str | None] | None = None

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(
        cls, value: dict[str, str | None] | None
    ) -> dict[str, str | None] | None:
        return _validate_supporting_files(value, allow_deletes=True)


class BundleFileInfo(BaseModel):
    path: str
    size: int
    executable: bool = False
    is_text: bool = True


class SkillResponse(SkillBase):
    id: str
    content: str
    supporting_files: dict[str, str] | None = None
    directory_path: str
    created_at: datetime
    last_modified: datetime
    version: int
    bundle_files: list[BundleFileInfo] | None = None

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
    def validate_supporting_files(
        cls, value: dict[str, str] | None
    ) -> dict[str, str] | None:
        validated = _validate_supporting_files(value, allow_deletes=False)
        return validated  # type: ignore[return-value]


class SkillContextPayload(BaseModel):
    available_skills: list[SkillSummary]
    context_text: str

    model_config = ConfigDict(from_attributes=True, extra="allow")
