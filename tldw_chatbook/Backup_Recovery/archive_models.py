"""Strict inert version-one recovery manifest; no imported executable policy."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
]


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class Metadata(Record):
    version: int = Field(ge=1, le=1)
    mtime_ns: int = Field(ge=0)
    mode: int = Field(ge=0, le=0o777)


class Directory(Record):
    logical_id: Identifier
    root_id: Identifier
    parent_id: Identifier | None
    relative_path: str
    metadata: Metadata


class Payload(Record):
    logical_id: Identifier
    root_id: Identifier
    parent_id: Identifier
    relative_path: str
    owner_id: Identifier
    payload: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: Metadata | None = None


class ProducerItem(Record):
    """Archived ownership observations; never local destination authority."""

    logical_id: Identifier
    owner_id: Identifier
    status: Literal[
        "included",
        "included_directory",
        "intentionally_excluded",
        "unused",
        "intentionally_deleted",
        "unavailable",
        "unsupported",
        "missing_required",
    ]
    dependencies: tuple[Identifier, ...]
    shared_group: Identifier | None = None


class Owner(Record):
    owner_id: Identifier
    schema_version: int = Field(ge=0)
    capabilities: tuple[Identifier, ...]


class DependencyGroup(Record):
    group_id: Identifier
    members: tuple[Identifier, ...] = Field(min_length=1)
    complete: bool


class Exclusion(Record):
    logical_id: Identifier
    reason: Identifier


class RecoveryReport(Record):
    version: int = Field(ge=1, le=1)
    lines: tuple[str, ...]

    @field_validator("lines")
    @classmethod
    def inert_lines(cls, lines):
        """Reports are plain text, never terminal control or rich markup."""
        if any(
            any(ord(c) < 32 or ord(c) == 127 or c in "[]<>" for c in line)
            for line in lines
        ):
            raise ValueError("unsafe_report")
        return lines


class Relocation(Record):
    logical_id: Identifier
    locator: str


class ArchiveManifest(Record):
    format_version: int = Field(ge=1, le=1)
    producer_version: str = Field(min_length=1, max_length=128)
    captured_at: str = Field(min_length=1, max_length=64)
    profile_ids: tuple[Identifier, ...] = Field(min_length=1)
    owners: tuple[Owner, ...]
    directories: tuple[Directory, ...]
    files: tuple[Payload, ...]
    dependency_groups: tuple[DependencyGroup, ...]
    consistency: Literal["coherent", "partial"]
    exclusions: tuple[Exclusion, ...]
    credential_policy: Literal["exclude", "include", "rollback"]
    required_capabilities: tuple[Identifier, ...]
    report: RecoveryReport
    relocations: tuple[Relocation, ...]
    producer_inventory: tuple[ProducerItem, ...] = ()


@dataclass(frozen=True)
class SealedArchive:
    path: Path
    digest: str
    manifest_bytes: bytes
