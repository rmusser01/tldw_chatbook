"""Process-local typed handoff values for audio.cpp Model Library selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
import re
import unicodedata

from ...Model_Artifacts.service import ArtifactRef


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)


def _validate_token(value: object) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError("audio.cpp Model Library token is invalid")


def _validate_draft_revision(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("audio.cpp Model Library draft revision is invalid")


def _validate_canonical_root(value: object) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 4096
        or any(
            character in {"\x00", "\r", "\n"}
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in value
        )
    ):
        raise ValueError("audio.cpp Model Library root is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if not (posix.is_absolute() or windows.is_absolute()):
        raise ValueError("audio.cpp Model Library root must be absolute")
    if any(part in {".", ".."} for part in (*posix.parts, *windows.parts)):
        raise ValueError("audio.cpp Model Library root must be canonical")
    if posix.is_absolute() and (
        len(posix.parts) < 2 or "\\" in value or posix.as_posix() != value
    ):
        raise ValueError("audio.cpp Model Library root must be canonical")
    if windows.is_absolute() and (len(windows.parts) < 2 or str(windows) != value):
        raise ValueError("audio.cpp Model Library root must be canonical")


@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryRequest:
    """Opaque Settings request to browse reviewed audio.cpp packages."""

    token: str
    draft_revision: int

    def __post_init__(self) -> None:
        _validate_token(self.token)
        _validate_draft_revision(self.draft_revision)


@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryResult:
    """Exact installed package returned to the originating Settings draft."""

    token: str
    draft_revision: int
    artifact_id: str
    revision: str
    variant: str
    canonical_root: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_token(self.token)
        _validate_draft_revision(self.draft_revision)
        ArtifactRef(self.artifact_id, self.revision, self.variant)
        _validate_canonical_root(self.canonical_root)
