"""Bounded, immutable persistent listening values shared with inert recovery validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Literal

from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.TTS.audio_limits import MAX_BUFFERED_AUDIO_BYTES
from tldw_chatbook.TTS.controlled_wav_profile import PADDING_FRAMES, RATE, WavBookmark

MAX_ENTRIES = 256
MAX_TEXT_BYTES = 256 * 1024
MAX_SESSION_BYTES = 2 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_KEY = re.compile(r"[0-9a-f]{32}\.wav")


class ListeningConflict(ValueError):
    """A captured authority, session, source or reservation is no longer current."""


def _int(value, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError("invalid_listening_integer")
    return value


def _id(value):
    if type(value) is not str or not _SAFE_ID.fullmatch(value):
        raise ValueError("invalid_listening_identifier")
    return value


def _digest(value):
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError("invalid_listening_digest")
    return value


def _encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def listening_profile_key(selector: Path) -> str:
    """Return the exact lexical selector key used by recovery inventory."""
    return _hash(str(lexical_path(selector)))[:24]


@dataclass(frozen=True, slots=True)
class ListeningAuthority:
    profile: str
    backend: Literal["local"]
    policy_epoch: int

    def __post_init__(self):
        if type(self.profile) is not str or not re.fullmatch(
            r"[0-9a-f]{24}", self.profile
        ):
            raise ValueError("invalid_listening_profile")
        _int(self.policy_epoch)
        if self.backend != "local":
            raise ValueError("local_listening_authority_required")


@dataclass(frozen=True, slots=True)
class VoiceDependency:
    kind: Literal["profile", "reference"]
    identifier: str
    revision: str
    sha256: str

    def __post_init__(self):
        if self.kind not in ("profile", "reference"):
            raise ValueError("invalid_voice_dependency")
        _id(self.identifier)
        _id(self.revision)
        _digest(self.sha256)


@dataclass(frozen=True, slots=True)
class FrozenVoice:
    provider: str
    model: str
    voice: str
    format: str = "wav"
    speed: float = 1.0
    dependencies: tuple[VoiceDependency, ...] = ()
    options: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for value in (self.provider, self.model, self.voice):
            _id(value)
        if self.format not in ("wav", "pcm", "mp3", "flac", "opus", "aac") or (
            type(self.speed) not in (int, float) or not 0.25 <= self.speed <= 4
        ):
            raise ValueError("invalid_frozen_voice")
        if (
            type(self.dependencies) is not tuple
            or len(self.dependencies) > 8
            or any(type(dep) is not VoiceDependency for dep in self.dependencies)
        ):
            raise ValueError("invalid_voice_dependencies")
        if type(self.options) is not tuple or len(self.options) > 3:
            raise ValueError("invalid_voice_options")
        keys = set()
        for pair in self.options:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("invalid_voice_options")
            name, value = pair
            if (
                name not in {"stability", "similarity_boost", "style"}
                or name in keys
                or (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                )
            ):
                raise ValueError("invalid_voice_options")
            keys.add(name)
        object.__setattr__(self, "options", tuple(sorted(self.options)))
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                sorted(self.dependencies, key=lambda dep: (dep.kind, dep.identifier))
            ),
        )
        if len({(d.kind, d.identifier) for d in self.dependencies}) != len(
            self.dependencies
        ):
            raise ValueError("invalid_voice_dependencies")

    @classmethod
    def from_json(cls, value):
        if len(value) > 8192:
            raise ValueError("invalid_frozen_voice")
        fields = json.loads(value)
        fields["dependencies"] = tuple(
            VoiceDependency(**dep) for dep in fields["dependencies"]
        )
        fields["options"] = tuple(tuple(pair) for pair in fields["options"])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class EntryCandidate:
    item_id: int | None
    source_id: int | None
    title: str | None
    text: str | None
    converter: str | None
    voice: FrozenVoice | None
    origin: tuple[ListeningAuthority, Path] | None = None

    def __post_init__(self):
        if self.origin is not None and (
            type(self.origin) is not tuple
            or len(self.origin) != 2
            or type(self.origin[0]) is not ListeningAuthority
            or not isinstance(self.origin[1], Path)
        ):
            raise ValueError("invalid_listening_origin")
        if self.item_id is None:
            if any(
                v is not None
                for v in (
                    self.source_id,
                    self.title,
                    self.text,
                    self.converter,
                    self.voice,
                )
            ):
                raise ValueError("invalid_unavailable_entry")
            return
        _int(self.item_id, 1)
        _int(self.source_id, 1)
        _id(self.converter)
        if (
            type(self.voice) is not FrozenVoice
            or type(self.title) is not str
            or len(self.title.encode()) > 2048
            or (
                type(self.text) is not str
                or not self.text.strip()
                or len(self.text.encode()) > MAX_TEXT_BYTES
            )
        ):
            raise ValueError("invalid_listening_entry")


_UNAVAILABLE = EntryCandidate(None, None, None, None, None, None)


@dataclass(frozen=True, slots=True)
class ListeningCandidate:
    authority: ListeningAuthority
    database: Path
    observed_revision: int
    observed_incarnation: str
    voice: FrozenVoice
    brief_id: int | None
    queue: tuple[EntryCandidate, ...]
    foreground: EntryCandidate | None

    def __post_init__(self):
        if (
            type(self.voice) is not FrozenVoice
            or type(self.authority) is not ListeningAuthority
            or type(self.queue) is not tuple
            or (
                len(self.queue) > MAX_ENTRIES
                or bool(self.queue) != (self.brief_id is not None)
            )
        ):
            raise ValueError("invalid_listening_candidate")
        if self.brief_id is not None:
            _int(self.brief_id, 1)
        entries = (*self.queue, self.foreground) if self.foreground else self.queue
        if (
            any(type(e) is not EntryCandidate for e in entries)
            or sum(len((e.text or "").encode()) for e in self.queue) > MAX_SESSION_BYTES
        ):
            raise ValueError("invalid_listening_candidate")


@dataclass(frozen=True, slots=True)
class ListeningEntry:
    entry_id: str
    item_id: int | None
    source_id: int | None
    title: str | None
    text: str | None
    converter: str | None
    voice: FrozenVoice | None
    artifact: int | None
    bookmark: WavBookmark | None
    availability: Literal["pending", "ready", "missing", "unavailable"]


@dataclass(frozen=True, slots=True)
class ListeningSession:
    revision: int
    incarnation: str
    authority: ListeningAuthority
    voice: FrozenVoice
    playback_generation: int
    brief_id: int | None
    queue: tuple[ListeningEntry, ...]
    foreground: ListeningEntry | None
    cursor: int | None
    resume_queue: bool
    transport: Literal["paused", "completed"]


@dataclass(frozen=True, slots=True)
class ListeningCheckpoint:
    authority: ListeningAuthority
    incarnation: str
    playback_generation: int
    cursor: int | None
    resume_queue: bool
    transport: Literal["paused", "playing", "preparing", "completed"]
    bookmarks: tuple[tuple[str, WavBookmark], ...] = ()
    foreground: EntryCandidate | Literal["keep", "remove"] = "keep"

    def __post_init__(self):
        if (
            type(self.authority) is not ListeningAuthority
            or type(self.resume_queue) is not bool
            or (
                self.transport not in ("paused", "playing", "preparing", "completed")
                or type(self.bookmarks) is not tuple
                or len(self.bookmarks) > 2
            )
        ):
            raise ValueError("invalid_listening_checkpoint")
        _id(self.incarnation)
        _int(self.playback_generation)
        if self.cursor is not None:
            _int(self.cursor)
        if type(self.foreground) is not EntryCandidate and self.foreground not in (
            "keep",
            "remove",
        ):
            raise ValueError("invalid_listening_checkpoint")
        ids = set()
        for pair in self.bookmarks:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("invalid_listening_checkpoint")
            entry_id, point = pair
            _id(entry_id)
            if (
                type(point) is not WavBookmark
                or entry_id in ids
                or point.sample_rate != RATE
            ):
                raise ValueError("invalid_listening_bookmark")
            _digest(point.sha256)
            _int(point.frame)
            ids.add(entry_id)


@dataclass(frozen=True, slots=True)
class AudioReservation:
    generation: int
    file_key: str
    entry_id: str
    item_id: int
    revision: int
    incarnation: str
    authority: ListeningAuthority
    database: Path


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    sha256: str
    sample_rate: int
    content_frames: int
    file_frames: int
    byte_size: int

    def __post_init__(self):
        _digest(self.sha256)
        for n in (
            self.sample_rate,
            self.content_frames,
            self.file_frames,
            self.byte_size,
        ):
            _int(n, 1)
        if (
            self.sample_rate != RATE
            or self.file_frames != self.content_frames + PADDING_FRAMES
            or not (
                44 + self.file_frames * 4 <= self.byte_size <= MAX_BUFFERED_AUDIO_BYTES
            )
        ):
            raise ValueError("invalid_article_audio_shape")
