from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias
from uuid import UUID

from tldw_chatbook.TTS.effective_settings import TTSStudioDraftSelection
from tldw_chatbook.TTS.adapter_types import TTSCloneGenerationEvidence
from tldw_chatbook.TTS.profile_types import (
    AUDIO_CPP_PROFILE_SPEED,
    PROFILE_PROVIDER_FORMATS,
    PROFILE_PROVIDER_IDS,
    PROFILE_PROVIDER_REQUIRES_EXACT_VOICE,
)
from tldw_chatbook.TTS.profile_reference_types import CanonicalTTSCloneReference
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot

AudioMetadataValue = str | int | float | bool | None

ProfileSaveBlockCode: TypeAlias = Literal["provider_options"]
#: The one reason a successful generation is refused profile provenance that
#: the user can act on: slice 1 profiles hold empty options by design, so a
#: generation that used provider-specific options cannot be reproduced by one.
PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS: Final[ProfileSaveBlockCode] = "provider_options"
PROFILE_SAVE_BLOCK_CODES: frozenset[str] = frozenset(
    {PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS}
)


def _freeze_option(value: Any) -> Any:
    """Recursively isolate mutable option containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                deepcopy(key): _freeze_option(nested_value)
                for key, nested_value in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_option(item) for item in value)
    return deepcopy(value)


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_exact_identifier(
    name: str,
    value: object,
    *,
    nullable: bool = False,
) -> None:
    if value is None and nullable:
        return
    if type(value) is not str or not value:
        raise ValueError(f"{name} must not be empty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError(f"{name} is invalid") from None


@dataclass(frozen=True, slots=True)
class TTSRequestedSelectionSnapshot:
    """Immutable text-free provenance for one exact admitted request.

    Covers all seven providers the profile system recognizes (`audio_cpp`
    plus the six legacy-bridge providers), not native-only: `audio_cpp`
    keeps its exact WAV / speed-1.0 contract and its server-default voice;
    legacy providers accept any format in their catalog set and any speed in
    [0.25, 4.0] but must name an exact voice, matching
    `profile_types.PROFILE_PROVIDER_FORMATS`,
    `profile_types.PROFILE_PROVIDER_REQUIRES_EXACT_VOICE`, and
    `profile_service._selection_is_profile_safe`.
    """

    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    configuration_revision: int

    def __post_init__(self) -> None:
        _require_exact_identifier("provider_id", self.provider_id)
        if self.provider_id not in PROFILE_PROVIDER_IDS:
            raise ValueError("Requested selection requires a recognized provider")
        _require_exact_identifier("model_id", self.model_id)
        _require_exact_identifier("voice_id", self.voice_id, nullable=True)
        if (
            self.voice_id is None
            and PROFILE_PROVIDER_REQUIRES_EXACT_VOICE[self.provider_id]
        ):
            raise ValueError("Requested selection requires an exact voice")
        _require_exact_identifier("response_format", self.response_format)
        if self.response_format not in PROFILE_PROVIDER_FORMATS[self.provider_id]:
            raise ValueError(
                "Requested selection format is not valid for this provider"
            )
        if type(self.speed) is not float or not math.isfinite(self.speed):
            raise ValueError("Requested selection requires a finite speed")
        if self.provider_id == "audio_cpp":
            if self.speed != AUDIO_CPP_PROFILE_SPEED:
                raise ValueError("Requested selection requires speed 1.0")
        elif not 0.25 <= self.speed <= 4.0:
            raise ValueError("Requested selection speed is out of range")
        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")
        try:
            next(iter(self.options))
        except StopIteration:
            pass
        except Exception:
            raise TypeError("options must be an empty mapping") from None
        else:
            raise ValueError("Requested selection options must be empty")
        if type(self.configuration_revision) is not int:
            raise TypeError("configuration_revision must be an integer")
        if self.configuration_revision < 0:
            raise ValueError("configuration_revision must be nonnegative")
        object.__setattr__(self, "options", MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class STTSPlaygroundCloneSnapshot:
    """One exact private clone draft admitted by a Playground request."""

    draft_revision: int
    canonical_reference: CanonicalTTSCloneReference = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.draft_revision) is not int:
            raise TypeError("draft_revision must be an integer")
        if self.draft_revision < 0:
            raise ValueError("draft_revision must be nonnegative")
        if type(self.canonical_reference) is not CanonicalTTSCloneReference:
            raise TypeError("canonical_reference must be an exact canonical reference")


@dataclass(frozen=True, slots=True)
class STTSPlaygroundProfilePreview:
    """Path-free identity for one reference-bearing profile preview."""

    profile_id: UUID
    repository_generation: int
    profile_revision: int

    def __post_init__(self) -> None:
        if type(self.profile_id) is not UUID:
            raise TypeError("profile_id must be a UUID")
        if type(self.repository_generation) is not int:
            raise TypeError("repository_generation must be an integer")
        if self.repository_generation < 0:
            raise ValueError("repository_generation must be nonnegative")
        if type(self.profile_revision) is not int:
            raise TypeError("profile_revision must be an integer")
        if self.profile_revision < 1:
            raise ValueError("profile_revision must be positive")


@dataclass(frozen=True, slots=True)
class STTSPlaygroundRequest:
    """Immutable snapshot of one Playground generation request."""

    operation_id: str
    provider_id: str
    model_id: str
    text: str
    voice_id: str | None
    response_format: str
    speed: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)
    studio_draft: TTSStudioDraftSelection | None = None
    studio_preferences: StudioTTSPreferencesSnapshot | None = None
    clone_audition: STTSPlaygroundCloneSnapshot | None = field(
        default=None,
        repr=False,
    )
    profile_preview: STTSPlaygroundProfilePreview | None = None

    def __post_init__(self) -> None:
        for name in ("operation_id", "provider_id", "model_id", "response_format"):
            _require_identifier(name, getattr(self, name))
        object.__setattr__(
            self,
            "options",
            _freeze_option(self.options),
        )
        if (self.studio_draft is None) is not (self.studio_preferences is None):
            raise ValueError(
                "Studio Playground requests require both draft and saved preferences"
            )
        if self.studio_draft is not None:
            if type(self.studio_draft) is not TTSStudioDraftSelection:
                raise TypeError("studio_draft must be an exact Studio draft")
            if type(self.studio_preferences) is not StudioTTSPreferencesSnapshot:
                raise TypeError("studio_preferences must be an exact Studio snapshot")
            studio_preferences = self.studio_preferences
            assert studio_preferences is not None
            if self.studio_draft.base_revision != studio_preferences.revision:
                raise ValueError("Studio Playground preference revisions must match")
        if self.clone_audition is not None:
            if type(self.clone_audition) is not STTSPlaygroundCloneSnapshot:
                raise TypeError("clone_audition must be an exact clone snapshot")
            if (
                self.provider_id != "audio_cpp"
                or self.studio_draft is None
                or self.studio_preferences is None
                or self.studio_draft.selection.provider_id != "audio_cpp"
                or self.studio_draft.selection.model_id != self.model_id
            ):
                raise ValueError(
                    "Clone auditions require a complete matching audio.cpp Studio request"
                )
        if self.profile_preview is not None:
            if type(self.profile_preview) is not STTSPlaygroundProfilePreview:
                raise TypeError("profile_preview must be an exact profile preview")
            if (
                self.provider_id != "audio_cpp"
                or self.studio_draft is None
                or self.studio_preferences is None
                or not self.studio_draft.preview
                or self.studio_draft.selection.provider_id != "audio_cpp"
                or self.studio_draft.selection.model_id != self.model_id
            ):
                raise ValueError(
                    "Profile preview requires a complete matching audio.cpp Studio request"
                )
        if self.clone_audition is not None and self.profile_preview is not None:
            raise ValueError("Clone audition and profile preview are mutually exclusive")


@dataclass(frozen=True, slots=True)
class STTSGeneratedAudio:
    """Immutable generated-audio artifact with request provenance."""

    path: Path
    provider_id: str
    model_id: str
    voice_id: str | None
    source_text: str
    operation_id: str
    audio_format: str
    content_type: str
    metadata: Mapping[str, AudioMetadataValue] = field(default_factory=dict)
    requested_selection: TTSRequestedSelectionSnapshot | None = None
    clone_evidence: TTSCloneGenerationEvidence | None = field(
        default=None,
        repr=False,
    )
    #: Why provenance was refused, when the reason is actionable to explain.
    #:
    #: `None` covers both "provenance attached" and "dropped for a reason the
    #: user cannot act on" (a momentarily unreadable registry revision). The
    #: one bounded code says the generation used provider-specific options,
    #: which a slice-1 profile fixes empty and therefore cannot reproduce.
    profile_save_block_code: ProfileSaveBlockCode | None = None

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "provider_id",
            "model_id",
            "audio_format",
            "content_type",
        ):
            _require_identifier(name, getattr(self, name))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )
        if (
            self.requested_selection is not None
            and type(self.requested_selection) is not TTSRequestedSelectionSnapshot
        ):
            raise TypeError(
                "requested_selection must be a requested selection snapshot"
            )
        if self.clone_evidence is not None and (
            type(self.clone_evidence) is not TTSCloneGenerationEvidence
        ):
            raise TypeError("clone_evidence must be exact clone generation evidence")
        if self.profile_save_block_code is not None:
            if self.profile_save_block_code not in PROFILE_SAVE_BLOCK_CODES:
                raise ValueError("profile_save_block_code is not a known code")
            if self.requested_selection is not None:
                raise ValueError(
                    "profile_save_block_code cannot accompany attached provenance"
                )

    @property
    def file_suffix(self) -> str:
        """Return the suffix implied by the actual response format."""
        return f".{self.audio_format.removeprefix('.')}"

    @property
    def profile_save_eligible(self) -> bool:
        """Return whether exact native request provenance can seed a profile."""
        return type(self.requested_selection) is TTSRequestedSelectionSnapshot


@dataclass(frozen=True, slots=True)
class STTSPlaygroundResultProjection:
    """Sanitized playback facts that carry no clone-reference authority."""

    path: Path = field(repr=False)
    provider_id: str
    model_id: str
    voice_id: str | None
    operation_id: str
    audio_format: str
    metadata: Mapping[str, AudioMetadataValue] = field(default_factory=dict)
    requested_selection: TTSRequestedSelectionSnapshot | None = None
    profile_save_block_code: ProfileSaveBlockCode | None = None
    clone_profile_save_eligible: bool = False

    def __post_init__(self) -> None:
        for name in ("operation_id", "provider_id", "model_id", "audio_format"):
            _require_identifier(name, getattr(self, name))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )
        if self.requested_selection is not None and (
            type(self.requested_selection) is not TTSRequestedSelectionSnapshot
        ):
            raise TypeError("requested_selection must be an exact snapshot")
        if self.profile_save_block_code is not None:
            if self.profile_save_block_code not in PROFILE_SAVE_BLOCK_CODES:
                raise ValueError("profile_save_block_code is not a known code")
            if self.requested_selection is not None:
                raise ValueError(
                    "profile_save_block_code cannot accompany attached provenance"
                )
        if type(self.clone_profile_save_eligible) is not bool:
            raise TypeError("clone_profile_save_eligible must be a boolean")

    @property
    def profile_save_eligible(self) -> bool:
        """Return whether the handler retained exact save provenance."""

        return bool(
            self.clone_profile_save_eligible
            or type(self.requested_selection) is TTSRequestedSelectionSnapshot
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: STTSGeneratedAudio,
    ) -> STTSPlaygroundResultProjection:
        """Copy only playback-safe facts from one handler-owned artifact."""

        if type(artifact) is not STTSGeneratedAudio:
            raise TypeError("generated audio artifact must be exact")
        return cls(
            path=artifact.path,
            provider_id=artifact.provider_id,
            model_id=artifact.model_id,
            voice_id=artifact.voice_id,
            operation_id=artifact.operation_id,
            audio_format=artifact.audio_format,
            metadata=artifact.metadata,
            requested_selection=artifact.requested_selection,
            profile_save_block_code=artifact.profile_save_block_code,
            clone_profile_save_eligible=artifact.clone_evidence is not None,
        )
