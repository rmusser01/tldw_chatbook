"""Sealed package recipes for guided audio.cpp setup.

Recipes are inert data. Matching consumes a bounded pre-scanned description;
managed acceptance delegates exact identity validation to the offline catalog.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from uuid import uuid4

from .audio_cpp_artifact_catalog import audio_cpp_artifact_identity_matches_recipe
from .audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppBackendPreference,
    AudioCppManagedArtifactIdentity,
    AudioCppRecipeOption,
    AudioCppSafeModelProjection,
    AudioCppSettingsConfig,
)


AUDIO_CPP_PINNED_RELEASE = "release-0.5.1"
AUDIO_CPP_PINNED_COMMIT = "238ab6a9e321c17de8e120559f57efeedaeb1345"
AUDIO_CPP_RECIPE_SCHEMA_VERSION = 1

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PINNED_MODEL_SPEC_BASE = (
    f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}/model_specs"
)
_RECIPE_EVIDENCE_REFERENCE = (
    "Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md"
)
_MODEL_LIBRARY_ADMITTED_VARIANTS = frozenset(
    {
        "dramabox_q8_0",
        "fish_audio_s2_pro_bf16",
        "fish_audio_s2_pro_q8_0",
        "higgs_audio_tts_4b_bf16",
        "higgs_audio_tts_4b_q8_0",
        "index_tts2_f16",
        "index_tts2_orig",
        "index_tts2_q8_0",
        "inflect_micro_v2_orig",
        "irodori_tts_500m_v3_f16",
        "irodori_tts_500m_v3_q8_0",
        "irodori_tts_600m_v3_voicedesign_f16",
        "irodori_tts_600m_v3_voicedesign_q8_0",
        "irodori_tts_v4_small_f16",
        "irodori_tts_v4_small_q8_0",
        "moss_tts_local_v1_5_bf16",
        "moss_tts_local_v1_5_q8_0",
        "moss_tts_nano_100m_bf16",
        "moss_tts_nano_100m_q8_0",
        "omnivoice_bf16",
        "omnivoice_f16",
        "omnivoice_q8_0",
        "pocket_tts_english_bf16",
        "pocket_tts_english_q8_0",
        "pocket_tts_german_bf16",
        "pocket_tts_german_q8_0",
        "pocket_tts_italian_bf16",
        "pocket_tts_italian_q8_0",
        "pocket_tts_portuguese_bf16",
        "pocket_tts_portuguese_q8_0",
        "pocket_tts_spanish_bf16",
        "pocket_tts_spanish_q8_0",
        "qwen3_tts_1_7b_base_bf16",
        "qwen3_tts_1_7b_base_orig",
        "qwen3_tts_1_7b_base_q8_0",
        "supertonic_3_f16",
        "supertonic_3_orig",
        "vevo2_f16",
        "vevo2_orig",
        "vevo2_q8_0",
        "vibevoice_1_5b_bf16",
        "vibevoice_1_5b_q8_0",
        "voxcpm2_bf16",
        "voxcpm2_orig",
        "voxcpm2_q8_0",
    }
)


class AudioCppFileKind(StrEnum):
    """Bounded metadata inspection rule for a required package file."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    JSON = "json"
    TOKENIZER = "tokenizer"
    OTHER = "other"


class AudioCppFileRole(StrEnum):
    """Identity role used when fencing an accepted package."""

    CONFIGURATION = "configuration"
    WEIGHT = "weight"
    VOICE = "voice"
    OTHER = "other"


class AudioCppBackendEvidenceState(StrEnum):
    """Truthful posture for one exact recipe/platform/backend tuple."""

    VERIFIED = "verified"
    EXPECTED = "expected"
    UNTESTED = "untested"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"


class AudioCppRecipeSupportState(StrEnum):
    """Auditable release-accounting state for one package variant."""

    APPROVED = "approved"
    EXPLICITLY_UNSUPPORTED = "explicitly_unsupported"
    OPEN_GAP = "open_gap"


class AudioCppMatchState(StrEnum):
    """Fail-closed result of matching one pre-scanned package description."""

    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    INCOMPLETE = "incomplete"
    PERMISSION_LIMITED = "permission_limited"


class AudioCppReferenceRequirement(StrEnum):
    """Recipe-declared voice-reference posture."""

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class AudioCppVoiceReferencePolicy(StrEnum):
    """Exact native-voice/reference combinations admitted by one recipe."""

    TEXT_ONLY = "text_only"
    NATIVE_ONLY = "native_only"
    REFERENCE_ONLY = "reference_only"
    OPTIONAL_REFERENCE_ONLY = "optional_reference_only"
    VOICE_OR_REFERENCE_REQUIRED = "voice_or_reference_required"
    EITHER = "either"
    BOTH_REQUIRED_COMBINED = "both_required_combined"


def _unsafe_text(value: str) -> bool:
    return any(
        character in {"\x00", "\r", "\n"}
        or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    )


def _require_token(value: str, label: str) -> None:
    if type(value) is not str or not _TOKEN.fullmatch(value) or _unsafe_text(value):
        raise ValueError(f"audio.cpp {label} is invalid")


def _require_digest(value: str, label: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"audio.cpp {label} must be a lowercase SHA-256 identity")


def _require_relative_path(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 1024
        or "\\" in value
        or "$" in value
        or "%" in value
        or _unsafe_text(value)
    ):
        raise ValueError(f"audio.cpp {label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError(f"audio.cpp {label} must be a safe relative path")
    if path.as_posix() != value:
        raise ValueError(f"audio.cpp {label} must be a safe relative path")


def _require_absolute_path(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 4096
        or _unsafe_text(value)
        or not (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
        )
    ):
        raise ValueError(f"audio.cpp {label} must be an absolute path")


def _safe_name(value: str) -> str:
    if type(value) is not str:
        raise ValueError("audio.cpp package display name is invalid")
    cleaned = "".join(
        character
        for character in value
        if not unicodedata.category(character).startswith("C")
    ).strip()
    if not cleaned:
        return "Selected package"
    return cleaned[:128]


@dataclass(frozen=True, slots=True)
class AudioCppFileSignal:
    """Exact allowlisted relative-file signal for one package recipe."""

    relative_path: str
    kind: AudioCppFileKind
    role: AudioCppFileRole
    minimum_size_bytes: int = 1

    def __post_init__(self) -> None:
        _require_relative_path(self.relative_path, "recipe file path")
        if not isinstance(self.kind, AudioCppFileKind) or not isinstance(
            self.role, AudioCppFileRole
        ):
            raise ValueError("audio.cpp recipe file signal type is invalid")
        if type(self.minimum_size_bytes) is not int or self.minimum_size_bytes < 1:
            raise ValueError("audio.cpp recipe file minimum size must be positive")


@dataclass(frozen=True, slots=True)
class AudioCppBackendEvidence:
    """One exact platform/backend compatibility posture."""

    system: str
    architecture: str
    backend: AudioCppBackendPreference
    state: AudioCppBackendEvidenceState
    evidence_reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend, AudioCppBackendPreference) or not isinstance(
            self.state, AudioCppBackendEvidenceState
        ):
            raise ValueError("audio.cpp backend evidence type is invalid")
        _require_token(self.system, "backend system")
        _require_token(self.architecture, "backend architecture")
        if self.backend is AudioCppBackendPreference.AUTO:
            raise ValueError("audio.cpp recipe backend evidence cannot use auto")
        if (
            type(self.evidence_reference) is not str
            or not self.evidence_reference
            or len(self.evidence_reference) > 512
            or _unsafe_text(self.evidence_reference)
        ):
            raise ValueError("audio.cpp backend evidence reference is invalid")


@dataclass(frozen=True, slots=True)
class AudioCppPackageRecipe:
    """Immutable reviewed projection for one exact upstream package variant."""

    schema_version: int
    recipe_id: str
    recipe_revision: int
    audio_cpp_release: str
    audio_cpp_commit: str
    family: str
    package_variant: str
    display_name: str
    package_format: AudioCppFileKind
    precision: str
    capabilities: tuple[str, ...]
    required_files: tuple[AudioCppFileSignal, ...]
    optional_files: tuple[AudioCppFileSignal, ...]
    projection: AudioCppSafeModelProjection
    default_public_model_id: str
    reference_requirement: AudioCppReferenceRequirement
    voice_reference_policy: AudioCppVoiceReferencePolicy
    backend_evidence: tuple[AudioCppBackendEvidence, ...]
    support_state: AudioCppRecipeSupportState
    evidence_reference: str
    source_links: tuple[str, ...]
    model_library_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tuple_fields = (
            self.capabilities,
            self.required_files,
            self.optional_files,
            self.backend_evidence,
            self.source_links,
            self.model_library_artifact_ids,
        )
        if any(type(value) is not tuple for value in tuple_fields):
            raise ValueError("audio.cpp recipe collections must be immutable tuples")
        if (
            not isinstance(self.package_format, AudioCppFileKind)
            or not isinstance(
                self.reference_requirement,
                AudioCppReferenceRequirement,
            )
            or not isinstance(
                self.voice_reference_policy,
                AudioCppVoiceReferencePolicy,
            )
        ):
            raise ValueError("audio.cpp recipe classification is invalid")
        if not all(
            isinstance(item, AudioCppFileSignal)
            for item in (*self.required_files, *self.optional_files)
        ) or not all(
            isinstance(item, AudioCppBackendEvidence) for item in self.backend_evidence
        ):
            raise ValueError("audio.cpp recipe record type is invalid")
        if self.schema_version != AUDIO_CPP_RECIPE_SCHEMA_VERSION:
            raise ValueError("audio.cpp recipe schema version is unsupported")
        if type(self.recipe_revision) is not int or self.recipe_revision < 1:
            raise ValueError("audio.cpp recipe revision must be positive")
        for value, label in (
            (self.recipe_id, "recipe id"),
            (self.family, "recipe family"),
            (self.package_variant, "package variant"),
            (self.precision, "package precision"),
            (self.default_public_model_id, "default public model id"),
        ):
            _require_token(value, label)
        if (
            type(self.display_name) is not str
            or not self.display_name.strip()
            or len(self.display_name) > 256
            or _unsafe_text(self.display_name)
        ):
            raise ValueError("audio.cpp recipe display name is invalid")
        if self.audio_cpp_release != AUDIO_CPP_PINNED_RELEASE:
            raise ValueError("audio.cpp recipe release is not pinned")
        if self.audio_cpp_commit != AUDIO_CPP_PINNED_COMMIT:
            raise ValueError("audio.cpp recipe commit is not pinned")
        if self.support_state is not AudioCppRecipeSupportState.APPROVED:
            raise ValueError("only approved entries may enter the recipe registry")
        if self.projection.family != self.family:
            raise ValueError("audio.cpp recipe projection family does not match")
        if not self.required_files:
            raise ValueError("audio.cpp recipe requires at least one file")
        all_paths = tuple(
            signal.relative_path
            for signal in (*self.required_files, *self.optional_files)
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("audio.cpp recipe file signals must be unique")
        if (
            self.projection.model_relative_path is not None
            and self.projection.model_relative_path
            not in {signal.relative_path for signal in self.required_files}
        ):
            raise ValueError("audio.cpp recipe model path must be a required file")
        if not self.capabilities or not set(self.capabilities) <= {
            "tts",
            "clone",
            "design",
        }:
            raise ValueError("audio.cpp recipe capabilities are invalid")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("audio.cpp recipe capabilities must be unique")
        expected_reference_contracts = {
            AudioCppVoiceReferencePolicy.TEXT_ONLY: (
                AudioCppReferenceRequirement.NONE,
                False,
            ),
            AudioCppVoiceReferencePolicy.NATIVE_ONLY: (
                AudioCppReferenceRequirement.NONE,
                False,
            ),
            AudioCppVoiceReferencePolicy.REFERENCE_ONLY: (
                AudioCppReferenceRequirement.REQUIRED,
                True,
            ),
            AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY: (
                AudioCppReferenceRequirement.OPTIONAL,
                True,
            ),
            AudioCppVoiceReferencePolicy.VOICE_OR_REFERENCE_REQUIRED: (
                AudioCppReferenceRequirement.OPTIONAL,
                True,
            ),
            AudioCppVoiceReferencePolicy.EITHER: (
                AudioCppReferenceRequirement.OPTIONAL,
                True,
            ),
            AudioCppVoiceReferencePolicy.BOTH_REQUIRED_COMBINED: (
                AudioCppReferenceRequirement.REQUIRED,
                True,
            ),
        }
        expected_requirement, requires_clone = expected_reference_contracts[
            self.voice_reference_policy
        ]
        if (
            self.reference_requirement is not expected_requirement
            or ("clone" in self.capabilities) is not requires_clone
        ):
            raise ValueError("audio.cpp recipe voice reference policy is invalid")
        if not self.backend_evidence:
            raise ValueError("audio.cpp recipe requires backend posture")
        if (
            type(self.evidence_reference) is not str
            or not self.evidence_reference
            or len(self.evidence_reference) > 512
            or _unsafe_text(self.evidence_reference)
        ):
            raise ValueError("audio.cpp recipe evidence reference is invalid")
        if not self.source_links or any(
            not link.startswith("https://")
            or AUDIO_CPP_PINNED_COMMIT not in link
            or len(link) > 1024
            for link in self.source_links
        ):
            raise ValueError("audio.cpp recipe source links must be pinned HTTPS links")
        for artifact_id in self.model_library_artifact_ids:
            _require_token(artifact_id, "model library artifact id")

    def admits_voice_reference(
        self,
        *,
        has_voice: bool,
        has_reference: bool,
    ) -> bool:
        """Return whether this recipe admits the exact request combination."""

        if type(has_voice) is not bool or type(has_reference) is not bool:
            raise TypeError("audio.cpp voice/reference markers must be boolean")
        if self.voice_reference_policy is AudioCppVoiceReferencePolicy.TEXT_ONLY:
            return not has_voice and not has_reference
        if self.voice_reference_policy is AudioCppVoiceReferencePolicy.NATIVE_ONLY:
            return not has_reference
        if self.voice_reference_policy is AudioCppVoiceReferencePolicy.REFERENCE_ONLY:
            return has_reference and not has_voice
        if (
            self.voice_reference_policy
            is AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY
        ):
            return not has_voice
        if (
            self.voice_reference_policy
            is AudioCppVoiceReferencePolicy.VOICE_OR_REFERENCE_REQUIRED
        ):
            return has_voice is not has_reference
        if self.voice_reference_policy is AudioCppVoiceReferencePolicy.EITHER:
            return not (has_voice and has_reference)
        return has_voice and has_reference


@dataclass(frozen=True, slots=True)
class AudioCppPackageFileEvidence:
    """Bounded identity and readiness evidence for one allowlisted file."""

    relative_path: str
    size_bytes: int
    identity: str
    readable: bool
    metadata_valid: bool

    def __post_init__(self) -> None:
        _require_relative_path(self.relative_path, "scanned evidence path")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("audio.cpp scanned file size is invalid")
        _require_digest(self.identity, "scanned file identity")
        if type(self.readable) is not bool or type(self.metadata_valid) is not bool:
            raise ValueError("audio.cpp scanned file readiness is invalid")


@dataclass(frozen=True, slots=True)
class AudioCppPackageDescription:
    """Pre-scanned exact-root description consumed by the pure matcher."""

    canonical_root: str = field(repr=False)
    canonical_root_identity: str
    safe_name: str
    files: tuple[AudioCppPackageFileEvidence, ...]
    partial: bool = False
    permission_limited: bool = False

    def __post_init__(self) -> None:
        if type(self.files) is not tuple or not all(
            isinstance(item, AudioCppPackageFileEvidence) for item in self.files
        ):
            raise ValueError("audio.cpp scanned evidence must be an immutable tuple")
        _require_absolute_path(self.canonical_root, "canonical package root")
        _require_digest(self.canonical_root_identity, "canonical root identity")
        object.__setattr__(self, "safe_name", _safe_name(self.safe_name))
        paths = tuple(item.relative_path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("audio.cpp scanned evidence paths must be unique")
        if type(self.partial) is not bool or type(self.permission_limited) is not bool:
            raise ValueError("audio.cpp scan state is invalid")


def _identity(parts: tuple[str, ...]) -> str:
    return sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _managed_artifact_matches_recipe(
    recipe: AudioCppPackageRecipe,
    identity: AudioCppManagedArtifactIdentity,
) -> bool:
    if type(identity) is not AudioCppManagedArtifactIdentity:
        return False
    return audio_cpp_artifact_identity_matches_recipe(
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        package_variant=recipe.package_variant,
        recipe_artifact_ids=recipe.model_library_artifact_ids,
        recipe_precision=recipe.precision,
        artifact_id=identity.artifact_id,
        revision=identity.revision,
        variant=identity.variant,
    )


@dataclass(frozen=True, slots=True)
class AudioCppPackageCandidate:
    """One exact candidate suitable for explicit user acceptance."""

    recipe: AudioCppPackageRecipe
    canonical_root: str = field(repr=False)
    canonical_root_identity: str
    configuration_identity: str
    weight_identity: str
    safe_name: str
    evidence_relative_paths: tuple[str, ...]

    def accept(
        self,
        *,
        public_model_id: str | None = None,
        managed_artifact: AudioCppManagedArtifactIdentity | None = None,
    ) -> AudioCppAcceptedPackage:
        """Freeze this exact match into a durable accepted snapshot.

        Args:
            public_model_id: Optional user-facing model ID override.
            managed_artifact: Optional exact managed-store identity.

        Returns:
            A new immutable accepted-package snapshot with a durable UUID.

        Raises:
            ValueError: If the requested public model ID or managed identity is
                invalid for this exact recipe.
        """
        if managed_artifact is not None and not _managed_artifact_matches_recipe(
            self.recipe,
            managed_artifact,
        ):
            raise ValueError("audio.cpp managed artifact does not match recipe")
        return AudioCppAcceptedPackage(
            package_uuid=str(uuid4()),
            recipe_id=self.recipe.recipe_id,
            recipe_revision=self.recipe.recipe_revision,
            package_variant=self.recipe.package_variant,
            public_model_id=(
                self.recipe.default_public_model_id
                if public_model_id is None
                else public_model_id
            ),
            canonical_root=self.canonical_root,
            canonical_root_identity=self.canonical_root_identity,
            configuration_identity=self.configuration_identity,
            weight_identity=self.weight_identity,
            projection=self.recipe.projection,
            managed_artifact=managed_artifact,
        )


@dataclass(frozen=True, slots=True)
class AudioCppMatchResult:
    """Pure fail-closed recipe matching result."""

    state: AudioCppMatchState
    recipe_ids: tuple[str, ...] = ()
    candidates: tuple[AudioCppPackageCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class AudioCppVerifiedSupportClaim:
    """One user-facing support claim backed by an exact evidence tuple."""

    recipe_id: str
    family: str
    package_variant: str
    system: str
    architecture: str
    backend: AudioCppBackendPreference
    evidence_reference: str


class AudioCppRecipeRegistry:
    """Small immutable registry with pure matching and snapshot validation."""

    __slots__ = ("_by_id", "_by_package", "_recipes")
    _by_id: MappingProxyType[str, AudioCppPackageRecipe]
    _by_package: MappingProxyType[str, AudioCppPackageRecipe]
    _recipes: tuple[AudioCppPackageRecipe, ...]

    def __init__(self, recipes: tuple[AudioCppPackageRecipe, ...]) -> None:
        ordered = tuple(sorted(recipes, key=lambda item: item.recipe_id))
        if len({recipe.recipe_id for recipe in ordered}) != len(ordered):
            raise ValueError("audio.cpp recipe IDs must be unique")
        if len({recipe.package_variant for recipe in ordered}) != len(ordered):
            raise ValueError("audio.cpp package variants must be unique")
        path_contracts: dict[str, tuple[AudioCppFileKind, int]] = {}
        for recipe in ordered:
            for signal in recipe.required_files:
                contract = (signal.kind, signal.minimum_size_bytes)
                previous = path_contracts.setdefault(signal.relative_path, contract)
                if previous != contract:
                    raise ValueError(
                        "audio.cpp recipes have conflicting file validation contracts"
                    )
        object.__setattr__(self, "_recipes", ordered)
        object.__setattr__(
            self,
            "_by_id",
            MappingProxyType({recipe.recipe_id: recipe for recipe in ordered}),
        )
        object.__setattr__(
            self,
            "_by_package",
            MappingProxyType({recipe.package_variant: recipe for recipe in ordered}),
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("audio.cpp recipe registry is immutable")

    @property
    def recipes(self) -> tuple[AudioCppPackageRecipe, ...]:
        """Return the immutable ordered recipe collection.

        Returns:
            Every installed recipe ordered by recipe ID.
        """
        return self._recipes

    def for_package(self, package_variant: str) -> AudioCppPackageRecipe:
        """Return one exact package recipe or raise a stable lookup error.

        Args:
            package_variant: Exact pinned package variant identifier.

        Returns:
            The reviewed recipe for the requested package variant.

        Raises:
            ValueError: If the package variant has no installed recipe.
        """
        try:
            return self._by_package[package_variant]
        except (KeyError, TypeError):
            raise ValueError("audio.cpp package recipe is unavailable") from None

    @staticmethod
    def _candidate(
        recipe: AudioCppPackageRecipe,
        description: AudioCppPackageDescription,
        evidence: dict[str, AudioCppPackageFileEvidence],
    ) -> AudioCppPackageCandidate:
        configuration_parts = tuple(
            f"{signal.relative_path}:{evidence[signal.relative_path].identity}"
            for signal in recipe.required_files
            if signal.role in {AudioCppFileRole.CONFIGURATION, AudioCppFileRole.OTHER}
        ) or (f"{recipe.recipe_id}:{recipe.recipe_revision}",)
        weight_parts = tuple(
            f"{signal.relative_path}:{evidence[signal.relative_path].identity}"
            for signal in recipe.required_files
            if signal.role in {AudioCppFileRole.WEIGHT, AudioCppFileRole.VOICE}
        ) or (f"{recipe.recipe_id}:{recipe.recipe_revision}:no-weight",)
        return AudioCppPackageCandidate(
            recipe=recipe,
            canonical_root=description.canonical_root,
            canonical_root_identity=description.canonical_root_identity,
            configuration_identity=_identity(configuration_parts),
            weight_identity=_identity(weight_parts),
            safe_name=description.safe_name,
            evidence_relative_paths=tuple(
                signal.relative_path for signal in recipe.required_files
            ),
        )

    def match(self, description: AudioCppPackageDescription) -> AudioCppMatchResult:
        """Match one bounded description without fuzzy or closest selection.

        Args:
            description: Pre-scanned, bounded evidence for one candidate root.

        Returns:
            An exact, ambiguous, incomplete, permission-limited, or unknown
            match result.

        Raises:
            TypeError: If ``description`` is not package evidence.
        """
        if not isinstance(description, AudioCppPackageDescription):
            raise TypeError("audio.cpp package description is required")
        evidence = {item.relative_path: item for item in description.files}
        recognizable: list[AudioCppPackageRecipe] = []
        exact: list[AudioCppPackageRecipe] = []
        for recipe in self.recipes:
            required_paths = {signal.relative_path for signal in recipe.required_files}
            if required_paths & evidence.keys():
                recognizable.append(recipe)
            if all(
                signal.relative_path in evidence
                and evidence[signal.relative_path].readable
                and evidence[signal.relative_path].metadata_valid
                and evidence[signal.relative_path].size_bytes
                >= signal.minimum_size_bytes
                for signal in recipe.required_files
            ):
                exact.append(recipe)

        candidate_recipes = exact or recognizable
        recipe_ids = tuple(sorted(recipe.recipe_id for recipe in candidate_recipes))
        if description.permission_limited:
            return AudioCppMatchResult(
                AudioCppMatchState.PERMISSION_LIMITED,
                recipe_ids=recipe_ids,
            )
        if description.partial:
            return AudioCppMatchResult(
                AudioCppMatchState.INCOMPLETE,
                recipe_ids=recipe_ids,
            )
        candidates = tuple(
            self._candidate(recipe, description, evidence)
            for recipe in sorted(exact, key=lambda item: item.recipe_id)
        )
        if len(candidates) == 1:
            return AudioCppMatchResult(
                AudioCppMatchState.EXACT,
                recipe_ids=(candidates[0].recipe.recipe_id,),
                candidates=candidates,
            )
        if len(candidates) > 1:
            return AudioCppMatchResult(
                AudioCppMatchState.AMBIGUOUS,
                recipe_ids=tuple(
                    candidate.recipe.recipe_id for candidate in candidates
                ),
                candidates=candidates,
            )
        if recognizable:
            return AudioCppMatchResult(
                AudioCppMatchState.INCOMPLETE,
                recipe_ids=recipe_ids,
            )
        return AudioCppMatchResult(AudioCppMatchState.UNKNOWN)

    def validate_accepted(
        self,
        accepted: AudioCppAcceptedPackage,
    ) -> AudioCppPackageRecipe:
        """Require an accepted snapshot to equal the installed recipe exactly.

        Args:
            accepted: Durable package snapshot to validate.

        Returns:
            The exact currently installed recipe.

        Raises:
            ValueError: If the recipe is absent or its frozen projection changed.
        """
        try:
            recipe = self._by_id[accepted.recipe_id]
        except (AttributeError, KeyError, TypeError):
            raise ValueError(
                "audio.cpp accepted package requires recipe review"
            ) from None
        if (
            accepted.recipe_revision != recipe.recipe_revision
            or accepted.package_variant != recipe.package_variant
            or accepted.projection != recipe.projection
            or (
                accepted.managed_artifact is not None
                and not _managed_artifact_matches_recipe(
                    recipe,
                    accepted.managed_artifact,
                )
            )
        ):
            raise ValueError("audio.cpp accepted package requires recipe review")
        return recipe

    def verified_support_claims(self) -> tuple[AudioCppVerifiedSupportClaim, ...]:
        """Return only exact tuples carrying retained Verified evidence.

        Returns:
            Sorted support claims whose backend evidence is explicitly verified.
        """
        claims = [
            AudioCppVerifiedSupportClaim(
                recipe_id=recipe.recipe_id,
                family=recipe.family,
                package_variant=recipe.package_variant,
                system=evidence.system,
                architecture=evidence.architecture,
                backend=evidence.backend,
                evidence_reference=evidence.evidence_reference,
            )
            for recipe in self.recipes
            for evidence in recipe.backend_evidence
            if evidence.state is AudioCppBackendEvidenceState.VERIFIED
        ]
        return tuple(
            sorted(
                claims,
                key=lambda item: (
                    item.recipe_id,
                    item.system,
                    item.architecture,
                    item.backend.value,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class AudioCppReleaseAccountingEntry:
    """One package row in the complete pinned-release accounting matrix."""

    family: str
    package_variant: str
    state: AudioCppRecipeSupportState
    recipe_id: str | None = None
    reason: str | None = None
    evidence_reference: str | None = None


_EXPECTED_CPU_BACKENDS = tuple(
    AudioCppBackendEvidence(
        system=system,
        architecture=architecture,
        backend=AudioCppBackendPreference.CPU,
        state=AudioCppBackendEvidenceState.EXPECTED,
        evidence_reference=(
            "backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md"
            "#decision"
        ),
    )
    for system, architecture in (
        ("darwin", "arm64"),
        ("darwin", "x86_64"),
        ("linux", "aarch64"),
        ("linux", "x86_64"),
        ("windows", "x86_64"),
    )
)


def _gguf_file(path: str) -> AudioCppFileSignal:
    return AudioCppFileSignal(path, AudioCppFileKind.GGUF, AudioCppFileRole.WEIGHT, 8)


def _safetensors_file(path: str, role: AudioCppFileRole) -> AudioCppFileSignal:
    return AudioCppFileSignal(path, AudioCppFileKind.SAFETENSORS, role, 10)


def _config_file(path: str) -> AudioCppFileSignal:
    kind = (
        AudioCppFileKind.JSON if path.endswith(".json") else AudioCppFileKind.TOKENIZER
    )
    return AudioCppFileSignal(path, kind, AudioCppFileRole.CONFIGURATION, 1)


def _recipe(
    *,
    family: str,
    package_variant: str,
    display_name: str,
    package_format: AudioCppFileKind,
    precision: str,
    capabilities: tuple[str, ...],
    required_files: tuple[AudioCppFileSignal, ...],
    model_relative_path: str | None,
    language: str | None = None,
    recipe_revision: int = 1,
    reference_requirement: AudioCppReferenceRequirement | None = None,
    voice_reference_policy: AudioCppVoiceReferencePolicy | None = None,
    task: str = "tts",
) -> AudioCppPackageRecipe:
    options = (
        ()
        if language is None
        else (AudioCppRecipeOption(name="language", value=language),)
    )
    projection = AudioCppSafeModelProjection(
        family=family,
        task=task,
        mode="offline",
        model_relative_path=model_relative_path,
        load_options=options,
        session_options=options,
    )
    return AudioCppPackageRecipe(
        schema_version=AUDIO_CPP_RECIPE_SCHEMA_VERSION,
        recipe_id=f"audio-cpp-0.5.1.{family}.{package_variant}",
        recipe_revision=recipe_revision,
        audio_cpp_release=AUDIO_CPP_PINNED_RELEASE,
        audio_cpp_commit=AUDIO_CPP_PINNED_COMMIT,
        family=family,
        package_variant=package_variant,
        display_name=display_name,
        package_format=package_format,
        precision=precision,
        capabilities=capabilities,
        required_files=required_files,
        optional_files=(),
        projection=projection,
        default_public_model_id=package_variant.replace("_", "-"),
        reference_requirement=(
            reference_requirement
            if reference_requirement is not None
            else AudioCppReferenceRequirement.OPTIONAL
            if "clone" in capabilities
            else AudioCppReferenceRequirement.NONE
        ),
        voice_reference_policy=(
            voice_reference_policy
            if voice_reference_policy is not None
            else AudioCppVoiceReferencePolicy.REFERENCE_ONLY
            if reference_requirement is AudioCppReferenceRequirement.REQUIRED
            else AudioCppVoiceReferencePolicy.EITHER
            if "clone" in capabilities
            else AudioCppVoiceReferencePolicy.NATIVE_ONLY
        ),
        backend_evidence=_EXPECTED_CPU_BACKENDS,
        support_state=AudioCppRecipeSupportState.APPROVED,
        evidence_reference=_RECIPE_EVIDENCE_REFERENCE,
        source_links=(f"{_PINNED_MODEL_SPEC_BASE}/{family}.json",),
        model_library_artifact_ids=(
            (f"audio-cpp-{package_variant.replace('_', '-')}",)
            if package_variant in _MODEL_LIBRARY_ADMITTED_VARIANTS
            else ()
        ),
    )


_SUPERTONIC_VARIANTS = (
    ("supertonic_3_q8_0", "Supertonic 3 Q8_0 GGUF", "q8_0", "supertonic-3-q8_0.gguf"),
    ("supertonic_3_f16", "Supertonic 3 F16 GGUF", "f16", "supertonic-3-f16.gguf"),
    (
        "supertonic_3_orig",
        "Supertonic 3 Original-Dtype GGUF",
        "orig",
        "supertonic-3-orig.gguf",
    ),
)
_POCKET_GGUF_VARIANTS = tuple(
    (
        f"pocket_tts_{language}_{precision}",
        f"PocketTTS {language.title()} {precision.upper()} GGUF",
        precision,
        language,
        f"pocket-tts-{language}-{precision}.gguf",
    )
    for language in ("english", "german", "italian", "portuguese", "spanish")
    for precision in ("q8_0", "bf16")
)
_POCKET_VOICES = (
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "estelle",
    "eve",
    "fantine",
    "george",
    "giovanni",
    "jane",
    "javert",
    "jean",
    "juergen",
    "lola",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "rafael",
    "stuart_bell",
    "vera",
)

_INITIAL_RECIPES = (
    *(
        _recipe(
            family="supertonic",
            package_variant=variant,
            display_name=display,
            package_format=AudioCppFileKind.GGUF,
            precision=precision,
            capabilities=("tts",),
            required_files=(_gguf_file(filename),),
            model_relative_path=filename,
        )
        for variant, display, precision, filename in _SUPERTONIC_VARIANTS
    ),
    _recipe(
        family="supertonic",
        package_variant="supertonic_3_safetensors",
        display_name="Supertonic 3 Safetensors",
        package_format=AudioCppFileKind.SAFETENSORS,
        precision="native",
        capabilities=("tts",),
        required_files=(
            _config_file("config/tts.json"),
            _config_file("config/unicode_indexer.json"),
            _safetensors_file("ggml/supertonic.safetensors", AudioCppFileRole.WEIGHT),
        ),
        model_relative_path=None,
    ),
    *(
        _recipe(
            family="pocket_tts",
            package_variant=variant,
            display_name=display,
            package_format=AudioCppFileKind.GGUF,
            precision=precision,
            capabilities=("tts", "clone"),
            required_files=(_gguf_file(filename),),
            model_relative_path=filename,
            language=language,
            recipe_revision=2,
            reference_requirement=AudioCppReferenceRequirement.REQUIRED,
        )
        for variant, display, precision, language, filename in _POCKET_GGUF_VARIANTS
    ),
    _recipe(
        family="pocket_tts",
        package_variant="pocket_tts_english_safetensors",
        display_name="PocketTTS English Safetensors",
        package_format=AudioCppFileKind.SAFETENSORS,
        precision="native",
        capabilities=("tts", "clone"),
        required_files=(
            *(
                _safetensors_file(
                    f"languages/english/embeddings/{voice}.safetensors",
                    AudioCppFileRole.VOICE,
                )
                for voice in _POCKET_VOICES
            ),
            _safetensors_file(
                "languages/english/model.safetensors",
                AudioCppFileRole.WEIGHT,
            ),
            _config_file("languages/english/tokenizer.model"),
        ),
        model_relative_path=None,
        language="english",
        voice_reference_policy=(
            AudioCppVoiceReferencePolicy.VOICE_OR_REFERENCE_REQUIRED
        ),
    ),
)


_ADDITIONAL_VOICE_POLICIES = {
    "dramabox_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "vibevoice_1_5b_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "vibevoice_1_5b_bf16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "moss_tts_nano_100m_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "moss_tts_nano_100m_bf16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "fish_audio_s2_pro_q8_0": AudioCppVoiceReferencePolicy.EITHER,
    "fish_audio_s2_pro_bf16": AudioCppVoiceReferencePolicy.EITHER,
    "omnivoice_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "omnivoice_bf16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "omnivoice_f16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "omnivoice_safetensors": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "voxcpm2_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "voxcpm2_bf16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "voxcpm2_orig": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "voxcpm2_safetensors": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_v4_small_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_v4_small_f16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_600m_v3_voicedesign_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_600m_v3_voicedesign_f16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_500m_v3_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "irodori_tts_500m_v3_f16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "moss_tts_local_v1_5_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "moss_tts_local_v1_5_bf16": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "outetts_1_0_1b_q8_0": AudioCppVoiceReferencePolicy.OPTIONAL_REFERENCE_ONLY,
    "inflect_micro_v2_orig": AudioCppVoiceReferencePolicy.TEXT_ONLY,
}


_ADDITIONAL_GGUF_VARIANTS = (
    # family, variant, display name, precision, relative model path, capabilities,
    # reference requirement, projected task
    (
        "dramabox",
        "dramabox_q8_0",
        "DramaBox Q8_0 GGUF",
        "q8_0",
        "dramabox-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "vibevoice",
        "vibevoice_1_5b_q8_0",
        "VibeVoice 1.5B Q8_0 GGUF",
        "q8_0",
        "vibevoice-1.5b-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "vibevoice",
        "vibevoice_1_5b_bf16",
        "VibeVoice 1.5B BF16 GGUF",
        "bf16",
        "vibevoice-1.5b-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "moss_tts_nano",
        "moss_tts_nano_100m_q8_0",
        "MOSS-TTS-Nano 100M Q8_0 GGUF",
        "q8_0",
        "moss-tts-nano-100m-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "moss_tts_nano",
        "moss_tts_nano_100m_bf16",
        "MOSS-TTS-Nano 100M BF16 GGUF",
        "bf16",
        "moss-tts-nano-100m-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "fish_audio",
        "fish_audio_s2_pro_q8_0",
        "Fish Audio S2 Pro Q8_0 GGUF",
        "q8_0",
        "fish-audio-s2-pro-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "fish_audio",
        "fish_audio_s2_pro_bf16",
        "Fish Audio S2 Pro BF16 GGUF",
        "bf16",
        "fish-audio-s2-pro-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "higgs_audio_tts",
        "higgs_audio_tts_4b_q8_0",
        "Higgs Audio v3 TTS 4B Q8_0 GGUF",
        "q8_0",
        "higgs-audio-v3-tts-4b-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "higgs_audio_tts",
        "higgs_audio_tts_4b_bf16",
        "Higgs Audio v3 TTS 4B BF16 GGUF",
        "bf16",
        "higgs-audio-v3-tts-4b-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "omnivoice",
        "omnivoice_q8_0",
        "OmniVoice Q8_0 GGUF",
        "q8_0",
        "omnivoice-q8_0.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "omnivoice",
        "omnivoice_bf16",
        "OmniVoice BF16 GGUF",
        "bf16",
        "omnivoice-bf16.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "omnivoice",
        "omnivoice_f16",
        "OmniVoice F16 GGUF",
        "f16",
        "omnivoice-f16.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "qwen3_tts",
        "qwen3_tts_1_7b_base_q8_0",
        "Qwen3 TTS 12Hz 1.7B Base Q8_0 GGUF",
        "q8_0",
        "qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "qwen3_tts",
        "qwen3_tts_1_7b_base_bf16",
        "Qwen3 TTS 12Hz 1.7B Base BF16 GGUF",
        "bf16",
        "qwen3-tts-12hz-1.7b-base-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "qwen3_tts",
        "qwen3_tts_1_7b_base_orig",
        "Qwen3 TTS 12Hz 1.7B Base Original-Dtype GGUF",
        "orig",
        "qwen3-tts-12hz-1.7b-base-orig.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "voxcpm2",
        "voxcpm2_q8_0",
        "VoxCPM2 Q8_0 GGUF",
        "q8_0",
        "voxcpm2-q8_0.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "voxcpm2",
        "voxcpm2_bf16",
        "VoxCPM2 BF16 GGUF",
        "bf16",
        "voxcpm2-bf16.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "voxcpm2",
        "voxcpm2_orig",
        "VoxCPM2 Original-Dtype GGUF",
        "orig",
        "voxcpm2-orig.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "vevo2",
        "vevo2_q8_0",
        "Vevo2 Q8_0 GGUF",
        "q8_0",
        "vevo2-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "vevo2",
        "vevo2_f16",
        "Vevo2 F16 GGUF",
        "f16",
        "vevo2-f16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "vevo2",
        "vevo2_orig",
        "Vevo2 Original-Dtype GGUF",
        "orig",
        "vevo2-orig.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "index_tts2",
        "index_tts2_q8_0",
        "IndexTTS2 Q8_0 GGUF",
        "q8_0",
        "index-tts2-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "index_tts2",
        "index_tts2_f16",
        "IndexTTS2 F16 GGUF",
        "f16",
        "index-tts2-f16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "index_tts2",
        "index_tts2_orig",
        "IndexTTS2 Original-Dtype GGUF",
        "orig",
        "index-tts2-orig.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_v4_small_q8_0",
        "Irodori-TTS v4 Small Q8_0 GGUF",
        "q8_0",
        "irodori-tts-v4-small-q8_0.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_v4_small_f16",
        "Irodori-TTS v4 Small F16 GGUF",
        "f16",
        "irodori-tts-v4-small-f16.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_600m_v3_voicedesign_q8_0",
        "Irodori-TTS 600M v3 VoiceDesign Q8_0 GGUF",
        "q8_0",
        "irodori-tts-600m-v3-voicedesign-q8_0.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_600m_v3_voicedesign_f16",
        "Irodori-TTS 600M v3 VoiceDesign F16 GGUF",
        "f16",
        "irodori-tts-600m-v3-voicedesign-f16.gguf",
        ("tts", "clone", "design"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_500m_v3_q8_0",
        "Irodori-TTS 500M v3 Q8_0 GGUF",
        "q8_0",
        "irodori-tts-500m-v3-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "irodori_tts",
        "irodori_tts_500m_v3_f16",
        "Irodori-TTS 500M v3 F16 GGUF",
        "f16",
        "irodori-tts-500m-v3-f16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "moss_tts_local",
        "moss_tts_local_v1_5_q8_0",
        "MOSS-TTS-Local v1.5 Q8_0 GGUF",
        "q8_0",
        "moss-tts-local-v1.5-q8_0.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "moss_tts_local",
        "moss_tts_local_v1_5_bf16",
        "MOSS-TTS-Local v1.5 BF16 GGUF",
        "bf16",
        "moss-tts-local-v1.5-bf16.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
    (
        "glm_tts",
        "glm_tts_q8_0",
        "GLM-TTS mixed Q8_0/F16 GGUF",
        "q8_0",
        "Text to audio (TTS)/GLM-TTS_Q8.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.REQUIRED,
        "tts",
    ),
    (
        "inflect_v2",
        "inflect_micro_v2_orig",
        "Inflect Micro v2 Original-Dtype GGUF",
        "orig",
        "inflect-micro-v2-orig.gguf",
        ("tts",),
        AudioCppReferenceRequirement.NONE,
        "tts",
    ),
    (
        "outetts",
        "outetts_1_0_1b_q8_0",
        "Llama-OuteTTS 1.0 1B Q8_0 GGUF",
        "q8_0",
        "Text to audio (TTS)/Llama-OuteTTS-1.0-1B_Q8.gguf",
        ("tts", "clone"),
        AudioCppReferenceRequirement.OPTIONAL,
        "tts",
    ),
)


def _package_files(*paths: str) -> tuple[AudioCppFileSignal, ...]:
    return tuple(
        _safetensors_file(path, AudioCppFileRole.WEIGHT)
        if path.endswith(".safetensors")
        else _config_file(path)
        for path in paths
    )


_ADDITIONAL_RECIPES = (
    *(
        _recipe(
            family=family,
            package_variant=variant,
            display_name=display_name,
            package_format=AudioCppFileKind.GGUF,
            precision=precision,
            capabilities=capabilities,
            required_files=(_gguf_file(model_path),),
            model_relative_path=model_path,
            reference_requirement=reference_requirement,
            voice_reference_policy=_ADDITIONAL_VOICE_POLICIES.get(variant),
            task=task,
        )
        for (
            family,
            variant,
            display_name,
            precision,
            model_path,
            capabilities,
            reference_requirement,
            task,
        ) in _ADDITIONAL_GGUF_VARIANTS
    ),
    _recipe(
        family="omnivoice",
        package_variant="omnivoice_safetensors",
        display_name="OmniVoice Safetensors",
        package_format=AudioCppFileKind.SAFETENSORS,
        precision="native",
        capabilities=("tts", "clone", "design"),
        required_files=_package_files(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "audio_tokenizer/config.json",
            "audio_tokenizer/preprocessor_config.json",
            "model.safetensors",
            "audio_tokenizer/model.safetensors",
        ),
        model_relative_path=None,
        reference_requirement=AudioCppReferenceRequirement.OPTIONAL,
        voice_reference_policy=_ADDITIONAL_VOICE_POLICIES["omnivoice_safetensors"],
    ),
    _recipe(
        family="voxcpm2",
        package_variant="voxcpm2_safetensors",
        display_name="VoxCPM2 Safetensors",
        package_format=AudioCppFileKind.SAFETENSORS,
        precision="native",
        capabilities=("tts", "clone", "design"),
        required_files=_package_files(
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "model.safetensors",
            "audiovae.safetensors",
        ),
        model_relative_path=None,
        reference_requirement=AudioCppReferenceRequirement.OPTIONAL,
        voice_reference_policy=_ADDITIONAL_VOICE_POLICIES["voxcpm2_safetensors"],
    ),
    _recipe(
        family="index_tts2",
        package_variant="index_tts2_safetensors",
        display_name="IndexTTS2 Safetensors",
        package_format=AudioCppFileKind.SAFETENSORS,
        precision="native",
        capabilities=("tts", "clone"),
        required_files=_package_files(
            "config.yaml",
            "bpe.model",
            "w2v-bert-2.0/config.json",
            "w2v-bert-2.0/preprocessor_config.json",
            "bigvgan/config.json",
            "qwen0.6bemo4-merge/config.json",
            "qwen0.6bemo4-merge/generation_config.json",
            "qwen0.6bemo4-merge/tokenizer.json",
            "qwen0.6bemo4-merge/tokenizer_config.json",
            "qwen0.6bemo4-merge/vocab.json",
            "qwen0.6bemo4-merge/merges.txt",
            "gpt.safetensors",
            "s2mel.safetensors",
            "feat1.safetensors",
            "feat2.safetensors",
            "wav2vec2bert_stats.safetensors",
            "w2v-bert-2.0/model.safetensors",
            "semantic_codec_model.safetensors",
            "campplus.safetensors",
            "bigvgan/model.safetensors",
            "qwen0.6bemo4-merge/model.safetensors",
        ),
        model_relative_path=None,
        reference_requirement=AudioCppReferenceRequirement.REQUIRED,
    ),
)

AUDIO_CPP_RECIPE_REGISTRY = AudioCppRecipeRegistry(
    _INITIAL_RECIPES + _ADDITIONAL_RECIPES
)


def audio_cpp_guided_default_is_text_ready(
    settings: AudioCppSettingsConfig,
    *,
    registry: AudioCppRecipeRegistry = AUDIO_CPP_RECIPE_REGISTRY,
) -> bool:
    """Return whether the exact Guided default can generate without voice inputs.

    Args:
        settings: Validated full audio.cpp Settings snapshot.
        registry: Sealed recipe registry used to validate accepted identities.

    Returns:
        ``True`` only when the exact default package is a reviewed text-to-speech
        recipe that admits a request with neither a native voice nor a reference.
    """

    default_model_id = settings.guided_default_model_id
    if default_model_id is None:
        return False
    for package in settings.guided_packages:
        if package.public_model_id != default_model_id:
            continue
        try:
            recipe = registry.validate_accepted(package)
        except ValueError:
            return False
        return bool(
            recipe.projection.task == "tts"
            and recipe.admits_voice_reference(
                has_voice=False,
                has_reference=False,
            )
        )
    return False


_RELEASE_PACKAGES: dict[str, tuple[str, ...]] = {
    "supertonic": tuple(item[0] for item in _SUPERTONIC_VARIANTS)
    + ("supertonic_3_safetensors",),
    "pocket_tts": tuple(item[0] for item in _POCKET_GGUF_VARIANTS)
    + ("pocket_tts_english_safetensors",),
    "chatterbox": ("chatterbox_q8_0", "chatterbox_f16", "chatterbox_safetensors"),
    "dramabox": ("dramabox_q8_0",),
    "miotts": ("miotts_1_7b_q8_0", "miotts_1_7b_bf16", "miotts_1_7b_orig"),
    "vibevoice": ("vibevoice_1_5b_q8_0", "vibevoice_1_5b_bf16"),
    "moss_tts_nano": ("moss_tts_nano_100m_q8_0", "moss_tts_nano_100m_bf16"),
    "fish_audio": ("fish_audio_s2_pro_q8_0", "fish_audio_s2_pro_bf16"),
    "higgs_audio_tts": ("higgs_audio_tts_4b_q8_0", "higgs_audio_tts_4b_bf16"),
    "omnivoice": (
        "omnivoice_q8_0",
        "omnivoice_bf16",
        "omnivoice_f16",
        "omnivoice_safetensors",
    ),
    "qwen3_tts": (
        "qwen3_tts_1_7b_base_q8_0",
        "qwen3_tts_1_7b_base_bf16",
        "qwen3_tts_1_7b_base_orig",
        "qwen3_tts_1_7b_customvoice_q8_0",
        "qwen3_tts_1_7b_customvoice_bf16",
        "qwen3_tts_1_7b_voicedesign_q8_0",
        "qwen3_tts_1_7b_voicedesign_bf16",
        "qwen3_tts_1_7b_base_safetensors",
        "qwen3_tts_0_6b_base_safetensors",
    ),
    "voxcpm2": ("voxcpm2_q8_0", "voxcpm2_bf16", "voxcpm2_orig", "voxcpm2_safetensors"),
    "confucius4_tts": ("confucius4_tts_orig",),
    "vevo2": ("vevo2_q8_0", "vevo2_f16", "vevo2_orig"),
    "index_tts2": (
        "index_tts2_q8_0",
        "index_tts2_f16",
        "index_tts2_orig",
        "index_tts2_safetensors",
    ),
    "irodori_tts": (
        "irodori_tts_v4_small_q8_0",
        "irodori_tts_v4_small_f16",
        "irodori_tts_600m_v3_voicedesign_q8_0",
        "irodori_tts_600m_v3_voicedesign_f16",
        "irodori_tts_500m_v3_q8_0",
        "irodori_tts_500m_v3_f16",
    ),
    "moss_tts_local": ("moss_tts_local_v1_5_q8_0", "moss_tts_local_v1_5_bf16"),
    "glm_tts": ("glm_tts_q8_0",),
    "inflect_v2": ("inflect_micro_v2_orig",),
    "outetts": ("outetts_1_0_1b_q8_0",),
    "vietneu_tts": ("vietneu_tts_v3_turbo_q8_0",),
}

_UNSUPPORTED_RELEASE_PACKAGES = {
    "chatterbox_q8_0": (
        "The exact clon/vc task tokens are outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#chatterbox",
    ),
    "chatterbox_f16": (
        "The exact clon/vc task tokens are outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#chatterbox",
    ),
    "chatterbox_safetensors": (
        "The exact clon/vc task tokens are outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#chatterbox",
    ),
    "confucius4_tts_orig": (
        "The exact clon task token is outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#confucius4-tts",
    ),
    "miotts_1_7b_q8_0": (
        "Guided projection cannot resolve the required sibling MioCodec path.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#miotts",
    ),
    "miotts_1_7b_bf16": (
        "Guided projection cannot resolve the required sibling MioCodec path.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#miotts",
    ),
    "miotts_1_7b_orig": (
        "Guided projection cannot resolve the required sibling MioCodec path.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/tts.md#miotts",
    ),
    "qwen3_tts_1_7b_base_safetensors": (
        "Bounded file signals cannot distinguish the two Base Safetensors sizes.",
        f"{_PINNED_MODEL_SPEC_BASE}/qwen3_tts.json",
    ),
    "qwen3_tts_0_6b_base_safetensors": (
        "Bounded file signals cannot distinguish the two Base Safetensors sizes.",
        f"{_PINNED_MODEL_SPEC_BASE}/qwen3_tts.json",
    ),
    "qwen3_tts_1_7b_customvoice_q8_0": (
        "Packaged speaker input is required, but the typed policy permits no voice.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/models/qwen3.md#qwen3-tts-customvoice",
    ),
    "qwen3_tts_1_7b_customvoice_bf16": (
        "Packaged speaker input is required, but the typed policy permits no voice.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/models/qwen3.md#qwen3-tts-customvoice",
    ),
    "qwen3_tts_1_7b_voicedesign_q8_0": (
        "The exact vdes task token is outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/models/qwen3.md#qwen3-tts-voicedesign",
    ),
    "qwen3_tts_1_7b_voicedesign_bf16": (
        "The exact vdes task token is outside the typed guided projection.",
        f"https://github.com/0xShug0/audio.cpp/blob/{AUDIO_CPP_PINNED_COMMIT}"
        "/docs/models/qwen3.md#qwen3-tts-voicedesign",
    ),
    "vietneu_tts_v3_turbo_q8_0": (
        "The official package exposes only generic model.gguf as a bounded signal.",
        f"{_PINNED_MODEL_SPEC_BASE}/vietneu_tts.json",
    ),
}


def _release_accounting() -> tuple[AudioCppReleaseAccountingEntry, ...]:
    approved = {
        recipe.package_variant: recipe.recipe_id
        for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
    }
    rows = tuple(
        AudioCppReleaseAccountingEntry(
            family=family,
            package_variant=package_variant,
            state=(
                AudioCppRecipeSupportState.APPROVED
                if package_variant in approved
                else AudioCppRecipeSupportState.EXPLICITLY_UNSUPPORTED
                if package_variant in _UNSUPPORTED_RELEASE_PACKAGES
                else AudioCppRecipeSupportState.OPEN_GAP
            ),
            recipe_id=approved.get(package_variant),
            reason=(
                None
                if package_variant in approved
                else _UNSUPPORTED_RELEASE_PACKAGES.get(
                    package_variant,
                    ("Recipe review is not complete.", None),
                )[0]
            ),
            evidence_reference=(
                None
                if package_variant in approved
                else _UNSUPPORTED_RELEASE_PACKAGES.get(
                    package_variant,
                    (None, None),
                )[1]
            ),
        )
        for family, packages in _RELEASE_PACKAGES.items()
        for package_variant in packages
    )
    if len(_RELEASE_PACKAGES) != 21 or len(rows) != 67:
        raise RuntimeError("audio.cpp pinned release accounting is incomplete")
    return rows


AUDIO_CPP_RELEASE_ACCOUNTING = _release_accounting()


__all__ = (
    "AUDIO_CPP_PINNED_COMMIT",
    "AUDIO_CPP_PINNED_RELEASE",
    "AUDIO_CPP_RECIPE_REGISTRY",
    "AUDIO_CPP_RECIPE_SCHEMA_VERSION",
    "AUDIO_CPP_RELEASE_ACCOUNTING",
    "AudioCppBackendEvidence",
    "AudioCppBackendEvidenceState",
    "AudioCppFileKind",
    "AudioCppFileRole",
    "AudioCppFileSignal",
    "AudioCppMatchResult",
    "AudioCppMatchState",
    "AudioCppPackageCandidate",
    "AudioCppPackageDescription",
    "AudioCppPackageFileEvidence",
    "AudioCppPackageRecipe",
    "AudioCppRecipeRegistry",
    "AudioCppRecipeSupportState",
    "AudioCppReferenceRequirement",
    "AudioCppReleaseAccountingEntry",
    "AudioCppVerifiedSupportClaim",
    "AudioCppVoiceReferencePolicy",
    "audio_cpp_guided_default_is_text_ready",
)
