"""Bounded audio.cpp artifact consumers and shared-root mutation leases."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactNotInstalledError,
    ArtifactRef,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedArtifactIdentity,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.profile_reference_types import TTSCloneRecipeRequirement
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

_MAX_CONSUMERS = 200
_MAX_LABEL_CHARACTERS = 256
_MAX_ID_CHARACTERS = 256


class AudioCppArtifactDependencyError(RuntimeError):
    """Report one bounded dependency-projection or lease failure."""


class _InstalledRootHandle(Protocol):
    def close(self) -> None: ...


class _ArtifactService(Protocol):
    def acquire_installed_root(
        self, reference: ArtifactRef
    ) -> _InstalledRootHandle: ...


@dataclass(frozen=True, slots=True)
class AudioCppArtifactRemovalEvidence:
    """Stable bounded consumer facts for one exact managed package."""

    reference: ArtifactRef
    settings_consumers: tuple[tuple[str, str, str], ...] = ()
    profile_consumers: tuple[tuple[str, str, int, bool], ...] = ()
    staged_runtime_ids: tuple[str, ...] = ()
    live_runtime_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference) is not ArtifactRef:
            raise AudioCppArtifactDependencyError("invalid dependency evidence")
        if (
            type(self.settings_consumers) is not tuple
            or type(self.profile_consumers) is not tuple
            or type(self.staged_runtime_ids) is not tuple
            or type(self.live_runtime_ids) is not tuple
            or sum(
                map(
                    len,
                    (
                        self.settings_consumers,
                        self.profile_consumers,
                        self.staged_runtime_ids,
                        self.live_runtime_ids,
                    ),
                )
            )
            > _MAX_CONSUMERS
        ):
            raise AudioCppArtifactDependencyError("invalid dependency evidence")
        invalid = False
        try:
            settings = tuple(
                (
                    _bounded_identifier(scope),
                    _bounded_label(label),
                    _bounded_identifier(identity),
                )
                for scope, label, identity in self.settings_consumers
            )
            profiles = tuple(
                (
                    _bounded_identifier(identity),
                    _bounded_label(label),
                    _bounded_count(assignments),
                    _exact_bool(has_clone),
                )
                for identity, label, assignments, has_clone in self.profile_consumers
            )
            staged = tuple(
                _bounded_identifier(value) for value in self.staged_runtime_ids
            )
            live = tuple(_bounded_identifier(value) for value in self.live_runtime_ids)
        except (TypeError, ValueError):
            invalid = True
            settings = ()
            profiles = ()
            staged = ()
            live = ()
        if invalid:
            raise AudioCppArtifactDependencyError("invalid dependency evidence")
        object.__setattr__(self, "settings_consumers", tuple(sorted(settings)))
        object.__setattr__(self, "profile_consumers", tuple(sorted(profiles)))
        object.__setattr__(self, "staged_runtime_ids", tuple(sorted(set(staged))))
        object.__setattr__(self, "live_runtime_ids", tuple(sorted(set(live))))


@dataclass(frozen=True, slots=True)
class AudioCppArtifactRemovalPreview:
    """Immutable public dependency preview bound to stable consumer evidence."""

    reference: ArtifactRef
    fingerprint: str
    settings_labels: tuple[str, ...]
    profile_labels: tuple[str, ...]
    assignment_count: int
    clone_reference_count: int
    staged_or_live: bool
    generic_lease_blocked: bool


@dataclass(frozen=True, slots=True)
class AudioCppArtifactConsumerRequirement:
    """Exact or settings-resolvable model dependency for one consumer mutation."""

    provider_id: str
    model_id: str
    recipe_requirement: TTSCloneRecipeRequirement | None = None

    def __post_init__(self) -> None:
        try:
            provider_id = _bounded_identifier(self.provider_id)
            model_id = _bounded_identifier(self.model_id)
        except (TypeError, ValueError):
            raise AudioCppArtifactDependencyError(
                "invalid consumer requirement"
            ) from None
        if (
            self.recipe_requirement is not None
            and type(self.recipe_requirement) is not TTSCloneRecipeRequirement
        ):
            raise AudioCppArtifactDependencyError("invalid consumer requirement")
        if (
            self.recipe_requirement is not None
            and self.recipe_requirement.model_id != model_id
        ):
            raise AudioCppArtifactDependencyError("invalid consumer requirement")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id)


@dataclass(frozen=True, slots=True)
class AudioCppManagedConsumerIdentity:
    """Immutable saved-Settings package identity used at mutation boundaries."""

    recipe_id: str
    recipe_revision: int
    model_id: str
    managed_artifact: AudioCppManagedArtifactIdentity

    def __post_init__(self) -> None:
        try:
            recipe_id = _bounded_identifier(self.recipe_id)
            model_id = _bounded_identifier(self.model_id)
            if type(self.recipe_revision) is not int or self.recipe_revision < 1:
                raise ValueError
            if type(self.managed_artifact) is not AudioCppManagedArtifactIdentity:
                raise TypeError
        except (TypeError, ValueError):
            raise AudioCppArtifactDependencyError(
                "invalid saved package identity"
            ) from None
        object.__setattr__(self, "recipe_id", recipe_id)
        object.__setattr__(self, "model_id", model_id)


def _bounded_identifier(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_ID_CHARACTERS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError
    value.encode("utf-8", errors="strict")
    return value


def _bounded_label(value: object) -> str:
    label = _bounded_identifier(value)
    absolute_path = label.startswith(("/", "\\")) or (
        len(label) >= 3 and label[1] == ":" and label[2] in "/\\"
    )
    if len(label) > _MAX_LABEL_CHARACTERS or absolute_path:
        raise ValueError
    return label


def _bounded_count(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_CONSUMERS:
        raise ValueError
    return value


def _exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError
    return value


def _hash_field(digest: object, value: str) -> None:
    encoded = value.encode("utf-8", errors="strict")
    digest.update(len(encoded).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(encoded)  # type: ignore[attr-defined]


def _consumer_fingerprint(evidence: AudioCppArtifactRemovalEvidence) -> str:
    digest = hashlib.sha256()
    for value in (
        "audio-cpp-removal-preview-v1",
        evidence.reference.artifact_id,
        evidence.reference.revision,
        evidence.reference.variant,
    ):
        _hash_field(digest, value)
    for scope, label, identity in evidence.settings_consumers:
        for value in ("settings", scope, label, identity):
            _hash_field(digest, value)
    for identity, label, assignments, has_clone in evidence.profile_consumers:
        for value in (
            "profile",
            identity,
            label,
            str(assignments),
            "clone" if has_clone else "plain",
        ):
            _hash_field(digest, value)
    for kind, identities in (
        ("staged", evidence.staged_runtime_ids),
        ("live", evidence.live_runtime_ids),
    ):
        for identity in identities:
            _hash_field(digest, kind)
            _hash_field(digest, identity)
    return digest.hexdigest()


def build_audio_cpp_artifact_removal_preview(
    evidence: AudioCppArtifactRemovalEvidence,
    *,
    generic_lease_blocked: bool = False,
) -> AudioCppArtifactRemovalPreview:
    """Build one public preview; the volatile lease advisory is not fingerprinted."""

    if type(evidence) is not AudioCppArtifactRemovalEvidence:
        raise AudioCppArtifactDependencyError("invalid dependency evidence")
    if type(generic_lease_blocked) is not bool:
        raise AudioCppArtifactDependencyError("invalid dependency evidence")
    return AudioCppArtifactRemovalPreview(
        reference=evidence.reference,
        fingerprint=_consumer_fingerprint(evidence),
        settings_labels=tuple(item[1] for item in evidence.settings_consumers),
        profile_labels=tuple(item[1] for item in evidence.profile_consumers),
        assignment_count=sum(item[2] for item in evidence.profile_consumers),
        clone_reference_count=sum(item[3] for item in evidence.profile_consumers),
        staged_or_live=bool(evidence.staged_runtime_ids or evidence.live_runtime_ids),
        generic_lease_blocked=generic_lease_blocked,
    )


def project_audio_cpp_artifact_removal_evidence(
    reference: ArtifactRef,
    *,
    saved_settings: AudioCppSettingsConfig,
    draft_settings: AudioCppSettingsConfig | None = None,
    profiles: tuple[tuple[TTSGenerationProfile, int], ...] = (),
    staged_runtime_ids: tuple[str, ...] = (),
    live_runtime_ids: tuple[str, ...] = (),
) -> AudioCppArtifactRemovalEvidence:
    """Project exact settings/profile consumers without private package material."""

    if (
        type(reference) is not ArtifactRef
        or type(saved_settings) is not AudioCppSettingsConfig
    ):
        raise AudioCppArtifactDependencyError("dependency projection failed")
    if (
        draft_settings is not None
        and type(draft_settings) is not AudioCppSettingsConfig
    ):
        raise AudioCppArtifactDependencyError("dependency projection failed")
    settings: list[tuple[str, str, str]] = []
    try:
        from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
            audio_cpp_curated_entries,
        )

        relevant_models = {
            descriptor.model_id
            for descriptor, _sources in audio_cpp_curated_entries()
            if descriptor.reference == reference
        }
    except (OSError, TypeError, ValueError):
        raise AudioCppArtifactDependencyError("dependency projection failed") from None
    for scope, label, config in (
        ("saved", "Guided Settings", saved_settings),
        ("draft", "Unsaved Guided Settings", draft_settings),
    ):
        if config is None:
            continue
        for package in config.guided_packages:
            identity = package.managed_artifact
            if identity is None or _reference_from_identity(identity) != reference:
                continue
            relevant_models.add(package.public_model_id)
            display = (
                f"{label} default"
                if config.guided_default_model_id == package.public_model_id
                else label
            )
            settings.append((scope, display, package.package_uuid))
    projected_profiles: list[tuple[str, str, int, bool]] = []
    try:
        for profile, assignment_count in profiles:
            if type(profile) is not TTSGenerationProfile:
                raise TypeError
            if (
                profile.provider_id != "audio_cpp"
                or profile.model_id not in relevant_models
            ):
                continue
            projected_profiles.append(
                (
                    str(profile.profile_id),
                    profile.display_name,
                    assignment_count,
                    profile.reference is not None,
                )
            )
    except (TypeError, ValueError):
        raise AudioCppArtifactDependencyError("dependency projection failed") from None
    return AudioCppArtifactRemovalEvidence(
        reference=reference,
        settings_consumers=tuple(settings),
        profile_consumers=tuple(projected_profiles),
        staged_runtime_ids=staged_runtime_ids,
        live_runtime_ids=live_runtime_ids,
    )


def _reference_from_identity(identity: AudioCppManagedArtifactIdentity) -> ArtifactRef:
    return ArtifactRef(identity.artifact_id, identity.revision, identity.variant)


def _reference_key(reference: ArtifactRef) -> tuple[str, str, str]:
    return reference.artifact_id, reference.revision, reference.variant


def is_curated_audio_cpp_artifact_reference(reference: ArtifactRef) -> bool:
    """Return network-free exact membership in the checked-in audio.cpp catalog."""

    if type(reference) is not ArtifactRef:
        return False
    try:
        from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
            audio_cpp_curated_entries,
        )

        return any(
            descriptor.reference == reference
            for descriptor, _sources in audio_cpp_curated_entries()
        )
    except (OSError, TypeError, ValueError):
        return False


class AudioCppArtifactLeaseCoordinator:
    """Lease exact installed roots around consumer mutation commits."""

    def __init__(
        self,
        artifact_service: _ArtifactService,
        *,
        saved_settings_snapshot: Callable[
            [], tuple[AudioCppManagedConsumerIdentity, ...]
        ],
        catalog_entries: Callable[[], tuple[object, ...]] | None = None,
    ) -> None:
        if not callable(saved_settings_snapshot) or (
            catalog_entries is not None and not callable(catalog_entries)
        ):
            raise AudioCppArtifactDependencyError("invalid lease coordinator")
        self._artifact_service = artifact_service
        self._saved_settings_snapshot = saved_settings_snapshot
        self._catalog_entries = catalog_entries or self._default_catalog_entries

    @staticmethod
    def _default_catalog_entries() -> tuple[object, ...]:
        from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
            audio_cpp_curated_entries,
        )
        from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

        recipes = {
            recipe.recipe_id: recipe for recipe in AUDIO_CPP_RECIPE_REGISTRY.recipes
        }
        return tuple(
            (descriptor, sources, recipes[package.recipe_id])
            for descriptor, sources in audio_cpp_curated_entries()
            for package in AUDIO_CPP_RECIPE_REGISTRY.recipes
            if package.model_library_artifact_ids == (descriptor.reference.artifact_id,)
        )

    def _resolved_references(
        self,
        consumers: Iterable[AudioCppArtifactConsumerRequirement],
    ) -> tuple[ArtifactRef, ...]:
        try:
            exact_consumers = tuple(consumers)
            saved = tuple(self._saved_settings_snapshot())
            entries = tuple(self._catalog_entries())
        except Exception:
            raise AudioCppArtifactDependencyError(
                "dependency resolution failed"
            ) from None
        if len(exact_consumers) > _MAX_CONSUMERS or len(saved) > _MAX_CONSUMERS:
            raise AudioCppArtifactDependencyError("dependency resolution failed")
        if not all(
            type(item) is AudioCppArtifactConsumerRequirement
            for item in exact_consumers
        ) or not all(type(item) is AudioCppManagedConsumerIdentity for item in saved):
            raise AudioCppArtifactDependencyError("dependency resolution failed")

        resolved: dict[tuple[str, str, str], ArtifactRef] = {}
        for consumer in exact_consumers:
            if consumer.provider_id != "audio_cpp":
                continue
            requirement = consumer.recipe_requirement
            candidates: list[tuple[ArtifactRef, object]] = []
            for entry in entries:
                try:
                    descriptor, _sources, recipe = cast(tuple[Any, Any, Any], entry)
                    matches = descriptor.model_id == consumer.model_id
                    if requirement is not None:
                        matches = matches and (
                            recipe.recipe_id == requirement.recipe_id
                            and recipe.recipe_revision == requirement.recipe_revision
                            and recipe.default_public_model_id == requirement.model_id
                        )
                    if matches:
                        candidates.append((descriptor.reference, recipe))
                except Exception:
                    raise AudioCppArtifactDependencyError(
                        "dependency resolution failed"
                    ) from None
            if len(candidates) > 1:
                raise AudioCppArtifactDependencyError("dependency resolution failed")
            catalog_reference = candidates[0][0] if candidates else None
            saved_matches = tuple(
                item
                for item in saved
                if item.model_id == consumer.model_id
                and (
                    requirement is None
                    or (
                        item.recipe_id == requirement.recipe_id
                        and item.recipe_revision == requirement.recipe_revision
                    )
                )
            )
            if len(saved_matches) > 1:
                raise AudioCppArtifactDependencyError("dependency resolution failed")
            persisted_reference = (
                None
                if not saved_matches
                else _reference_from_identity(saved_matches[0].managed_artifact)
            )
            if (
                catalog_reference is not None
                and persisted_reference is not None
                and catalog_reference != persisted_reference
            ):
                raise AudioCppArtifactDependencyError(
                    "managed artifact identity disagreement"
                )
            reference = persisted_reference or catalog_reference
            if reference is not None:
                resolved[_reference_key(reference)] = reference
        return tuple(resolved[key] for key in sorted(resolved))

    @contextmanager
    def lease_consumers(
        self,
        consumers: Iterable[AudioCppArtifactConsumerRequirement],
    ) -> Iterator[None]:
        """Hold sorted exact shared-root leases through caller commit/rollback."""

        handles: list[_InstalledRootHandle] = []
        primary_error: BaseException | None = None
        try:
            for reference in self._resolved_references(consumers):
                try:
                    handle = self._artifact_service.acquire_installed_root(reference)
                except ArtifactNotInstalledError:
                    continue
                handles.append(handle)
            yield
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error: BaseException | None = None
            for handle in reversed(handles):
                try:
                    handle.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                if primary_error is not None:
                    primary_error.add_note("audio.cpp artifact lease cleanup failed")
                else:
                    raise cleanup_error


__all__ = [
    "AudioCppArtifactConsumerRequirement",
    "AudioCppArtifactDependencyError",
    "AudioCppArtifactLeaseCoordinator",
    "AudioCppArtifactRemovalEvidence",
    "AudioCppArtifactRemovalPreview",
    "AudioCppManagedConsumerIdentity",
    "build_audio_cpp_artifact_removal_preview",
    "project_audio_cpp_artifact_removal_evidence",
    "is_curated_audio_cpp_artifact_reference",
]
