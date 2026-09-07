"""Bounded audio.cpp artifact consumers and shared-root mutation leases."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactNotInstalledError,
    ArtifactRemovalAvailability,
    ArtifactRef,
    take_artifact_removal_cleanup_owner,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedArtifactIdentity,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.profile_reference_types import TTSCloneRecipeRequirement
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

_MAX_CONSUMERS = 200
_MAX_LABEL_CHARACTERS = 256
_MAX_ID_CHARACTERS = 256


class AudioCppArtifactDependencyError(RuntimeError):
    """Report one bounded dependency-projection or lease failure."""


class _InstalledRootHandle(Protocol):
    def close(self) -> None: ...


class _RemovalAuthority(Protocol):
    def commit(self) -> None: ...

    def close(self) -> None: ...


class _ArtifactService(Protocol):
    def acquire_installed_root(
        self, reference: ArtifactRef
    ) -> _InstalledRootHandle: ...

    def probe_removal_availability(
        self, reference: ArtifactRef
    ) -> ArtifactRemovalAvailability: ...

    def acquire_removal_authority(
        self, reference: ArtifactRef
    ) -> _RemovalAuthority: ...


@dataclass(frozen=True, slots=True)
class AudioCppArtifactRemovalEvidence:
    """Stable bounded consumer facts for one exact managed package."""

    reference: ArtifactRef
    settings_consumers: tuple[tuple[str, str, str], ...] = ()
    profile_consumers: tuple[tuple[str, str, int, bool], ...] = ()
    profile_provenance: tuple[tuple[str, str], ...] = ()
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
            or type(self.profile_provenance) is not tuple
            or len(self.profile_provenance) > _MAX_CONSUMERS
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
            provenance = tuple(
                (_bounded_identifier(identity), _bounded_identifier(requirement))
                for identity, requirement in self.profile_provenance
            )
        except (TypeError, ValueError):
            invalid = True
            settings = ()
            profiles = ()
            staged = ()
            live = ()
            provenance = ()
        if invalid:
            raise AudioCppArtifactDependencyError("invalid dependency evidence")
        object.__setattr__(self, "settings_consumers", tuple(sorted(settings)))
        object.__setattr__(self, "profile_consumers", tuple(sorted(profiles)))
        object.__setattr__(self, "staged_runtime_ids", tuple(sorted(set(staged))))
        object.__setattr__(self, "live_runtime_ids", tuple(sorted(set(live))))
        object.__setattr__(self, "profile_provenance", tuple(sorted(provenance)))


@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryObservationSnapshot:
    """One immutable, generation-local batch of exact package observations."""

    observations: tuple[AudioCppArtifactRemovalEvidence, ...]

    def __post_init__(self) -> None:
        if (
            type(self.observations) is not tuple
            or len(self.observations) > _MAX_CONSUMERS
            or any(
                type(observation) is not AudioCppArtifactRemovalEvidence
                for observation in self.observations
            )
            or len({item.reference for item in self.observations})
            != len(self.observations)
        ):
            raise AudioCppArtifactDependencyError("invalid model library observation")


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
    for identity, requirement in evidence.profile_provenance:
        for value in ("profile-provenance", identity, requirement):
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
    saved_preferences: TTSPreferencesSnapshot | None = None,
    draft_preferences: TTSPreferencesSnapshot | None = None,
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
    if any(
        value is not None and type(value) is not TTSPreferencesSnapshot
        for value in (saved_preferences, draft_preferences)
    ):
        raise AudioCppArtifactDependencyError("dependency projection failed")
    settings: list[tuple[str, str, str]] = []
    try:
        from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
            audio_cpp_curated_entries,
        )
        from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

        descriptors = tuple(audio_cpp_curated_entries())
        recipes = tuple(AUDIO_CPP_RECIPE_REGISTRY.recipes)
        catalog_model_references: dict[str, set[ArtifactRef]] = {}
        catalog_requirement_references: dict[
            tuple[str, int, str], set[ArtifactRef]
        ] = {}
        for descriptor, _sources in descriptors:
            catalog_model_references.setdefault(descriptor.model_id, set()).add(
                descriptor.reference
            )
            for recipe in recipes:
                if (
                    descriptor.reference.artifact_id
                    in recipe.model_library_artifact_ids
                    and recipe.default_public_model_id == descriptor.model_id
                ):
                    key = (
                        recipe.recipe_id,
                        recipe.recipe_revision,
                        descriptor.model_id,
                    )
                    catalog_requirement_references.setdefault(key, set()).add(
                        descriptor.reference
                    )
    except (OSError, TypeError, ValueError):
        raise AudioCppArtifactDependencyError("dependency projection failed") from None
    managed_model_references: dict[str, set[ArtifactRef]] = {}
    managed_requirement_references: dict[tuple[str, int, str], set[ArtifactRef]] = {}
    scoped_managed_model_references: dict[str, dict[str, set[ArtifactRef]]] = {
        "saved": {},
        "draft": {},
    }
    for scope, label, config in (
        ("saved", "Guided Settings", saved_settings),
        ("draft", "Unsaved Guided Settings", draft_settings),
    ):
        if config is None:
            continue
        for package in config.guided_packages:
            identity = package.managed_artifact
            if identity is None:
                continue
            package_reference = _reference_from_identity(identity)
            managed_model_references.setdefault(package.public_model_id, set()).add(
                package_reference
            )
            managed_requirement_references.setdefault(
                (
                    package.recipe_id,
                    package.recipe_revision,
                    package.public_model_id,
                ),
                set(),
            ).add(package_reference)
            scoped_managed_model_references[scope].setdefault(
                package.public_model_id, set()
            ).add(package_reference)
            if package_reference != reference:
                continue
            display = (
                f"{label} default"
                if config.guided_default_model_id == package.public_model_id
                else label
            )
            settings.append((scope, display, package.package_uuid))
    for scope, label, preferences in (
        ("saved", "Saved global TTS default", saved_preferences),
        ("draft", "Unsaved global TTS default", draft_preferences),
    ):
        if (
            preferences is None
            or preferences.provider_id != "audio_cpp"
            or preferences.model_mode != "exact"
            or preferences.model_id is None
        ):
            continue
        preference_reference = _resolve_exact_projection_reference(
            catalog_model_references.get(preferences.model_id, set()),
            scoped_managed_model_references[scope].get(preferences.model_id, set()),
        )
        if preference_reference == reference:
            settings.append((f"{scope}-default", label, preferences.model_id))
    projected_profiles: list[tuple[str, str, int, bool]] = []
    projected_provenance: list[tuple[str, str]] = []
    try:
        for profile, assignment_count in profiles:
            if type(profile) is not TTSGenerationProfile:
                raise TypeError
            if profile.provider_id != "audio_cpp":
                continue
            profile_reference = profile.reference
            requirement = (
                None
                if profile_reference is None
                else profile_reference.recipe_requirement
            )
            provenance = ""
            if requirement is None:
                profile_reference_match = _resolve_exact_projection_reference(
                    catalog_model_references.get(profile.model_id, set()),
                    managed_model_references.get(profile.model_id, set()),
                )
            else:
                requirement_key = (
                    requirement.recipe_id,
                    requirement.recipe_revision,
                    requirement.model_id,
                )
                profile_reference_match = _resolve_exact_projection_reference(
                    catalog_requirement_references.get(requirement_key, set()),
                    managed_requirement_references.get(requirement_key, set()),
                )
                provenance = (
                    f"{requirement.recipe_id}@{requirement.recipe_revision}:"
                    f"{requirement.model_id}"
                )
            if profile_reference_match != reference:
                continue
            profile_identity = str(profile.profile_id)
            projected_profiles.append(
                (
                    profile_identity,
                    profile.display_name,
                    assignment_count,
                    profile_reference is not None,
                )
            )
            if provenance:
                projected_provenance.append((profile_identity, provenance))
    except (TypeError, ValueError):
        raise AudioCppArtifactDependencyError("dependency projection failed") from None
    return AudioCppArtifactRemovalEvidence(
        reference=reference,
        settings_consumers=tuple(settings),
        profile_consumers=tuple(projected_profiles),
        profile_provenance=tuple(projected_provenance),
        staged_runtime_ids=staged_runtime_ids,
        live_runtime_ids=live_runtime_ids,
    )


def _reference_from_identity(identity: AudioCppManagedArtifactIdentity) -> ArtifactRef:
    return ArtifactRef(identity.artifact_id, identity.revision, identity.variant)


def _reference_key(reference: ArtifactRef) -> tuple[str, str, str]:
    return reference.artifact_id, reference.revision, reference.variant


def _resolve_exact_projection_reference(
    catalog_references: Iterable[ArtifactRef],
    persisted_references: Iterable[ArtifactRef],
) -> ArtifactRef | None:
    """Resolve one exact catalog/persisted join or fail closed on ambiguity."""

    try:
        catalog = set(catalog_references)
        persisted = set(persisted_references)
    except (TypeError, ValueError):
        raise AudioCppArtifactDependencyError("dependency resolution failed") from None
    if len(catalog) > 1 or len(persisted) > 1:
        raise AudioCppArtifactDependencyError("dependency resolution failed")
    catalog_reference = None if not catalog else next(iter(catalog))
    persisted_reference = None if not persisted else next(iter(persisted))
    if (
        catalog_reference is not None
        and persisted_reference is not None
        and catalog_reference != persisted_reference
    ):
        raise AudioCppArtifactDependencyError("managed artifact identity disagreement")
    return persisted_reference or catalog_reference


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
        self._cleanup_owners: list[_InstalledRootHandle | _RemovalAuthority] = []
        self._cleanup_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._blocking_tasks: set[asyncio.Task[object]] = set()
        self._operations: dict[asyncio.Future[None], asyncio.Task[Any]] = {}
        self._removal_tasks: set[asyncio.Task[str]] = set()
        self._shutdown_task: asyncio.Task[None] | None = None
        self._closed = False

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
            catalog_references: set[ArtifactRef] = set()
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
                        catalog_references.add(descriptor.reference)
                except Exception:
                    raise AudioCppArtifactDependencyError(
                        "dependency resolution failed"
                    ) from None
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
            persisted_references = {
                _reference_from_identity(item.managed_artifact)
                for item in saved_matches
            }
            reference = _resolve_exact_projection_reference(
                catalog_references,
                persisted_references,
            )
            if reference is not None:
                resolved[_reference_key(reference)] = reference
        return tuple(resolved[key] for key in sorted(resolved))

    async def _settle_task(
        self,
        task: asyncio.Future[Any],
    ) -> tuple[asyncio.CancelledError | None, object | None, BaseException | None]:
        cancellation: asyncio.CancelledError | None = None
        waiter = asyncio.current_task()
        requests = waiter.cancelling() if waiter is not None else 0
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                current = waiter.cancelling() if waiter is not None else 0
                if current > requests:
                    cancellation = cancellation or error
                    requests = current
            except BaseException:
                if not task.done():
                    raise
        try:
            return cancellation, task.result(), None
        except BaseException as error:
            return cancellation, None, error

    async def _settle_blocking(
        self,
        function: Callable[..., object],
        *args: object,
    ) -> tuple[asyncio.CancelledError | None, object | None, BaseException | None]:
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._blocking_tasks.add(task)
        try:
            return await self._settle_task(task)
        finally:
            self._blocking_tasks.discard(task)

    def _retain_error_owner(self, error: BaseException) -> None:
        owner = take_artifact_removal_cleanup_owner(error)
        if owner is not None:
            self._retain_cleanup_owner(owner)

    def _retain_cleanup_owner(
        self,
        owner: _InstalledRootHandle | _RemovalAuthority,
    ) -> None:
        if all(retained is not owner for retained in self._cleanup_owners):
            self._cleanup_owners.append(owner)

    async def _close_owner(
        self,
        owner: _InstalledRootHandle | _RemovalAuthority,
    ) -> tuple[asyncio.CancelledError | None, BaseException | None]:
        cancellation, _result, error = await self._settle_blocking(owner.close)
        if error is not None:
            self._retain_error_owner(error)
        return cancellation, error

    async def _drain_cleanup_locked(self) -> None:
        retained = self._cleanup_owners
        self._cleanup_owners = []
        cancellation: asyncio.CancelledError | None = None
        for owner in retained:
            owner_cancellation, error = await self._close_owner(owner)
            cancellation = cancellation or owner_cancellation
            if error is not None:
                self._retain_cleanup_owner(owner)
        if cancellation is not None:
            raise cancellation
        if self._cleanup_owners:
            raise AudioCppArtifactDependencyError("artifact cleanup is incomplete")

    async def drain_cleanup(self) -> None:
        """Retry every app-owned cleanup authority exactly once."""

        async with self._cleanup_lock:
            await self._drain_cleanup_locked()

    async def _admit_operation(self) -> asyncio.Future[None]:
        owner = asyncio.current_task()
        if owner is None:
            raise AudioCppArtifactDependencyError("artifact operation admission failed")
        async with self._admission_lock:
            if self._closed:
                raise AudioCppArtifactDependencyError("lease coordinator is closed")
            completion = asyncio.get_running_loop().create_future()
            self._operations[completion] = owner
            return completion

    def _complete_operation(self, completion: asyncio.Future[None]) -> None:
        self._operations.pop(completion, None)
        if not completion.done():
            completion.set_result(None)

    @asynccontextmanager
    async def lease_consumers(
        self,
        consumers: Iterable[AudioCppArtifactConsumerRequirement],
    ) -> AsyncIterator[None]:
        """Hold sorted exact shared-root leases through caller commit/rollback."""

        completion = await self._admit_operation()
        try:
            async with self._lease_consumer_operation(consumers):
                yield
        finally:
            self._complete_operation(completion)

    @asynccontextmanager
    async def _lease_consumer_operation(
        self,
        consumers: Iterable[AudioCppArtifactConsumerRequirement],
    ) -> AsyncIterator[None]:
        """Run one admitted consumer lease through definitive cleanup."""

        await self.drain_cleanup()
        handles: list[_InstalledRootHandle] = []
        primary_error: BaseException | None = None
        try:
            for reference in self._resolved_references(consumers):
                cancellation, value, error = await self._settle_blocking(
                    self._artifact_service.acquire_installed_root,
                    reference,
                )
                if isinstance(error, ArtifactNotInstalledError):
                    if cancellation is not None:
                        raise cancellation
                    continue
                if error is not None:
                    self._retain_error_owner(error)
                    if cancellation is not None:
                        raise cancellation
                    raise AudioCppArtifactDependencyError(
                        "artifact lease acquisition failed"
                    ) from None
                handle = cast(_InstalledRootHandle, value)
                handles.append(handle)
                if cancellation is not None:
                    raise cancellation
            yield
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_failed = False
            cleanup_cancellation: asyncio.CancelledError | None = None
            for handle in reversed(handles):
                cancellation, close_error = await self._close_owner(handle)
                cleanup_cancellation = cleanup_cancellation or cancellation
                if close_error is not None:
                    self._retain_cleanup_owner(handle)
                    cleanup_failed = True
            if cleanup_failed:
                if primary_error is not None:
                    primary_error.add_note("audio.cpp artifact lease cleanup failed")
                else:
                    raise AudioCppArtifactDependencyError(
                        "artifact cleanup is incomplete"
                    ) from None
            if cleanup_cancellation is not None and primary_error is None:
                raise cleanup_cancellation

    async def probe_removal_availability(
        self,
        reference: ArtifactRef,
    ) -> ArtifactRemovalAvailability:
        """Probe off-loop while retaining any escaped cleanup authority."""

        completion = await self._admit_operation()
        try:
            await self.drain_cleanup()
            cancellation, value, error = await self._settle_blocking(
                self._artifact_service.probe_removal_availability,
                reference,
            )
            if error is not None:
                self._retain_error_owner(error)
                if cancellation is not None:
                    raise cancellation
                raise AudioCppArtifactDependencyError("removal probe failed") from None
            if cancellation is not None:
                raise cancellation
            if type(value) is not ArtifactRemovalAvailability:
                raise AudioCppArtifactDependencyError("removal probe failed")
            return value
        finally:
            self._complete_operation(completion)

    async def _remove_if_unchanged(
        self,
        reference: ArtifactRef,
        fingerprint: str,
        collect_fingerprint: Callable[[], Awaitable[str]],
    ) -> str:
        await self.drain_cleanup()
        authority: _RemovalAuthority | None = None
        primary_error: BaseException | None = None
        close_failed = False
        try:
            cancellation, value, error = await self._settle_blocking(
                self._artifact_service.acquire_removal_authority,
                reference,
            )
            if error is not None:
                self._retain_error_owner(error)
                if cancellation is not None:
                    raise cancellation
                raise AudioCppArtifactDependencyError(
                    "removal authority is unavailable"
                ) from None
            authority = cast(_RemovalAuthority, value)
            if cancellation is not None:
                raise cancellation
            current = await collect_fingerprint()
            if current != fingerprint:
                return "changed"
            cancellation, _value, error = await self._settle_blocking(authority.commit)
            if error is not None:
                self._retain_error_owner(error)
                if cancellation is not None:
                    raise cancellation
                raise AudioCppArtifactDependencyError(
                    "artifact removal failed"
                ) from None
            if cancellation is not None:
                raise cancellation
            return "committed"
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if authority is not None:
                cancellation, close_error = await self._close_owner(authority)
                if close_error is not None:
                    self._retain_cleanup_owner(authority)
                    close_failed = True
                if close_failed and primary_error is None:
                    raise AudioCppArtifactDependencyError(
                        "artifact cleanup is incomplete"
                    ) from None
                if cancellation is not None and primary_error is None:
                    raise cancellation

    async def remove_if_unchanged(
        self,
        reference: ArtifactRef,
        fingerprint: str,
        collect_fingerprint: Callable[[], Awaitable[str]],
    ) -> str:
        """Own one retained acquire/revalidate/commit/close operation."""

        async with self._admission_lock:
            if self._closed:
                raise AudioCppArtifactDependencyError("lease coordinator is closed")
            task = asyncio.create_task(
                self._remove_if_unchanged(reference, fingerprint, collect_fingerprint)
            )
            self._removal_tasks.add(task)
        cancellation: asyncio.CancelledError | None = None
        waiter = asyncio.current_task()
        requests = waiter.cancelling() if waiter is not None else 0
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    current = waiter.cancelling() if waiter is not None else 0
                    if current > requests:
                        cancellation = cancellation or error
                        requests = current
                        task.cancel()
                except BaseException:
                    if not task.done():
                        raise
            result = task.result()
        finally:
            self._removal_tasks.discard(task)
        if cancellation is not None:
            raise cancellation
        return result

    async def _shutdown_owned(self) -> None:
        """Seal admission and definitively settle every admitted operation."""

        async with self._admission_lock:
            self._closed = True
        while self._operations or self._removal_tasks:
            operations = tuple(self._operations.items())
            removals = tuple(self._removal_tasks)
            for _completion, owner in operations:
                owner.cancel()
            for task in removals:
                task.cancel()
            for completion, _owner in operations:
                await self._settle_task(completion)
            for task in removals:
                await self._settle_task(task)
                self._removal_tasks.discard(task)
        for blocking_task in tuple(self._blocking_tasks):
            await self._settle_task(blocking_task)
        await self.drain_cleanup()

    async def shutdown(self) -> None:
        """Stop admission and retain shutdown through definitive cleanup."""

        current = asyncio.current_task()
        if any(owner is current for owner in self._operations.values()):
            raise AudioCppArtifactDependencyError(
                "cannot shut down from an active artifact operation"
            )
        task = self._shutdown_task
        if task is None or (
            task.done() and (task.cancelled() or task.exception() is not None)
        ):
            task = asyncio.create_task(self._shutdown_owned())
            self._shutdown_task = task
        cancellation, _result, error = await self._settle_task(task)
        if error is not None:
            if self._shutdown_task is task:
                self._shutdown_task = None
            if cancellation is not None:
                error.add_note("shutdown caller was cancelled")
            raise error
        if cancellation is not None:
            raise cancellation


__all__ = [
    "AudioCppArtifactConsumerRequirement",
    "AudioCppArtifactDependencyError",
    "AudioCppArtifactLeaseCoordinator",
    "AudioCppArtifactRemovalEvidence",
    "AudioCppArtifactRemovalPreview",
    "AudioCppModelLibraryObservationSnapshot",
    "AudioCppManagedConsumerIdentity",
    "build_audio_cpp_artifact_removal_preview",
    "project_audio_cpp_artifact_removal_evidence",
    "is_curated_audio_cpp_artifact_reference",
]
