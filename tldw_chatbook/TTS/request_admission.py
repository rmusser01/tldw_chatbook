"""Coherent default-request resolution and TTS resource admission."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast
from uuid import UUID

from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSCloneGenerationEvidence,
    TTSAudioResponse,
    TTSConfigurationRevisionError,
    TTSNativeCapabilitySnapshot,
    TTSProviderUnavailableError,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.audio_cpp_supervisor import _AudioCppGenerationChanged
from tldw_chatbook.TTS.effective_settings import (
    NativeCapabilityReader,
    TTSCharacterProfileSelection,
    TTSDefaultProfileSelection,
    TTSEffectiveResolutionError,
    TTSEffectiveSelectionSnapshot,
    TTSEffectiveSettingsResolver,
    TTSSelectionOverrides,
    TTSSelectionSource,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.legacy_bridge import (
    openai_internal_model_id,
    resolve_legacy_route,
)
from tldw_chatbook.TTS.playground_types import (
    STTSPlaygroundCloneSnapshot,
    STTSPlaygroundProfilePreview,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot

if TYPE_CHECKING:
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterLease
    from tldw_chatbook.TTS.TTS_Generation import (
        TTSService,
        _AdmittedTTSOperation,
        _AudioCppPreparation,
        _OperationCapacityReservation,
    )

_AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
_VALID_AUDIO_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
TTSAdmissionAuthorizer = Callable[[str, str], bool]


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedTTSCloneExecution:
    """Exact profile/reference authority frozen with effective selection."""

    profile_id: UUID
    repository_generation: int
    profile_revision: int
    reference: TTSCloneReference

    def __repr__(self) -> str:
        return "_ResolvedTTSCloneExecution(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedTransientTTSCloneExecution:
    """Exact transient canonical authority without fabricated profile identity."""

    reference: CanonicalTTSCloneReference

    def __post_init__(self) -> None:
        if type(self.reference) is not CanonicalTTSCloneReference:
            raise TypeError("Transient clone reference is invalid")

    def __repr__(self) -> str:
        return "_ResolvedTransientTTSCloneExecution(<private>)"


_ResolvedTTSCloneExecutionAuthority: TypeAlias = (
    _ResolvedTTSCloneExecution | _ResolvedTransientTTSCloneExecution
)


class TTSProfileReferenceResolver(Protocol):
    """Resolve one exact profile identity without exposing it to the UI."""

    def __call__(
        self,
        profile_id: UUID,
        repository_generation: int,
        profile_revision: int,
    ) -> Awaitable[TTSCloneReference]: ...


class _WriterPreferredGate:
    """A cancellation-safe shared/exclusive gate with writer preference."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._reader_count = 0
        self._active_writer = False
        self._waiting_writer_count = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        """Enter the shared side without holding the condition during work."""
        admitted = False
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._active_writer and self._waiting_writer_count == 0
            )
            self._reader_count += 1
            admitted = True

        try:
            yield
        finally:
            if admitted:

                async def release_reader() -> None:
                    async with self._condition:
                        self._reader_count -= 1
                        if self._reader_count == 0:
                            self._condition.notify_all()

                release_task = asyncio.create_task(release_reader())
                await join_retained_task(release_task)

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        """Enter the exclusive side and prevent later readers from bypassing."""
        admitted = False
        async with self._condition:
            self._waiting_writer_count += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._active_writer and self._reader_count == 0
                )
            finally:
                self._waiting_writer_count -= 1
                self._condition.notify_all()
            self._active_writer = True
            admitted = True

        try:
            yield
        finally:
            if admitted:

                async def release_writer() -> None:
                    async with self._condition:
                        self._active_writer = False
                        self._condition.notify_all()

                release_task = asyncio.create_task(release_writer())
                await join_retained_task(release_task)


class TTSRequestAdmissionCoordinator:
    """Freeze preferences and acquire matching service resources atomically."""

    def __init__(
        self,
        service: TTSService,
        preferences: TTSPreferencesSnapshot | None,
        studio_preferences_loader: Callable[[], StudioTTSPreferencesSnapshot]
        | None = None,
        native_capability_reader: NativeCapabilityReader | None = None,
    ) -> None:
        self._service = service
        self._preferences = preferences
        self._preferences_generation = 0
        self._gate = _WriterPreferredGate()
        self._publication_lock = asyncio.Lock()
        self._effective_settings = TTSEffectiveSettingsResolver()
        self._studio_preferences_loader = studio_preferences_loader
        self._native_capability_reader = native_capability_reader

    def preferences_snapshot(self) -> TTSPreferencesSnapshot | None:
        """Return canonical preferences, or None while unconfigured."""
        return self._preferences

    def preferences_generation(self) -> int:
        """Return the latest saved settings generation published in memory."""
        return self._preferences_generation

    async def _read_native_capability(
        self,
        provider_id: str,
        model_id: str,
        voice_id: str | None,
    ) -> TTSNativeCapabilitySnapshot:
        if self._native_capability_reader is not None:
            return await self._native_capability_reader(
                provider_id,
                model_id,
                voice_id,
            )
        model_ids = (model_id,) if voice_id is not None else ()
        return await self._service._get_native_capability_snapshot_already_prepared(
            provider_id,
            model_ids,
        )

    def _publish_preferences(
        self,
        preferences: TTSPreferencesSnapshot,
        generation: int,
    ) -> None:
        """Publish one already-persisted immutable snapshot exactly once."""
        if generation <= self._preferences_generation:
            return
        self._preferences = preferences
        self._preferences_generation = generation

    async def synthesize_effective(
        self,
        *,
        text: str,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        clone_audition: STTSPlaygroundCloneSnapshot | None = None,
        profile_preview: STTSPlaygroundProfilePreview | None = None,
        profile_reference_resolver: TTSProfileReferenceResolver | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> tuple[TTSAudioResponse, TTSEffectiveSelectionSnapshot]:
        """Resolve and synthesize while preserving the established public API."""

        response, selection, _evidence = await self.synthesize_effective_with_evidence(
            text=text,
            explicit=explicit,
            character_profile=character_profile,
            default_profile=default_profile,
            studio_draft=studio_draft,
            studio_preferences=studio_preferences,
            clone_audition=clone_audition,
            profile_preview=profile_preview,
            profile_reference_resolver=profile_reference_resolver,
            progress_sink=progress_sink,
            admission_authorizer=admission_authorizer,
        )
        return response, selection

    async def synthesize_effective_with_evidence(
        self,
        *,
        text: str,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        clone_audition: STTSPlaygroundCloneSnapshot | None = None,
        profile_preview: STTSPlaygroundProfilePreview | None = None,
        profile_reference_resolver: TTSProfileReferenceResolver | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> tuple[
        TTSAudioResponse,
        TTSEffectiveSelectionSnapshot,
        TTSCloneGenerationEvidence | None,
    ]:
        """Resolve, admit, and retain exact clone-success evidence internally."""

        if explicit is not None and type(explicit) is not TTSSelectionOverrides:
            raise TypeError("Explicit TTS selection is invalid")
        if character_profile is not None and (
            type(character_profile) is not TTSCharacterProfileSelection
        ):
            raise TypeError("Character TTS profile selection is invalid")
        if default_profile is not None and (
            type(default_profile) is not TTSDefaultProfileSelection
        ):
            raise TypeError("Default-profile TTS selection is invalid")
        if (
            studio_draft is not None
            and type(studio_draft) is not TTSStudioDraftSelection
        ):
            raise TypeError("Studio TTS draft is invalid")
        if studio_preferences is not None and (
            type(studio_preferences) is not StudioTTSPreferencesSnapshot
        ):
            raise TypeError("Saved Studio TTS preferences are invalid")
        if clone_audition is not None and (
            type(clone_audition) is not STTSPlaygroundCloneSnapshot
        ):
            raise TypeError("Clone audition is invalid")
        if profile_preview is not None and (
            type(profile_preview) is not STTSPlaygroundProfilePreview
        ):
            raise TypeError("Profile preview is invalid")
        if (profile_preview is None) is not (profile_reference_resolver is None):
            raise TypeError("Profile preview requires one private reference resolver")
        if profile_reference_resolver is not None and not callable(
            profile_reference_resolver
        ):
            raise TypeError("Profile reference resolver is invalid")
        if admission_authorizer is not None and not callable(admission_authorizer):
            raise TypeError("TTS admission authorizer is invalid")
        if clone_audition is not None and profile_preview is not None:
            raise TypeError("Clone audition and profile preview are mutually exclusive")

        studio_request = studio_draft is not None or studio_preferences is not None
        if studio_request and (
            explicit is not None
            or character_profile is not None
            or default_profile is not None
        ):
            raise TypeError("Studio TTS resolution cannot use non-Studio layers")
        if studio_request and studio_preferences is None:
            raise TypeError("Studio TTS resolution requires saved preferences")
        if clone_audition is not None and not studio_request:
            raise TypeError("Clone audition requires Studio TTS resolution")
        if profile_preview is not None and not studio_request:
            raise TypeError("Profile preview requires Studio TTS resolution")
        if clone_audition is not None or profile_preview is not None:
            if (
                studio_draft is None
                or studio_draft.selection.provider_id != "audio_cpp"
                or studio_draft.selection.model_mode != "exact"
                or studio_draft.selection.model_id is None
            ):
                raise TypeError("Clone synthesis requires exact audio.cpp Studio state")
        if profile_preview is not None:
            assert studio_draft is not None
            if not studio_draft.preview:
                raise TypeError("Profile preview requires a Studio preview draft")

        preview_execution: _ResolvedTTSCloneExecution | None = None
        higher_scope_provider = next(
            (
                provider_id
                for provider_id in (
                    explicit.provider_id if explicit is not None else None,
                    (
                        character_profile.selection.provider_id
                        if character_profile is not None
                        else None
                    ),
                    (
                        default_profile.selection.provider_id
                        if default_profile is not None
                        else None
                    ),
                    (
                        studio_draft.selection.provider_id
                        if studio_draft is not None
                        else None
                    ),
                    (
                        studio_preferences.selection.provider_id
                        if studio_preferences is not None
                        else None
                    ),
                )
                if provider_id is not None
            ),
            None,
        )

        reservation: _OperationCapacityReservation | None = None
        operation: _AdmittedTTSOperation | None = None
        try:
            self._service._require_operation_admission_open()
            if profile_preview is not None:
                assert profile_reference_resolver is not None
                preview_execution = await self._resolve_profile_preview_reference(
                    profile_preview,
                    profile_reference_resolver,
                )
            requirement = self._candidate_clone_requirement(
                explicit=explicit,
                character_profile=character_profile,
                default_profile=default_profile,
                profile_preview_execution=preview_execution,
            )
            if requirement is not None:
                await self._service._require_audio_cpp_clone_dependency(requirement)
                await self._service._preflight_audio_cpp_clone_dependency(requirement)
            reservation = await self._service._reserve_operation_capacity()
            while True:
                projected_provider = self._effective_settings.project_provider(
                    global_preferences=self._preferences,
                    explicit=explicit,
                    character_profile=character_profile,
                    default_profile=default_profile,
                    studio_preferences=studio_preferences,
                    studio_draft=studio_draft,
                )
                try:
                    async with self._service._prepared_provider_read(
                        projected_provider,
                        deliberate=True,
                    ):
                        clone_candidate = (
                            clone_audition is not None
                            or preview_execution is not None
                            or self._profile_clone_candidate(
                                explicit=explicit,
                                character_profile=character_profile,
                                default_profile=default_profile,
                            )
                        )
                        if (
                            projected_provider == "audio_cpp"
                            and clone_candidate
                            and requirement is None
                        ):
                            await self._service._preflight_audio_cpp_clone_source()
                        preferences = self._preferences
                        if preferences is None and higher_scope_provider is None:
                            raise TTSProviderUnavailableError(
                                "TTS default provider is not configured"
                            )
                        if (
                            self._effective_settings.project_provider(
                                global_preferences=preferences,
                                explicit=explicit,
                                character_profile=character_profile,
                                default_profile=default_profile,
                                studio_preferences=studio_preferences,
                                studio_draft=studio_draft,
                            )
                            != projected_provider
                        ):
                            continue

                        if studio_request:
                            assert studio_preferences is not None
                            loader = self._studio_preferences_loader
                            if loader is None:
                                raise TTSEffectiveResolutionError(
                                    code="revision_incoherent",
                                    axis="studio_preferences",
                                    source=(
                                        TTSSelectionSource.STUDIO_DRAFT
                                        if studio_draft is not None
                                        else TTSSelectionSource.STUDIO_SAVED
                                    ),
                                )
                            try:
                                current_studio_preferences = loader()
                            except Exception:
                                raise TTSEffectiveResolutionError(
                                    code="revision_incoherent",
                                    axis="studio_preferences",
                                    source=(
                                        TTSSelectionSource.STUDIO_DRAFT
                                        if studio_draft is not None
                                        else TTSSelectionSource.STUDIO_SAVED
                                    ),
                                ) from None
                            if (
                                type(current_studio_preferences)
                                is not StudioTTSPreferencesSnapshot
                                or current_studio_preferences != studio_preferences
                            ):
                                raise TTSEffectiveResolutionError(
                                    code="revision_incoherent",
                                    axis="studio_preferences",
                                    source=(
                                        TTSSelectionSource.STUDIO_DRAFT
                                        if studio_draft is not None
                                        else TTSSelectionSource.STUDIO_SAVED
                                    ),
                                )
                            selection = await self._effective_settings.resolve_studio(
                                studio_draft=studio_draft,
                                studio_preferences=current_studio_preferences,
                                global_preferences=preferences,
                                global_preferences_revision=(
                                    self._preferences_generation
                                ),
                                provider_revision_reader=(
                                    self._service.configuration_revision
                                ),
                                catalog_reader=(
                                    self._service._get_catalog_already_prepared
                                ),
                                native_capability_reader=(self._read_native_capability),
                            )
                        else:
                            selection = (
                                await self._effective_settings.resolve_non_studio(
                                    explicit=explicit,
                                    character_profile=character_profile,
                                    default_profile=default_profile,
                                    global_preferences=preferences,
                                    global_preferences_revision=(
                                        self._preferences_generation
                                    ),
                                    provider_revision_reader=(
                                        self._service.configuration_revision
                                    ),
                                    catalog_reader=(
                                        self._service._get_catalog_already_prepared
                                    ),
                                    native_capability_reader=(
                                        self._read_native_capability
                                    ),
                                )
                            )
                        request = self._build_request(selection, text=text)
                        clone_execution = self._resolve_clone_execution(
                            selection,
                            character_profile=character_profile,
                            default_profile=default_profile,
                            clone_audition=clone_audition,
                            profile_preview_execution=preview_execution,
                        )
                        operation = await self._service._admit_reserved(
                            request,
                            reservation,
                            expected_configuration_revision=(
                                selection.revisions.provider_configuration
                            ),
                            clone_execution=clone_execution,
                            admission_authorizer=admission_authorizer,
                        )
                        operation.claim()
                    break
                except _AudioCppGenerationChanged:
                    continue
        except BaseException as error:
            if operation is None:
                if reservation is not None:
                    reservation.release_if_untransferred()
            else:
                await self._service._close_admitted_operation_preserving_primary(
                    operation,
                    error,
                )
            raise

        assert operation is not None
        response, evidence = await operation.synthesize_with_evidence(progress_sink)
        return response, selection, evidence

    @staticmethod
    async def _resolve_profile_preview_reference(
        preview: STTSPlaygroundProfilePreview,
        resolver: TTSProfileReferenceResolver,
    ) -> _ResolvedTTSCloneExecution:
        """Resolve private profile authority before any provider admission."""

        failed = False
        reference: object = None
        try:
            reference = await resolver(
                preview.profile_id,
                preview.repository_generation,
                preview.profile_revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        if failed or type(reference) is not TTSCloneReference:
            raise TTSEffectiveResolutionError(
                code="revision_incoherent",
                axis="profile_reference",
                source=TTSSelectionSource.STUDIO_DRAFT,
            ) from None
        return _ResolvedTTSCloneExecution(
            profile_id=preview.profile_id,
            repository_generation=preview.repository_generation,
            profile_revision=preview.profile_revision,
            reference=cast(TTSCloneReference, reference),
        )

    @staticmethod
    def _profile_clone_candidate(
        *,
        explicit: TTSSelectionOverrides | None,
        character_profile: TTSCharacterProfileSelection | None,
        default_profile: TTSDefaultProfileSelection | None,
    ) -> bool:
        """Return whether profile precedence can select a clone reference."""
        if explicit is not None and (
            explicit.provider_id is not None or explicit.model_id is not None
        ):
            return False
        if character_profile is not None:
            return character_profile.reference is not None
        return default_profile is not None and default_profile.reference is not None

    @staticmethod
    def _candidate_clone_requirement(
        *,
        explicit: TTSSelectionOverrides | None,
        character_profile: TTSCharacterProfileSelection | None,
        default_profile: TTSDefaultProfileSelection | None,
        profile_preview_execution: _ResolvedTTSCloneExecution | None,
    ) -> TTSCloneRecipeRequirement | None:
        """Return persisted provenance before any provider authority is acquired."""

        if profile_preview_execution is not None:
            return profile_preview_execution.reference.recipe_requirement
        if explicit is not None and (
            explicit.provider_id is not None or explicit.model_id is not None
        ):
            return None
        profile = character_profile or default_profile
        if profile is None or profile.reference is None:
            return None
        return profile.reference.recipe_requirement

    @staticmethod
    def _resolve_clone_execution(
        selection: TTSEffectiveSelectionSnapshot,
        *,
        character_profile: TTSCharacterProfileSelection | None,
        default_profile: TTSDefaultProfileSelection | None,
        clone_audition: STTSPlaygroundCloneSnapshot | None,
        profile_preview_execution: _ResolvedTTSCloneExecution | None,
    ) -> _ResolvedTTSCloneExecutionAuthority | None:
        """Select a reference only when one profile owns provider and model."""
        if selection.provider_id != "audio_cpp":
            return None
        if clone_audition is not None:
            if (
                selection.sources["provider_id"] is not TTSSelectionSource.STUDIO_DRAFT
                or selection.sources["model_id"] is not TTSSelectionSource.STUDIO_DRAFT
            ):
                raise TTSEffectiveResolutionError(
                    code="revision_incoherent",
                    axis="clone_audition",
                    source=TTSSelectionSource.STUDIO_DRAFT,
                )
            return _ResolvedTransientTTSCloneExecution(
                reference=clone_audition.canonical_reference,
            )
        if profile_preview_execution is not None:
            if (
                not selection.studio_preview
                or selection.sources["provider_id"]
                is not TTSSelectionSource.STUDIO_DRAFT
                or selection.sources["model_id"] is not TTSSelectionSource.STUDIO_DRAFT
            ):
                raise TTSEffectiveResolutionError(
                    code="revision_incoherent",
                    axis="profile_reference",
                    source=TTSSelectionSource.STUDIO_DRAFT,
                )
            return profile_preview_execution
        source_pair = (
            selection.sources["provider_id"],
            selection.sources["model_id"],
        )
        profile: TTSCharacterProfileSelection | TTSDefaultProfileSelection | None
        if source_pair == (
            TTSSelectionSource.CHARACTER_PROFILE,
            TTSSelectionSource.CHARACTER_PROFILE,
        ):
            profile = character_profile
        elif source_pair == (
            TTSSelectionSource.DEFAULT_PROFILE,
            TTSSelectionSource.DEFAULT_PROFILE,
        ):
            profile = default_profile
        else:
            profile = None
        if profile is None or profile.reference is None:
            return None
        return _ResolvedTTSCloneExecution(
            profile_id=profile.profile_id,
            repository_generation=profile.repository_generation,
            profile_revision=profile.profile_revision,
            reference=profile.reference,
        )

    async def synthesize_default(
        self,
        *,
        text: str,
        voice_override: str | None = None,
        progress_sink: ProgressSink | None = None,
        admission_authorizer: TTSAdmissionAuthorizer | None = None,
    ) -> TTSAudioResponse:
        """Resolve and admit one coherent default request, then execute it.

        Args:
            text: Text to synthesize.
            voice_override: Optional request-scoped voice identifier.
            progress_sink: Optional callback for bounded synthesis progress.

        Returns:
            The admitted provider response. The caller owns and must close it.

        Raises:
            TTSProviderUnavailableError: If no default provider is configured
                or its dynamic model catalog is empty.
            TTSRegistryClosedError: If service shutdown has begun.
            TTSProviderReconfiguringError: If the selected provider is in an
                exclusive settings handoff.
            TTSConfigurationRevisionError: If the selected provider revision
                changes before admission completes.
            TTSOperationError: If admission or synthesis fails.
            ValueError: If the published default preferences are invalid.
        """
        explicit = (
            None
            if voice_override is None
            else TTSSelectionOverrides(
                voice_mode="exact",
                voice_id=voice_override,
            )
        )
        response, _selection = await self.synthesize_effective(
            text=text,
            explicit=explicit,
            progress_sink=progress_sink,
            admission_authorizer=admission_authorizer,
        )
        return response

    async def acquire_native_capability_lease(
        self,
        provider_id: str,
    ) -> tuple[int, TTSAdapterLease, _AudioCppPreparation | None]:
        """Acquire one revision-matched lease and passive process fence."""
        lease: TTSAdapterLease | None = None
        preparation: _AudioCppPreparation | None = None
        try:
            async with self._service._prepared_provider_read(
                provider_id,
                deliberate=False,
            ) as preparation:
                revision = self._service.configuration_revision(provider_id)
                lease = await self._service.registry.acquire(
                    provider_id,
                    expected_revision=revision,
                )
        except BaseException as error:
            if lease is not None:
                await self._service._release_lease_preserving_primary(
                    lease,
                    error,
                )
            raise

        assert lease is not None
        return revision, lease, preparation

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]:
        """Freeze and admit one exact request under the shared read side."""
        response, effective = await self.synthesize_effective(
            text=request.text,
            explicit=TTSSelectionOverrides(
                provider_id=request.provider_id,
                model_mode="exact",
                model_id=request.model_id,
                voice_mode="server_default" if request.voice is None else "exact",
                voice_id=request.voice,
                response_format=request.response_format,
                speed=request.speed,
                provider_options=request.options,
            ),
            progress_sink=progress_sink,
        )
        selection = TTSRequestedSelectionSnapshot(
            provider_id=effective.provider_id,
            model_id=effective.model_id,
            voice_id=effective.voice_id,
            response_format=effective.response_format,
            speed=effective.speed,
            options=effective.provider_options,
            configuration_revision=effective.revisions.provider_configuration,
        )
        return response, selection

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        """Make one writer-ordered provider revision decision."""
        async with self._gate.read():
            current_revision = self._service.configuration_revision(provider_id)
            if current_revision != expected_revision:
                raise TTSConfigurationRevisionError(
                    f"TTS provider configuration changed: {provider_id}"
                )

    @staticmethod
    def _build_request(
        selection: TTSEffectiveSelectionSnapshot,
        *,
        text: str,
    ) -> TTSRequest:
        if selection.provider_id == "audio_cpp":
            return TTSRequest(
                provider_id="audio_cpp",
                model_id=selection.model_id,
                text=text,
                voice=selection.voice_id,
                response_format="wav",
                speed=1.0,
                options={},
            )

        legacy_request, internal_model_id = _legacy_request(
            selection,
            text=text,
        )
        route = resolve_legacy_route(internal_model_id)
        if route.provider_id != selection.provider_id:
            raise ValueError("Legacy TTS route does not match provider")
        return TTSRequest(
            provider_id=selection.provider_id,
            model_id=legacy_request.model,
            text=text,
            voice=legacy_request.voice,
            response_format=legacy_request.response_format,
            speed=legacy_request.speed,
            options={
                "_legacy_openai_request": legacy_request,
                "_legacy_internal_model_id": route.internal_model_id,
            },
        )


def _legacy_request(
    selection: TTSEffectiveSelectionSnapshot,
    *,
    text: str,
) -> tuple[OpenAISpeechRequest, str]:
    if selection.voice_id is None:
        raise ValueError("Legacy TTS providers require an exact voice")

    provider_id = selection.provider_id
    request_model = selection.model_id
    response_format = selection.response_format
    if response_format not in _VALID_AUDIO_FORMATS:
        raise ValueError("Legacy TTS response format is invalid")

    request = OpenAISpeechRequest(
        model=request_model,
        input=text,
        voice=selection.voice_id,
        response_format=cast(_AudioFormat, response_format),
        speed=selection.speed,
        extra_params=(
            dict(selection.provider_options) if selection.provider_options else None
        ),
    )
    if provider_id == "openai":
        internal_model_id = openai_internal_model_id(request.model)
    elif provider_id == "elevenlabs":
        internal_model_id = f"elevenlabs_{request.model}"
    elif provider_id == "kokoro":
        engine = (
            "onnx" if selection.provider_options.get("use_onnx", True) else "pytorch"
        )
        internal_model_id = f"local_kokoro_default_{engine}"
    elif provider_id == "chatterbox":
        internal_model_id = "local_chatterbox_default"
    elif provider_id == "higgs":
        internal_model_id = "local_higgs_v2"
    elif provider_id == "alltalk":
        internal_model_id = "alltalk_default"
    else:
        internal_model_id = request.model
    return request, internal_model_id
