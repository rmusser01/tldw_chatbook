from __future__ import annotations

import asyncio
import threading
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

import tldw_chatbook.TTS.audio_cpp_artifact_dependencies as dependency_module
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactNotInstalledError,
    ArtifactRef,
)
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactConsumerRequirement,
    AudioCppArtifactDependencyError,
    AudioCppArtifactLeaseCoordinator,
    AudioCppArtifactRemovalEvidence,
    AudioCppManagedConsumerIdentity,
    build_audio_cpp_artifact_removal_preview,
    project_audio_cpp_artifact_removal_evidence,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedArtifactIdentity,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile


REFERENCE = ArtifactRef("audio-cpp-model-a", "a" * 40, "q8_0")
OTHER = ArtifactRef("audio-cpp-model-a", "a" * 40, "f16")


def _catalog(reference: ArtifactRef = REFERENCE):
    recipe = SimpleNamespace(
        recipe_id="audio-cpp-0.5.1.family.variant",
        recipe_revision=1,
        default_public_model_id="model-a",
    )
    descriptor = SimpleNamespace(reference=reference, model_id="model-a")
    return ((descriptor, {}, recipe),)


def _evidence(**changes: object) -> AudioCppArtifactRemovalEvidence:
    values: dict[str, object] = {
        "reference": REFERENCE,
        "settings_consumers": (("saved", "Guided Settings", "package-1"),),
        "profile_consumers": (("profile-1", "Narrator", 2, True),),
        "staged_runtime_ids": ("stage-1",),
        "live_runtime_ids": (),
    }
    values.update(changes)
    return AudioCppArtifactRemovalEvidence(**values)  # type: ignore[arg-type]


def test_removal_preview_is_frozen_bounded_and_fingerprint_excludes_probe_bit() -> None:
    available = build_audio_cpp_artifact_removal_preview(
        _evidence(), generic_lease_blocked=False
    )
    busy = build_audio_cpp_artifact_removal_preview(
        _evidence(), generic_lease_blocked=True
    )

    assert available.fingerprint == busy.fingerprint
    assert available.settings_labels == ("Guided Settings",)
    assert available.profile_labels == ("Narrator",)
    assert available.assignment_count == 2
    assert available.clone_reference_count == 1
    assert available.staged_or_live is True
    assert busy.generic_lease_blocked is True
    with pytest.raises(FrozenInstanceError):
        busy.assignment_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"settings_consumers": (("saved", "Guided Settings", "package-2"),)},
        {"profile_consumers": (("profile-1", "Narrator", 3, True),)},
        {"profile_provenance": (("profile-1", "other-recipe@2:model-a"),)},
        {"staged_runtime_ids": (), "live_runtime_ids": ("live-1",)},
    ),
)
def test_removal_fingerprint_detects_every_stable_consumer_change(
    changes: dict[str, object],
) -> None:
    assert (
        build_audio_cpp_artifact_removal_preview(_evidence()).fingerprint
        != build_audio_cpp_artifact_removal_preview(_evidence(**changes)).fingerprint
    )


def _catalog_profile(*, with_requirement: bool) -> tuple[object, TTSGenerationProfile]:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    descriptor, _sources = audio_cpp_curated_entries()[0]
    now = datetime(2026, 1, 1, tzinfo=UTC)
    profile = TTSGenerationProfile(
        profile_id=UUID("00000000-0000-0000-0000-000000000001"),
        display_name="Narrator",
        normalized_name="narrator",
        provider_id="audio_cpp",
        model_id=descriptor.model_id,
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        revision=1,
        created_at=now,
        updated_at=now,
    )
    if with_requirement:
        recipe = next(
            item
            for item in AUDIO_CPP_RECIPE_REGISTRY.recipes
            if descriptor.reference.artifact_id in item.model_library_artifact_ids
        )
        profile = replace(
            profile,
            reference=TTSCloneReferenceSummary(
                reference_id=UUID("00000000-0000-0000-0000-000000000002"),
                byte_length=2,
                duration_ms=1,
                sample_rate_hz=24_000,
                channels=1,
                sample_encoding="pcm_s16le",
                created_at=now,
                updated_at=now,
                recipe_requirement=TTSCloneRecipeRequirement(
                    recipe.recipe_id,
                    recipe.recipe_revision,
                    descriptor.model_id,
                ),
            ),
        )
    return descriptor, profile


def test_reference_free_profile_requires_exact_managed_settings_identity() -> None:
    descriptor, profile = _catalog_profile(with_requirement=False)

    evidence = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
        profiles=((profile, 0),),
    )

    assert evidence.profile_consumers == ()


def test_clone_profile_joins_exact_recipe_and_future_recipe_is_not_a_consumer() -> None:
    descriptor, exact = _catalog_profile(with_requirement=True)
    assert exact.reference is not None
    future = replace(
        exact,
        reference=replace(
            exact.reference,
            recipe_requirement=TTSCloneRecipeRequirement(
                "audio-cpp-99.future.recipe",
                99,
                exact.model_id,
            ),
        ),
    )

    exact_evidence = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
        profiles=((exact, 1),),
    )
    future_evidence = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
        profiles=((future, 1),),
    )

    assert len(exact_evidence.profile_consumers) == 1
    assert future_evidence.profile_consumers == ()
    assert (
        build_audio_cpp_artifact_removal_preview(exact_evidence).fingerprint
        != build_audio_cpp_artifact_removal_preview(future_evidence).fingerprint
    )


def test_global_saved_and_draft_defaults_are_fingerprinted_exact_consumers() -> None:
    descriptor, _profile = _catalog_profile(with_requirement=False)
    selected = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id=descriptor.model_id,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    without = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
    )
    with_defaults = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
        saved_preferences=selected,
        draft_preferences=selected,
    )

    preview = build_audio_cpp_artifact_removal_preview(with_defaults)
    assert preview.settings_labels == (
        "Unsaved global TTS default",
        "Saved global TTS default",
    )
    assert (
        preview.fingerprint
        != build_audio_cpp_artifact_removal_preview(without).fingerprint
    )


@pytest.mark.parametrize(
    "private_value",
    (
        "/Users/private/models/model.gguf",
        "private transcript\nsecond line",
        "x" * 257,
    ),
)
def test_removal_evidence_rejects_unbounded_or_path_shaped_public_labels(
    private_value: str,
) -> None:
    with pytest.raises(AudioCppArtifactDependencyError) as caught:
        _evidence(settings_consumers=(("saved", private_value, "package-1"),))

    assert private_value not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_removal_evidence_bounds_recipe_provenance_with_all_consumers() -> None:
    with pytest.raises(AudioCppArtifactDependencyError):
        _evidence(
            settings_consumers=(),
            profile_consumers=(),
            staged_runtime_ids=(),
            profile_provenance=tuple(
                (f"profile-{index}", "recipe@1:model") for index in range(201)
            ),
        )


class _Handle:
    def __init__(self, reference: ArtifactRef, closed: list[ArtifactRef]) -> None:
        self.reference = reference
        self.closed = closed

    def close(self) -> None:
        self.closed.append(self.reference)


class _Service:
    def __init__(self, *, missing: set[ArtifactRef] | None = None) -> None:
        self.missing = missing or set()
        self.acquired: list[ArtifactRef] = []
        self.closed: list[ArtifactRef] = []

    def acquire_installed_root(self, reference: ArtifactRef) -> _Handle:
        self.acquired.append(reference)
        if reference in self.missing:
            raise ArtifactNotInstalledError("PRIVATE missing path")
        return _Handle(reference, self.closed)


@pytest.mark.asyncio
async def test_async_coordinator_does_not_block_event_loop_while_acquiring() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingService(_Service):
        def acquire_installed_root(self, reference: ArtifactRef) -> _Handle:
            entered.set()
            assert release.wait(timeout=3.0)
            return super().acquire_installed_root(reference)

    service = BlockingService()
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )
    heartbeat = asyncio.Event()

    async def mutate() -> None:
        async with coordinator.lease_consumers((_requirement(),)):
            pass

    task = asyncio.create_task(mutate())
    assert await asyncio.to_thread(entered.wait, 1.0)
    await asyncio.sleep(0)
    heartbeat.set()
    assert heartbeat.is_set()
    release.set()
    await task


@pytest.mark.asyncio
async def test_failed_handle_close_is_retained_and_blocks_until_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class RetryHandle:
        attempts = 0

        def close(self) -> None:
            self.attempts += 1
            events.append(f"close-{self.attempts}")
            if self.attempts == 1:
                raise RuntimeError("PRIVATE /managed/root")

    handle = RetryHandle()
    monkeypatch.setattr(
        dependency_module,
        "take_artifact_removal_cleanup_owner",
        lambda error: handle if isinstance(error, RuntimeError) else None,
    )

    class RetryService:
        def acquire_installed_root(self, _reference: ArtifactRef) -> RetryHandle:
            events.append("acquire")
            return handle

    coordinator = AudioCppArtifactLeaseCoordinator(
        RetryService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )

    with pytest.raises(AudioCppArtifactDependencyError) as caught:
        async with coordinator.lease_consumers((_requirement(),)):
            events.append("commit-1")
    assert "PRIVATE" not in str(caught.value)
    assert events == ["acquire", "commit-1", "close-1"]

    async with coordinator.lease_consumers((_requirement(),)):
        events.append("commit-2")
    assert events == [
        "acquire",
        "commit-1",
        "close-1",
        "close-2",
        "acquire",
        "commit-2",
        "close-3",
    ]


@pytest.mark.asyncio
async def test_identical_saved_package_instances_are_deduped_not_disagreement() -> None:
    service = _Service()
    saved = AudioCppManagedConsumerIdentity(
        recipe_id="audio-cpp-0.5.1.family.variant",
        recipe_revision=1,
        model_id="model-a",
        managed_artifact=AudioCppManagedArtifactIdentity(
            artifact_id=REFERENCE.artifact_id,
            revision=REFERENCE.revision,
            variant=REFERENCE.variant,
        ),
    )
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (saved, saved),
        catalog_entries=_catalog,
    )

    async with coordinator.lease_consumers((_requirement(), _requirement())):
        assert service.acquired == [REFERENCE]

    assert service.closed == [REFERENCE]


@pytest.mark.asyncio
async def test_cancel_during_removal_acquire_joins_and_closes_returned_authority() -> (
    None
):
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class Authority:
        def commit(self) -> None:
            events.append("commit")

        def close(self) -> None:
            events.append("close")

    class BlockingRemovalService(_Service):
        def acquire_removal_authority(self, reference: ArtifactRef) -> Authority:
            assert reference == REFERENCE
            events.append("acquire-start")
            entered.set()
            assert release.wait(timeout=3.0)
            events.append("acquire-end")
            return Authority()

    coordinator = AudioCppArtifactLeaseCoordinator(
        BlockingRemovalService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )

    async def collect() -> str:
        events.append("evidence")
        return "fingerprint"

    task = asyncio.create_task(
        coordinator.remove_if_unchanged(REFERENCE, "fingerprint", collect)
    )
    assert await asyncio.to_thread(entered.wait, 1.0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["acquire-start", "acquire-end", "close"]


@pytest.mark.asyncio
async def test_shutdown_cancels_blocked_removal_then_joins_authority_cleanup() -> None:
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class Authority:
        def commit(self) -> None:
            events.append("commit")

        def close(self) -> None:
            events.append("close")

    class BlockingRemovalService(_Service):
        def acquire_removal_authority(self, _reference: ArtifactRef) -> Authority:
            events.append("acquire-start")
            entered.set()
            assert release.wait(timeout=3.0)
            events.append("acquire-end")
            return Authority()

    coordinator = AudioCppArtifactLeaseCoordinator(
        BlockingRemovalService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )
    removal = asyncio.create_task(
        coordinator.remove_if_unchanged(
            REFERENCE,
            "fingerprint",
            lambda: asyncio.sleep(0, result="fingerprint"),
        )
    )
    assert await asyncio.to_thread(entered.wait, 1.0)

    shutdown = asyncio.create_task(coordinator.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    release.set()
    await shutdown
    with pytest.raises(asyncio.CancelledError):
        await removal
    assert events == ["acquire-start", "acquire-end", "close"]


@pytest.mark.asyncio
async def test_removal_success_is_not_published_until_authority_close_settles() -> None:
    close_entered = threading.Event()
    release_close = threading.Event()
    events: list[str] = []

    class Authority:
        def commit(self) -> None:
            events.append("commit")

        def close(self) -> None:
            events.append("close-start")
            close_entered.set()
            assert release_close.wait(timeout=3.0)
            events.append("close-end")

    class RemovalService(_Service):
        def acquire_removal_authority(self, _reference: ArtifactRef) -> Authority:
            return Authority()

    coordinator = AudioCppArtifactLeaseCoordinator(
        RemovalService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )

    task = asyncio.create_task(
        coordinator.remove_if_unchanged(
            REFERENCE,
            "fingerprint",
            lambda: asyncio.sleep(0, result="fingerprint"),
        )
    )
    assert await asyncio.to_thread(close_entered.wait, 1.0)
    assert task.done() is False
    assert events == ["commit", "close-start"]
    release_close.set()
    assert await task == "committed"
    assert events == ["commit", "close-start", "close-end"]


@pytest.mark.asyncio
async def test_removal_close_failure_blocks_success_and_retries_before_reentry() -> (
    None
):
    events: list[str] = []

    class Authority:
        def __init__(self, *, fail_close: bool) -> None:
            self.fail_close = fail_close

        def commit(self) -> None:
            events.append("commit")

        def close(self) -> None:
            events.append("close")
            if self.fail_close:
                self.fail_close = False
                raise RuntimeError("PRIVATE /managed/root")

    authorities = iter((Authority(fail_close=True), Authority(fail_close=False)))

    class RemovalService(_Service):
        def acquire_removal_authority(self, _reference: ArtifactRef) -> Authority:
            events.append("acquire")
            return next(authorities)

    coordinator = AudioCppArtifactLeaseCoordinator(
        RemovalService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )

    async def collect() -> str:
        return "fingerprint"

    with pytest.raises(AudioCppArtifactDependencyError) as caught:
        await coordinator.remove_if_unchanged(REFERENCE, "fingerprint", collect)
    assert "PRIVATE" not in str(caught.value)
    assert events == ["acquire", "commit", "close"]

    assert (
        await coordinator.remove_if_unchanged(REFERENCE, "fingerprint", collect)
        == "committed"
    )
    assert events == [
        "acquire",
        "commit",
        "close",
        "close",
        "acquire",
        "commit",
        "close",
    ]


@pytest.mark.asyncio
async def test_probe_failure_cleanup_owner_is_retained_and_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class CleanupOwner:
        def close(self) -> None:
            events.append("cleanup-close")

    cleanup_owner = CleanupOwner()

    class ProbeService(_Service):
        def probe_removal_availability(self, _reference: ArtifactRef) -> object:
            raise RuntimeError("PRIVATE /managed/root")

    monkeypatch.setattr(
        dependency_module,
        "take_artifact_removal_cleanup_owner",
        lambda _error: cleanup_owner,
        raising=False,
    )
    coordinator = AudioCppArtifactLeaseCoordinator(
        ProbeService(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=_catalog,
    )

    with pytest.raises(AudioCppArtifactDependencyError) as caught:
        await coordinator.probe_removal_availability(REFERENCE)
    assert "PRIVATE" not in str(caught.value)
    assert events == []

    await coordinator.drain_cleanup()
    assert events == ["cleanup-close"]


def _requirement(
    *,
    model_id: str = "model-a",
    recipe_id: str = "audio-cpp-0.5.1.family.variant",
) -> AudioCppArtifactConsumerRequirement:
    return AudioCppArtifactConsumerRequirement(
        provider_id="audio_cpp",
        model_id=model_id,
        recipe_requirement=TTSCloneRecipeRequirement(recipe_id, 1, model_id),
    )


@pytest.mark.asyncio
async def test_coordinator_resolves_catalog_deduplicates_sorts_and_releases_after_scope() -> (
    None
):
    lower = ArtifactRef("a-model", "a" * 40, "q8_0")
    higher = ArtifactRef("z-model", "a" * 40, "q8_0")
    entries = _catalog(higher) + (
        (
            SimpleNamespace(reference=lower, model_id="model-b"),
            {},
            SimpleNamespace(
                recipe_id="audio-cpp-0.5.1.family.variant-b",
                recipe_revision=1,
                default_public_model_id="model-b",
            ),
        ),
    )
    service = _Service()
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: entries,
    )

    async with coordinator.lease_consumers(
        (
            _requirement(),
            _requirement(),
            _requirement(
                model_id="model-b",
                recipe_id="audio-cpp-0.5.1.family.variant-b",
            ),
        )
    ):
        assert service.acquired == [lower, higher]
        assert service.closed == []

    assert service.closed == [higher, lower]


@pytest.mark.asyncio
async def test_coordinator_tolerates_only_missing_artifact_and_leases_remaining_roots() -> (
    None
):
    service = _Service(missing={REFERENCE})
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: _catalog(),
    )

    async with coordinator.lease_consumers((_requirement(),)):
        pass

    assert service.acquired == [REFERENCE]


@pytest.mark.asyncio
async def test_coordinator_rejects_persisted_identity_disagreement_before_store_work() -> (
    None
):
    service = _Service()
    persisted = AudioCppManagedConsumerIdentity(
        recipe_id="audio-cpp-0.5.1.family.variant",
        recipe_revision=1,
        model_id="model-a",
        managed_artifact=AudioCppManagedArtifactIdentity(
            artifact_id=OTHER.artifact_id,
            revision=OTHER.revision,
            variant=OTHER.variant,
        ),
    )
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (persisted,),
        catalog_entries=lambda: _catalog(),
    )

    with pytest.raises(AudioCppArtifactDependencyError, match="identity disagreement"):
        async with coordinator.lease_consumers((_requirement(),)):
            raise AssertionError("unreachable")

    assert service.acquired == []


@pytest.mark.parametrize(
    "requirement",
    (
        AudioCppArtifactConsumerRequirement("openai", "model-a"),
        _requirement(model_id="local-model", recipe_id="local-only-recipe"),
    ),
)
@pytest.mark.asyncio
async def test_local_or_unsupported_consumers_do_not_acquire_managed_leases(
    requirement: AudioCppArtifactConsumerRequirement,
) -> None:
    service = _Service()
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: _catalog(),
    )

    async with coordinator.lease_consumers((requirement,)):
        pass

    assert service.acquired == []
