from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

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
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile
from tldw_chatbook.TTS.profile_reference_types import TTSCloneRecipeRequirement


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


def test_projection_keeps_catalog_profile_when_saved_package_was_removed() -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
    )

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

    evidence = project_audio_cpp_artifact_removal_evidence(
        descriptor.reference,
        saved_settings=AudioCppSettingsConfig(),
        profiles=((profile, 3),),
    )

    assert evidence.profile_consumers == (
        (str(profile.profile_id), "Narrator", 3, False),
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


def test_coordinator_resolves_catalog_deduplicates_sorts_and_releases_after_scope() -> (
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

    with coordinator.lease_consumers(
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


def test_coordinator_tolerates_only_missing_artifact_and_leases_remaining_roots() -> (
    None
):
    service = _Service(missing={REFERENCE})
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: _catalog(),
    )

    with coordinator.lease_consumers((_requirement(),)):
        pass

    assert service.acquired == [REFERENCE]


def test_coordinator_rejects_persisted_identity_disagreement_before_store_work() -> (
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
        with coordinator.lease_consumers((_requirement(),)):
            raise AssertionError("unreachable")

    assert service.acquired == []


@pytest.mark.parametrize(
    "requirement",
    (
        AudioCppArtifactConsumerRequirement("openai", "model-a"),
        _requirement(model_id="local-model", recipe_id="local-only-recipe"),
    ),
)
def test_local_or_unsupported_consumers_do_not_acquire_managed_leases(
    requirement: AudioCppArtifactConsumerRequirement,
) -> None:
    service = _Service()
    coordinator = AudioCppArtifactLeaseCoordinator(
        service,
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: _catalog(),
    )

    with coordinator.lease_consumers((requirement,)):
        pass

    assert service.acquired == []
