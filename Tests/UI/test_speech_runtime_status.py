"""Truthful, revision-bound Speech status and navigation projections."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.UI.Speech import speech_runtime_status
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    SpeechTTSRuntimeStatusStore,
    project_speech_tts_status,
    speech_tts_navigation_context,
    speech_tts_navigation_target_from_context,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
    speech_tts_model_scope,
)

_OBSERVED_AT = datetime(2026, 8, 1, 16, 0, tzinfo=timezone.utc)


def _observation(
    *,
    configuration_revision: int = 4,
    catalog_revision: int = 7,
    health: ProviderHealth | None = None,
) -> TTSNativeCapabilityObservation:
    catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=catalog_revision,
        health=health or ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id="model-a",
                display_name="Model A",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=("voice-a",),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )
    return TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=configuration_revision,
            state="complete" if catalog.health.fresh else "unverified",
            catalog=catalog,
            voice_results={},
        ),
        observed_at=_OBSERVED_AT,
    )


def _missing_local_dependencies() -> SpeechLocalDependencyAvailability:
    return SpeechLocalDependencyAvailability(
        stt=False,
        kokoro=False,
        chatterbox=False,
        higgs=False,
    )


def _all_missing() -> SpeechLocalDependencyAvailability:
    return _missing_local_dependencies()


def _ready_openai_status() -> SpeechTTSRuntimeStatus:
    return SpeechTTSRuntimeStatus(
        provider_id="openai",
        saved_configuration_revision=1,
        runtime_revision=1,
        catalog_revision=None,
        model_scope=None,
        runtime_state=SpeechTTSRuntimeState.READY,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )


def _catalog_status(
    *,
    saved_revision: int,
    observed_at: datetime,
    state: SpeechTTSRuntimeState,
    runtime_revision: int = 4,
    catalog_revision: int = 7,
) -> SpeechTTSRuntimeStatus:
    return SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=saved_revision,
        runtime_revision=runtime_revision,
        catalog_revision=catalog_revision,
        model_scope=speech_tts_model_scope("model-a"),
        runtime_state=state,
        observed_at=observed_at,
        freshness=(
            SpeechTTSStatusFreshness.STALE
            if state is SpeechTTSRuntimeState.STALE
            else SpeechTTSStatusFreshness.FRESH
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize("configuration_state", tuple(SpeechTTSConfigurationState))
def test_every_configuration_state_uses_the_canonical_vocabulary(
    configuration_state: SpeechTTSConfigurationState,
) -> None:
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=configuration_state,
        current_configuration_revision=4,
        model_id=None,
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.rows()[0].state is configuration_state


@pytest.mark.unit
@pytest.mark.parametrize("runtime_state", tuple(SpeechTTSRuntimeState))
def test_every_runtime_state_uses_the_canonical_vocabulary(
    runtime_state: SpeechTTSRuntimeState,
) -> None:
    status = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=2,
        catalog_revision=None,
        model_scope=None,
        runtime_state=runtime_state,
        observed_at=_OBSERVED_AT,
        freshness=(
            SpeechTTSStatusFreshness.STALE
            if runtime_state is SpeechTTSRuntimeState.STALE
            else SpeechTTSStatusFreshness.FRESH
        ),
    )
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id=None,
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        runtime_status=status,
    )

    assert projection.runtime_state is runtime_state


@pytest.mark.unit
def test_external_audio_cpp_readiness_is_independent_of_every_local_dependency() -> (
    None
):
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id="model-a",
        observation=_observation(),
        local_dependencies=_missing_local_dependencies(),
    )

    assert projection.configuration_state is SpeechTTSConfigurationState.SAVED
    assert projection.runtime_state is SpeechTTSRuntimeState.READY
    assert projection.catalog_state is SpeechTTSRuntimeState.READY
    assert projection.local_dependencies.stt is False
    assert projection.local_dependencies.kokoro is False
    assert projection.local_dependencies.chatterbox is False
    assert projection.local_dependencies.higgs is False
    rows = {row.row_id: row for row in projection.rows()}
    assert rows["provider-runtime"].state is SpeechTTSRuntimeState.READY
    assert rows["stt-dependency"].state is SpeechTTSRuntimeState.UNAVAILABLE
    assert rows["kokoro-dependency"].state is SpeechTTSRuntimeState.UNAVAILABLE
    assert rows["chatterbox-dependency"].state is SpeechTTSRuntimeState.UNAVAILABLE
    assert rows["higgs-dependency"].state is SpeechTTSRuntimeState.UNAVAILABLE


@pytest.mark.unit
def test_remote_openai_compatible_remains_available_without_local_packages() -> None:
    assert not hasattr(speech_runtime_status, "build_speech_runtime_projection")
    projection = project_speech_tts_status(
        provider_id="openai",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=1,
        model_id=None,
        observation=None,
        local_dependencies=_all_missing(),
        runtime_status=_ready_openai_status(),
    )

    assert projection.runtime_ready
    assert "OpenAI-compatible" in projection.summary
    assert all(
        row.state is SpeechTTSRuntimeState.UNAVAILABLE
        for row in projection.rows()
        if row.row_id.endswith("-dependency")
    )


@pytest.mark.unit
def test_no_observation_is_not_checked_and_never_ready() -> None:
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id="model-a",
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.runtime_status is None
    assert projection.runtime_state is SpeechTTSRuntimeState.NOT_CHECKED
    assert projection.catalog_state is SpeechTTSRuntimeState.NOT_CHECKED


@pytest.mark.unit
def test_old_or_nonfresh_observation_is_stale_not_ready() -> None:
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=5,
        current_runtime_revision=9,
        applied_configuration_revision=5,
        model_id="model-a",
        observation=_observation(
            configuration_revision=4,
            health=ProviderHealth(state="available", fresh=False),
        ),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.runtime_state is SpeechTTSRuntimeState.STALE
    assert projection.catalog_state is SpeechTTSRuntimeState.STALE
    assert projection.runtime_status is not None
    assert projection.runtime_status.saved_configuration_revision == 5
    assert projection.runtime_status.model_scope is None
    assert projection.catalog_status is not None
    assert projection.catalog_status.model_scope == speech_tts_model_scope("model-a")
    assert projection.runtime_status.freshness is SpeechTTSStatusFreshness.STALE
    assert projection.runtime_status.diagnostic_category is (
        SpeechTTSDiagnosticCategory.CONFIGURATION
    )
    assert projection.runtime_status.recovery_action is SpeechTTSNavigationIntent.TEST
    assert projection.catalog_status.diagnostic_category is (
        SpeechTTSDiagnosticCategory.CATALOG
    )
    assert projection.catalog_status.recovery_action is (
        SpeechTTSNavigationIntent.REFRESH_MODELS
    )


@pytest.mark.unit
def test_nonfresh_catalog_does_not_downgrade_reachable_provider_runtime() -> None:
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=_observation(
            configuration_revision=4,
            health=ProviderHealth(state="available", fresh=False),
        ),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.runtime_state is SpeechTTSRuntimeState.READY
    assert projection.catalog_state is SpeechTTSRuntimeState.STALE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("health", "expected"),
    (
        (
            ProviderHealth(state="unavailable", fresh=True),
            SpeechTTSRuntimeState.UNAVAILABLE,
        ),
        (
            ProviderHealth(state="reconfiguring", fresh=True),
            SpeechTTSRuntimeState.RECONFIGURING,
        ),
    ),
)
def test_saved_configuration_remains_saved_for_nonready_runtime_states(
    health: ProviderHealth,
    expected: SpeechTTSRuntimeState,
) -> None:
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id=None,
        observation=_observation(health=health),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.configuration_state is SpeechTTSConfigurationState.SAVED
    assert projection.runtime_state is expected


@pytest.mark.unit
def test_runtime_reconfiguration_does_not_make_old_catalog_current() -> None:
    runtime_status = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=5,
        runtime_revision=9,
        catalog_revision=None,
        model_scope=None,
        runtime_state=SpeechTTSRuntimeState.RECONFIGURING,
        observed_at=_OBSERVED_AT + timedelta(seconds=1),
        freshness=SpeechTTSStatusFreshness.FRESH,
        diagnostic_category=SpeechTTSDiagnosticCategory.CONFIGURATION,
        recovery_action=SpeechTTSNavigationIntent.TEST,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=5,
        current_runtime_revision=9,
        applied_configuration_revision=5,
        model_id="model-a",
        observation=_observation(configuration_revision=4),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        runtime_status=runtime_status,
    )

    assert projection.runtime_state is SpeechTTSRuntimeState.RECONFIGURING
    assert projection.catalog_state is SpeechTTSRuntimeState.STALE


@pytest.mark.unit
def test_newer_shared_catalog_status_wins_over_an_older_cached_observation() -> None:
    shared = _catalog_status(
        saved_revision=4,
        observed_at=_OBSERVED_AT + timedelta(seconds=5),
        state=SpeechTTSRuntimeState.CHECKING,
        catalog_revision=8,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=_observation(),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        catalog_status=shared,
    )

    assert projection.catalog_status is shared
    assert projection.catalog_state is SpeechTTSRuntimeState.CHECKING


@pytest.mark.unit
def test_first_catalog_check_is_checking_before_a_revision_exists() -> None:
    checking = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=4,
        catalog_revision=None,
        model_scope=speech_tts_model_scope("model-a"),
        runtime_state=SpeechTTSRuntimeState.CHECKING,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        catalog_status=checking,
    )

    assert projection.catalog_status is checking
    assert projection.catalog_state is SpeechTTSRuntimeState.CHECKING


@pytest.mark.unit
def test_unverified_selected_model_voice_evidence_is_stale_not_ready() -> None:
    catalog = _observation().snapshot.catalog
    assert catalog is not None
    observation = TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=4,
            state="unverified",
            catalog=catalog,
            voice_results={
                "model-a": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model-a",
                    catalog_revision=catalog.revision,
                    voices=("voice-a",),
                    state="unverified",
                )
            },
        ),
        observed_at=_OBSERVED_AT,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=observation,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.runtime_state is SpeechTTSRuntimeState.READY
    assert projection.catalog_state is SpeechTTSRuntimeState.STALE
    assert projection.catalog_status is not None
    assert projection.catalog_status.recovery_action is (
        SpeechTTSNavigationIntent.REFRESH_VOICES
    )


@pytest.mark.unit
def test_authoritative_missing_selected_model_is_unavailable() -> None:
    catalog = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=7,
        health=ProviderHealth(state="available", fresh=True),
        models=(),
    )
    observation = TTSNativeCapabilityObservation(
        snapshot=TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=4,
            state="complete",
            catalog=catalog,
            voice_results={
                "model-a": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model-a",
                    catalog_revision=7,
                    voices=(),
                    state="model_missing",
                )
            },
        ),
        observed_at=_OBSERVED_AT,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=observation,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    assert projection.runtime_state is SpeechTTSRuntimeState.READY
    assert projection.catalog_state is SpeechTTSRuntimeState.UNAVAILABLE


@pytest.mark.unit
def test_status_store_rejects_older_revisions_and_times() -> None:
    store = SpeechTTSRuntimeStatusStore()
    newest = _catalog_status(
        saved_revision=5,
        observed_at=_OBSERVED_AT + timedelta(seconds=5),
        state=SpeechTTSRuntimeState.READY,
    )
    older_revision = _catalog_status(
        saved_revision=4,
        observed_at=_OBSERVED_AT + timedelta(seconds=10),
        state=SpeechTTSRuntimeState.UNAVAILABLE,
    )
    older_time = _catalog_status(
        saved_revision=5,
        observed_at=_OBSERVED_AT,
        state=SpeechTTSRuntimeState.UNAVAILABLE,
    )

    store.publish_catalog(newest)
    store.publish_catalog(older_revision)
    store.publish_catalog(older_time)

    assert store.catalog_status("audio_cpp", "model-a") is newest


@pytest.mark.unit
def test_revision_order_beats_later_completion_time_in_store_and_projection() -> None:
    store = SpeechTTSRuntimeStatusStore()
    newer_runtime = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=9,
        catalog_revision=None,
        model_scope=None,
        runtime_state=SpeechTTSRuntimeState.READY,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )
    late_old_runtime = replace(
        newer_runtime,
        runtime_revision=8,
        runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
        observed_at=_OBSERVED_AT + timedelta(seconds=10),
    )
    newer_catalog = _catalog_status(
        saved_revision=4,
        observed_at=_OBSERVED_AT,
        state=SpeechTTSRuntimeState.READY,
        runtime_revision=9,
        catalog_revision=8,
    )
    late_old_catalog = replace(
        newer_catalog,
        catalog_revision=7,
        runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
        observed_at=_OBSERVED_AT + timedelta(seconds=10),
    )

    store.publish_runtime(newer_runtime)
    store.publish_runtime(late_old_runtime)
    store.publish_catalog(newer_catalog)
    store.publish_catalog(late_old_catalog)

    assert store.runtime_status("audio_cpp") is newer_runtime
    assert store.catalog_status("audio_cpp", "model-a") is newer_catalog

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id="model-a",
        observation=_observation(
            configuration_revision=8,
            catalog_revision=7,
        ),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        runtime_status=newer_runtime,
        catalog_status=newer_catalog,
    )

    assert projection.runtime_status is newer_runtime
    assert projection.catalog_status is newer_catalog


@pytest.mark.unit
def test_later_operation_without_optional_catalog_revision_replaces_ready() -> None:
    store = SpeechTTSRuntimeStatusStore()
    ready_runtime = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=4,
        catalog_revision=7,
        model_scope=None,
        runtime_state=SpeechTTSRuntimeState.READY,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )
    checking_runtime = replace(
        ready_runtime,
        catalog_revision=None,
        runtime_state=SpeechTTSRuntimeState.CHECKING,
        observed_at=_OBSERVED_AT + timedelta(seconds=1),
    )
    ready_catalog = _catalog_status(
        saved_revision=4,
        observed_at=_OBSERVED_AT,
        state=SpeechTTSRuntimeState.READY,
        catalog_revision=7,
    )
    failed_catalog = replace(
        ready_catalog,
        catalog_revision=None,
        runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
        observed_at=_OBSERVED_AT + timedelta(seconds=1),
    )

    store.publish_runtime(ready_runtime)
    store.publish_runtime(checking_runtime)
    store.publish_catalog(ready_catalog)
    store.publish_catalog(failed_catalog)

    assert store.runtime_status("audio_cpp") is checking_runtime
    assert store.catalog_status("audio_cpp", "model-a") is failed_catalog

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=4,
        applied_configuration_revision=4,
        model_id="model-a",
        observation=_observation(),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        runtime_status=checking_runtime,
        catalog_status=failed_catalog,
    )

    assert projection.runtime_status is checking_runtime
    assert projection.catalog_status is failed_catalog


@pytest.mark.unit
def test_terminal_status_wins_a_same_operation_transition_tie() -> None:
    pending = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=9,
        catalog_revision=None,
        model_scope=None,
        runtime_state=SpeechTTSRuntimeState.RECONFIGURING,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )
    terminal = replace(
        pending,
        runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
    )

    pending_then_terminal = SpeechTTSRuntimeStatusStore()
    pending_then_terminal.publish_runtime(pending)
    pending_then_terminal.publish_runtime(terminal)
    terminal_then_pending = SpeechTTSRuntimeStatusStore()
    terminal_then_pending.publish_runtime(terminal)
    terminal_then_pending.publish_runtime(pending)

    assert pending_then_terminal.runtime_status("audio_cpp") is terminal
    assert terminal_then_pending.runtime_status("audio_cpp") is terminal


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runtime_state", "catalog_revision"),
    (
        (SpeechTTSRuntimeState.STALE, 7),
        (SpeechTTSRuntimeState.UNAVAILABLE, None),
    ),
)
def test_catalog_row_includes_bounded_observation_and_recovery_metadata(
    runtime_state: SpeechTTSRuntimeState,
    catalog_revision: int | None,
) -> None:
    status = SpeechTTSRuntimeStatus(
        provider_id="audio_cpp",
        saved_configuration_revision=4,
        runtime_revision=9,
        catalog_revision=catalog_revision,
        model_scope=speech_tts_model_scope("private-model-id"),
        runtime_state=runtime_state,
        observed_at=_OBSERVED_AT,
        freshness=(
            SpeechTTSStatusFreshness.STALE
            if runtime_state is SpeechTTSRuntimeState.STALE
            else SpeechTTSStatusFreshness.FRESH
        ),
        diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
        recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
    )
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        current_runtime_revision=9,
        applied_configuration_revision=4,
        model_id="private-model-id",
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        catalog_status=status,
    )

    catalog_copy = next(
        row.copy for row in projection.rows() if row.row_id == "catalog-freshness"
    )
    assert "saved revision 4" in catalog_copy
    assert "runtime revision 9" in catalog_copy
    assert "observed 2026-08-01 16:00 UTC" in catalog_copy
    assert "catalog" in catalog_copy
    assert "recovery refresh-models" in catalog_copy
    assert "selected model scope" in catalog_copy
    assert "private-model-id" not in catalog_copy
    if catalog_revision is None:
        assert "No accepted catalog or voice observation" not in catalog_copy
    else:
        assert "catalog revision 7" in catalog_copy


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_provider", "status_model"),
    (("openai", "model-a"), ("audio_cpp", "model-b")),
)
def test_runtime_status_outside_the_selected_scope_is_not_displayed(
    status_provider: str,
    status_model: str,
) -> None:
    status = SpeechTTSRuntimeStatus(
        provider_id=status_provider,
        saved_configuration_revision=4,
        runtime_revision=7,
        catalog_revision=8,
        model_scope=speech_tts_model_scope(status_model),
        runtime_state=SpeechTTSRuntimeState.READY,
        observed_at=_OBSERVED_AT,
        freshness=SpeechTTSStatusFreshness.FRESH,
    )

    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.SAVED,
        current_configuration_revision=4,
        model_id="model-a",
        observation=None,
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
        catalog_status=status,
    )

    assert projection.runtime_status is None
    assert projection.runtime_state is SpeechTTSRuntimeState.NOT_CHECKED
    assert projection.catalog_state is SpeechTTSRuntimeState.NOT_CHECKED


@pytest.mark.unit
def test_status_rows_are_bounded_and_attribute_a_dirty_draft_to_saved_values() -> None:
    sensitive_model_id = (
        "https://user:secret@example.invalid/model?token=private-model-token"
    )
    projection = project_speech_tts_status(
        provider_id="audio_cpp",
        configuration_state=SpeechTTSConfigurationState.UNSAVED,
        current_configuration_revision=4,
        model_id=sensitive_model_id,
        observation=_observation(
            health=ProviderHealth(
                state="unavailable",
                fresh=True,
                diagnostic=(
                    "https://user:secret@example.invalid/path?token=secret "
                    "submitted synthesis text and raw response body"
                ),
            )
        ),
        local_dependencies=SpeechLocalDependencyAvailability.all_available(),
    )

    rendered = "\n".join(row.copy for row in projection.rows(dirty_draft=True))
    assert "saved revision 4" in rendered
    assert "does not validate this unsaved draft" in rendered
    assert "2026-08-01 16:00 UTC" in rendered
    assert "connection" in rendered
    assert "secret" not in rendered
    assert "submitted synthesis text" not in rendered
    assert "raw response body" not in rendered
    assert "example.invalid" not in rendered
    assert "private-model-token" not in rendered
    assert sensitive_model_id not in rendered
    assert "selected model scope" in rendered


@pytest.mark.unit
@pytest.mark.parametrize("intent", tuple(SpeechTTSNavigationIntent))
def test_bounded_navigation_round_trips_every_allowed_intent(
    intent: SpeechTTSNavigationIntent,
) -> None:
    target = SpeechTTSNavigationTarget("audio_cpp", intent)
    context = speech_tts_navigation_context(target)

    assert context == {"provider": "audio_cpp", "intent": intent.value}
    assert speech_tts_navigation_target_from_context(context) == target


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    (
        {"provider": "audio.cpp", "intent": "test"},
        {"provider": "audio_cpp", "intent": "generate"},
        {"provider": "audio_cpp", "intent": "test", "text": "private"},
        {"provider": "audio_cpp", "intent": "test", "api_key": "private"},
        {"provider": "audio_cpp", "intent": "test", "url": "https://secret"},
    ),
)
def test_navigation_context_rejects_invalid_or_extra_payload(
    context: dict[str, object],
) -> None:
    assert speech_tts_navigation_target_from_context(context) is None
