"""Contracts for Speech/TTS ownership, state, navigation, and safe status."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone

import pytest

from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    BUILT_IN_TTS_PROVIDER_IDS,
    CONFIGURE_TTS_PROVIDER_CONTROL_ID,
    DEFAULT_TTS_PROVIDER_CONTROL_ID,
    SHARED_TTS_DEFAULTS_OWNER_ID,
    SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID,
    SPEECH_TTS_OWNERSHIP_INVENTORY,
    SpeechTTSConfigurationState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSOwnershipRecord,
    SpeechTTSOwnershipScope,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
    validate_speech_tts_ownership_inventory,
)
from tldw_chatbook.UI.Speech.speech_settings_model import ALL_SETTINGS_CONTROLS


EXPECTED_CONTROL_IDS_BY_SCOPE = {
    SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION: {
        "alltalk-url-input",
        "audio-cpp-base-url-input",
        "audio-cpp-connect-timeout-input",
        "audio-cpp-max-catalog-models-input",
        "audio-cpp-max-identifier-characters-input",
        "audio-cpp-max-input-characters-input",
        "audio-cpp-max-metadata-bytes-input",
        "audio-cpp-max-response-bytes-input",
        "audio-cpp-max-voices-per-model-input",
        "audio-cpp-mode-value",
        "audio-cpp-privacy-notice",
        "audio-cpp-synthesis-timeout-input",
        "chatterbox-browse-voice-dir-btn",
        "chatterbox-candidates-input",
        "chatterbox-chunk-size-input",
        "chatterbox-crossfade-ms-input",
        "chatterbox-crossfade-switch",
        "chatterbox-device-select",
        "chatterbox-max-chunk-input",
        "chatterbox-normalize-switch",
        "chatterbox-preprocess-switch",
        "chatterbox-seed-input",
        "chatterbox-stream-chunk-input",
        "chatterbox-streaming-switch",
        "chatterbox-target-db-input",
        "chatterbox-temperature-input",
        "chatterbox-whisper-switch",
        "default-format-select",
        "default-model-select",
        "default-provider-select",
        "default-speed-input",
        "default-voice-select",
        "elevenlabs-api-key-input",
        "elevenlabs-format-select",
        "elevenlabs-similarity-input",
        "elevenlabs-speaker-boost-switch",
        "elevenlabs-stability-input",
        "elevenlabs-style-input",
        "higgs-delimiter-input",
        "higgs-device-select",
        "higgs-dtype-select",
        "higgs-flash-attn-switch",
        "higgs-language-select",
        "higgs-max-ref-duration-input",
        "higgs-max-tokens-input",
        "higgs-model-path-input",
        "higgs-multi-speaker-switch",
        "higgs-repetition-penalty-input",
        "higgs-temperature-input",
        "higgs-top-p-input",
        "higgs-track-performance-switch",
        "higgs-voice-cloning-switch",
        "higgs-voices-browse-btn",
        "higgs-voices-dir-input",
        "kokoro-browse-model-btn",
        "kokoro-browse-voices-btn",
        "kokoro-device-select",
        "kokoro-max-tokens-input",
        "kokoro-performance-switch",
        "kokoro-use-onnx-switch",
        "kokoro-voice-mixing-switch",
        "openai-api-key-input",
        "openai-base-url-input",
        "openai-org-id-input",
        "alltalk-language-select",
    },
    SpeechTTSOwnershipScope.STUDIO_PREFERENCE: {
        "alltalk-format-select",
        "alltalk-voice-input",
        "chatterbox-cfg-weight-input",
        "chatterbox-exaggeration-input",
        "elevenlabs-model-select",
    },
    SpeechTTSOwnershipScope.VOICE_PROFILE_OPERATION: {
        "add-voice-blend-btn",
        "export-blends-btn",
        "import-blends-btn",
        "kokoro-voice-blends-list",
    },
    SpeechTTSOwnershipScope.RUNTIME_OPERATION_OR_READOUT: {
        "audio-cpp-discovery-status",
        "audio-cpp-refresh-models-btn",
        "audio-cpp-test-connection-btn",
    },
    SpeechTTSOwnershipScope.RETIRED: {
        "audio-cpp-settings",
        "save-settings-btn",
    },
}


@pytest.mark.unit
def test_built_in_provider_ids_and_shared_defaults_are_exact() -> None:
    assert BUILT_IN_TTS_PROVIDER_IDS == (
        "audio_cpp",
        "openai",
        "elevenlabs",
        "kokoro",
        "chatterbox",
        "higgs",
        "alltalk",
    )
    assert SHARED_TTS_DEFAULTS_OWNER_ID == "defaults"
    assert SHARED_TTS_DEFAULTS_OWNER_ID not in BUILT_IN_TTS_PROVIDER_IDS


@pytest.mark.unit
def test_every_current_control_is_classified_exactly_once() -> None:
    assert len(SPEECH_TTS_OWNERSHIP_INVENTORY) == len(ALL_SETTINGS_CONTROLS)
    assert set(SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID) == ALL_SETTINGS_CONTROLS
    assert len(SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID) == len(
        SPEECH_TTS_OWNERSHIP_INVENTORY
    )


@pytest.mark.unit
def test_every_adr_039_scope_partition_is_exact() -> None:
    actual = {
        scope: {
            record.control_id
            for record in SPEECH_TTS_OWNERSHIP_INVENTORY
            if record.scope is scope
        }
        for scope in SpeechTTSOwnershipScope
    }
    assert actual == EXPECTED_CONTROL_IDS_BY_SCOPE


@pytest.mark.unit
def test_studio_partition_is_limited_to_current_end_to_end_request_values() -> None:
    """Constructor-only values must not become persisted Studio no-ops.

    This list is intentionally conservative: the selections traverse the
    legacy request, while Chatterbox reliably consumes only exaggeration and
    CFG weight across every generation branch today.
    """
    actual = {
        record.control_id
        for record in SPEECH_TTS_OWNERSHIP_INVENTORY
        if record.scope is SpeechTTSOwnershipScope.STUDIO_PREFERENCE
    }
    assert actual == {
        "alltalk-format-select",
        "alltalk-voice-input",
        "chatterbox-cfg-weight-input",
        "chatterbox-exaggeration-input",
        "elevenlabs-model-select",
    }


@pytest.mark.unit
def test_every_provider_and_shared_defaults_are_represented() -> None:
    actual = {record.owner_id for record in SPEECH_TTS_OWNERSHIP_INVENTORY}
    assert actual == set(BUILT_IN_TTS_PROVIDER_IDS) | {
        SHARED_TTS_DEFAULTS_OWNER_ID
    }


@pytest.mark.unit
def test_every_control_retains_its_exact_provider_owner() -> None:
    special_owners = {
        "add-voice-blend-btn": "kokoro",
        "export-blends-btn": "kokoro",
        "import-blends-btn": "kokoro",
        "save-settings-btn": SHARED_TTS_DEFAULTS_OWNER_ID,
    }
    owner_prefixes = (
        ("default-", SHARED_TTS_DEFAULTS_OWNER_ID),
        ("audio-cpp-", "audio_cpp"),
        ("openai-", "openai"),
        ("elevenlabs-", "elevenlabs"),
        ("kokoro-", "kokoro"),
        ("chatterbox-", "chatterbox"),
        ("higgs-", "higgs"),
        ("alltalk-", "alltalk"),
    )

    for record in SPEECH_TTS_OWNERSHIP_INVENTORY:
        expected = special_owners.get(record.control_id)
        if expected is None:
            expected = next(
                owner_id
                for prefix, owner_id in owner_prefixes
                if record.control_id.startswith(prefix)
            )
        assert record.owner_id == expected, record.control_id


@pytest.mark.unit
def test_retired_entries_have_specific_replacement_reasons() -> None:
    retired = {
        record.control_id: record.reason
        for record in SPEECH_TTS_OWNERSHIP_INVENTORY
        if record.scope is SpeechTTSOwnershipScope.RETIRED
    }
    assert retired.keys() == {"audio-cpp-settings", "save-settings-btn"}
    assert all(reason.strip() for reason in retired.values())
    assert "mixed-scope" in retired["save-settings-btn"]
    assert "structural" in retired["audio-cpp-settings"]


@pytest.mark.unit
def test_default_provider_and_configure_provider_have_distinct_ids() -> None:
    assert DEFAULT_TTS_PROVIDER_CONTROL_ID == "default-provider-select"
    assert CONFIGURE_TTS_PROVIDER_CONTROL_ID == "configure-provider-select"
    assert DEFAULT_TTS_PROVIDER_CONTROL_ID != CONFIGURE_TTS_PROVIDER_CONTROL_ID
    assert (
        SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID[DEFAULT_TTS_PROVIDER_CONTROL_ID].scope
        is SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION
    )
    assert CONFIGURE_TTS_PROVIDER_CONTROL_ID not in ALL_SETTINGS_CONTROLS


@pytest.mark.unit
def test_validator_rejects_an_unclassified_current_control() -> None:
    with pytest.raises(ValueError, match="unclassified"):
        validate_speech_tts_ownership_inventory(
            SPEECH_TTS_OWNERSHIP_INVENTORY[:-1]
        )


@pytest.mark.unit
def test_validator_rejects_a_multiply_classified_control() -> None:
    duplicate = SPEECH_TTS_OWNERSHIP_INVENTORY[0]
    with pytest.raises(ValueError, match="multiple"):
        validate_speech_tts_ownership_inventory(
            (*SPEECH_TTS_OWNERSHIP_INVENTORY, duplicate)
        )


@pytest.mark.unit
def test_validator_rejects_an_unknown_control() -> None:
    unknown = SpeechTTSOwnershipRecord(
        control_id="plugin-secret-input",
        owner_id="openai",
        scope=SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_speech_tts_ownership_inventory(
            (*SPEECH_TTS_OWNERSHIP_INVENTORY, unknown)
        )


@pytest.mark.unit
def test_record_rejects_unknown_owner_and_scope_classifications() -> None:
    with pytest.raises(ValueError, match="owner"):
        SpeechTTSOwnershipRecord(
            control_id="openai-api-key-input",
            owner_id="plugin_provider",
            scope=SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        )
    with pytest.raises(TypeError, match="scope"):
        SpeechTTSOwnershipRecord(
            control_id="openai-api-key-input",
            owner_id="openai",
            scope="plugin-scope",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_validator_rejects_a_scope_that_contradicts_adr_039() -> None:
    records = list(SPEECH_TTS_OWNERSHIP_INVENTORY)
    index = next(
        index
        for index, record in enumerate(records)
        if record.control_id == "openai-api-key-input"
    )
    records[index] = replace(
        records[index], scope=SpeechTTSOwnershipScope.STUDIO_PREFERENCE
    )

    with pytest.raises(ValueError, match="ADR-039"):
        validate_speech_tts_ownership_inventory(records)


@pytest.mark.unit
def test_validator_rejects_a_provider_owner_that_contradicts_adr_039() -> None:
    records = list(SPEECH_TTS_OWNERSHIP_INVENTORY)
    index = next(
        index
        for index, record in enumerate(records)
        if record.control_id == "openai-api-key-input"
    )
    records[index] = replace(records[index], owner_id="elevenlabs")

    with pytest.raises(ValueError, match="ADR-039"):
        validate_speech_tts_ownership_inventory(records)


@pytest.mark.unit
def test_configuration_and_runtime_vocabularies_are_exact_and_independent() -> None:
    assert tuple(SpeechTTSConfigurationState) == (
        SpeechTTSConfigurationState.INHERITED,
        SpeechTTSConfigurationState.DEFAULT,
        SpeechTTSConfigurationState.SAVED,
        SpeechTTSConfigurationState.UNSAVED,
        SpeechTTSConfigurationState.INCOMPLETE,
        SpeechTTSConfigurationState.INVALID,
    )
    assert tuple(state.value for state in SpeechTTSConfigurationState) == (
        "Inherited",
        "Default",
        "Saved",
        "Unsaved",
        "Incomplete",
        "Invalid",
    )
    assert tuple(state.value for state in SpeechTTSRuntimeState) == (
        "Not checked",
        "Checking",
        "Ready",
        "Stale",
        "Unavailable",
        "Reconfiguring",
    )
    assert set(SpeechTTSConfigurationState).isdisjoint(SpeechTTSRuntimeState)

    with pytest.raises(ValueError):
        SpeechTTSConfigurationState("Ready")
    with pytest.raises(ValueError):
        SpeechTTSRuntimeState("Saved")


@pytest.mark.unit
def test_navigation_target_carries_only_provider_and_bounded_intent() -> None:
    target = SpeechTTSNavigationTarget(
        provider_id="audio_cpp",
        intent=SpeechTTSNavigationIntent.REFRESH_MODELS,
    )

    assert target.provider_id == "audio_cpp"
    assert target.intent is SpeechTTSNavigationIntent.REFRESH_MODELS
    assert {field.name for field in fields(target)} == {"provider_id", "intent"}
    assert tuple(intent.value for intent in SpeechTTSNavigationIntent) == (
        "configure",
        "test",
        "refresh-models",
        "refresh-voices",
    )


@pytest.mark.unit
@pytest.mark.parametrize("provider_id", BUILT_IN_TTS_PROVIDER_IDS)
def test_navigation_target_accepts_each_exact_built_in_provider(
    provider_id: str,
) -> None:
    target = SpeechTTSNavigationTarget(provider_id=provider_id)

    assert target.provider_id == provider_id
    assert target.intent is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_id",
    ("defaults", "audio.cpp", "Audio_cpp", "plugin_provider", ""),
)
def test_navigation_target_rejects_non_provider_or_noncanonical_ids(
    provider_id: str,
) -> None:
    with pytest.raises(ValueError, match="provider"):
        SpeechTTSNavigationTarget(provider_id=provider_id)


@pytest.mark.unit
def test_navigation_target_rejects_unbounded_values_and_extra_payload() -> None:
    class ProviderId(str):
        pass

    with pytest.raises(TypeError, match="provider"):
        SpeechTTSNavigationTarget(provider_id=ProviderId("audio_cpp"))
    with pytest.raises(TypeError, match="intent"):
        SpeechTTSNavigationTarget(
            provider_id="audio_cpp",
            intent="generate",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        SpeechTTSNavigationTarget(  # type: ignore[call-arg]
            provider_id="audio_cpp",
            intent=SpeechTTSNavigationIntent.TEST,
            synthesis_text="do not carry me",
        )


def _status(**updates: object) -> SpeechTTSRuntimeStatus:
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "saved_configuration_revision": 4,
        "runtime_revision": 7,
        "catalog_revision": 11,
        "runtime_state": SpeechTTSRuntimeState.READY,
        "observed_at": datetime(2026, 7, 31, 22, 0, tzinfo=timezone.utc),
        "freshness": SpeechTTSStatusFreshness.FRESH,
        "diagnostic_category": None,
        "recovery_action": None,
    }
    values.update(updates)
    return SpeechTTSRuntimeStatus(**values)  # type: ignore[arg-type]


@pytest.mark.unit
def test_safe_status_is_revisioned_frozen_and_has_no_free_form_payload() -> None:
    status = _status(
        runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
        freshness=SpeechTTSStatusFreshness.FRESH,
        diagnostic_category=SpeechTTSDiagnosticCategory.CONNECTION,
        recovery_action=SpeechTTSNavigationIntent.TEST,
    )

    assert status.provider_id == "audio_cpp"
    assert status.saved_configuration_revision == 4
    assert status.runtime_revision == 7
    assert status.catalog_revision == 11
    assert status.diagnostic_category is SpeechTTSDiagnosticCategory.CONNECTION
    assert status.recovery_action is SpeechTTSNavigationIntent.TEST
    assert {field.name for field in fields(status)} == {
        "provider_id",
        "saved_configuration_revision",
        "runtime_revision",
        "catalog_revision",
        "runtime_state",
        "observed_at",
        "freshness",
        "diagnostic_category",
        "recovery_action",
    }
    with pytest.raises(AttributeError):
        status.runtime_revision = 8  # type: ignore[misc]


@pytest.mark.unit
def test_safe_status_accepts_unavailable_runtime_and_catalog_revisions() -> None:
    status = _status(runtime_revision=None, catalog_revision=None)

    assert status.runtime_revision is None
    assert status.catalog_revision is None


@pytest.mark.unit
def test_safe_status_represents_a_stale_observation_as_stale_not_ready() -> None:
    status = _status(
        runtime_state=SpeechTTSRuntimeState.STALE,
        freshness=SpeechTTSStatusFreshness.STALE,
    )

    assert status.runtime_state is SpeechTTSRuntimeState.STALE
    assert status.freshness is SpeechTTSStatusFreshness.STALE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runtime_state", "freshness"),
    (
        (SpeechTTSRuntimeState.READY, SpeechTTSStatusFreshness.STALE),
        (SpeechTTSRuntimeState.UNAVAILABLE, SpeechTTSStatusFreshness.STALE),
        (SpeechTTSRuntimeState.STALE, SpeechTTSStatusFreshness.FRESH),
    ),
)
def test_safe_status_rejects_contradictory_state_and_freshness(
    runtime_state: SpeechTTSRuntimeState,
    freshness: SpeechTTSStatusFreshness,
) -> None:
    with pytest.raises(ValueError, match="Stale runtime observations"):
        _status(runtime_state=runtime_state, freshness=freshness)


@pytest.mark.unit
@pytest.mark.parametrize(
    "updates",
    (
        {"provider_id": "defaults"},
        {"provider_id": "audio.cpp"},
        {"saved_configuration_revision": -1},
        {"saved_configuration_revision": True},
        {"runtime_revision": -1},
        {"runtime_revision": 1.5},
        {"catalog_revision": -1},
        {"catalog_revision": False},
        {"runtime_state": "Ready"},
        {"observed_at": datetime(2026, 7, 31, 22, 0)},
        {"freshness": "fresh"},
        {"diagnostic_category": "raw exception text"},
        {"recovery_action": "https://secret.invalid/path?token=x"},
    ),
)
def test_safe_status_rejects_malformed_or_free_form_values(
    updates: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _status(**updates)


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe_field",
    ("url", "exception_text", "secret", "synthesis_text", "response_body"),
)
def test_safe_status_cannot_accept_prohibited_payload_fields(
    unsafe_field: str,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        _status(**{unsafe_field: "private"})
