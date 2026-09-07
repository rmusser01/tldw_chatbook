from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    _TTS_SETTING_BINDINGS,
    STTSEventHandler,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import (
    TTSSettingsPersistenceOutcome,
    TTSSettingsPublication,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    build_global_speech_tts_save_proposal,
    load_global_speech_tts_state,
)
from tldw_chatbook.UI.Speech.speech_settings_mixin import (
    LAB_STUDIO_COMPATIBILITY_SETTING_KEYS,
)


def test_settings_save_event_copies_bounded_delete_intent_and_reply_metadata() -> None:
    settings = {"OPENAI_BASE_URL": "https://api.openai.com/v1/audio/speech"}
    deletes = ["openai_api_key"]
    reply_target = object()

    event = STTSSettingsSaveEvent(
        settings,
        delete_setting_keys=deletes,
        request_id=7,
        reply_to=reply_target,
    )
    settings["OPENAI_BASE_URL"] = "https://mutated.invalid"
    deletes.append("elevenlabs_api_key")

    assert event.settings == {
        "OPENAI_BASE_URL": "https://api.openai.com/v1/audio/speech"
    }
    assert event.delete_setting_keys == ("openai_api_key",)
    assert event.request_id == 7
    assert event.reply_to is reply_target


def test_configuration_changed_message_carries_optional_global_revision() -> None:
    legacy = STTSProviderConfigurationChanged("openai", 8)
    current = STTSProviderConfigurationChanged("openai", 8, 12)

    assert legacy.global_preferences_revision is None
    assert current.global_preferences_revision == 12
    assert current.provider_id == "openai"
    assert current.configuration_revision == 8


def test_successful_default_only_publication_posts_one_global_refresh_signal() -> None:
    posted: list[object] = []
    app = SimpleNamespace(post_message=posted.append)
    handler = STTSEventHandler(app)
    preferences = load_global_speech_tts_state({}).defaults.snapshot()
    service = SimpleNamespace(configuration_revision=Mock(return_value=41))
    publication = TTSSettingsPublication(
        generation=12,
        preferences=preferences,
        persistence=TTSSettingsPersistenceOutcome(True, True, None),
        provider_statuses={},
        provider_revisions={},
        published=True,
    )

    handler._post_applied_settings_changes(service, publication)

    assert len(posted) == 1
    message = posted[0]
    assert type(message) is STTSProviderConfigurationChanged
    assert message.provider_id == preferences.provider_id
    assert message.configuration_revision == 41
    assert message.global_preferences_revision == 12


def test_provider_switch_posts_one_refresh_bearing_signal() -> None:
    posted: list[object] = []
    handler = STTSEventHandler(SimpleNamespace(post_message=posted.append))
    preferences = TTSPreferencesSnapshot(
        provider_id="elevenlabs",
        model_mode="exact",
        model_id="eleven_multilingual_v2",
        voice_mode="exact",
        voice_id="rachel",
        response_format="mp3",
        speed=1.0,
    )
    publication = TTSSettingsPublication(
        generation=12,
        preferences=preferences,
        persistence=TTSSettingsPersistenceOutcome(True, True, None),
        provider_statuses={"openai": "applied"},
        provider_revisions={"openai": 41},
        published=True,
    )

    handler._post_applied_settings_changes(
        SimpleNamespace(configuration_revision=Mock(return_value=7)),
        publication,
    )

    assert len(posted) == 1
    assert posted[0].provider_id == "openai"
    assert posted[0].global_preferences_revision == 12


@pytest.mark.parametrize(
    ("status", "published", "expected_global_revision"),
    (("unavailable", False, None), ("applied", False, None)),
)
def test_failed_or_unpublished_handoff_never_claims_global_refresh(
    status: str,
    published: bool,
    expected_global_revision: None,
) -> None:
    posted: list[object] = []
    handler = STTSEventHandler(SimpleNamespace(post_message=posted.append))
    preferences = load_global_speech_tts_state({}).defaults.snapshot()
    publication = TTSSettingsPublication(
        generation=12,
        preferences=preferences,
        persistence=TTSSettingsPersistenceOutcome(True, True, None),
        provider_statuses={"openai": status},
        provider_revisions={"openai": 41},
        published=published,
    )

    handler._post_applied_settings_changes(SimpleNamespace(), publication)

    assert all(
        message.global_preferences_revision == expected_global_revision
        for message in posted
        if type(message) is STTSProviderConfigurationChanged
    )
    if status == "unavailable":
        assert posted == []


def test_handler_forwards_global_refresh_once_without_exposing_configuration() -> None:
    refreshes: list[STTSProviderConfigurationChanged] = []
    invalidations: list[tuple[str, int]] = []
    window = SimpleNamespace(
        receive_provider_configuration_changed=refreshes.append,
    )
    playground = SimpleNamespace(
        mark_provider_configuration_changed=(
            lambda provider_id, revision: invalidations.append((provider_id, revision))
        )
    )

    class _App:
        def query(self, selector: str) -> list[object]:
            return {
                "STTSWindow": [window],
                "SpeechPlaygroundPane": [playground],
            }.get(selector, [])

    event = STTSProviderConfigurationChanged("openai", 8, 12)
    STTSEventHandler(_App()).on_stts_provider_configuration_changed(event)

    assert refreshes == [event]
    assert invalidations == [("openai", 8)]
    assert vars(event) == {
        "provider_id": "openai",
        "configuration_revision": 8,
        "global_preferences_revision": 12,
    }


@pytest.mark.parametrize("request_id", (True, -1, "1"))
def test_settings_save_event_rejects_unbounded_request_ids(request_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        STTSSettingsSaveEvent({}, request_id=request_id)  # type: ignore[arg-type]


def test_settings_save_result_is_safe_immutable_and_separates_persistence() -> None:
    result = STTSSettingsSaveResult(
        request_id=7,
        persisted=True,
        provider_statuses={"openai": "unavailable"},
        failure_phase=None,
        provider_configuration_revisions={"openai": 4},
        provider_runtime_revisions={"openai": 9},
        defaults_activated=False,
    )

    assert result.persisted is True
    assert result.provider_statuses == {"openai": "unavailable"}
    assert result.provider_configuration_revisions == {"openai": 4}
    assert result.provider_runtime_revisions == {"openai": 9}
    assert result.failure_phase is None
    assert result.defaults_activated is False
    with pytest.raises(TypeError):
        result.provider_statuses["openai"] = "applied"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.provider_configuration_revisions["openai"] = 5  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.persisted = False  # type: ignore[misc]


def test_voice_setup_save_intent_is_explicit_and_bounded() -> None:
    preferences = load_global_speech_tts_state({}).defaults.snapshot()

    event = STTSSettingsSaveEvent(
        {"OPENAI_BASE_URL": "http://127.0.0.1:8765"},
        preferences=preferences,
        commit_defaults_after_handoff=True,
    )

    assert event.commit_defaults_after_handoff is True

    with pytest.raises(TypeError):
        STTSSettingsSaveEvent(
            {},
            commit_defaults_after_handoff=1,  # type: ignore[arg-type]
        )


def test_every_global_provider_mutation_targets_exactly_its_adapter() -> None:
    for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
        original = load_global_speech_tts_state({})
        draft = load_global_speech_tts_state({})
        if provider_id == "audio_cpp":
            draft.providers[provider_id]["connect_timeout_seconds"] = 9.0
        elif provider_id == "openai":
            draft.providers[provider_id]["organization_id"] = "org-new"
        elif provider_id == "elevenlabs":
            draft.providers[provider_id]["stability"] = 0.7
        elif provider_id == "kokoro":
            draft.providers[provider_id]["use_onnx"] = False
        elif provider_id == "chatterbox":
            draft.providers[provider_id]["temperature"] = 0.7
        elif provider_id == "higgs":
            draft.providers[provider_id]["dtype"] = "float32"
        else:
            draft.providers[provider_id]["server_url"] = "http://127.0.0.1:7852"

        proposal = build_global_speech_tts_save_proposal(
            original,
            draft,
            configure_provider=provider_id,
        )

        assert proposal.settings
        assert {
            _TTS_SETTING_BINDINGS[key].provider_id for key in proposal.settings
        } == {provider_id}


def test_credential_bindings_set_canonical_and_clear_all_local_aliases() -> None:
    assert _TTS_SETTING_BINDINGS["OPENAI_AUTH_MODE"].destinations == (
        ("app_tts", "OPENAI_AUTH_MODE"),
    )
    assert _TTS_SETTING_BINDINGS["OPENAI_AUTH_MODE"].provider_id == "openai"
    assert _TTS_SETTING_BINDINGS["OPENAI_NONE_HTTP_CONFIRMATION"].destinations == (
        ("app_tts", "OPENAI_NONE_HTTP_CONFIRMATION"),
    )
    assert _TTS_SETTING_BINDINGS["openai_api_key"].destinations == (
        ("api_settings.openai", "api_key"),
    )
    assert set(_TTS_SETTING_BINDINGS["openai_api_key"].delete_destinations) == {
        ("api_settings.openai", "api_key"),
        ("openai_api", "api_key"),
        ("API", "openai_api_key"),
    }
    assert _TTS_SETTING_BINDINGS["elevenlabs_api_key"].destinations == (
        ("api_settings.elevenlabs", "api_key"),
    )
    assert set(_TTS_SETTING_BINDINGS["elevenlabs_api_key"].delete_destinations) == {
        ("api_settings.elevenlabs", "api_key"),
        ("elevenlabs_api", "api_key"),
        ("API", "elevenlabs_api_key"),
    }


def test_studio_compatibility_keys_do_not_request_adapter_reconfiguration() -> None:
    assert {
        _TTS_SETTING_BINDINGS[key].provider_id
        for key in LAB_STUDIO_COMPATIBILITY_SETTING_KEYS
    } == {None}
