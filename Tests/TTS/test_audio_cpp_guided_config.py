"""Typed durable settings for guided audio.cpp setup."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig


def _api():
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppAcceptedPackage,
        AudioCppBackendPreference,
        AudioCppBinarySelectionSource,
        AudioCppManagedSetupSource,
        AudioCppRecipeOption,
        AudioCppSafeModelProjection,
        AudioCppSettingsConfig,
    )

    return (
        AudioCppAcceptedPackage,
        AudioCppBackendPreference,
        AudioCppBinarySelectionSource,
        AudioCppManagedSetupSource,
        AudioCppRecipeOption,
        AudioCppSafeModelProjection,
        AudioCppSettingsConfig,
    )


def _accepted_package(*, public_model_id: str = "supertonic-3-orig"):
    (
        AudioCppAcceptedPackage,
        _,
        _,
        _,
        _,
        AudioCppSafeModelProjection,
        _,
    ) = _api()
    return AudioCppAcceptedPackage(
        package_uuid="d3f6d610-6fd9-4cde-9ea7-cc5175ca445b",
        recipe_id="audio-cpp-0.5.1.supertonic.supertonic_3_orig",
        recipe_revision=1,
        package_variant="supertonic_3_orig",
        public_model_id=public_model_id,
        canonical_root="/models/Supertonic-3-GGUF",
        canonical_root_identity="1" * 64,
        configuration_identity="2" * 64,
        weight_identity="3" * 64,
        projection=AudioCppSafeModelProjection(
            family="supertonic",
            task="tts",
            mode="offline",
            model_relative_path="supertonic-3-orig.gguf",
        ),
    )


def _managed_identity():
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )

    return AudioCppManagedArtifactIdentity(
        artifact_id="audio-cpp-supertonic-3-orig",
        revision=AUDIO_CPP_ARTIFACT_COMMIT,
        variant="orig",
    )


def test_legacy_accepted_package_json_shape_and_bytes_remain_unchanged() -> None:
    accepted = _accepted_package()
    expected_json = (
        '{"package_uuid":"d3f6d610-6fd9-4cde-9ea7-cc5175ca445b",'
        '"recipe_id":"audio-cpp-0.5.1.supertonic.supertonic_3_orig",'
        '"recipe_revision":1,"package_variant":"supertonic_3_orig",'
        '"public_model_id":"supertonic-3-orig",'
        '"canonical_root":"/models/Supertonic-3-GGUF",'
        '"canonical_root_identity":"1111111111111111111111111111111111111111111111111111111111111111",'
        '"configuration_identity":"2222222222222222222222222222222222222222222222222222222222222222",'
        '"weight_identity":"3333333333333333333333333333333333333333333333333333333333333333",'
        '"projection":{"family":"supertonic","task":"tts","mode":"offline",'
        '"model_relative_path":"supertonic-3-orig.gguf",'
        '"model_spec_override_relative_path":null,"busy_timeout_ms":null,'
        '"load_options":[],"session_options":[]}}'
    )

    serialized = accepted.model_dump_json()

    assert accepted.managed_artifact is None
    assert serialized == expected_json
    assert '"managed_artifact":' not in serialized


def test_exact_managed_artifact_identity_round_trips_frozen() -> None:
    AudioCppAcceptedPackage, *_ = _api()
    identity = _managed_identity()
    accepted = _accepted_package().model_copy(update={"managed_artifact": identity})

    dumped = accepted.model_dump(mode="json")
    reparsed = AudioCppAcceptedPackage.model_validate(dumped)

    assert reparsed == accepted
    assert reparsed.managed_artifact == identity
    assert dumped["managed_artifact"] == {
        "artifact_id": identity.artifact_id,
        "revision": identity.revision,
        "variant": identity.variant,
    }
    with pytest.raises(ValidationError):
        identity.variant = "changed"  # type: ignore[misc]


def test_persisted_managed_identity_must_still_match_the_exact_recipe() -> None:
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    AudioCppAcceptedPackage, *_ = _api()
    values = _accepted_package().model_dump(mode="json")
    values["managed_artifact"] = _managed_identity().model_dump(mode="json")
    values["managed_artifact"]["artifact_id"] = "audio-cpp-supertonic-3-f16"
    accepted = AudioCppAcceptedPackage.model_validate(values)

    with pytest.raises(ValueError, match="requires recipe review"):
        AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(accepted)


@pytest.mark.parametrize(
    "managed_artifact",
    (
        {},
        {"artifact_id": "audio-cpp-supertonic-3-orig"},
        {
            "artifact_id": "audio-cpp-supertonic-3-orig",
            "revision": "597048d9a920592808d7d4e2acd7b9c4596a143a",
        },
        {
            "artifact_id": "",
            "revision": "597048d9a920592808d7d4e2acd7b9c4596a143a",
            "variant": "orig",
        },
        {
            "artifact_id": "Audio-Cpp-Supertonic",
            "revision": "597048d9a920592808d7d4e2acd7b9c4596a143a",
            "variant": "orig",
        },
        {
            "artifact_id": "audio-cpp-supertonic-3-orig",
            "revision": "main",
            "variant": "orig/escape",
        },
        {
            "artifact_id": "audio-cpp-supertonic-3-orig",
            "revision": "597048d9a920592808d7d4e2acd7b9c4596a143a",
            "variant": "orig",
            "recipe_id": "unexpected",
        },
    ),
)
def test_partial_or_malformed_managed_identity_is_rejected_boundedly(
    managed_artifact: dict[str, object],
) -> None:
    AudioCppAcceptedPackage, *_ = _api()
    values = _accepted_package().model_dump(mode="json")
    values["managed_artifact"] = managed_artifact

    with pytest.raises(ValidationError) as raised:
        AudioCppAcceptedPackage.model_validate(values)

    assert "/models/Supertonic-3-GGUF" not in str(raised.value)


@pytest.mark.parametrize(
    "component",
    (
        "a",
        "a.b-c_d",
        "a" * 1024,
        "",
        "-a",
        "a-",
        "A",
        "a/b",
        "con",
        "com1.txt",
        "nul",
    ),
)
def test_managed_identity_components_match_artifact_ref_semantics(
    component: str,
) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )

    def accepts(factory: type) -> bool:
        try:
            factory(
                artifact_id=component,
                revision=component,
                variant=component,
            )
        except (TypeError, ValueError):
            return False
        return True

    assert accepts(ArtifactRef) == accepts(AudioCppManagedArtifactIdentity)


def test_existing_external_and_managed_json_mappings_keep_legacy_runtime_meaning() -> (
    None
):
    *_, AudioCppManagedSetupSource, _, _, AudioCppSettingsConfig = _api()
    external = {
        "mode": "external",
        "base_url": "http://127.0.0.1:9000",
        "connect_timeout_seconds": 7.0,
    }
    manual = {
        "mode": "managed",
        "managed_binary_path": "/opt/audio.cpp/audiocpp_server",
        "managed_server_json_path": "/opt/audio.cpp/server.json",
        "managed_startup_timeout_seconds": 12.0,
    }

    external_settings = AudioCppSettingsConfig.from_mapping(external)
    manual_settings = AudioCppSettingsConfig.from_mapping(manual)

    assert (
        external_settings.managed_setup_source is AudioCppManagedSetupSource.USER_JSON
    )
    assert manual_settings.managed_setup_source is AudioCppManagedSetupSource.USER_JSON
    assert AudioCppConfig.from_mapping(external).to_mapping() == {
        "mode": "external",
        "base_url": "http://127.0.0.1:9000",
        "connect_timeout_seconds": 7.0,
        "synthesis_timeout_seconds": 600.0,
        "max_input_characters": 10_000,
        "max_response_bytes": 128 * 1024 * 1024,
        "max_metadata_bytes": 1024 * 1024,
        "max_catalog_models": 1000,
        "max_voices_per_model": 1000,
        "max_identifier_characters": 256,
    }
    assert (
        AudioCppConfig.from_mapping(manual).managed_server_json_path
        == manual["managed_server_json_path"]
    )


def test_full_guided_and_dormant_values_round_trip_defensively() -> None:
    (
        _,
        AudioCppBackendPreference,
        AudioCppBinarySelectionSource,
        AudioCppManagedSetupSource,
        _,
        _,
        AudioCppSettingsConfig,
    ) = _api()
    package = _accepted_package()
    raw = {
        "mode": "external",
        "base_url": "https://external.example.test",
        "managed_setup_source": "guided",
        "managed_binary_path": "/manual/audiocpp_server",
        "managed_server_json_path": "/manual/server.json",
        "guided_binary_path": "/guided/audiocpp_server",
        "guided_binary_source": "homebrew",
        "guided_packages": [package.model_dump(mode="json")],
        "guided_default_model_id": package.public_model_id,
        "guided_backend_preference": "metal",
        "guided_device": 1,
        "guided_threads": 6,
        "guided_max_request_body_bytes": 64 * 1024 * 1024,
        "guided_busy_timeout_ms": 90_000,
        "managed_startup_timeout_seconds": 45.0,
        "managed_health_check_interval_seconds": 15.0,
        "managed_termination_grace_seconds": 8.0,
        "connect_timeout_seconds": 4.0,
        "synthesis_timeout_seconds": 120.0,
        "max_input_characters": 2_000,
        "max_response_bytes": 32 * 1024 * 1024,
        "max_metadata_bytes": 512 * 1024,
        "max_catalog_models": 12,
        "max_voices_per_model": 24,
        "max_identifier_characters": 128,
    }

    settings = AudioCppSettingsConfig.from_mapping(raw)
    dumped = settings.to_mapping()
    reparsed = AudioCppSettingsConfig.from_mapping(dumped)

    assert settings == reparsed
    assert settings.managed_setup_source is AudioCppManagedSetupSource.GUIDED
    assert settings.guided_binary_source is AudioCppBinarySelectionSource.HOMEBREW
    assert settings.guided_backend_preference is AudioCppBackendPreference.METAL
    assert settings.managed_binary_path == "/manual/audiocpp_server"
    assert settings.base_url == "https://external.example.test"
    dumped["guided_packages"][0]["public_model_id"] = "mutated"  # type: ignore[index]
    assert settings.guided_packages[0].public_model_id == "supertonic-3-orig"


def test_unknown_runtime_and_extension_fields_are_not_retained() -> None:
    *_, AudioCppSettingsConfig = _api()
    settings = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:8080",
            "runtime_port": 5555,
            "pid": 123,
            "health": "running",
            "headers": {"Authorization": "secret"},
        }
    )

    dumped = settings.to_mapping()

    assert not {"runtime_port", "pid", "health", "headers"} & dumped.keys()
    with pytest.raises(ValidationError):
        AudioCppSettingsConfig(runtime_port=5555)  # type: ignore[call-arg]


def test_safe_projection_is_frozen_bounded_and_has_no_arbitrary_json() -> None:
    (
        _,
        _,
        _,
        _,
        AudioCppRecipeOption,
        AudioCppSafeModelProjection,
        _,
    ) = _api()
    projection = AudioCppSafeModelProjection(
        family="pocket_tts",
        task="tts",
        mode="offline",
        model_relative_path=None,
        load_options=(AudioCppRecipeOption(name="language", value="english"),),
        session_options=(AudioCppRecipeOption(name="language", value="english"),),
        busy_timeout_ms=60_000,
    )

    assert projection.load_options[0].value == "english"
    with pytest.raises(ValidationError):
        projection.family = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        AudioCppSafeModelProjection(
            family="pocket_tts",
            task="tts",
            mode="offline",
            model_relative_path="../escape.gguf",
        )
    with pytest.raises(ValidationError):
        AudioCppSafeModelProjection(
            family="pocket_tts",
            task="tts",
            mode="offline",
            shell="curl example.test",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("guided_backend_preference", "best"),
        ("guided_device", -1),
        ("guided_device", True),
        ("guided_threads", 0),
        ("guided_threads", True),
        ("guided_max_request_body_bytes", 0),
        ("guided_max_request_body_bytes", 2**53),
        ("guided_busy_timeout_ms", -1),
        ("guided_busy_timeout_ms", 2**31),
    ),
)
def test_guided_compute_and_server_limits_are_bounded(
    field: str, value: object
) -> None:
    *_, AudioCppSettingsConfig = _api()
    values = AudioCppSettingsConfig().to_mapping()
    values[field] = value

    with pytest.raises(ValueError):
        AudioCppSettingsConfig.from_mapping(values)


def test_guided_default_must_name_one_unique_accepted_package() -> None:
    *_, AudioCppSettingsConfig = _api()
    first = _accepted_package(public_model_id="voice")
    duplicate = _accepted_package(public_model_id="voice")
    base = AudioCppSettingsConfig().to_mapping()

    duplicate_values = deepcopy(base)
    duplicate_values["guided_packages"] = [
        first.model_dump(mode="json"),
        duplicate.model_dump(mode="json"),
    ]
    duplicate_values["guided_default_model_id"] = "voice"
    with pytest.raises(ValueError, match="unique"):
        AudioCppSettingsConfig.from_mapping(duplicate_values)

    missing_default = deepcopy(base)
    missing_default["guided_packages"] = [first.model_dump(mode="json")]
    missing_default["guided_default_model_id"] = "other"
    with pytest.raises(ValueError, match="default"):
        AudioCppSettingsConfig.from_mapping(missing_default)


def test_guided_packages_require_unique_internal_uuids() -> None:
    *_, AudioCppSettingsConfig = _api()
    first = _accepted_package(public_model_id="first")
    second = first.model_copy(
        update={
            "public_model_id": "second",
            "canonical_root": "/models/other",
            "canonical_root_identity": "4" * 64,
            "configuration_identity": "5" * 64,
            "weight_identity": "6" * 64,
        }
    )
    values = AudioCppSettingsConfig().to_mapping()
    values["guided_packages"] = [
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
    ]

    with pytest.raises(ValueError, match="internal UUIDs"):
        AudioCppSettingsConfig.from_mapping(values)


def test_accepted_package_requires_absolute_root_safe_ids_and_digest_identities() -> (
    None
):
    AudioCppAcceptedPackage, *_ = _api()
    accepted = _accepted_package()

    with pytest.raises(ValidationError):
        accepted.public_model_id = "changed"  # type: ignore[misc]
    for changes in (
        {"package_uuid": "not-a-uuid"},
        {"package_uuid": "D3F6D610-6FD9-4CDE-9EA7-CC5175CA445B"},
        {"canonical_root": "relative/model"},
        {"canonical_root": "/models/with\ncontrol"},
        {"public_model_id": "bad model id"},
        {"weight_identity": "not-a-digest"},
        {"recipe_revision": 0},
    ):
        values = accepted.model_dump(mode="python")
        values.update(changes)
        with pytest.raises(ValidationError):
            AudioCppAcceptedPackage(**values)
