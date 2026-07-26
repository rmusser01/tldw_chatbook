"""Tests for immutable TTS generation-profile domain values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType
from uuid import uuid4

import pytest

import tldw_chatbook.TTS as tts_package
import tldw_chatbook.TTS.profile_errors as profile_errors
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileValidationError,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
    canonical_json_options,
)


def _draft(**overrides: object) -> TTSProfileDraft:
    values: dict[str, object] = {
        "display_name": "Profile",
        "provider_id": "openai",
        "model_id": "tts-1",
        "voice_id": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
    }
    values.update(overrides)
    return TTSProfileDraft(**values)  # type: ignore[arg-type]


def _profile(**overrides: object) -> TTSGenerationProfile:
    now = datetime(2026, 7, 26, tzinfo=UTC)
    values: dict[str, object] = {
        "profile_id": uuid4(),
        "display_name": "Profile",
        "normalized_name": "profile",
        "provider_id": "openai",
        "model_id": "tts-1",
        "voice_id": None,
        "response_format": "mp3",
        "speed": 1.0,
        "options": {},
        "revision": 1,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return TTSGenerationProfile(**values)  # type: ignore[arg-type]


def test_profile_name_uses_nfkc_casefold_uniqueness() -> None:
    first = _draft(display_name="  Café  ")
    second = _draft(display_name="CAFE\u0301")

    assert first.display_name == "Café"
    assert first.normalized_name == second.normalized_name == "café"


def test_profile_name_collides_for_non_ascii_casefold_equivalents() -> None:
    assert (
        _draft(display_name="Straße").normalized_name
        == _draft(display_name="STRASSE").normalized_name
    )


@pytest.mark.parametrize("display_name", ["   ", "a" * 129])
def test_profile_name_has_safe_bounds(display_name: str) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: display_name$"
    ):
        _draft(display_name=display_name)


@pytest.mark.parametrize(
    "character", ["\x00", "\u200e", "\ud800", "\ufdd0", "\U0001fffe"]
)
def test_profile_name_rejects_unsafe_unicode(character: str) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: display_name$"
    ):
        _draft(display_name=f"a{character}b")


@pytest.mark.parametrize("display_name", ["\tProfile", "Profile\n", "\x1cProfile\x1f"])
def test_profile_name_rejects_control_whitespace_before_trimming(
    display_name: str,
) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: display_name$"
    ):
        _draft(display_name=display_name)


@pytest.mark.parametrize("provider_id", ["OpenAI", "open-ai", "1openai", "a" * 65, ""])
def test_provider_identifier_must_be_canonical_lower_snake(provider_id: str) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: provider_id$"
    ):
        _draft(provider_id=provider_id)


def test_provider_identifier_is_preserved_without_normalization() -> None:
    draft = _draft(provider_id="audio_cpp", response_format="wav")

    assert draft.provider_id == "audio_cpp"


@pytest.mark.parametrize("field_name", ["model_id", "voice_id"])
@pytest.mark.parametrize("value", ["", "x" * 257])
def test_exact_model_and_voice_identifiers_have_bounds(
    field_name: str, value: str
) -> None:
    with pytest.raises(
        ProfileValidationError, match=rf"^TTS profile validation failed: {field_name}$"
    ):
        _draft(**{field_name: value})


def test_exact_model_and_voice_identifiers_remain_opaque() -> None:
    draft = _draft(model_id=" model/v1 ", voice_id=" M1 ")

    assert draft.model_id == " model/v1 "
    assert draft.voice_id == " M1 "
    assert _draft(voice_id=None).voice_id is None


@pytest.mark.parametrize(
    "speed", [True, "1", math.nan, math.inf, -math.inf, 0.24, 4.01]
)
def test_speed_rejects_values_outside_the_provider_neutral_contract(
    speed: object,
) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: speed$"
    ):
        _draft(speed=speed)


def test_speed_never_leaks_an_overflow_from_a_huge_integer() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: speed$"
    ):
        _draft(speed=10**10000)


@pytest.mark.parametrize("speed", [0.25, 4, 1])
def test_speed_is_float_in_the_inclusive_provider_neutral_range(
    speed: float | int,
) -> None:
    assert _draft(speed=speed).speed == float(speed)
    assert isinstance(_draft(speed=speed).speed, float)


@pytest.mark.parametrize(
    "response_format", ["", "   ", "1wav", "wave-format", "a" * 33]
)
def test_response_format_has_a_canonical_safe_identifier_contract(
    response_format: str,
) -> None:
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: response_format$",
    ):
        _draft(response_format=response_format)


def test_response_format_is_trimmed_and_lowercased() -> None:
    assert _draft(response_format="  WAV_16 ").response_format == "wav_16"


@pytest.mark.parametrize("revision", [True, 0, -1, 1.0, "1"])
def test_persisted_profile_revision_must_be_a_positive_integer(
    revision: object,
) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: revision$"
    ):
        _profile(revision=revision)


def test_character_ref_preserves_full_authority_scoped_identity() -> None:
    reference = CharacterRef(
        source="server", authority_id="account:42", character_id="card-1"
    )

    assert reference.source == "server"
    assert reference.authority_id == "account:42"
    assert reference.character_id == "card-1"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", "remote"),
        ("authority_id", ""),
        ("authority_id", " authority"),
        ("authority_id", "a" * 257),
        ("character_id", "\x00"),
        ("character_id", "b" * 257),
    ],
)
def test_character_ref_rejects_noncanonical_or_unbounded_identity(
    field_name: str, value: str
) -> None:
    values = {"source": "local", "authority_id": "main", "character_id": "card"}
    values[field_name] = value

    with pytest.raises(
        ProfileValidationError, match=rf"^TTS profile validation failed: {field_name}$"
    ):
        CharacterRef(**values)


@pytest.mark.parametrize("source", [[], {}])
def test_character_ref_never_leaks_unhashable_source_errors(source: object) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: source$"
    ):
        CharacterRef(source=source, authority_id="main", character_id="card")  # type: ignore[arg-type]


def test_options_are_canonical_json_and_are_defensively_frozen() -> None:
    source = {"z": [1, {"name": "Café"}], "a": True}
    draft = _draft(options=source)
    source["z"][1]["name"] = "changed"  # type: ignore[index]

    assert isinstance(draft.options, MappingProxyType)
    assert draft.options["z"] == (1, MappingProxyType({"name": "Café"}))
    assert (
        canonical_json_options({"z": [1, {"name": "Café"}], "a": True})
        == '{"a":true,"z":[1,{"name":"Café"}]}'
    )
    with pytest.raises(TypeError):
        draft.options["a"] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        draft.speed = 2.0


@pytest.mark.parametrize(
    "options", [b"{}", {"bad": {1, 2}}, {"bad": (1,)}, {1: "bad"}, {"bad": math.nan}]
)
def test_options_reject_non_json_input(options: object) -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(options=options)


def test_canonical_options_validator_rejects_caller_tuple_input() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        canonical_json_options({"bad": (1,)})


def test_canonical_options_validator_rejects_caller_mappingproxy_tuple_input() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        canonical_json_options(MappingProxyType({"bad": (1,)}))


def test_canonical_options_validator_reencodes_domain_frozen_arrays() -> None:
    draft = _draft(options={"items": ["first", "second"]})

    assert isinstance(draft.options["items"], tuple)
    assert canonical_json_options(draft.options) == '{"items":["first","second"]}'


def test_options_never_leak_a_caller_mapping_exception() -> None:
    class ExplosiveMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret mapping failure")

        def __iter__(self):
            return iter(("secret",))

        def __len__(self) -> int:
            return 1

        def items(self):
            raise RuntimeError("secret mapping failure")

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(options=ExplosiveMapping())


def test_options_never_leak_a_hostile_string_key_sort_error() -> None:
    class ExplosiveSortKey(str):
        def __lt__(self, other: object) -> bool:
            raise RuntimeError("secret sort failure")

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        canonical_json_options({"safe": False, ExplosiveSortKey("bad"): True})


def test_options_reject_scalar_subclasses_without_retaining_caller_state() -> None:
    class StringScalar(str):
        pass

    class IntegerScalar(int):
        pass

    class FloatScalar(float):
        pass

    for scalar in (StringScalar("value"), IntegerScalar(1), FloatScalar(1.0)):
        with pytest.raises(
            ProfileValidationError, match=r"^TTS profile validation failed: options$"
        ):
            _draft(options={"value": scalar})


def test_options_reject_mutable_hash_string_keys_before_retaining_them() -> None:
    class MutableHashKey(str):
        hash_value = 1

        def __hash__(self) -> int:
            return self.hash_value

    key = MutableHashKey("key")
    options = {key: "value"}
    MutableHashKey.hash_value = 2

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(options=options)


def test_options_normalize_second_serialization_failures() -> None:
    class StatefulSortKey(str):
        comparisons = 0

        def __lt__(self, other: object) -> bool:
            type(self).comparisons += 1
            if type(self).comparisons > 1:
                raise RuntimeError("secret second serialization failure")
            return super().__lt__(other)

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        canonical_json_options({"safe": False, StatefulSortKey("bad"): True})


def test_options_reject_excessive_nesting_and_canonical_size() -> None:
    nested: object = {"level": []}
    for _ in range(4):
        nested = {"level": nested}

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(options=nested)
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(options={"value": "é" * 8193})


@pytest.mark.parametrize(
    "overrides",
    [
        {"options": {"quality": "high"}},
        {"response_format": "mp3"},
        {"speed": 1.1},
    ],
)
def test_audio_cpp_profile_contract_is_stricter_than_provider_neutral_contract(
    overrides: dict[str, object],
) -> None:
    values = {"provider_id": "audio_cpp", "response_format": "wav", "speed": 1.0}
    values.update(overrides)

    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: audio_cpp$"
    ):
        _draft(**values)


def test_audio_cpp_profile_contract_accepts_only_exact_first_release_selection() -> (
    None
):
    draft = _draft(
        provider_id="audio_cpp", response_format=" WAV ", speed=1, options={}
    )

    assert draft.response_format == "wav"
    assert draft.speed == 1.0


def test_generation_profile_validates_identity_timestamps_and_normalized_name() -> None:
    profile_id = uuid4()
    profile = _profile(
        profile_id=profile_id, display_name="  Café ", normalized_name="café"
    )

    assert profile.profile_id == profile_id
    assert profile.display_name == "Café"
    assert profile.normalized_name == "café"
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: profile_id$"
    ):
        _profile(profile_id=str(profile_id))
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: normalized_name$",
    ):
        _profile(normalized_name="not-the-name")
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: created_at$"
    ):
        _profile(created_at=datetime(2026, 7, 26))
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: updated_at$"
    ):
        _profile(updated_at=datetime(2026, 7, 26, tzinfo=timezone(timedelta(hours=1))))
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: timestamps$"
    ):
        _profile(created_at=datetime(2026, 7, 27, tzinfo=UTC))


def test_generation_profile_rejects_hostile_non_string_normalized_names() -> None:
    class EqualsAnything:
        def __eq__(self, other: object) -> bool:
            return True

    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: normalized_name$",
    ):
        _profile(normalized_name=EqualsAnything())


def test_generation_profile_rejects_normalized_name_subclasses() -> None:
    class MutableNormalizedName(str):
        pass

    supplied = MutableNormalizedName("profile")
    with pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: normalized_name$",
    ):
        _profile(normalized_name=supplied)


def test_repository_values_are_frozen_and_minimal() -> None:
    profile = _profile()
    reference = CharacterRef(
        source="local", authority_id="library-1", character_id="card-1"
    )
    assignment = CharacterTTSAssignment(
        character_ref=reference, profile_id=profile.profile_id
    )
    snapshot = AssignedTTSProfileSnapshot(assignment=assignment, profile=profile)
    page = TTSProfilePage(profiles=[profile], total=1)
    result = ProfileStoreResult(generation=0, value=page)
    receipt_time = datetime(2026, 7, 26, tzinfo=UTC)
    backup = ProfileBackupReceipt(created_at=receipt_time, byte_count=1)
    restore = ProfileRestoreReceipt(
        restored_at=receipt_time, profile_count=1, assignment_count=1
    )

    assert snapshot.assignment == assignment
    assert snapshot.profile is profile
    assert page.profiles == (profile,)
    assert result.value is page
    assert backup.byte_count == 1
    assert restore.assignment_count == 1
    assert set(ProfileRepositoryState) == {
        ProfileRepositoryState.OPEN,
        ProfileRepositoryState.RESTORING,
        ProfileRepositoryState.UNAVAILABLE,
        ProfileRepositoryState.CLOSED,
    }
    with pytest.raises(FrozenInstanceError):
        assignment.profile_id = uuid4()


def test_restore_receipt_validates_its_own_timestamp_field() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: restored_at$"
    ):
        ProfileRestoreReceipt(
            restored_at=datetime(2026, 7, 26), profile_count=0, assignment_count=0
        )


def test_profile_page_never_leaks_an_invalid_profiles_container() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: profiles$"
    ):
        TTSProfilePage(profiles=None, total=0)  # type: ignore[arg-type]


def test_safe_profile_errors_never_include_caller_values() -> None:
    validation = ProfileValidationError("secret-name")
    repository = ProfileRepositoryError("secret-upstream-text")

    assert str(validation) == "TTS profile validation failed: options"
    assert str(repository) == "TTS profile repository failed: operation_failed"
    assert "secret-name" not in repr(validation)
    assert "secret-upstream-text" not in repr(repository)


def test_profile_error_base_cannot_be_constructed_with_arbitrary_payloads() -> None:
    assert not hasattr(tts_package, "ProfileError")
    assert not hasattr(profile_errors, "ProfileError")
