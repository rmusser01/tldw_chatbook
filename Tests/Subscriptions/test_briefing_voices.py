"""Tests for resolving a briefing script's roster to concrete voices.

`briefing_voices` gives phase 2a's dormant `voice_profile_id` roster field
its meaning: turning a stored, stringified profile UUID into a concrete
`VoiceSelection` Task 5 can synthesize from. The only faked seam is the
profile service (one async `get_profile`) -- mirroring `briefing_cast`'s
own rule of faking exactly one collaborator; the profiles it hands back
are real `TTSGenerationProfile` instances, so this suite also exercises
the real freeze/validate path those profiles go through.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from tldw_chatbook.Subscriptions.briefing_voices import (
    VoiceResolutionError,
    VoiceSelection,
    dump_voice_snapshot,
    resolve_roster_voices,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_service import LoadedTTSProfile
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

pytestmark = pytest.mark.unit

_CREATED_AT = datetime(2026, 7, 31, 12, tzinfo=UTC)
_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_OTHER_PROFILE_ID = UUID("22222222-2222-4222-8222-222222222222")


def _profile(
    *,
    profile_id: UUID = _PROFILE_ID,
    display_name: str = "Narrator",
    provider_id: str = "audio_cpp",
    model_id: str = "model-a",
    voice_id: str | None = "voice-a",
    response_format: str = "wav",
    speed: float = 1.0,
    options: dict[str, Any] | None = None,
    revision: int = 3,
) -> TTSGenerationProfile:
    """Build a real, fully-validated `TTSGenerationProfile` fixture."""

    return TTSGenerationProfile(
        profile_id=profile_id,
        display_name=display_name,
        normalized_name=display_name.casefold(),
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        options={} if options is None else options,
        revision=revision,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


class _FakeProfileService:
    """The one faked seam: mirrors `TTSProfileService.get_profile` exactly.

    Raises `ProfileRepositoryError("missing")` for an unknown id, matching
    the real service's own contract (`TTS/profile_service.py`), so
    `resolve_roster_voices`'s error-mapping is exercised against the real
    error shape it will see in production, not a test-only stand-in.
    """

    def __init__(self, profiles: dict[UUID, TTSGenerationProfile]) -> None:
        self._profiles = profiles
        self.calls: list[UUID] = []

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        self.calls.append(profile_id)
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProfileRepositoryError("missing")
        return LoadedTTSProfile(repository_generation=1, profile=profile)


class _ExplodingProfileService:
    """A profile service whose `get_profile` fails in an unexpected way."""

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        raise RuntimeError(f"boom {profile_id}")


def _roster_entry(
    *,
    name: str = "Host",
    voice_profile_id: str | None = str(_PROFILE_ID),
    character_card_id: int | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "role_prompt": "",
        "character_card_id": character_card_id,
        "voice_profile_id": voice_profile_id,
        "character_name": None,
    }


@pytest.mark.asyncio
async def test_resolves_a_speaker_with_a_valid_voice_profile_id() -> None:
    profile = _profile(voice_id="voice-a", speed=1.0, revision=6)
    service = _FakeProfileService({_PROFILE_ID: profile})

    selections = await resolve_roster_voices(
        [_roster_entry(name="Host")],
        profile_service=service,
    )

    assert selections == [
        VoiceSelection(
            speaker="Host",
            provider_id="audio_cpp",
            model_id="model-a",
            voice_id="voice-a",
            response_format="wav",
            speed=1.0,
            options={},
            profile_id=str(_PROFILE_ID),
            profile_revision=6,
        )
    ]
    assert service.calls == [_PROFILE_ID]


@pytest.mark.asyncio
async def test_response_format_is_forced_to_wav_regardless_of_the_profile() -> None:
    # A legacy (non-audio_cpp) provider so a non-wav response_format is
    # even legal on the stored profile in the first place -- audio_cpp
    # profiles are constrained to wav already, which would make this
    # assertion trivially true rather than a real test of the override.
    profile = _profile(
        provider_id="kokoro",
        response_format="mp3",
        speed=1.25,
    )
    service = _FakeProfileService({_PROFILE_ID: profile})

    [selection] = await resolve_roster_voices(
        [_roster_entry()],
        profile_service=service,
    )

    assert selection.response_format == "wav"
    assert selection.provider_id == "kokoro"
    assert selection.speed == 1.25


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_value", (None, ""), ids=("none", "empty-string"))
async def test_no_voice_profile_id_raises_naming_the_speaker(
    missing_value: str | None,
) -> None:
    service = _FakeProfileService({})

    with pytest.raises(VoiceResolutionError, match="Host") as caught:
        await resolve_roster_voices(
            [_roster_entry(name="Host", voice_profile_id=missing_value)],
            profile_service=service,
        )

    assert "no voice profile" in str(caught.value)
    assert service.calls == []


@pytest.mark.asyncio
async def test_deleted_profile_raises_naming_speaker_and_id() -> None:
    service = _FakeProfileService({})  # empty store: the id never resolves

    with pytest.raises(VoiceResolutionError) as caught:
        await resolve_roster_voices(
            [_roster_entry(name="Narrator", voice_profile_id=str(_PROFILE_ID))],
            profile_service=service,
        )

    message = str(caught.value)
    assert "Narrator" in message
    assert str(_PROFILE_ID) in message
    assert service.calls == [_PROFILE_ID]


@pytest.mark.asyncio
async def test_malformed_profile_id_raises_naming_the_speaker() -> None:
    service = _FakeProfileService({})

    with pytest.raises(VoiceResolutionError, match="Host") as caught:
        await resolve_roster_voices(
            [_roster_entry(name="Host", voice_profile_id="not-a-uuid")],
            profile_service=service,
        )

    assert "not-a-uuid" in str(caught.value)
    # A malformed id must fail before ever reaching the service.
    assert service.calls == []


@pytest.mark.asyncio
async def test_unbound_profile_service_raises_naming_the_speaker() -> None:
    with pytest.raises(VoiceResolutionError, match="Host") as caught:
        await resolve_roster_voices(
            [_roster_entry(name="Host")],
            profile_service=None,
        )

    assert "no voice service" in str(caught.value)


@pytest.mark.asyncio
async def test_unexpected_resolution_failure_is_wrapped_naming_the_speaker() -> None:
    with pytest.raises(VoiceResolutionError, match="Host"):
        await resolve_roster_voices(
            [_roster_entry(name="Host")],
            profile_service=_ExplodingProfileService(),
        )


@pytest.mark.asyncio
async def test_resolves_multiple_speakers_in_roster_order_and_stops_at_first_failure() -> (
    None
):
    service = _FakeProfileService({_PROFILE_ID: _profile()})
    roster = [
        _roster_entry(name="Host", voice_profile_id=str(_PROFILE_ID)),
        _roster_entry(name="Guest", voice_profile_id=str(_OTHER_PROFILE_ID)),
        _roster_entry(name="Never Reached", voice_profile_id=str(_PROFILE_ID)),
    ]

    with pytest.raises(VoiceResolutionError, match="Guest"):
        await resolve_roster_voices(roster, profile_service=service)

    # The third speaker's id must never have been looked up.
    assert service.calls == [_PROFILE_ID, _OTHER_PROFILE_ID]


@pytest.mark.asyncio
async def test_resolves_all_speakers_when_every_voice_is_available() -> None:
    service = _FakeProfileService(
        {
            _PROFILE_ID: _profile(profile_id=_PROFILE_ID, display_name="Narrator"),
            _OTHER_PROFILE_ID: _profile(
                profile_id=_OTHER_PROFILE_ID,
                display_name="Guest voice",
                voice_id="voice-b",
            ),
        }
    )
    roster = [
        _roster_entry(name="Host", voice_profile_id=str(_PROFILE_ID)),
        _roster_entry(name="Guest", voice_profile_id=str(_OTHER_PROFILE_ID)),
    ]

    selections = await resolve_roster_voices(roster, profile_service=service)

    assert [selection.speaker for selection in selections] == ["Host", "Guest"]
    assert [selection.profile_id for selection in selections] == [
        str(_PROFILE_ID),
        str(_OTHER_PROFILE_ID),
    ]


def test_is_exact_provider_is_true_only_for_audio_cpp() -> None:
    exact = VoiceSelection(
        speaker="Host",
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id="voice-a",
        response_format="wav",
        speed=1.0,
        options={},
        profile_id=str(_PROFILE_ID),
        profile_revision=1,
    )
    legacy = VoiceSelection(
        speaker="Host",
        provider_id="kokoro",
        model_id="model-a",
        voice_id="voice-a",
        response_format="wav",
        speed=1.0,
        options={},
        profile_id=str(_PROFILE_ID),
        profile_revision=1,
    )

    assert exact.is_exact_provider() is True
    assert legacy.is_exact_provider() is False


def test_dump_voice_snapshot_round_trips_and_includes_profile_revision() -> None:
    selections = [
        VoiceSelection(
            speaker="Host",
            provider_id="audio_cpp",
            model_id="model-a",
            voice_id="voice-a",
            response_format="wav",
            speed=1.0,
            options={},
            profile_id=str(_PROFILE_ID),
            profile_revision=6,
        ),
        VoiceSelection(
            speaker="Guest",
            provider_id="kokoro",
            model_id="model-b",
            voice_id=None,
            response_format="wav",
            speed=1.25,
            options={"stability": 0.5},
            profile_id=str(_OTHER_PROFILE_ID),
            profile_revision=2,
        ),
    ]

    dumped = dump_voice_snapshot(selections)
    restored = json.loads(dumped)

    assert restored == [
        {
            "speaker": "Host",
            "provider_id": "audio_cpp",
            "model_id": "model-a",
            "voice_id": "voice-a",
            "response_format": "wav",
            "speed": 1.0,
            "options": {},
            "profile_id": str(_PROFILE_ID),
            "profile_revision": 6,
        },
        {
            "speaker": "Guest",
            "provider_id": "kokoro",
            "model_id": "model-b",
            "voice_id": None,
            "response_format": "wav",
            "speed": 1.25,
            "options": {"stability": 0.5},
            "profile_id": str(_OTHER_PROFILE_ID),
            "profile_revision": 2,
        },
    ]
    assert all("profile_revision" in entry for entry in restored)


def test_dump_voice_snapshot_is_deterministic_across_equal_calls() -> None:
    selections = [
        VoiceSelection(
            speaker="Host",
            provider_id="audio_cpp",
            model_id="model-a",
            voice_id="voice-a",
            response_format="wav",
            speed=1.0,
            options={"b": 1, "a": 2},
            profile_id=str(_PROFILE_ID),
            profile_revision=1,
        )
    ]

    first = dump_voice_snapshot(selections)
    second = dump_voice_snapshot(list(selections))

    assert first == second
    # sort_keys=True: key order in the source dict must not affect output.
    assert '"a": 2' in first.replace(" ", "") or '"a":2' in first.replace(" ", "")


@pytest.mark.asyncio
async def test_resolved_selection_carries_the_profiles_frozen_options() -> None:
    profile = _profile(
        provider_id="kokoro",
        response_format="mp3",
        options={"stability": 0.4, "tags": ["warm", "slow"]},
    )
    service = _FakeProfileService({_PROFILE_ID: profile})

    [selection] = await resolve_roster_voices(
        [_roster_entry()],
        profile_service=service,
    )

    dumped = dump_voice_snapshot([selection])
    restored = json.loads(dumped)
    assert restored[0]["options"] == {"stability": 0.4, "tags": ["warm", "slow"]}
