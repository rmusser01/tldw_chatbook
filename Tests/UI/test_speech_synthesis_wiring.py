"""Generate must reach the shared synthesis path.

The button existing and being above the fold is not the same as it working:
before `SpeechActionStrip`, it rendered perfectly and its handler could never
match its id. These tests press it and assert something downstream happened.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID
from hashlib import sha256

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_synthesis_mixin import SpeechSynthesisMixin
from tldw_chatbook.TTS.effective_settings import (
    TTSSelectionOverrides,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.profile_service import TTSPlaygroundSelectionPreset
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.TTS import CanonicalTTSCloneReference, STTSPlaygroundCloneSnapshot


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(provider="audio_cpp")


@pytest.mark.unit
def test_synthesis_comes_from_the_shared_mixin():
    """The 322-line generate path is inherited, not copied into the pane.

    It lived in the legacy widget until that was retired; keeping it in the
    mixin is what let the pane adopt it whole rather than reimplementing it,
    and is what the remaining Speech surfaces will inherit in turn."""
    assert issubclass(SpeechPlaygroundPane, SpeechSynthesisMixin)
    assert (
        SpeechPlaygroundPane._generate_tts is SpeechSynthesisMixin._generate_tts
    ), "the pane redefined generate instead of inheriting it"


@pytest.mark.asyncio
async def test_pressing_generate_invokes_the_synthesis_path(monkeypatch):
    """The wiring this whole phase turns on.

    Asserts the press reaches `_generate_tts` -- not that synthesis
    succeeds, which needs a configured provider.
    """
    called: list[bool] = []
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_generate_tts",
        lambda self: called.append(True),
        raising=True,
    )

    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Enable it deliberately. With no catalog loaded the button is
        # CORRECTLY disabled -- generation cannot resolve a provider -- and
        # a disabled button swallows the press, so the test would pass or
        # fail on the button's state rather than on the wiring it is about.
        button = app.query_one("#tts-generate-btn")
        button.disabled = False
        button.press()
        await pilot.pause()

    assert called == [True], "Generate did not reach the synthesis path"


@pytest.mark.asyncio
async def test_the_synthesis_state_exists_before_any_press():
    """`_generate_tts` reads these four with no guard; a host that forgets
    `init_synthesis_state` fails with AttributeError mid-generation."""
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        for attribute in (
            "reference_audio_path",
            "higgs_reference_audio_path",
            "_provider_ids",
            "_generation_operation_id",
        ):
            assert hasattr(pane, attribute), attribute


def test_reference_profile_preview_builds_only_a_path_free_request_token() -> None:
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="clone-model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
        repository_generation=7,
        profile_revision=4,
    )
    host = SimpleNamespace(_profile_preset=preset)

    preview = SpeechSynthesisMixin._profile_preview_for_request(host)

    assert preview.profile_id == preset.profile_id
    assert preview.repository_generation == preset.repository_generation
    assert preview.profile_revision == preset.profile_revision
    assert not hasattr(preview, "reference")
    assert not hasattr(preview, "wav_bytes")


def test_reference_free_profile_preview_builds_no_private_request_token() -> None:
    preset = TTSPlaygroundSelectionPreset(
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
        speed=1.0,
        options={},
        availability="available",
    )
    host = SimpleNamespace(_profile_preset=preset)

    assert SpeechSynthesisMixin._profile_preview_for_request(host) is None


def test_playground_request_builder_attaches_only_profile_preview_identity() -> None:
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="clone-model",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
        repository_generation=7,
        profile_revision=4,
    )
    host = object.__new__(SpeechSynthesisMixin)
    host._profile_preset = preset
    preferences = StudioTTSPreferencesSnapshot(revision=4)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=4,
        preview=True,
    )

    request = host._build_playground_request(
        operation_id="preview-op",
        provider="audio_cpp",
        model="clone-model",
        text="hello",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        studio_draft=draft,
        studio_preferences=preferences,
    )

    assert request.profile_preview is not None
    assert request.profile_preview.profile_id == preset.profile_id
    assert not hasattr(request.profile_preview, "reference")
    assert request.clone_audition is None


def test_playground_request_builder_attaches_exact_private_clone_snapshot() -> None:
    payload = b"RIFF"
    canonical = CanonicalTTSCloneReference(
        wav_bytes=payload,
        reference_text="Exact local transcript",
        sha256=sha256(payload).hexdigest(),
        byte_length=len(payload),
        duration_ms=1,
        sample_rate_hz=16_000,
        channels=1,
        sample_encoding="pcm_s16le",
    )
    snapshot = STTSPlaygroundCloneSnapshot(
        draft_revision=8,
        canonical_reference=canonical,
    )
    host = object.__new__(SpeechSynthesisMixin)
    host._profile_preset = None
    host._clone_audition_for_request = lambda provider, model: (
        snapshot if (provider, model) == ("audio_cpp", "clone-model") else None
    )
    preferences = StudioTTSPreferencesSnapshot(revision=4)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=4,
    )

    request = host._build_playground_request(
        operation_id="clone-op",
        provider="audio_cpp",
        model="clone-model",
        text="hello",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        studio_draft=draft,
        studio_preferences=preferences,
    )

    assert request.clone_audition is snapshot
    assert request.profile_preview is None
    assert "Exact local transcript" not in repr(request)
