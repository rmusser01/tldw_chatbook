"""`dictation.model` -- a dictation-specific fast model default.

Measured live, on real hardware, against the same 1s "console stop" WAV
(warm, on a loaded machine): the transcription stack's own default
faster-whisper model, `distil-large-v3`, took 11.47s to transcribe it --
commands feel dead, and (before Fix 2 in this same change) there was zero
feedback while it ran. `base` measured 1.43s on the identical WAV,
transcribing it correctly. `resolve()` used to always inherit
`transcription.default_model` for dictation; this file pins the new
dictation-specific default and the explicit override that can still opt
back into any model, including the slow one, on purpose.

New file rather than extending `Tests/Chat/test_console_voice_input.py`:
that file's `resolve()` tests are a tight, deliberately-worded pinning suite
(see e.g. `test_resolve_fallback_prefers_the_first_declared_provider`'s own
docstring) for a different axis of `resolve()` (provider selection); keeping
the model-default tests separate avoids fighting over shared fixture setup
with whichever task's tests land there next.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat import console_voice_input as cvi

pytestmark = pytest.mark.unit


def _stub_settings(monkeypatch, values: dict[str, object]) -> None:
    """Route console_voice_input's config reads through a dict.

    Mirrors `Tests/Chat/test_console_voice_input.py`'s helper of the same
    name and shape.
    """

    def fake_get(section, key=None, default=None):
        if key is not None and not isinstance(key, str):
            default = key
            key = None
        lookup = section if key is None else f"{section}.{key}"
        return values.get(lookup, default)

    monkeypatch.setattr(cvi, "get_cli_setting", fake_get)


def test_explicit_dictation_model_wins_for_the_fast_default_provider(monkeypatch):
    """`dictation.model` beats the dictation-specific fast default outright."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "transcription.default_model": "distil-large-v3",
            "dictation.model": "tiny",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.model == "tiny"
    assert effective.model_overridden_for_dictation is True


def test_explicit_dictation_model_wins_even_for_a_non_fast_provider(monkeypatch):
    """The override is provider-agnostic: it wins regardless of which provider resolved."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("parakeet-mlx",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "parakeet-mlx",
            "transcription.default_model": "v2",
            "dictation.model": "v3",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "parakeet-mlx"
    assert effective.model == "v3"
    assert effective.model_overridden_for_dictation is True


def test_unset_dictation_model_defaults_to_the_fast_model_for_faster_whisper(
    monkeypatch,
):
    """No `dictation.model` and the resolved provider is faster-whisper -> the fast default.

    This is the RED case this task exists to fix: before it, an unset
    `dictation.model` meant `resolve()` always inherited
    `transcription.default_model` (here, the slow `distil-large-v3`) for
    dictation too.
    """
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "transcription.default_model": "distil-large-v3",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.model == cvi.DICTATION_FAST_MODEL_DEFAULT
    assert effective.model != "distil-large-v3"
    assert effective.model_overridden_for_dictation is True


def test_unset_dictation_model_defaults_to_the_fast_model_when_faster_whisper_is_the_fallback(
    monkeypatch,
):
    """The fast default keys off the RESOLVED provider, not the configured one.

    `configured` names an unavailable provider, so `resolve()` falls back to
    the first installed one (`faster-whisper` here) -- the fast default must
    still apply to that resolved choice.
    """
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "parakeet-mlx",
            "transcription.default_model": "distil-large-v3",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.was_overridden is True
    assert effective.model == cvi.DICTATION_FAST_MODEL_DEFAULT
    assert effective.model_overridden_for_dictation is True


def test_unset_dictation_model_leaves_other_providers_unchanged(monkeypatch):
    """A resolved provider that is NOT faster-whisper keeps the old behavior."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("parakeet-mlx",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "parakeet-mlx",
            "transcription.default_model": "v2",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "parakeet-mlx"
    assert effective.model == "v2"
    assert effective.model_overridden_for_dictation is False


def test_unset_dictation_model_and_unset_transcription_model_stays_none_for_other_providers(
    monkeypatch,
):
    """Neither key set, non-fast provider: `model` stays `None`, exactly as before."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("parakeet-mlx",))
    _stub_settings(
        monkeypatch, {"transcription.default_provider": "parakeet-mlx"}
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.model is None
    assert effective.model_overridden_for_dictation is False


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_dictation_model_is_treated_as_unset(monkeypatch, blank):
    """A blank/whitespace-only override falls back to the fast default, not itself."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "dictation.model": blank,
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.model == cvi.DICTATION_FAST_MODEL_DEFAULT


def test_non_string_dictation_model_warns_and_falls_back(monkeypatch):
    """Same warn+fallback shape as this module's other config readers."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "dictation.model": 12345,
        },
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        cvi.logger, "warning", lambda *args, **kwargs: warnings.append(str(args))
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.model == cvi.DICTATION_FAST_MODEL_DEFAULT
    assert warnings, "a non-string dictation.model must be logged, not silently ignored"


def test_dictation_model_override_reader_direct_unit_coverage(monkeypatch):
    """Direct coverage of `_dictation_model_override`, independent of `resolve()`."""
    _stub_settings(monkeypatch, {})
    assert cvi._dictation_model_override() is None

    _stub_settings(monkeypatch, {"dictation.model": "  tiny  "})
    assert cvi._dictation_model_override() == "tiny"

    _stub_settings(monkeypatch, {"dictation.model": "   "})
    assert cvi._dictation_model_override() is None
