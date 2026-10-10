"""TASK-1240: the app records that it started and that it stopped cleanly.

Also TASK-2110: the `speech_stack_available` startup diagnostic must not
report `status=ok` when the configured STT provider was unavailable and the
resolver substituted another.
"""

from __future__ import annotations

import pytest

from Tests.app_module_patches import set_app_global

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
# TASK-2110: this boot test predates the `bootstrap_profile` opt-in
# (TASK-32873) and failed with RecoveryRequired("raw_source_selection_changed")
# whenever the file ran standalone -- the app import happens inside the test,
# under the per-test config redirect, while the config participant was bound
# at conftest bootstrap time. The marker is the established fix for exactly
# this signature; verified red-without/green-with at base before applying.
@pytest.mark.bootstrap_profile
async def test_mounting_the_app_records_app_started(monkeypatch):
    """Boot the real app and assert the event fired.

    Asserted by capturing the call rather than scanning source: this repo has
    been burned by name-matching guards that pass against unwired code.
    """
    from Tests.UI.app_factory import _build_test_app

    recorded: list[tuple[str, str]] = []
    set_app_global(monkeypatch, "persist_event",
        lambda component, event, **fields: recorded.append((component, event)),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    assert ("app", "app_started") in recorded
    assert ("app", "app_stopping") in recorded


def _capture_persist_events(monkeypatch) -> list[tuple[str, str, dict]]:
    """Capture every `persist_event` call with its fields."""
    recorded: list[tuple[str, str, dict]] = []
    set_app_global(
        monkeypatch,
        "persist_event",
        lambda component, event, **fields: recorded.append(
            (component, event, fields)
        ),
    )
    return recorded


def _force_webrtcvad(monkeypatch, *, present: bool) -> None:
    """Control only the `webrtcvad` lookup; delegate every other find_spec.

    `on_mount` imports `find_spec` from `importlib.util` at call time, so the
    module attribute patch reaches it. Delegation matters: app boot resolves
    many other modules through the same function.
    """
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "webrtcvad":
            return object() if present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _speech_stack_events(recorded):
    return [
        fields
        for component, event, fields in recorded
        if (component, event) == ("dictation", "speech_stack_available")
    ]


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_speech_stack_diagnostic_fallback_names_both_providers(monkeypatch):
    """TASK-2110 AC#1: a substituted provider must not report status=ok.

    The incident (2026-08-03 live gate): configured `parakeet-mlx`, a venv
    without parakeet_mlx, and the diagnostic read
    `event=speech_stack_available model=base provider=faster-whisper
    status=ok` -- indistinguishable from a healthy configuration. The
    effective stack must be reported as a fallback naming BOTH providers.
    """
    from tldw_chatbook.Chat import console_voice_input as cvi
    from tldw_chatbook.Chat.console_voice_input import EffectiveConfig

    from Tests.UI.app_factory import _build_test_app

    recorded = _capture_persist_events(monkeypatch)
    # `on_mount` imports `resolve` from console_voice_input at call time, so
    # patching the module attribute reaches the startup diagnostic.
    monkeypatch.setattr(
        cvi,
        "resolve",
        lambda: EffectiveConfig(
            provider="faster-whisper",
            model="base",
            language="en",
            configured_provider="parakeet-mlx",
            was_overridden=True,
        ),
    )
    _force_webrtcvad(monkeypatch, present=True)

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    events = _speech_stack_events(recorded)
    assert len(events) == 1, "startup must emit exactly one speech_stack_available"
    fields = events[0]
    assert fields["status"] != "ok"
    assert fields["status"] == "fallback"
    assert fields["provider"] == "faster-whisper"
    assert fields["configured_provider"] == "parakeet-mlx"


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_speech_stack_diagnostic_records_vad_and_substitution_together(
    monkeypatch,
):
    """Both degradations at once must both be recorded, not either/or.

    VAD missing is its own honesty signal (status=degraded today); a provider
    substitution on top of it must not be masked by the VAD half.
    """
    from tldw_chatbook.Chat import console_voice_input as cvi
    from tldw_chatbook.Chat.console_voice_input import EffectiveConfig

    from Tests.UI.app_factory import _build_test_app

    recorded = _capture_persist_events(monkeypatch)
    monkeypatch.setattr(
        cvi,
        "resolve",
        lambda: EffectiveConfig(
            provider="faster-whisper",
            model="base",
            language="en",
            configured_provider="parakeet-mlx",
            was_overridden=True,
        ),
    )
    _force_webrtcvad(monkeypatch, present=False)

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    events = _speech_stack_events(recorded)
    assert len(events) == 1
    fields = events[0]
    assert fields["status"] != "degraded", "VAD missing must not mask the fallback"
    assert fields["configured_provider"] == "parakeet-mlx"
    assert fields["provider"] == "faster-whisper"


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_speech_stack_diagnostic_healthy_path_is_unchanged(monkeypatch):
    """TASK-2110 AC#3: an available configured provider adds no noise.

    `status=ok`, the effective provider, and no substitution fields -- the
    healthy path keeps today's shape exactly.
    """
    from tldw_chatbook.Chat import console_voice_input as cvi
    from tldw_chatbook.Chat.console_voice_input import EffectiveConfig

    from Tests.UI.app_factory import _build_test_app

    recorded = _capture_persist_events(monkeypatch)
    monkeypatch.setattr(
        cvi,
        "resolve",
        lambda: EffectiveConfig(
            provider="parakeet-mlx",
            model=None,
            language="en",
            configured_provider="parakeet-mlx",
            was_overridden=False,
        ),
    )
    _force_webrtcvad(monkeypatch, present=True)

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    events = _speech_stack_events(recorded)
    assert len(events) == 1
    fields = events[0]
    assert fields["status"] == "ok"
    assert fields["provider"] == "parakeet-mlx"
    assert fields["model"] == "provider-default"
    assert "configured_provider" not in fields
