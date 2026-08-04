"""Config readers and engine-resolution logic for the realtime voice engine
(V4 task 1). See `.superpowers/sdd/2026-08-04-realtime-voice-engine/
task-1-brief.md`.

Follows the exact monkeypatch style of
`Tests/UI/test_console_hands_free_wiring.py:83` (`_spy_get_cli_setting`):
patch `console_voice_input.get_cli_setting` and assert each reader passes the
exact ``(section, key, default)`` triple, mirroring `handsfree_send_delay_
seconds`'s sibling-validation shape (invalid values log + fall back to the
default, never raise).
"""

from __future__ import annotations

import tldw_chatbook.Chat.console_voice_input as cvi


def _patch_setting(monkeypatch, mapping):
    """Install a fake `get_cli_setting` that records every call.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        mapping: `(section, key) -> value` overrides; anything not present
            falls through to the caller's own default.

    Returns:
        The list of `(section, key, default)` triples the fake was called
        with, in call order.
    """
    calls = []

    def fake(section, key, default=None):
        calls.append((section, key, default))
        return mapping.get((section, key), default)

    monkeypatch.setattr(cvi, "get_cli_setting", fake)
    return calls


# ---------------------------------------------------------------------------
# realtime_enabled
# ---------------------------------------------------------------------------


def test_realtime_enabled_reads_exact_key_and_defaults_false(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_enabled() is False
    assert ("realtime", "enabled", False) in calls


def test_realtime_enabled_accepts_truthy_string(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "enabled"): "true"})
    assert cvi.realtime_enabled() is True


def test_realtime_enabled_accepts_falsy_string(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "enabled"): "off"})
    assert cvi.realtime_enabled() is False


def test_realtime_enabled_accepts_bool_true(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "enabled"): True})
    assert cvi.realtime_enabled() is True


# ---------------------------------------------------------------------------
# realtime_provider
# ---------------------------------------------------------------------------


def test_realtime_provider_reads_exact_key_and_defaults(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_provider() == "openai"
    assert ("realtime", "provider", "openai") in calls


def test_realtime_provider_accepts_configured_value(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "provider"): "azure"})
    assert cvi.realtime_provider() == "azure"


# ---------------------------------------------------------------------------
# realtime_model
# ---------------------------------------------------------------------------


def test_realtime_model_reads_exact_key_and_defaults(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_model() == "gpt-realtime"
    assert ("realtime", "model", "gpt-realtime") in calls


def test_realtime_model_accepts_configured_value(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "model"): "gpt-realtime-mini"})
    assert cvi.realtime_model() == "gpt-realtime-mini"


# ---------------------------------------------------------------------------
# realtime_voice
# ---------------------------------------------------------------------------


def test_realtime_voice_reads_exact_key_and_defaults_none(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_voice() is None
    assert ("realtime", "voice", None) in calls


def test_realtime_voice_accepts_configured_value(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "voice"): "marin"})
    assert cvi.realtime_voice() == "marin"


# ---------------------------------------------------------------------------
# realtime_idle_timeout_seconds
# ---------------------------------------------------------------------------


def test_idle_timeout_reads_exact_key_and_defaults(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_idle_timeout_seconds() == 300.0
    assert ("realtime", "idle_timeout_minutes", 5) in calls


def test_idle_timeout_converts_minutes_and_rejects_nonpositive(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): 2})
    assert cvi.realtime_idle_timeout_seconds() == 120.0
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): -3})
    assert cvi.realtime_idle_timeout_seconds() == 300.0
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): "soon"})
    assert cvi.realtime_idle_timeout_seconds() == 300.0


def test_idle_timeout_rejects_zero(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): 0})
    assert cvi.realtime_idle_timeout_seconds() == 300.0


# ---------------------------------------------------------------------------
# handsfree_engine
# ---------------------------------------------------------------------------


def test_handsfree_engine_reads_exact_key_and_defaults_auto(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.handsfree_engine() == "auto"
    assert ("dictation", "handsfree_engine", "auto") in calls


def test_handsfree_engine_accepts_pipeline(monkeypatch):
    _patch_setting(monkeypatch, {("dictation", "handsfree_engine"): "pipeline"})
    assert cvi.handsfree_engine() == "pipeline"


def test_handsfree_engine_accepts_realtime(monkeypatch):
    _patch_setting(monkeypatch, {("dictation", "handsfree_engine"): "realtime"})
    assert cvi.handsfree_engine() == "realtime"


def test_handsfree_engine_rejects_unknown_values(monkeypatch):
    _patch_setting(monkeypatch, {("dictation", "handsfree_engine"): "hyperspace"})
    assert cvi.handsfree_engine() == "auto"


# ---------------------------------------------------------------------------
# resolve_handsfree_engine
# ---------------------------------------------------------------------------


def test_resolve_engine_matrix(monkeypatch):
    for engine, enabled, expect in [
        ("auto", False, "pipeline"),
        ("auto", True, "realtime"),
        ("pipeline", True, "pipeline"),
        ("realtime", False, "realtime"),
    ]:
        _patch_setting(
            monkeypatch,
            {
                ("dictation", "handsfree_engine"): engine,
                ("realtime", "enabled"): enabled,
            },
        )
        assert cvi.resolve_handsfree_engine() == expect, (engine, enabled)


def test_resolve_engine_forced_realtime_ignores_realtime_enabled(monkeypatch):
    # Forcing "realtime" while not realtime_enabled() still returns
    # "realtime" -- the wiring toasts and refuses there. This reader stays
    # a pure combination, not a gate.
    _patch_setting(
        monkeypatch,
        {
            ("dictation", "handsfree_engine"): "realtime",
            ("realtime", "enabled"): False,
        },
    )
    assert cvi.resolve_handsfree_engine() == "realtime"
