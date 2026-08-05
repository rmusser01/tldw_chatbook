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


# ---------------------------------------------------------------------------
# Turn detection (gate round 5)
#
# Probed live against the GA endpoint before any of this shipped
# (`Tests/LLM_Calls/openai_realtime_turn_detection_probe.py`): both modes
# are accepted, and the server_vad knobs are accepted ONLY for server_vad
# (`semantic_vad` + `threshold` is rejected `unknown_parameter`). The
# default is `semantic_vad` because the server's own server_vad defaults
# commit a turn after 200 ms of silence, which is what was chopping the
# owner's speech into fragments for whisper-1 to hallucinate from.
# ---------------------------------------------------------------------------


def test_turn_detection_defaults_to_semantic_vad(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_turn_detection() == "semantic_vad"
    assert ("realtime", "turn_detection", "semantic_vad") in calls


def test_turn_detection_accepts_server_vad(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "turn_detection"): "server_vad"})
    assert cvi.realtime_turn_detection() == "server_vad"


def test_turn_detection_normalizes_case_and_whitespace(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "turn_detection"): "  Server_VAD "})
    assert cvi.realtime_turn_detection() == "server_vad"


def test_turn_detection_rejects_unknown_values(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "turn_detection"): "psychic_vad"})
    assert cvi.realtime_turn_detection() == "semantic_vad"


def test_vad_threshold_is_unset_by_default(monkeypatch):
    """Unset means "let the provider decide" -- this app does not restate
    the provider's own default, which would freeze it at today's value."""
    _patch_setting(monkeypatch, {})
    assert cvi.realtime_vad_threshold() is None


def test_vad_threshold_accepts_the_configured_value(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "vad_threshold"): 0.6})
    assert cvi.realtime_vad_threshold() == 0.6


def test_vad_threshold_rejects_out_of_range_and_non_numeric(monkeypatch):
    for bad in (1.5, -0.1, "loud"):
        _patch_setting(monkeypatch, {("realtime", "vad_threshold"): bad})
        assert cvi.realtime_vad_threshold() is None, bad


def test_vad_silence_ms_is_unset_by_default(monkeypatch):
    _patch_setting(monkeypatch, {})
    assert cvi.realtime_vad_silence_ms() is None


def test_vad_silence_ms_accepts_a_positive_int(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "vad_silence_ms"): 700})
    assert cvi.realtime_vad_silence_ms() == 700


def test_vad_silence_ms_rejects_non_positive_and_non_numeric(monkeypatch):
    for bad in (0, -200, "later"):
        _patch_setting(monkeypatch, {("realtime", "vad_silence_ms"): bad})
        assert cvi.realtime_vad_silence_ms() is None, bad
