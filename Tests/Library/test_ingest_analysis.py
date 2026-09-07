"""Tests for the Library ingest analysis-provider resolution seam (task-3301).

``resolve_ingest_analysis_provider`` is the one place the Analyze-after-import
option's provider + credential are resolved: it reads the incumbent
``[analysis_defaults] provider`` config (the Media analysis feature's own
default) and runs it through ``Chat/provider_readiness.get_provider_readiness``
-- the same single definition of "ready" Console uses -- so the ingest
panel's hint, the job's skip reason, and the actual analysis call can never
disagree about the same config.
"""

from __future__ import annotations

from tldw_chatbook.Library.ingest_analysis import (
    resolve_ingest_analysis_provider,
)


def test_no_provider_configured_is_not_ready() -> None:
    resolution = resolve_ingest_analysis_provider({}, environ={})

    assert resolution.ready is False
    assert resolution.provider == ""
    assert resolution.api_key is None
    assert "provider" in resolution.short_reason
    assert resolution.hint  # a full sentence for the panel


def test_config_key_resolves_ready() -> None:
    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True
    assert resolution.provider == "OpenAI"
    assert resolution.api_key == "sk-test-configured"
    assert resolution.short_reason == ""
    assert resolution.hint == ""


def test_env_key_resolves_ready() -> None:
    config = {"analysis_defaults": {"provider": "OpenAI"}}

    resolution = resolve_ingest_analysis_provider(
        config, environ={"OPENAI_API_KEY": "sk-test-env"}
    )

    assert resolution.ready is True
    assert resolution.api_key == "sk-test-env"


def test_placeholder_key_is_not_ready() -> None:
    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "<API_KEY_HERE>"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is False
    assert resolution.api_key is None
    assert "OpenAI" in resolution.short_reason
    assert "OpenAI" in resolution.hint


def test_keyless_local_provider_is_ready_without_key() -> None:
    config = {"analysis_defaults": {"provider": "Ollama"}}

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True
    assert resolution.provider == "Ollama"
    assert resolution.api_key is None
    assert resolution.short_reason == ""


def test_non_mapping_config_degrades_to_unconfigured() -> None:
    resolution = resolve_ingest_analysis_provider(None, environ={})

    assert resolution.ready is False
    assert "provider" in resolution.short_reason


# ---------------------------------------------------------------------------
# task-3301 xhigh review round (F5): a "ready" resolution must name a
# provider the chat dispatcher actually accepts, or come back not-ready
# with a reason -- never a runtime "Error: Invalid API Name".
# ---------------------------------------------------------------------------


def test_ready_resolution_carries_a_dispatchable_name() -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True
    assert resolution.dispatch_name == "openai"
    assert resolution.dispatch_name in API_CALL_HANDLERS


def test_display_name_variants_normalize_to_dispatch_names() -> None:
    """Provider spellings the config/UI can hold whose plain ``.lower()``
    is NOT a dispatch key -- the exact F5 failure class."""
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    expected = {
        "MistralAI": "mistralai",
        "KoboldCpp": "koboldcpp",
        "Oobabooga": "oobabooga",
        "Aphrodite": "aphrodite",
        "MLX-LM": "mlx_lm",
        "Local-LLM": "local-llm",
    }
    for display, dispatch in expected.items():
        config = {
            "analysis_defaults": {"provider": display},
            "api_settings": {display.lower(): {"api_key": "sk-test"}},
        }
        resolution = resolve_ingest_analysis_provider(config, environ={})
        assert resolution.ready is True, display
        assert resolution.dispatch_name == dispatch, display
        assert resolution.dispatch_name in API_CALL_HANDLERS, display


def test_ready_but_undispatchable_provider_is_rejected_with_reason() -> None:
    """``custom`` is keyless-known to the readiness gate but has no chat
    dispatch handler: the seam must reject it, not let it error at runtime."""
    config = {"analysis_defaults": {"provider": "custom"}}

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is False
    assert "not supported for ingest analysis" in resolution.short_reason
    assert resolution.hint


def test_every_readiness_ready_provider_dispatches_or_is_rejected() -> None:
    """Pin BOTH universes from code: every provider name the readiness gate
    can mark ready (its own known-provider keys, plus an arbitrary name made
    ready by a configured credential) either resolves to a name in
    ``API_CALL_HANDLERS`` or is rejected by the seam with a reason."""
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.Chat.provider_readiness import KNOWN_PROVIDER_KEYS

    candidates = set(KNOWN_PROVIDER_KEYS) | {"a-made-up-provider"}
    for provider in sorted(candidates):
        config = {
            "analysis_defaults": {"provider": provider},
            # A configured credential makes ANY provider readiness-ready,
            # so this exercises the seam's own dispatchability constraint.
            "api_settings": {provider: {"api_key": "sk-test-universe"}},
        }
        resolution = resolve_ingest_analysis_provider(config, environ={})
        if resolution.ready:
            assert resolution.dispatch_name in API_CALL_HANDLERS, provider
        else:
            assert "not supported for ingest analysis" in resolution.short_reason, (
                provider
            )


# ---------------------------------------------------------------------------
# task-3301 xhigh review round (F10): the resolution carries the full
# [analysis_defaults] call shape, defaults mirroring the Media viewer.
# ---------------------------------------------------------------------------


def test_resolution_carries_full_call_settings() -> None:
    config = {
        "analysis_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "top_p": 0.9,
            "min_p": 0.01,
            "max_tokens": 512,
            "system_prompt": "Analyze thoroughly.",
        },
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.model == "gpt-4o-mini"
    assert resolution.temperature == 0.2
    assert resolution.top_p == 0.9
    assert resolution.min_p == 0.01
    assert resolution.max_tokens == 512
    assert resolution.system_prompt == "Analyze thoroughly."


def test_call_setting_defaults_mirror_the_media_viewer() -> None:
    from tldw_chatbook.Library.ingest_analysis import (
        ANALYSIS_DEFAULT_MAX_TOKENS,
        ANALYSIS_DEFAULT_MIN_P,
        ANALYSIS_DEFAULT_SYSTEM_PROMPT,
        ANALYSIS_DEFAULT_TEMPERATURE,
        ANALYSIS_DEFAULT_TOP_P,
    )

    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.model is None
    assert resolution.temperature == ANALYSIS_DEFAULT_TEMPERATURE == 0.7
    assert resolution.top_p == ANALYSIS_DEFAULT_TOP_P == 0.95
    assert resolution.min_p == ANALYSIS_DEFAULT_MIN_P == 0.05
    assert resolution.max_tokens == ANALYSIS_DEFAULT_MAX_TOKENS == 4096
    assert resolution.system_prompt == ANALYSIS_DEFAULT_SYSTEM_PROMPT


def test_display_string_settings_are_coerced() -> None:
    """Persisted configs may hold display strings; the seam types them."""
    config = {
        "analysis_defaults": {
            "provider": "OpenAI",
            "temperature": "0.3",
            "max_tokens": "1024",
        },
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.temperature == 0.3
    assert resolution.max_tokens == 1024


def test_keyless_ready_resolution_sets_keyless_flag() -> None:
    config = {"analysis_defaults": {"provider": "Ollama"}}

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True
    assert resolution.api_key is None
    assert resolution.keyless is True
    assert resolution.dispatch_name == "ollama"


def test_keyed_ready_resolution_is_not_keyless() -> None:
    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.keyless is False


# --- task-28007 AC#5: the reason both the control and the guard speak -------


def test_a_ready_resolution_has_no_unavailable_reason() -> None:
    """The seam's "" contract: callers gate on truthiness, not on ``ready``."""
    from tldw_chatbook.Library.ingest_analysis import analysis_unavailable_reason

    config = {
        "analysis_defaults": {"provider": "OpenAI"},
        "api_settings": {"openai": {"api_key": "sk-test-configured"}},
    }
    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True
    assert analysis_unavailable_reason(resolution) == ""


def test_every_not_ready_shape_yields_a_capitalised_sentence() -> None:
    """All three not-ready branches the resolver can actually produce, read
    off the resolver itself rather than hand-built fixtures -- a copy change
    to ``short_reason`` must not silently ship a lowercase fragment."""
    from tldw_chatbook.Library.ingest_analysis import analysis_unavailable_reason

    # 1. Nothing configured at all.
    none_configured = resolve_ingest_analysis_provider({}, environ={})
    # 2. Configured and credentialled, but no chat handler can dispatch it (F5).
    unsupported = resolve_ingest_analysis_provider(
        {
            "analysis_defaults": {"provider": "local_onnx"},
            "api_settings": {"local_onnx": {"api_key": "sk-anything"}},
        },
        environ={},
    )
    # 3. Configured and dispatchable, but no credential.
    unready = resolve_ingest_analysis_provider(
        {"analysis_defaults": {"provider": "OpenAI"}}, environ={}
    )

    for resolution in (none_configured, unsupported, unready):
        assert resolution.ready is False, resolution
        sentence = analysis_unavailable_reason(resolution)
        assert sentence, resolution
        assert sentence[0].isupper(), sentence
        assert sentence.endswith("."), sentence

    assert (
        analysis_unavailable_reason(none_configured)
        == "No analysis provider is configured."
    )
    assert "local_onnx" in analysis_unavailable_reason(unsupported)
    assert analysis_unavailable_reason(unready).startswith("OpenAI is not ready")


def test_a_blank_short_reason_falls_back_instead_of_raising() -> None:
    """This is a public seam other gates feed resolutions into, so a blank
    (or whitespace-only) ``short_reason`` must degrade to the generic reason
    rather than raising IndexError off ``reason[0]``."""
    from tldw_chatbook.Library.ingest_analysis import (
        IngestAnalysisResolution,
        analysis_unavailable_reason,
    )

    for blank in ("", "   ", "\n\t"):
        resolution = IngestAnalysisResolution(
            provider="Whatever",
            api_key=None,
            ready=False,
            short_reason=blank,
            hint="",
        )
        assert (
            analysis_unavailable_reason(resolution)
            == "No analysis provider is configured."
        ), blank
