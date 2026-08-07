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
