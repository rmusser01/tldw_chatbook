"""Analysis provider resolution over the REAL ``load_settings()`` (TASK-34000.5).

``Tests/Library/test_ingest_analysis.py`` covers ``resolve_ingest_analysis_
provider`` against hand-built mappings. That is the half the review's L-02
could not see: the app hands the resolver ``load_settings()``'s dict, and
that dict never carried ``[analysis_defaults]`` over from the TOML, so the
Media Analysis tab said "No analysis provider is configured" for every user
with a ready provider (qa/notes-library-ux-review-2026-10-02/verify/nl-v-l-02/
08-probe-output.txt). These cases resolve over the loader's real output.

The absent / not-ready cases are the AC#3 fence: the fix must not introduce a
false "ready".
"""

from __future__ import annotations

from pathlib import Path

from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Library.ingest_analysis import (
    NO_ANALYSIS_PROVIDER_REASON,
    resolve_ingest_analysis_provider,
)

READY_KEY = "sk-test-ready-0123456789abcdef"

ANALYSIS_DEFAULTS = '[analysis_defaults]\nprovider = "OpenAI"\nmodel = "gpt-4.1-mini"\n\n'
READY_CREDENTIAL = f'[api_settings.openai]\napi_key = "{READY_KEY}"\n'
PLACEHOLDER_CREDENTIAL = '[api_settings.openai]\napi_key = "<API_KEY_HERE>"\n'


def _load_settings_from(tmp_path: Path, monkeypatch, toml_text: str) -> dict:
    """``load_settings()`` over ``toml_text`` via the fresh-module recipe.

    The shared ``tldw_chatbook.config`` module is bound to the test
    bootstrap profile by the time any test runs; a redirected
    ``load_settings(force_reload=True)`` on it is refused by the config
    participant (``RecoveryRequired(raw_source_selection_changed)``).
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    fresh = install_config_source(monkeypatch)
    settings = fresh.load_settings(force_reload=True)
    assert settings["COMPREHENSIVE_CONFIG_RAW"].get("api_settings"), (
        "the temp TOML did not reach load_settings()"
    )
    return settings


def test_ready_provider_named_in_analysis_defaults_resolves_ready(tmp_path, monkeypatch):
    """AC#5: the TOML names a ready provider -> resolution over load_settings() is ready."""
    settings = _load_settings_from(
        tmp_path, monkeypatch, ANALYSIS_DEFAULTS + READY_CREDENTIAL
    )

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert resolution.ready, (
        f"not ready: {resolution.short_reason!r}; "
        f"'analysis_defaults' in settings = {'analysis_defaults' in settings}"
    )
    assert resolution.provider == "OpenAI"
    assert resolution.dispatch_name == "openai"
    assert resolution.model == "gpt-4.1-mini"
    assert resolution.api_key == READY_KEY
    assert resolution.short_reason == ""


def test_empty_provider_in_analysis_defaults_still_reports_unavailable(
    tmp_path, monkeypatch
):
    """AC#3: [analysis_defaults] names no provider -> unavailable, no-provider reason."""
    settings = _load_settings_from(
        tmp_path,
        monkeypatch,
        '[analysis_defaults]\nprovider = ""\n\n' + READY_CREDENTIAL,
    )

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert not resolution.ready
    assert resolution.short_reason == NO_ANALYSIS_PROVIDER_REASON
    assert resolution.api_key is None


def test_absent_analysis_defaults_without_credential_still_reports_unavailable(
    tmp_path, monkeypatch
):
    """AC#3: no user [analysis_defaults] and no credential -> unavailable.

    The loader merges the shipped default config, whose ``[analysis_defaults]``
    names OpenAI (config.py ``CONFIG_TOML_CONTENT``), so "absent from the
    user's file" resolves to that default provider -- which must still be
    unavailable until its credential is ready.
    """
    settings = _load_settings_from(tmp_path, monkeypatch, PLACEHOLDER_CREDENTIAL)

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert not resolution.ready
    assert resolution.provider == "OpenAI"  # the shipped default, not a user choice
    assert resolution.short_reason != NO_ANALYSIS_PROVIDER_REASON
    assert resolution.api_key is None


def test_absent_analysis_defaults_with_ready_credential_uses_shipped_default(
    tmp_path, monkeypatch
):
    """Documents the shipped default: no user table + a ready OpenAI key -> ready via OpenAI."""
    settings = _load_settings_from(tmp_path, monkeypatch, READY_CREDENTIAL)

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert resolution.ready, resolution.short_reason
    assert resolution.provider == "OpenAI"
    assert resolution.dispatch_name == "openai"


def test_not_ready_provider_in_analysis_defaults_still_reports_unavailable(
    tmp_path, monkeypatch
):
    """AC#3: the named provider has only a placeholder key -> unavailable, not a false ready."""
    settings = _load_settings_from(
        tmp_path, monkeypatch, ANALYSIS_DEFAULTS + PLACEHOLDER_CREDENTIAL
    )

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert not resolution.ready
    assert resolution.provider == "OpenAI"
    assert resolution.short_reason != NO_ANALYSIS_PROVIDER_REASON
    assert resolution.api_key is None


def test_analysis_defaults_call_shape_survives_load_settings(tmp_path, monkeypatch):
    """The full [analysis_defaults] call shape reaches the resolver unchanged."""
    toml_text = (
        '[analysis_defaults]\nprovider = "OpenAI"\nmodel = "gpt-4.1-mini"\n'
        "temperature = 0.2\ntop_p = 0.5\nmin_p = 0.01\nmax_tokens = 123\n"
        'system_prompt = "Summarize tersely."\n\n' + READY_CREDENTIAL
    )
    settings = _load_settings_from(tmp_path, monkeypatch, toml_text)

    resolution = resolve_ingest_analysis_provider(settings, environ={})

    assert resolution.ready, resolution.short_reason
    assert (resolution.temperature, resolution.top_p, resolution.min_p) == (0.2, 0.5, 0.01)
    assert resolution.max_tokens == 123
    assert resolution.system_prompt == "Summarize tersely."
