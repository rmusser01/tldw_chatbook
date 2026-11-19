"""TASK-33641.11: a recorded credential_source decides the loader's bridge.

``config.py::_normalize_legacy_provider_api_key`` builds the key the nine
bridged cloud providers' legacy ``<provider>_api`` dict holds, and copies a
legacy ``[API] <provider>_api_key`` into ``[api_settings.<provider>]``. With a
``credential_source`` recorded it now resolves as readiness does (ADR-012): a
key the user cleared ("none") or replaced with a variable ("environment") is
neither copied nor used. Real loader on a scratch config.
"""

from __future__ import annotations

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

_ENV_KEY = "sk-env-canary-33641-11"
_PROVIDERS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _load(tmp_path, monkeypatch, provider, source, table_key, legacy_key, exported):
    env_var = _PROVIDERS[provider]
    for name in _PROVIDERS.values():
        monkeypatch.delenv(name, raising=False)
    if exported:
        monkeypatch.setenv(env_var, _ENV_KEY)
    lines = ["[API]"]
    if legacy_key:
        lines.append(f'{provider}_api_key = "sk-legacy"')
    lines += [f"[api_settings.{provider}]", f'api_key_env_var = "{env_var}"']
    if source:
        lines.append(f'credential_source = "{source}"')
    if table_key:
        lines.append('api_key = "sk-table"')
    path = tmp_path / "legacy-credential-source.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    fresh = install_config_source(monkeypatch)
    fresh.load_cli_config_and_ensure_existence(force_reload=True)
    return fresh.load_settings(force_reload=True)


@pytest.mark.parametrize("source", ["none", "environment"])
@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_a_cleared_or_replaced_key_is_not_copied_from_legacy_api(
    tmp_path, monkeypatch, provider, source
):
    """AC#1: after Clear ("none") or an Env var choice with the variable
    unset ("environment"), the ``[API]`` key reaches neither the provider's
    table nor its legacy dict, and readiness resolves no key either."""
    settings = _load(tmp_path, monkeypatch, provider, source, False, True, False)

    assert "api_key" not in settings["api_settings"][provider]
    assert settings[f"{provider}_api"]["api_key"] is None
    assert provider not in settings["legacy_api_key_providers"]
    assert get_provider_readiness(provider, settings).api_key is None


#: Every key location for each source: the table, legacy [API] and the
#: variable the table names, each present or not.
_LOCATIONS = [
    (table, legacy, exported)
    for table in (False, True)
    for legacy in (False, True)
    for exported in (False, True)
]


@pytest.mark.parametrize(
    ("table_key", "legacy_key", "exported"),
    _LOCATIONS,
    ids=[
        "-".join(
            name for name, present in (("table", t), ("api", a), ("env", e)) if present
        )
        or "no-key"
        for t, a, e in _LOCATIONS
    ],
)
@pytest.mark.parametrize("source", [None, "none", "stored", "environment"])
def test_the_legacy_dict_holds_the_key_readiness_resolves(
    tmp_path, monkeypatch, source, table_key, legacy_key, exported
):
    """AC#3 (and AC#2 for the no-source rows): the key in ``openai_api`` --
    what the chat handlers that read the legacy dict spend -- is the key
    readiness resolves, for every source and key location."""
    settings = _load(
        tmp_path, monkeypatch, "openai", source, table_key, legacy_key, exported
    )

    readiness_key = get_provider_readiness("openai", settings).api_key
    assert settings["openai_api"]["api_key"] == readiness_key
