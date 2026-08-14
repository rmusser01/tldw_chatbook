"""Provider readiness tests for first-run Chat guidance."""

import os
from contextlib import contextmanager
from copy import deepcopy

import pytest

from tldw_chatbook import config as config_mod
from tldw_chatbook.Chat import provider_readiness as provider_readiness_module
from tldw_chatbook.Chat.provider_readiness import (
    ProviderReadiness,
    get_provider_readiness,
)


def test_key_required_provider_reports_missing_key_without_value_leakage():
    readiness = get_provider_readiness(
        "OpenAI",
        {
            "api_settings": {
                "openai": {
                    "api_key": "",
                    "api_key_env_var": "OPENAI_API_KEY",
                }
            }
        },
        environ={},
    )

    assert readiness == ProviderReadiness(
        provider="OpenAI",
        provider_key="openai",
        requires_api_key=True,
        ready=False,
        api_key=None,
        api_key_source=None,
        env_var="OPENAI_API_KEY",
        reason="Missing API key",
        recovery="Set OPENAI_API_KEY or add api_key under [api_settings.openai].",
    )
    assert "OPENAI_API_KEY" in readiness.user_message
    assert "api_settings.openai" in readiness.user_message
    assert "sk-" not in readiness.user_message


def test_key_required_provider_uses_environment_key_without_displaying_it():
    readiness = get_provider_readiness(
        "Anthropic",
        {"api_settings": {"anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"}}},
        environ={"ANTHROPIC_API_KEY": "sk-ant-secret"},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is True
    assert readiness.api_key == "sk-ant-secret"
    assert readiness.api_key_source == "env:ANTHROPIC_API_KEY"
    assert "sk-ant-secret" not in readiness.user_message


def test_key_required_provider_uses_standard_environment_key_when_config_only_has_model():
    readiness = get_provider_readiness(
        "Mistral",
        {"api_settings": {"mistral": {"model": "open-mistral-nemo"}}},
        environ={"MISTRAL_API_KEY": "mistral-secret"},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is True
    assert readiness.api_key == "mistral-secret"
    assert readiness.api_key_source == "env:MISTRAL_API_KEY"
    assert "mistral-secret" not in readiness.user_message


def test_mistralai_defaults_to_mistral_environment_key():
    readiness = get_provider_readiness(
        "MistralAI",
        {"api_settings": {"mistralai": {"model": "open-mistral-nemo"}}},
        environ={"MISTRAL_API_KEY": "mistral-secret"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "mistral-secret"
    assert readiness.env_var == "MISTRAL_API_KEY"


def test_placeholder_config_key_is_not_ready():
    readiness = get_provider_readiness(
        "OpenRouter",
        {
            "api_settings": {
                "openrouter": {
                    "api_key": "<API_KEY_HERE>",
                    "api_key_env_var": "OPENROUTER_API_KEY",
                }
            }
        },
        environ={},
    )

    assert readiness.ready is False
    assert readiness.api_key is None
    assert (
        readiness.recovery
        == "Set OPENROUTER_API_KEY or add api_key under [api_settings.openrouter]."
    )


@pytest.mark.parametrize(
    "value",
    ["", "<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key"],
)
def test_public_provider_api_key_validator_rejects_placeholder_values(value):
    assert provider_readiness_module.is_valid_provider_api_key(value) is False


def test_public_provider_api_key_validator_accepts_real_trimmed_key():
    assert (
        provider_readiness_module.is_valid_provider_api_key("  sk-real-key  ") is True
    )


def test_key_required_provider_names_are_case_insensitive():
    readiness = get_provider_readiness(
        "openai",
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.requires_api_key is True
    assert readiness.ready is False
    assert (
        readiness.recovery
        == "Set OPENAI_API_KEY or add api_key under [api_settings.openai]."
    )


def test_provider_settings_lookup_uses_normalized_config_key():
    readiness = get_provider_readiness(
        "Custom-2",
        {"api_settings": {"Custom-2": {"api_key": "configured-custom-key"}}},
        environ={},
    )

    assert readiness.provider_key == "custom_2"
    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key == "configured-custom-key"
    assert readiness.api_key_source == "config:api_settings.custom_2.api_key"


@pytest.mark.parametrize("canonical_first", [False, True])
def test_qwencloud_readiness_prefers_canonical_fields_over_normalized_alias(
    canonical_first,
):
    alias = {
        "api_key": "alias-secret",
        "api_key_env_var": "QWEN_ALIAS_API_KEY",
    }
    canonical = {"api_key": "canonical-secret"}
    entries = (
        [("qwencloud", canonical), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", canonical)]
    )

    readiness = get_provider_readiness(
        "QwenCloud",
        {"api_settings": dict(entries)},
        environ={"QWEN_ALIAS_API_KEY": "environment-secret"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "canonical-secret"
    assert readiness.api_key_source == "config:api_settings.qwencloud.api_key"


@pytest.mark.parametrize("canonical_first", [False, True])
def test_qwencloud_readiness_uses_alias_fields_missing_from_canonical(
    canonical_first,
):
    alias = {"api_key_env_var": "QWEN_ALIAS_API_KEY"}
    canonical = {"api_mode": "chat_completions"}
    entries = (
        [("qwencloud", canonical), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", canonical)]
    )

    readiness = get_provider_readiness(
        "QwenCloud",
        {"api_settings": dict(entries)},
        environ={"QWEN_ALIAS_API_KEY": "environment-secret"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "environment-secret"
    assert readiness.api_key_source == "env:QWEN_ALIAS_API_KEY"


@pytest.mark.parametrize("canonical_first", [False, True])
def test_qwencloud_malformed_canonical_table_fails_closed_without_alias_leakage(
    canonical_first,
):
    alias = {"api_key": "alias-secret-canary", "api_mode": "responses"}
    entries = (
        [("qwencloud", []), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", [])]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)

    with pytest.raises(ValueError) as exc_info:
        config_mod.provider_settings_for_key(source["api_settings"], "qwencloud")

    readiness = get_provider_readiness("QwenCloud", source, environ={})

    assert source == original
    assert "alias-secret-canary" not in str(exc_info.value)
    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.api_key_source is None
    assert readiness.reason == "Invalid provider settings"
    assert "api_settings.qwencloud" in readiness.user_message
    assert "alias-secret-canary" not in readiness.user_message


@pytest.mark.parametrize("canonical_first", [False, True])
def test_qwencloud_valid_canonical_table_ignores_malformed_alias(
    canonical_first,
):
    canonical = {"api_key": "canonical-secret", "api_mode": "responses"}
    entries = (
        [("qwencloud", canonical), ("QwenCloud", [])]
        if canonical_first
        else [("QwenCloud", []), ("qwencloud", canonical)]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)

    settings = config_mod.provider_settings_for_key(source["api_settings"], "qwencloud")
    readiness = get_provider_readiness("QwenCloud", source, environ={})

    assert source == original
    assert settings == canonical
    assert readiness.ready is True
    assert readiness.api_key == "canonical-secret"
    assert readiness.api_key_source == "config:api_settings.qwencloud.api_key"


def test_qwencloud_alias_only_malformed_table_fails_closed():
    source = {
        "api_settings": {
            "QwenCloud": ["secret-canary"],
            " QWENCLOUD ": {"api_key": "later-alias-secret-canary"},
        }
    }
    original = deepcopy(source)

    with pytest.raises(ValueError) as exc_info:
        config_mod.provider_settings_for_key(source["api_settings"], "qwencloud")

    readiness = get_provider_readiness("QwenCloud", source, environ={})

    assert source == original
    assert "secret-canary" not in str(exc_info.value)
    assert "later-alias-secret-canary" not in str(exc_info.value)
    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.reason == "Invalid provider settings"
    assert "api_settings.qwencloud" in readiness.user_message
    assert "later-alias-secret-canary" not in readiness.user_message


def test_qwencloud_multiple_aliases_keep_first_match_as_fallback():
    readiness = get_provider_readiness(
        "QwenCloud",
        {
            "api_settings": {
                "QwenCloud": {},
                "QWENCLOUD": {"api_key_env_var": "SECOND_ALIAS_API_KEY"},
                "qwencloud": {"api_mode": "responses"},
            }
        },
        environ={"SECOND_ALIAS_API_KEY": "must-not-win"},
    )

    assert readiness.ready is False
    assert readiness.env_var == "DASHSCOPE_API_KEY"


def test_non_qwen_provider_lookup_keeps_first_normalized_match():
    readiness = get_provider_readiness(
        "OpenAI",
        {
            "api_settings": {
                "OpenAI": {"api_key": "first-match-key"},
                "openai": {"api_key": "canonical-key"},
            }
        },
        environ={},
    )

    assert readiness.ready is True
    assert readiness.api_key == "first-match-key"


@pytest.mark.parametrize(
    ("provider", "provider_key", "model", "base_url", "env_var"),
    [
        (
            "Moonshot",
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
            "MOONSHOT_API_KEY",
        ),
        (
            "Z.ai",
            "zai",
            "glm-5.2",
            "https://api.z.ai/api/paas/v4",
            "ZAI_API_KEY",
        ),
    ],
)
def test_hosted_readiness_validates_the_same_full_contract_as_send(
    provider, provider_key, model, base_url, env_var
):
    readiness = get_provider_readiness(
        provider,
        {
            "api_settings": {
                provider_key: {
                    "api_key_env_var": env_var,
                    "model": model,
                    "api_base_url": base_url,
                    "timeout": 90,
                    "retries": 3,
                    "retry_delay": 1,
                    "streaming": True,
                }
            }
        },
        environ={env_var: "hosted-secret-canary"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "hosted-secret-canary"
    assert "hosted-secret-canary" not in readiness.user_message


@pytest.mark.parametrize(
    ("provider", "provider_key", "settings"),
    [
        ("Moonshot", "moonshot", {"model": " "}),
        ("Moonshot", "moonshot", {"api_base_url": "https://user:pass@example.test/v1"}),
        ("Moonshot", "moonshot", {"timeout": True}),
        ("Z.ai", "zai", {"retries": -1}),
        ("Z.ai", "zai", {"retry_delay": -1}),
        ("Z.ai", "zai", {"streaming": "true"}),
    ],
)
def test_hosted_readiness_blocks_malformed_send_settings(
    provider, provider_key, settings
):
    source = {
        "api_settings": {
            provider_key: {
                "api_key": "hosted-secret-canary",
                **settings,
            }
        }
    }
    original = deepcopy(source)

    readiness = get_provider_readiness(provider, source, environ={})

    assert source == original
    assert readiness.ready is False
    assert readiness.reason == "Invalid provider settings"
    assert f"api_settings.{provider_key}" in readiness.user_message
    assert "hosted-secret-canary" not in readiness.user_message


@pytest.mark.parametrize(
    ("provider", "provider_key", "alias"),
    [
        ("Moonshot", "moonshot", " MOONSHOT "),
        ("Z.ai", "zai", "ZAI"),
    ],
)
def test_hosted_readiness_rejects_normalized_alias_conflicts(
    provider, provider_key, alias
):
    source = {
        "api_settings": {
            alias: {"api_key": "alias-secret-canary"},
            provider_key: {"api_key": "canonical-secret-canary"},
        }
    }

    readiness = get_provider_readiness(provider, source, environ={})

    assert readiness.ready is False
    assert readiness.reason == "Invalid provider settings"
    assert "alias-secret-canary" not in readiness.user_message
    assert "canonical-secret-canary" not in readiness.user_message


def test_keyless_local_provider_is_ready_without_api_key():
    readiness = get_provider_readiness(
        "Ollama",
        {"api_settings": {"ollama": {"api_url": "http://localhost:11434"}}},
        environ={},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key is None
    assert readiness.user_message == "Ollama is ready. No API key is required."


@pytest.mark.parametrize("provider", ["vLLM", "Custom-2", "local-llm"])
def test_known_keyless_provider_aliases_are_ready_without_api_key(provider):
    readiness = get_provider_readiness(
        provider,
        {"api_settings": {}},
        environ={},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key is None


def test_unknown_provider_without_key_is_not_ready():
    readiness = get_provider_readiness(
        "OpenAi Typo",
        {"api_settings": {}},
        environ={},
    )

    assert readiness.ready is False
    assert readiness.requires_api_key is True
    assert readiness.api_key is None
    assert readiness.reason == "Unknown provider"
    assert readiness.recovery == (
        "Choose a supported provider or add api_key under [api_settings.openai_typo]."
    )


# --- PR-T2 Task 7: one truth for "is a provider configured?" ---------------
#
# The critique's harm: a config with ONLY `[API] anthropic_api_key` set spent
# real money through the Library path (`LLM_Calls/LLM_API_Calls.py`'s
# `chat_with_anthropic` reads the legacy `anthropic_api` dict, which DOES see
# that key -- `config.py` has always projected `[API]` into it) while
# Console's own readiness check -- this module's `get_provider_readiness`,
# reading only `api_settings.<provider>.api_key` -- showed a blocking
# "Connect a provider" wall for the identical config. `config.py`'s
# `load_settings()` now bridges a resolved legacy `[API] <provider>_api_key`
# into `api_settings.<provider>.api_key` when the modern key is absent, so
# both readers agree. These tests drive the REAL loader (same pattern as
# `Tests/Utils/test_config_api_key_resolution.py`'s `_real_config`) because
# the bridge lives inside `load_settings()` itself -- a hand-built config
# dict cannot exercise it.

#: Env vars this suite guarantees are unset so a developer machine's real
#: credentials cannot mask the branch under test (same rationale as
#: `test_config_api_key_resolution.py`'s `_clear_provider_env`).
_REAL_LOADER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@contextmanager
def _real_config(tmp_path, monkeypatch, toml_text: str):
    """Point the real config loader at a scratch TOML file; restore + reload
    afterwards. Copied deliberately from `Tests/Utils/test_config_api_key_
    resolution.py`'s `_real_config` -- same isolation contract, same
    teardown -- so this suite can never write to the live user config and
    cannot drift from that file on how "the real loader" is driven.
    """
    config_path = tmp_path / "scratch-provider-readiness-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod.load_cli_config_and_ensure_existence(force_reload=True)
    config_mod.load_settings(force_reload=True)
    try:
        yield
    finally:
        if original_env is not None:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        else:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        config_mod.load_cli_config_and_ensure_existence(force_reload=True)
        config_mod.load_settings(force_reload=True)


def _clear_provider_env(monkeypatch) -> None:
    for name in _REAL_LOADER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_legacy_API_section_only_anthropic_key_satisfies_provider_readiness(
    tmp_path, monkeypatch
):
    """With ONLY `[API] anthropic_api_key` set, `get_provider_readiness`
    reports the provider configured -- Console's "Connect a provider" wall
    disappears for the exact config that used to spend money silently
    through the Library path.
    """
    _clear_provider_env(monkeypatch)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[API]\nanthropic_api_key = "sk-ant-legacy-only-key"\n',
    ):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.ready is True
    assert readiness.api_key == "sk-ant-legacy-only-key"
    assert readiness.api_key_source == "config:api_settings.anthropic.api_key"


def test_legacy_API_section_only_openai_key_satisfies_provider_readiness(
    tmp_path, monkeypatch
):
    """Same bridge, a second provider -- proves the fix is not anthropic-
    specific special-casing."""
    _clear_provider_env(monkeypatch)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[API]\nopenai_api_key = "sk-oai-legacy-only-key"\n',
    ):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("openai", settings)

    assert readiness.ready is True
    assert readiness.api_key == "sk-oai-legacy-only-key"
    assert readiness.api_key_source == "config:api_settings.openai.api_key"


def test_legacy_API_section_only_anthropic_key_still_resolves_for_the_spending_path(
    tmp_path, monkeypatch
):
    """`LLM_Calls/LLM_API_Calls.py`'s `chat_with_anthropic` (~1218-1219)
    reads `settings["anthropic_api"]["api_key"]` directly -- this is the
    single most important non-regression in PR-T2 Task 7: the bridge must
    add a second place the key is visible from, not remove the first.
    """
    _clear_provider_env(monkeypatch)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[API]\nanthropic_api_key = "sk-ant-legacy-only-key"\n',
    ):
        settings = config_mod.load_settings()

    assert settings["anthropic_api"]["api_key"] == "sk-ant-legacy-only-key"


def test_modern_api_settings_anthropic_key_wins_over_legacy_API_section_key(
    tmp_path, monkeypatch
):
    """Precedence, named explicitly: a modern `api_settings.anthropic.
    api_key` wins where both exist -- for BOTH readers, since Task 7 makes
    them share one normalized value instead of two independent reads.
    """
    _clear_provider_env(monkeypatch)
    toml_text = (
        "[api_settings.anthropic]\n"
        'api_key = "sk-ant-modern-key"\n'
        "\n"
        "[API]\n"
        'anthropic_api_key = "sk-ant-legacy-key"\n'
    )
    with _real_config(tmp_path, monkeypatch, toml_text):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.api_key == "sk-ant-modern-key"
    assert settings["anthropic_api"]["api_key"] == "sk-ant-modern-key"


def test_env_var_only_anthropic_key_is_not_bridged_into_api_settings(
    tmp_path, monkeypatch
):
    """An env-var-only credential must NOT be written into `api_settings`.

    Doing so would flip its reported `api_key_source` from `env:...` to
    `config:...`, and `provider_readiness.chat_api_key_field_state` treats a
    `config:` source as safe to prefill and persist in the inline Chat-
    Defaults API-key field -- silently exposing a secret that was never
    typed into config in the first place. `get_provider_readiness`'s own
    environment fallback already reports the env-only case as ready without
    this rewrite, so nothing is lost by leaving `api_settings` untouched.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-only-key")
    with _real_config(tmp_path, monkeypatch, ""):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.ready is True
    assert readiness.api_key == "sk-ant-env-only-key"
    assert readiness.api_key_source == "env:ANTHROPIC_API_KEY"
    assert settings["anthropic_api"]["api_key"] == "sk-ant-env-only-key"
    assert (
        settings.get("api_settings", {}).get("anthropic", {}).get("api_key")
        != "sk-ant-env-only-key"
    )


def test_no_legacy_or_modern_key_leaves_api_settings_api_key_unset(
    tmp_path, monkeypatch
):
    """A config with no credential anywhere must not gain a fabricated
    `api_key` -- the bridge must never write a value it has no source for.

    (The default shipped config already carries a `[api_settings.anthropic]`
    table with `api_key_env_var`/`model` defaults but deliberately no
    `api_key` -- this asserts the bridge leaves that absence alone rather
    than asserting the whole table is empty.)
    """
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, ""):
        settings = config_mod.load_settings()

    assert not settings.get("api_settings", {}).get("anthropic", {}).get("api_key")


# --- PR-T2 Task 7 fix round: reviewer findings I1, I2, I4 -----------------


def test_modern_api_settings_key_outranks_the_env_var_for_the_spending_path(
    tmp_path, monkeypatch
):
    """I1: named precedence for the case CLAUDE.md's general "env vars ->
    config.toml -> defaults" ordering does NOT apply to.

    Before PR-T2 Task 7, `chat_with_anthropic` read ONLY the legacy
    `anthropic_api` dict, itself resolved env-before-TOML with no `api_
    settings` input at all -- an explicit `api_settings.anthropic.api_key`
    had zero effect on what was actually spent, even though `get_provider_
    readiness` displayed it as the ready-making value. This is the
    deliberate fix: an explicit, non-placeholder `api_settings.<provider>.
    api_key` now wins over the environment variable for BOTH readers, so
    the credential Console displays as "why you're ready" is the SAME one
    the spend actually uses. `chat_with_openai` already did exactly this
    overlay for its own dict before this task (`LLM_API_Calls.py:561-580`)
    -- Task 7 makes the other 8 bridged providers consistent with it,
    rather than the reverse (restoring env-first would leave `get_
    provider_readiness`'s displayed source and the actually-spent key able
    to diverge again, recreating a split this task exists to close).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stale-env-key")
    toml_text = '[api_settings.anthropic]\napi_key = "sk-ant-current-modern-key"\n'
    with _real_config(tmp_path, monkeypatch, toml_text):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.api_key == "sk-ant-current-modern-key"
    assert settings["anthropic_api"]["api_key"] == "sk-ant-current-modern-key"


def test_a_placeholder_modern_key_falls_through_to_a_real_legacy_key_for_both_readers(
    tmp_path, monkeypatch
):
    """I2: the bridge's placeholder detection must be the SAME check `get_
    provider_readiness` uses -- not a locally re-declared one that
    recognizes fewer placeholder spellings.

    Concrete failure this pins: `api_settings.anthropic.api_key = "YOUR_KEY"`
    (a placeholder `get_provider_readiness` already recognizes, but which
    an earlier revision of `_normalize_legacy_provider_api_key` did NOT,
    since it only special-cased the literal `"<API_KEY_HERE>"` string)
    alongside a REAL `[API] anthropic_api_key`. Before this fix, the bridge
    treated `"YOUR_KEY"` as "explicit modern config wins", writing the
    placeholder itself into the legacy `anthropic_api` dict `chat_with_
    anthropic` spends through -- while readiness correctly said not-ready.
    That is the exact split this task exists to close, recreated inside
    the function meant to end it. Now both readers fall through to the
    real legacy key instead.
    """
    _clear_provider_env(monkeypatch)
    toml_text = (
        "[api_settings.anthropic]\n"
        'api_key = "YOUR_KEY"\n'
        "\n"
        "[API]\n"
        'anthropic_api_key = "sk-ant-real-legacy-key"\n'
    )
    with _real_config(tmp_path, monkeypatch, toml_text):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.ready is True
    assert readiness.api_key == "sk-ant-real-legacy-key"
    assert settings["anthropic_api"]["api_key"] == "sk-ant-real-legacy-key"
    # The placeholder must never land anywhere a spend could read it.
    assert settings["api_settings"]["anthropic"]["api_key"] == "sk-ant-real-legacy-key"


def test_legacy_API_section_only_mistral_key_satisfies_readiness_and_spend(
    tmp_path, monkeypatch
):
    """I4: `mistral` IS bridged (a prior revision wrongly excluded it).

    `chat_with_mistral` (`LLM_API_Calls.py:~4617-4621`) reads `api_settings.
    mistral` -- via `get_runtime_config_snapshot().values.get("api_
    settings", {}).get("mistral", {})`, and `RuntimeConfigSnapshot.values`
    is a deep copy of `load_settings()`'s own return value (`config.py`'s
    `get_runtime_config_snapshot`), so `settings["api_settings"]["mistral"]`
    here IS that exact table -- NOT the `mistral_api` dict, and NOT the
    shipped default's decorative `[api_settings.mistralai]` table. `
    "mistral"` (what `provider_config_key("Mistral")` computes, and what
    this bridge writes into) IS the live table the spend path already
    reads; bridging under it closes a real gap rather than creating a
    disconnected table. (`settings["mistral_api"]["api_key"]` also ends up
    holding the same value -- the bridge's whole point -- but that dict is
    NOT what `chat_with_mistral` reads, so asserting against it would pin
    the wrong artifact as evidence.)
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[API]\nmistral_api_key = "sk-mistral-legacy-only-key"\n',
    ):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("mistral", settings)

    assert readiness.ready is True
    assert readiness.api_key == "sk-mistral-legacy-only-key"
    # The exact table `chat_with_mistral` reads (see docstring above).
    assert (
        settings["api_settings"]["mistral"]["api_key"]
        == "sk-mistral-legacy-only-key"
    )


# --- PR-T2 review round 3 -------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    ("custom-openai-api", "custom-openai-api-2", "mlx_lm"),
)
def test_dispatchable_hyphenated_provider_spellings_stay_ready(provider):
    """I2: the three dispatch-table spellings whose normalized key is NOT
    the `[api_settings.*]` table name must not be blocked.

    `Chat/Chat_Functions.API_CALL_HANDLERS` dispatches on the exact keys
    `"custom-openai-api"`, `"custom-openai-api-2"` and `"mlx_lm"`, and a
    self-hoster's `default_api_endpoint` carries one of those verbatim
    spellings -- they DISPATCH, and they need no credential. But
    `provider_config_key` normalizes them to `custom_openai_api`,
    `custom_openai_api_2` and `mlx_lm`, none of which matched the
    `custom`/`custom_2`/`local_mlx_lm` entries the keyless set had, so
    Task 7's stricter gate reported "Unknown provider" and permanently
    disabled a Run button that worked at the branch base -- with copy
    naming no remedy, since there is no key to add.
    """
    readiness = get_provider_readiness(provider, {"api_settings": {}}, environ={})

    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.reason == "Ready"


def test_widening_the_keyless_set_did_not_weaken_credential_rejection():
    """I2 guard: the fix must widen the KEYLESS set only.

    A provider that genuinely needs a key, and a genuinely unknown one,
    must both still be reported not-ready -- otherwise "no credential
    needed" would have become the default answer and the money-stopping
    gate would open for everything.
    """
    keyed = get_provider_readiness("anthropic", {"api_settings": {}}, environ={})
    assert keyed.ready is False
    assert keyed.reason == "Missing API key"

    unknown = get_provider_readiness("totally-made-up", {"api_settings": {}}, environ={})
    assert unknown.ready is False
    assert unknown.reason == "Unknown provider"


def test_placeholder_environment_key_never_reaches_the_spend_path(
    tmp_path, monkeypatch
):
    """I5: the env branch of the credential bridge must run through the
    SAME validity check every other source does.

    `_normalize_legacy_provider_api_key`'s env branch was a bare
    `os.getenv(env_var)` truth test, so `ANTHROPIC_API_KEY="YOUR_KEY"` was
    returned verbatim into the legacy `anthropic_api` dict `chat_with_
    anthropic` spends through, while `get_provider_readiness` (which runs
    every source through `resolve_provider_api_key`) reported not-ready.
    Gated surfaces failed safe; an ungated `chat_api_call` caller (Evals,
    briefings, agent runs) would have sent the placeholder as a
    credential. Same bug class as the placeholder-in-modern-config split
    pinned above, one branch below it in the same function.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "YOUR_KEY")
    with _real_config(tmp_path, monkeypatch, "[API]\n"):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.ready is False
    assert settings["anthropic_api"]["api_key"] is None


def test_surrounding_whitespace_is_stripped_for_readiness_and_spend_alike(
    tmp_path, monkeypatch
):
    """Minor 2: the bridge validated the STRIPPED form but returned the raw
    one, so `api_key = " sk-xyz "` showed `sk-xyz` in readiness while the
    spend path sent `" sk-xyz "` -- a 401 whose cause is invisible from
    every surface that reports the key as fine.
    """
    _clear_provider_env(monkeypatch)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[api_settings.anthropic]\napi_key = "  sk-ant-padded-key  "\n',
    ):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("anthropic", settings)

    assert readiness.api_key == "sk-ant-padded-key"
    assert settings["anthropic_api"]["api_key"] == "sk-ant-padded-key"


def test_legacy_only_google_key_lands_in_the_table_chat_with_google_reads(
    tmp_path, monkeypatch
):
    """I4: `chat_with_google` reads `api_settings.google` -- the table the
    bridge writes -- so the "one credential truth" claim holds for Google.

    It used to read `api_settings.google_api`, a table nothing in this app
    produces (the shipped default is `[api_settings.google]`, the bridge
    writes `api_settings["google"]`, and the legacy dict is the top-level
    `google_generative_api`). So a bridged `[API] google_api_key` could
    not reach the spend path even while readiness reported ready and the
    Library RAG gate opened -- the one provider for which three claim
    sites (`CLAUDE.md`, `config.py`'s bridge docstring, `Docs/User_Guide/
    library/search-and-rag.md`) stated something false. This pins the
    config half; `Tests/Chat/test_google_native_tools.py::test_google_
    api_key_comes_from_the_api_settings_google_table` pins the handler
    half.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with _real_config(
        tmp_path,
        monkeypatch,
        '[API]\ngoogle_api_key = "google-legacy-only-key"\n',
    ):
        settings = config_mod.load_settings()
        readiness = get_provider_readiness("google", settings)

    assert readiness.ready is True
    assert settings["api_settings"]["google"]["api_key"] == "google-legacy-only-key"
    # The table the handler used to read is still produced by nobody.
    assert "google_api" not in settings["api_settings"]
