"""Engine request resolution for engine-driven hosted providers (ADR-179).

Ports the ``resolve_zai_request`` resolution contracts from
``Tests/LLM_Calls/test_zai.py`` onto the generic engine parameterized by the
Databricks preset record. Databricks-specific divergences pinned here:

- No shipped default base URL (workspace hosts are per-account): a missing
  URL is an actionable configuration error, and a bare workspace host gets
  the record's ``/openai/v1`` suffix appended.
- A pasted terminal ``/chat/completions`` endpoint URL is rejected (zai
  accepts-and-strips it; the engine preset contract rejects the paste).
- No shipped default model: an unset model resolves to ``""`` (gated by the
  payload layer); a user-supplied blank model fails closed.
- ``provider_settings_for_key`` treats ``databricks`` as a generic provider
  (first-normalized-match), so zai's strict alias-conflict/null-table error
  cases port as fail-closed-without-env-rescue assertions instead.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    resolve_hosted_request,
)
from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.provider_registry import DATABRICKS

_BARE_HOST = "https://dbc-1.cloud.databricks.com"
_FULL_BASE_URL = "https://dbc-1.cloud.databricks.com/openai/v1"


def _config(api_settings: object) -> dict[str, object]:
    return {"api_settings": api_settings}


def test_explicit_args_win_over_settings_and_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-key")
    resolution = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="explicit-key",
        explicit_base_url="https://dbc-1.cloud.databricks.com",
        explicit_model="gpt-4o",
        app_config=_config({"databricks": {"api_key": "stored-key", "model": "claude-4-sonnet"}}),
        environ={"DATABRICKS_TOKEN": "env-key"},
    )
    assert resolution.api_key == "explicit-key"
    assert resolution.model == "gpt-4o"
    assert resolution.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"


def test_bare_workspace_host_gets_suffix_and_full_url_is_kept():
    bare = resolve_hosted_request(
        DATABRICKS, explicit_api_key="k", explicit_base_url="https://dbc-1.cloud.databricks.com",
    )
    full = resolve_hosted_request(
        DATABRICKS, explicit_api_key="k",
        explicit_base_url="https://dbc-1.cloud.databricks.com/openai/v1",
    )
    assert bare.base_url == full.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"


def test_terminal_chat_completions_paste_is_rejected():
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS, explicit_api_key="k",
            explicit_base_url="https://dbc-1.cloud.databricks.com/openai/v1/chat/completions",
        )


def test_missing_key_and_missing_url_are_actionable():
    with pytest.raises(ChatConfigurationError) as missing_key:
        resolve_hosted_request(DATABRICKS, app_config=_config({"databricks": {}}), environ={})
    assert "Databricks" in str(missing_key.value)
    with pytest.raises(ChatConfigurationError) as missing_url:
        resolve_hosted_request(
            DATABRICKS, explicit_api_key="k", app_config=_config({"databricks": {}}), environ={},
        )
    assert "Databricks" in str(missing_url.value)


def test_resolve_hosted_request_uses_canonical_precedence_and_record_defaults() -> None:
    config = {
        "api_settings": {
            "databricks": {
                "api_key": " config-key ",
                "api_key_env_var": "TEAM_DATABRICKS_TOKEN",
                "api_base_url": "https://config.cloud.databricks.com",
                "model": "databricks-gpt-4o",
                "timeout": 12,
                "retries": 4,
                "retry_delay": 0.5,
                "streaming": False,
            }
        }
    }
    original = deepcopy(config)

    configured = resolve_hosted_request(
        DATABRICKS,
        app_config=config,
        environ={
            "TEAM_DATABRICKS_TOKEN": "renamed-env-key",
            "DATABRICKS_TOKEN": "canonical-env-key",
        },
    )
    assert configured == HostedProviderResolution(
        provider="databricks",
        model="databricks-gpt-4o",
        api_key="ignored-by-compare",  # api_key is compare=False
        base_url="https://config.cloud.databricks.com/openai/v1",
        timeout=12.0,
        retries=4,
        retry_delay=0.5,
        streaming=False,
    )
    # Qodo finding 4: the engine follows the repo precedence rule
    # (env -> config -> defaults), so a USABLE env key beats the stored
    # settings key (the configured env name wins over the canonical one).
    assert configured.api_key == "renamed-env-key"

    explicit = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key=" explicit-key ",
        explicit_base_url="https://explicit.cloud.databricks.com",
        explicit_model="databricks-claude-sonnet",
        explicit_timeout=30,
        explicit_retries=1,
        explicit_retry_delay=2,
        app_config=config,
        environ={},
    )
    assert explicit.api_key == "explicit-key"
    assert explicit.model == "databricks-claude-sonnet"
    assert explicit.base_url == "https://explicit.cloud.databricks.com/openai/v1"
    assert explicit.timeout == 30.0
    assert explicit.retries == 1
    assert explicit.retry_delay == 2.0
    assert config == original
    assert "config-key" not in repr(configured)

    # Env fallback when no stored key: the record's canonical candidates.
    env_only = resolve_hosted_request(
        DATABRICKS,
        app_config=_config(
            {
                "databricks": {
                    "api_base_url": "https://env.cloud.databricks.com",
                    "model": "env-model",
                }
            }
        ),
        environ={"DATABRICKS_TOKEN": "env-key"},
    )
    assert env_only.api_key == "env-key"

    # Numeric/streaming defaults come from record.settings_defaults.
    defaults = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="k",
        explicit_base_url=_BARE_HOST,
        explicit_model="databricks-gpt-4o",
        app_config=_config({"databricks": {}}),
        environ={},
    )
    assert defaults.timeout == 90.0
    assert defaults.retries == 3
    assert defaults.retry_delay == 5.0
    assert defaults.streaming is True


def test_resolve_hosted_request_env_named_var_precedes_canonical() -> None:
    renamed = resolve_hosted_request(
        DATABRICKS,
        app_config=_config(
            {
                "databricks": {
                    "api_key_env_var": "TEAM_DATABRICKS_TOKEN",
                    "api_base_url": _BARE_HOST,
                    "model": "databricks-gpt-4o",
                }
            }
        ),
        environ={"TEAM_DATABRICKS_TOKEN": "renamed", "DATABRICKS_TOKEN": "canonical"},
    )
    assert renamed.api_key == "renamed"


def test_resolve_hosted_request_accepts_normalized_table_spelling() -> None:
    # "databricks" is a generic provider key: provider_settings_for_key
    # resolves the first normalized spelling, unlike zai's strict tables.
    resolution = resolve_hosted_request(
        DATABRICKS,
        app_config=_config(
            {
                "Databricks": {
                    "api_key": "alias-key",
                    "api_base_url": "https://alias.cloud.databricks.com",
                    "model": "alias-model",
                }
            }
        ),
        environ={},
    )
    assert resolution.api_key == "alias-key"
    assert resolution.model == "alias-model"
    assert resolution.base_url == "https://alias.cloud.databricks.com/openai/v1"


def test_resolve_hosted_request_unset_model_passes_through_empty() -> None:
    # Databricks ships no default model (gateway models are
    # workspace-configured; readiness requires key and URL, not a model), so
    # an unset model resolves to "" and is gated by the payload layer.
    resolution = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="k",
        explicit_base_url=_BARE_HOST,
        app_config=_config({"databricks": {}}),
        environ={},
    )
    assert resolution.model == ""


def test_resolve_hosted_request_blank_explicit_model_fails_closed() -> None:
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS,
            explicit_api_key="k",
            explicit_base_url=_BARE_HOST,
            explicit_model=" ",
            app_config=_config({"databricks": {}}),
            environ={},
        )


def test_resolve_hosted_request_null_table_fails_closed_without_env_rescue() -> None:
    # A null table is tolerated by the generic settings lookup (empty
    # settings), but must never enable an unvalidated resolution: the
    # missing workspace URL fails closed and the env key never rescues or
    # leaks (ported from zai's null-canonical-table case, adapted to the
    # generic provider path).
    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_hosted_request(
            DATABRICKS,
            explicit_api_key="k",
            app_config=_config({"databricks": None}),
            environ={"DATABRICKS_TOKEN": "must-not-rescue"},
        )
    assert exc_info.value.provider == "databricks"
    assert "must-not-rescue" not in str(exc_info.value)


def test_resolve_hosted_request_settings_null_base_url_is_required_error() -> None:
    with pytest.raises(ChatConfigurationError, match="workspace base URL is required"):
        resolve_hosted_request(
            DATABRICKS,
            explicit_api_key="k",
            app_config=_config({"databricks": {"api_base_url": None}}),
            environ={},
        )


def test_resolve_hosted_request_settings_base_url_gets_suffix() -> None:
    resolution = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="k",
        app_config=_config(
            {"databricks": {"api_base_url": "https://settings.cloud.databricks.com/", "model": "m0"}}
        ),
        environ={},
    )
    assert resolution.base_url == "https://settings.cloud.databricks.com/openai/v1"


@pytest.mark.parametrize(
    ("app_config", "kwargs"),
    [
        # api_settings itself is not a table.
        ({"api_settings": []}, {}),
        # Stored key is a placeholder (settings-supplied credential path).
        (
            {"api_settings": {"databricks": {"api_key": "YOUR_KEY"}}},
            {"explicit_base_url": _BARE_HOST},
        ),
        # Model is blank.
        (
            {"api_settings": {"databricks": {"model": " "}}},
            {"explicit_api_key": "k", "explicit_base_url": _BARE_HOST},
        ),
        # Endpoint-marker and non-URL base URLs.
        (
            {"api_settings": {"databricks": {"api_base_url": "https://bad/v4/responses"}}},
            {"explicit_api_key": "k"},
        ),
        (
            {"api_settings": {"databricks": {"api_base_url": "not-a-url"}}},
            {"explicit_api_key": "k"},
        ),
        (
            {"api_settings": {"databricks": {"api_base_url": 123}}},
            {"explicit_api_key": "k"},
        ),
        # Transport numeric/streaming settings are malformed.
        (
            {"api_settings": {"databricks": {"timeout": True}}},
            {"explicit_api_key": "k", "explicit_base_url": _BARE_HOST, "explicit_model": "m0"},
        ),
        (
            {"api_settings": {"databricks": {"retries": 1.5}}},
            {"explicit_api_key": "k", "explicit_base_url": _BARE_HOST, "explicit_model": "m0"},
        ),
        (
            {"api_settings": {"databricks": {"retry_delay": -1}}},
            {"explicit_api_key": "k", "explicit_base_url": _BARE_HOST, "explicit_model": "m0"},
        ),
        (
            {"api_settings": {"databricks": {"streaming": "yes"}}},
            {"explicit_api_key": "k", "explicit_base_url": _BARE_HOST, "explicit_model": "m0"},
        ),
    ],
)
def test_resolve_hosted_request_malformed_settings_fail_closed(
    app_config: object,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_hosted_request(
            DATABRICKS,
            app_config=app_config,  # type: ignore[arg-type]
            environ={"DATABRICKS_TOKEN": "must-not-rescue"},
            **kwargs,  # type: ignore[arg-type]
        )

    assert exc_info.value.provider == "databricks"
    assert "must-not-rescue" not in str(exc_info.value)


# --- Pydantic settings boundary (Qodo finding 10) ---

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_key", 123, "Databricks api_settings.databricks.api_key is invalid."),
        (
            "api_key_env_var",
            7,
            "Databricks api_settings.databricks.api_key_env_var is invalid.",
        ),
        ("model", 42, "Databricks model is invalid."),
        ("streaming", "yes", "Databricks streaming must be a boolean."),
        ("timeout", True, "Databricks timeout must be numeric."),
        ("timeout", -5, "Databricks timeout must be positive and finite."),
        ("retries", 1.5, "Databricks retries must be a non-negative integer."),
        ("retry_delay", "5", "Databricks retry_delay must be numeric."),
        ("retry_delay", -1, "Databricks retry_delay must be non-negative."),
    ],
)
def test_settings_boundary_translates_validation_errors_byte_for_byte(
    field: str, value: object, message: str
) -> None:
    """The Pydantic boundary's translated copy is exactly the copy the
    hand-rolled validators raised -- the tests are the contract."""
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import _typed_setting

    with pytest.raises(ChatConfigurationError) as exc_info:
        _typed_setting(DATABRICKS, {field: value}, field)
    assert str(exc_info.value.message) == message
    assert exc_info.value.provider == "databricks"


def test_settings_boundary_yields_typed_values_and_skips_unknown_keys() -> None:
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
        HostedProviderSettings,
        _typed_setting,
    )

    assert HostedProviderSettings.model_config.get("extra") == "ignore"
    settings: dict[str, object] = {
        "api_key": "k",
        "timeout": 12,  # int is a valid float setting (never coerced FROM str)
        "retries": 3,
        "retry_delay": 5,
        "streaming": False,
        "api_base_url": "https://config.cloud.databricks.com",
        "api_base": "https://shadowed.example",  # lower-precedence alias
        "totally_unknown": {"nested": True},  # extra: ignored
    }
    assert _typed_setting(DATABRICKS, settings, "timeout") == 12.0
    assert type(_typed_setting(DATABRICKS, settings, "timeout")) is float
    assert _typed_setting(DATABRICKS, settings, "retries") == 3
    assert _typed_setting(DATABRICKS, settings, "streaming") is False
    # Unknown keys are not boundary fields and never reach the model
    # (extra="ignore" keeps the unambiguous-table semantics).
    assert "totally_unknown" not in HostedProviderSettings.model_fields
    assert HostedProviderSettings.model_validate(settings).timeout == 12.0
    # The alias family resolves first-alias-wins from the typed view.
    resolution = resolve_hosted_request(
        DATABRICKS, app_config=_config({"databricks": settings}), environ={}
    )
    assert resolution.base_url == "https://config.cloud.databricks.com/openai/v1"


def test_settings_boundary_nonstring_alias_value_skips_not_fails() -> None:
    # A non-string alias value is skipped (the shared helper's semantics,
    # preserved by the model): the next alias is consulted.
    settings: dict[str, object] = {
        "api_base_url": 123,
        "base_url": "https://fallback-alias.cloud.databricks.com",
    }
    resolution = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="k",
        app_config=_config({"databricks": settings}),
        environ={},
    )
    assert (
        resolution.base_url == "https://fallback-alias.cloud.databricks.com/openai/v1"
    )
