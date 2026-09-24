# Tests/test_provider_registry.py
"""Registry parity tests: registry data must equal today's literal tables."""
from tldw_chatbook.provider_registry import (
    ALL_RECORDS,
    AUDITED_ENDPOINT_KEYS,
    AUTO_REFRESH_KEYS,
    CLOUD_PROVIDER_CONFIG_KEYS,
    DATABRICKS,
    ENGINE_RECORDS,
    NATIVE_TOOLS_KEYS,
    RECORDS_BY_KEY,
)
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)


def test_records_unique_and_complete():
    keys = [record.key for record in ALL_RECORDS]
    assert len(keys) == len(set(keys))
    # every dispatch key is a registry key or an alias of one
    assert set(API_CALL_HANDLERS) - set(RECORDS_BY_KEY) <= set()


def test_audited_set_matches_handlers():
    # Since Task 8 the audited set is derived from the registry (ADR-179),
    # so it tracks AUDITED_ENDPOINT_KEYS exactly.
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS


def test_cloud_classification_matches_config():
    # Since Task 9, config.py's _cloud_provider_keys is derived from the
    # registry (ADR-179), so full equality holds with no exclusions.
    from tldw_chatbook.config import _cloud_provider_keys
    assert tuple(sorted(_cloud_provider_keys)) == tuple(
        sorted(CLOUD_PROVIDER_CONFIG_KEYS)
    )


def test_auto_refresh_flags_match_catalog_settings():
    # Same style as the cloud-classification parity: the literal list is in
    # [providers]-key form, so compare it against the config keys of the
    # records flagged auto_refresh. Task 12 added "Databricks" to the literal
    # list, so full equality holds with no exclusions.
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        AUTO_REFRESH_PROVIDER_LIST_KEYS,
    )
    auto_refresh_config_keys = {
        RECORDS_BY_KEY[key].config_key for key in AUTO_REFRESH_KEYS
    }
    assert set(AUTO_REFRESH_PROVIDER_LIST_KEYS) == auto_refresh_config_keys


def test_native_tools_flags_match_native_tools_module():
    # NATIVE_TOOLS_PROVIDERS is keyed by dispatch (record) keys. Task 12
    # added "databricks" to the literal set, so full equality holds with no
    # exclusions.
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS
    assert NATIVE_TOOLS_PROVIDERS == NATIVE_TOOLS_KEYS


def test_identity_fields_match_config_tables():
    # api_key_env_var / default api_base_url must be transcribed EXACTLY
    # from the shipped [api_settings.*] tables (the e080a2fb92-class bug
    # guard: a wrong env var or URL must fail here, not at a user's first
    # call). A record's table is keyed by its dispatch key or its config
    # key (lowercased), e.g. mistral -> [api_settings.mistralai],
    # custom-openai-api -> [api_settings.custom]. databricks is excluded:
    # it ships no defaults (workspace host is per-account).
    import tomllib

    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    tables = tomllib.loads(CONFIG_TOML_CONTENT)["api_settings"]
    checked = 0
    for record in ALL_RECORDS:
        if record.key == "databricks":
            continue
        table_key = next(
            (k for k in (record.key, record.config_key.lower()) if k in tables),
            None,
        )
        assert table_key is not None, f"no [api_settings.*] table for {record.key}"
        table = tables[table_key]
        assert record.api_key_env_var == table.get("api_key_env_var"), (
            f"{record.key}: api_key_env_var {record.api_key_env_var!r} != "
            f"[api_settings.{table_key}] {table.get('api_key_env_var')!r}"
        )
        if "api_base_url" in table:
            assert record.default_base_url == table["api_base_url"], (
                f"{record.key}: default_base_url {record.default_base_url!r} != "
                f"[api_settings.{table_key}] {table['api_base_url']!r}"
            )
        checked += 1
    assert checked > 0


def test_databricks_preset_shape():
    record = DATABRICKS
    assert record.key == "databricks"
    assert record.classification == "cloud"
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.api_key_env_candidates == ("DATABRICKS_TOKEN",)
    assert record.base_url_suffix == "/openai/v1"
    assert record.default_base_url is None  # workspace host is per-account
    assert record.reasoning_disposition == "ignored"
    assert "databricks" in {r.key for r in ENGINE_RECORDS}
