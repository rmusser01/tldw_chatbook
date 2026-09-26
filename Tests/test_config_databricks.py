# Tests/test_config_databricks.py
"""Default-config tables for Databricks (Task 9, ADR-179).

The default-config assertions use ``tomllib.loads(CONFIG_TOML_CONTENT)`` --
the established pattern in ``Tests/test_config_model_catalog_defaults.py``
(``test_kimi_zai_provider_and_settings_defaults_are_current``) -- so they
pin the shipped template without touching the shared module caches.
"""
import tomllib

from tldw_chatbook.config import API_MODELS_BY_PROVIDER, CONFIG_TOML_CONTENT
from tldw_chatbook.provider_registry import CLOUD_PROVIDER_CONFIG_KEYS, DATABRICKS


def test_providers_table_has_empty_databricks_entry():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["providers"].get("Databricks") == []


def test_api_settings_databricks_defaults():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["api_settings"]["databricks"] == dict(
        DATABRICKS.settings_defaults
    )
    # The workspace host is per-account: the shipped table must NOT pin an
    # api_base_url (DATABRICKS.default_base_url is None).
    assert "api_base_url" not in parsed["api_settings"]["databricks"]
    # No shipped default model either: a PRESENT-but-blank value would fail
    # closed at engine resolution (only the UNSET key resolves to the
    # payload-gated ""), so the key must be absent, matching
    # settings_defaults (Task 12 review fix).
    assert "model" not in parsed["api_settings"]["databricks"]
    assert "model" not in DATABRICKS.settings_defaults


def test_databricks_classified_cloud():
    from tldw_chatbook.config import _cloud_provider_keys

    assert "Databricks" in _cloud_provider_keys


def test_cloud_classification_derived_from_registry():
    # ADR-179: config.py no longer hand-types the cloud list; it binds the
    # registry's CLOUD_PROVIDER_CONFIG_KEYS (order included).
    from tldw_chatbook.config import _cloud_provider_keys

    assert tuple(_cloud_provider_keys) == CLOUD_PROVIDER_CONFIG_KEYS


def test_databricks_lands_in_cloud_model_partition():
    # The import-time partition of [providers] into API_MODELS_BY_PROVIDER
    # vs LOCAL_PROVIDERS keys off _cloud_provider_keys, so Databricks' empty
    # model list must classify as cloud (it fills via discovery/seeding).
    assert API_MODELS_BY_PROVIDER.get("Databricks") == []
