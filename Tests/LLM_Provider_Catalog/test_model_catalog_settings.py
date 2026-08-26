from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
    SELECTOR_MERGE_CAP,
    load_model_catalog_settings,
)


def test_defaults_when_section_missing():
    settings = load_model_catalog_settings({})
    assert settings.auto_refresh_enabled is True
    assert settings.stale_after_hours == 24.0
    assert settings.auto_refresh_disabled == frozenset()
    assert settings.write_to_config == frozenset()
    # Confirm-first default: no consent until explicitly recorded.
    assert settings.refresh_consent_recorded is False


def test_consent_defaults_false_and_requires_explicit_true():
    assert load_model_catalog_settings(
        {"model_catalog": {"refresh_consent_recorded": True}}
    ).refresh_consent_recorded is True
    assert load_model_catalog_settings(
        {"model_catalog": {"refresh_consent_recorded": "yes"}}
    ).refresh_consent_recorded is False
    assert load_model_catalog_settings(
        {"model_catalog": {"refresh_consent_recorded": 1}}
    ).refresh_consent_recorded is False


def test_full_section_parsed_and_normalized():
    settings = load_model_catalog_settings(
        {
            "model_catalog": {
                "auto_refresh_enabled": False,
                "stale_after_hours": 12,
                "auto_refresh_disabled": ["ZAI"],
                "write_to_config": ["OpenRouter", "MistralAI"],
            }
        }
    )
    assert settings.auto_refresh_enabled is False
    assert settings.stale_after_hours == 12.0
    assert settings.auto_refresh_disabled == frozenset({"zai"})
    assert settings.write_to_config == frozenset({"openrouter", "mistralai"})


def test_garbage_values_fall_back_safely():
    settings = load_model_catalog_settings(
        {"model_catalog": {"stale_after_hours": "banana", "auto_refresh_disabled": "ZAI"}}
    )
    assert settings.stale_after_hours == 24.0
    assert settings.auto_refresh_disabled == frozenset()


def test_zero_stale_hours_is_allowed():
    settings = load_model_catalog_settings({"model_catalog": {"stale_after_hours": 0}})
    assert settings.stale_after_hours == 0.0


def test_qwencloud_is_seventh_auto_refresh_cloud_provider():
    assert set(AUTO_REFRESH_PROVIDER_LIST_KEYS) == {
        "OpenAI",
        "Anthropic",
        "MistralAI",
        "Moonshot",
        "OpenRouter",
        "QwenCloud",
        "ZAI",
    }
    assert len(AUTO_REFRESH_PROVIDER_LIST_KEYS) == 7
    assert SELECTOR_MERGE_CAP == 50
