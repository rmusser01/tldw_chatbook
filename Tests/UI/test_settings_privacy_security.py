from tldw_chatbook.UI.Screens.settings_privacy_security import (
    build_privacy_posture_rows,
    build_settings_privacy_posture,
)


DUMMY_ENV_SECRET = "env-secret-value-that-must-not-render"
DUMMY_CONFIG_SECRET = "config-secret-value-that-must-not-render"
DUMMY_SERVER_SECRET = "server-secret-value-that-must-not-render"


def test_privacy_posture_counts_secret_sources_without_exposing_values():
    config = {
        "encryption": {"enabled": True},
        "api_settings": {
            "openai": {
                "api_key_env_var": "OPENAI_API_KEY",
                "api_key": DUMMY_CONFIG_SECRET,
            },
            "groq": {"api_key_env_var": "GROQ_API_KEY"},
        },
        "tldw_api": {"auth_token": DUMMY_SERVER_SECRET},
    }

    posture = build_settings_privacy_posture(
        config,
        environ={"OPENAI_API_KEY": DUMMY_ENV_SECRET},
    )
    text = "\n".join(build_privacy_posture_rows(posture))

    assert posture.encryption_enabled is True
    assert posture.sensitive_config_fields == 2
    assert posture.provider_config_secrets == 1
    assert posture.provider_env_present == 1
    assert posture.provider_env_missing == 1
    assert posture.provider_env_configured == 2
    assert "Config encryption: enabled" in text
    assert "Provider env vars: 1 present / 1 missing / 2 configured" in text
    assert "Provider config secrets: 1 present" in text
    assert DUMMY_ENV_SECRET not in text
    assert DUMMY_CONFIG_SECRET not in text
    assert DUMMY_SERVER_SECRET not in text


def test_privacy_posture_ignores_non_secret_token_limits():
    config = {
        "api_settings": {
            "openai": {
                "api_key": DUMMY_CONFIG_SECRET,
                "max_tokens": 4096,
            },
        },
        "chat_defaults": {
            "max_tokens": 2048,
            "token_budget": 512,
        },
    }

    posture = build_settings_privacy_posture(config, environ={})

    assert posture.sensitive_config_fields == 1
    assert posture.provider_config_secrets == 1


def test_privacy_posture_handles_malformed_config_safely():
    posture = build_settings_privacy_posture(
        {
            "encryption": "invalid",
            "api_settings": {
                "openai": "invalid",
                "custom": {"api_key_env_var": ""},
            },
        },
        environ=None,
    )
    rows = build_privacy_posture_rows(posture)

    assert posture.encryption_enabled is False
    assert posture.sensitive_config_fields == 0
    assert posture.provider_env_configured == 0
    assert "Redaction: active; raw secret values hidden" in rows
