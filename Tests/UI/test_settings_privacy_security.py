from types import SimpleNamespace

from tldw_chatbook.UI.Screens.settings_privacy_security import (
    build_privacy_posture_rows,
    build_settings_privacy_posture,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


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


def test_privacy_posture_counts_numeric_secret_values_without_exposing_values():
    config = {
        "api_settings": {
            "numeric": {
                "api_key": 123456789,
                "token": 98765.0,
                "enabled": True,
            },
        },
    }

    posture = build_settings_privacy_posture(config, environ={})
    text = "\n".join(build_privacy_posture_rows(posture))

    assert posture.sensitive_config_fields == 2
    assert posture.provider_config_secrets == 2
    assert "123456789" not in text
    assert "98765" not in text


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


def test_privacy_posture_reports_skill_trust_without_leaking_paths():
    posture = build_settings_privacy_posture(
        {"encryption": {"enabled": True}},
        environ={},
        skill_trust={
            "enabled": True,
            "trust_status": "quarantined_modified",
            "keyring_convenience_enabled": True,
            "reduced_rollback_protection": False,
            "skills_dir": "/Users/example/private/skills",
        },
    )

    text = "\n".join(build_privacy_posture_rows(posture))

    assert "Skill trust: quarantined_modified" in text
    assert "Skill trust keyring convenience: enabled" in text
    assert "Skill trust rollback protection: full" in text
    assert "/Users/example/private/skills" not in text


def test_privacy_posture_reports_skill_trust_disabled_without_mapping():
    posture = build_settings_privacy_posture({}, environ={}, skill_trust=None)
    rows = build_privacy_posture_rows(posture)

    assert "Skill trust: disabled" in rows
    assert "Skill trust keyring convenience: disabled" in rows
    assert "Skill trust rollback protection: full" in rows


def test_privacy_posture_sanitizes_unknown_skill_trust_status():
    posture = build_settings_privacy_posture(
        {},
        environ={},
        skill_trust={
            "enabled": True,
            "trust_status": "trusted /Users/example/private/skills secret-token",
        },
    )
    text = "\n".join(build_privacy_posture_rows(posture))

    assert "Skill trust: unavailable" in text
    assert "/Users/example/private/skills" not in text
    assert "secret-token" not in text


def test_settings_screen_skill_trust_posture_uses_redacted_service_fields():
    class FakeSkillTrustService:
        keyring_convenience_enabled = True
        reduced_rollback_protection = True
        skills_dir = "/Users/example/private/skills"

        def overall_status(self):
            return "trusted"

    screen = SettingsScreen(
        SimpleNamespace(
            app_config={},
            local_skill_trust_service=FakeSkillTrustService(),
        )
    )

    posture = screen._skill_trust_posture()

    assert posture == {
        "enabled": True,
        "trust_status": "trusted",
        "keyring_convenience_enabled": True,
        "reduced_rollback_protection": True,
    }


def test_settings_screen_skill_trust_posture_handles_status_errors_safely():
    class RaisingSkillTrustService:
        keyring_convenience_enabled = False
        reduced_rollback_protection = False

        def overall_status(self):
            raise RuntimeError("/Users/example/private/skills secret-token")

    screen = SettingsScreen(
        SimpleNamespace(
            app_config={},
            local_skill_trust_service=RaisingSkillTrustService(),
        )
    )

    posture = screen._skill_trust_posture()

    assert posture == {
        "enabled": True,
        "trust_status": "unavailable_error",
        "keyring_convenience_enabled": False,
        "reduced_rollback_protection": False,
    }
