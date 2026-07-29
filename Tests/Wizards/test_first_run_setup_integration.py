"""Integration tests: wizard commit plans against a real TOML config file."""

import os
from pathlib import Path

import pytest

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


@pytest.fixture()
def temp_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    from tldw_chatbook import config as config_module

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    yield config_path
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def _reload():
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    return load_cli_config_and_ensure_existence(force_reload=True)


def _write(section_values):
    from tldw_chatbook.config import save_settings_to_cli_config

    assert wizard_state.commit_sections_allowed(section_values), section_values
    assert save_settings_to_cli_config(section_values) is True


class TestCommitRoundTrip:
    def test_provider_and_model_commits_land_in_toml(self, temp_config):
        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-integration", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        assert config["api_settings"]["openai"]["api_key"] == "sk-integration"
        assert config["chat_defaults"]["provider"] == "OpenAI"
        assert config["chat_defaults"]["model"] == "gpt-5.6-terra"

    def test_wizard_state_flags_land_and_gate_offers(self, temp_config):
        _write(wizard_state.build_wizard_state_commit(started=True))
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False
        assert wizard_state.should_show_resume_toast(config, {}) is True
        _write(wizard_state.build_wizard_state_commit(completed=True))
        config = _reload()
        assert wizard_state.should_show_resume_toast(config, {}) is False

    def test_rerun_prefill_round_trip_without_secret_leak(self, temp_config):
        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-secret", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        prefill = wizard_state.read_wizard_prefill(config)
        assert prefill.provider_value == "OpenAI"
        assert "sk-secret" not in repr(prefill)
        presence = wizard_state.read_provider_secret_presence(
            config, {}, provider_key="openai"
        )
        assert presence.configured is True
        assert "sk-secret" not in repr(presence)

    def test_upgrader_config_never_auto_offers(self, temp_config):
        _write({"api_settings.anthropic": {"api_key": "sk-upgrader"}})
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False

    def test_summary_rows_match_persisted_state(self, temp_config):
        _write(wizard_state.build_notes_commit(
            sync_directory="~/N", auto_sync_enabled=True
        ))
        config = _reload()
        rows = {r.label: r for r in wizard_state.build_summary_rows(
            config, {}, rag_deps_installed=False
        )}
        assert rows["Notes sync"].ok is True


class TestEncryptionAtRest:
    def test_enable_encryption_encrypts_stored_key(self, temp_config):
        from tldw_chatbook.config import enable_config_encryption

        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-to-encrypt", api_url=None
        ))
        assert enable_config_encryption("integration-test-password") is True
        raw = Path(os.environ["TLDW_CONFIG_PATH"]).read_text()
        assert "sk-to-encrypt" not in raw
        assert "enc:" in raw or "password_verifier" in raw
