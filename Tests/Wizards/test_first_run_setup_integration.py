"""Integration tests: wizard commit plans against a real TOML config file."""

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
            provider_key="openai", api_key="wizard-test-key-alpha", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        assert config["api_settings"]["openai"]["api_key"] == "wizard-test-key-alpha"
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
            provider_key="openai", api_key="wizard-test-key-beta", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        prefill = wizard_state.read_wizard_prefill(config)
        assert prefill.provider_value == "OpenAI"
        assert "wizard-test-key-beta" not in repr(prefill)
        presence = wizard_state.read_provider_secret_presence(
            config, {}, provider_key="openai"
        )
        assert presence.configured is True
        assert "wizard-test-key-beta" not in repr(presence)

    def test_upgrader_config_never_auto_offers(self, temp_config):
        _write({"api_settings.anthropic": {"api_key": "wizard-test-key-gamma"}})
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


class TestFreshTemplateOfferGuard:
    """UAT regression pin (root cause of the live-app bug): every other test
    in this module builds its ``app_config`` from scratch as a Python dict.
    The shipped ``config.toml`` template (``config.py``'s
    ``CONFIG_TOML_CONTENT``) additionally pre-populates ~12
    ``[api_settings.*]`` blocks with default endpoint URLs (llama.cpp
    ``http://localhost:8080``, Ollama, vLLM, the HuggingFace router, etc.)
    that no synthetic-dict test ever reproduced. Loading the REAL generated
    template via ``temp_config``/``load_cli_config_and_ensure_existence`` is
    the only way to catch a regression where those default endpoints get
    miscounted as "configured" and the wizard silently never auto-offers."""

    def test_fresh_template_offers_wizard(self, temp_config):
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is True

    def test_template_with_one_real_inline_key_does_not_offer(self, temp_config):
        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="wizard-test-key-epsilon", api_url=None
        ))
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False


class TestLoadSettingsProjectsFirstRun:
    """UAT regression pin (F-E): ``app.py`` repoints ``self.app_config`` at
    ``load_settings()`` (a differently-shaped, hand-curated projection of the
    raw TOML), not at ``load_cli_config_and_ensure_existence()`` directly.
    ``load_settings`` builds its return dict section-by-section (see
    ``config.py``'s ``config_dict = {...}`` literal) and, before this fix,
    never listed ``first_run`` among the sections it passes through -- every
    other section the wizard depends on (``chat_defaults``, ``notes``,
    ``console``, ...) IS listed. Every other test in this module reads back
    via ``load_cli_config_and_ensure_existence`` (the raw loader), which does
    carry ``first_run`` -- masking this exact gap. In the live app, the
    dropped section meant ``should_offer_wizard``/``should_show_resume_toast``
    never saw the persisted flags, so the wizard re-offered on every launch
    even after a real completion."""

    def test_completed_flag_survives_the_load_settings_projection(self, temp_config):
        from tldw_chatbook.config import load_settings

        _write(wizard_state.build_wizard_state_commit(completed=True))
        settings = load_settings(force_reload=True)
        assert settings["first_run"]["setup_completed"] is True
        assert wizard_state.should_offer_wizard(settings, {}) is False

    def test_started_only_flag_still_gates_offer_and_shows_resume_toast(
        self, temp_config
    ):
        from tldw_chatbook.config import load_settings

        _write(wizard_state.build_wizard_state_commit(started=True))
        settings = load_settings(force_reload=True)
        assert settings["first_run"]["setup_started"] is True
        assert wizard_state.should_offer_wizard(settings, {}) is False
        assert wizard_state.should_show_resume_toast(settings, {}) is True


class TestEncryptionAtRest:
    def test_enable_encryption_encrypts_stored_key(self, temp_config):
        from tldw_chatbook.config import enable_config_encryption

        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="wizard-test-key-delta", api_url=None
        ))
        assert enable_config_encryption("integration-test-password") is True
        raw = temp_config.read_text()
        assert "wizard-test-key-delta" not in raw
        assert "enc:" in raw or "password_verifier" in raw
