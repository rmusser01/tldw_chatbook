"""Unit tests for the pure first-run setup wizard state module."""

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    any_provider_configured,
    coerce_wizard_flag,
    should_offer_wizard,
    should_show_resume_toast,
)


def _config(api_settings=None, first_run=None):
    cfg = {}
    if api_settings is not None:
        cfg["api_settings"] = api_settings
    if first_run is not None:
        cfg["first_run"] = first_run
    return cfg


class TestCoerceWizardFlag:
    def test_truthy_values(self):
        assert coerce_wizard_flag(True) is True
        assert coerce_wizard_flag("true") is True
        assert coerce_wizard_flag(1) is True

    def test_falsy_and_garbage_values(self):
        assert coerce_wizard_flag(False) is False
        assert coerce_wizard_flag(None) is False
        assert coerce_wizard_flag("nope") is False
        assert coerce_wizard_flag({}) is False


class TestAnyProviderConfigured:
    def test_empty_config_is_unconfigured(self):
        assert any_provider_configured(_config(), {}) is False

    def test_placeholder_key_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key": "<API_KEY_HERE>"}})
        assert any_provider_configured(cfg, {}) is False

    def test_real_inline_key_counts(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert any_provider_configured(cfg, {}) is True

    def test_env_var_present_counts(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {"OPENAI_API_KEY": "sk-x"}) is True

    def test_env_var_declared_but_unset_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {}) is False

    def test_local_endpoint_url_counts(self):
        cfg = _config(api_settings={"llama_cpp": {"api_url": "http://127.0.0.1:8080"}})
        assert any_provider_configured(cfg, {}) is True


class TestShouldOfferWizard:
    def test_fresh_config_offers(self):
        assert should_offer_wizard(_config(), {}) is True

    def test_configured_provider_blocks_offer(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert should_offer_wizard(cfg, {}) is False

    def test_completed_blocks_offer(self):
        cfg = _config(first_run={"setup_completed": True})
        assert should_offer_wizard(cfg, {}) is False

    def test_started_but_not_completed_blocks_reoffer(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_offer_wizard(cfg, {}) is False


class TestShouldShowResumeToast:
    def test_started_not_completed_shows_toast(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_show_resume_toast(cfg, {}) is True

    def test_completed_never_shows_toast(self):
        cfg = _config(first_run={"setup_started": True, "setup_completed": True})
        assert should_show_resume_toast(cfg, {}) is False

    def test_never_started_never_shows_toast(self):
        assert should_show_resume_toast(_config(), {}) is False
