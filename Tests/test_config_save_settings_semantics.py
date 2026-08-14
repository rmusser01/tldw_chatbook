"""Boolean contract of config_module.save_settings_to_cli_config."""

from tldw_chatbook import config as config_module
from tldw_chatbook.config import ConfigMutationResult


def _patch_result(monkeypatch, result: ConfigMutationResult) -> None:
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        lambda *args, **kwargs: result,
    )


def test_identity_conflict_reports_failure(monkeypatch):
    _patch_result(
        monkeypatch,
        ConfigMutationResult(
            False, False, None, conflict=True, conflict_reason="identity_changed"
        ),
    )
    assert (
        config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}})
        is False
    )


def test_fully_applied_reports_success(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(True, True, None))
    assert (
        config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}})
        is True
    )


def test_noop_reports_success(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(False, False, None))
    assert config_module.save_settings_to_cli_config({}) is True


def test_before_replace_failure_reports_failure(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(False, False, "before_replace"))
    assert (
        config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}})
        is False
    )


def test_cache_reload_failure_reports_failure(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(True, False, "cache_reload"))
    assert (
        config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}})
        is False
    )
