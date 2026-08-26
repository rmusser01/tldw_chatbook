"""Boolean contract of config_module.save_settings_to_cli_config."""

from tldw_chatbook import config as config_module
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail


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


def test_safe_capture_settings_publish_before_failed_save(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        lambda *args, **kwargs: (
            kwargs["locked_snapshot_precondition"](
                config_module.AtomicConfigSnapshot(3, {})
            ),
            kwargs["before_replace"](),
            ConfigMutationResult(False, False, "before_replace"),
        )[-1],
    )

    result = config_module.apply_console_capture_settings(
        enabled=True,
        detail=CaptureDetail.SAFE,
        expected_generation=3,
    )

    assert result.failure_phase == "before_replace"
    assert config_module.runtime_capture_policy().detail is CaptureDetail.SAFE


def test_full_capture_settings_publish_after_replacement_despite_cache_failure(
    monkeypatch,
):
    def mutate(*args, **kwargs):
        assert config_module.runtime_capture_policy().detail is CaptureDetail.SAFE
        assert kwargs["locked_snapshot_precondition"](
            config_module.AtomicConfigSnapshot(4, {})
        )
        kwargs["after_replace"]()
        return ConfigMutationResult(True, False, "cache_reload")

    config_module._publish_runtime_capture_policy(True, CaptureDetail.SAFE, 4)
    monkeypatch.setattr(config_module, "apply_settings_mutation_to_cli_config", mutate)

    result = config_module.apply_console_capture_settings(
        enabled=True,
        detail=CaptureDetail.FULL,
        expected_generation=4,
    )

    assert result.file_replaced is True
    assert config_module.runtime_capture_policy().detail is CaptureDetail.FULL


def test_stale_safe_capture_generation_does_not_publish(monkeypatch):
    config_module._publish_runtime_capture_policy(True, CaptureDetail.FULL, 7)

    def mutate(*args, **kwargs):
        assert not kwargs["locked_snapshot_precondition"](
            config_module.AtomicConfigSnapshot(8, {})
        )
        return ConfigMutationResult(
            False,
            False,
            None,
            conflict=True,
            conflict_reason="identity_changed",
        )

    monkeypatch.setattr(config_module, "apply_settings_mutation_to_cli_config", mutate)

    result = config_module.apply_console_capture_settings(
        enabled=True,
        detail=CaptureDetail.SAFE,
        expected_generation=7,
    )

    assert result.conflict is True
    assert config_module.runtime_capture_policy().detail is CaptureDetail.FULL
