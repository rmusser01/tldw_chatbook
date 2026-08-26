"""Boolean contract of config_module.save_settings_to_cli_config."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from tldw_chatbook import config as config_module
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail


def test_config_import_isolated_from_eager_chat_package(tmp_path):
    """A fresh config-first process must not cycle through Chat.__init__."""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tldw_chatbook.config; import tldw_chatbook.app",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
    generation = 4

    def snapshot():
        policy = config_module._RUNTIME_CAPTURE_POLICY
        assert policy is not None
        return config_module.RuntimeConfigSnapshot(
            generation,
            {
                "console": {
                    "exchange_capture": policy.enabled,
                    "exchange_capture_detail": policy.detail.value,
                }
            },
        )

    def mutate(*args, **kwargs):
        nonlocal generation
        assert config_module.runtime_capture_policy().detail is CaptureDetail.SAFE
        assert kwargs["locked_snapshot_precondition"](
            config_module.AtomicConfigSnapshot(4, {})
        )
        kwargs["after_replace"]()
        generation = 5
        return ConfigMutationResult(True, False, "cache_reload")

    config_module._publish_runtime_capture_policy(True, CaptureDetail.SAFE, 4)
    monkeypatch.setattr(config_module, "_published_runtime_config_snapshot", snapshot)
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
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            7,
            {
                "console": {
                    "exchange_capture": True,
                    "exchange_capture_detail": "full",
                }
            },
        ),
    )

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


def test_runtime_capture_policy_rebuilds_from_new_canonical_generation(monkeypatch):
    config_module._publish_runtime_capture_policy(True, CaptureDetail.FULL, 40)
    snapshots = iter(
        [
            config_module.RuntimeConfigSnapshot(
                41,
                {
                    "console": {
                        "exchange_capture": False,
                        "exchange_capture_detail": "safe",
                    }
                },
            ),
            config_module.RuntimeConfigSnapshot(
                42,
                {
                    "console": {
                        "exchange_capture": True,
                        "exchange_capture_detail": "full",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: next(snapshots),
    )

    off = config_module.runtime_capture_policy()
    full = config_module.runtime_capture_policy()

    assert (off.enabled, off.detail, off.generation) == (
        False,
        CaptureDetail.SAFE,
        41,
    )
    assert (full.enabled, full.detail, full.generation) == (
        True,
        CaptureDetail.FULL,
        42,
    )


def test_runtime_capture_policy_concurrent_generation_refresh_is_equivalent(
    monkeypatch,
):
    config_module._publish_runtime_capture_policy(True, CaptureDetail.FULL, 50)
    snapshot = config_module.RuntimeConfigSnapshot(
        51,
        {
            "console": {
                "exchange_capture": False,
                "exchange_capture_detail": "safe",
            }
        },
    )
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: snapshot,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        policies = list(pool.map(lambda _index: config_module.runtime_capture_policy(), range(32)))

    assert set(policies) == {
        config_module.RuntimeCapturePolicy(False, CaptureDetail.SAFE, 51)
    }
