"""Boolean contract of config_module.save_settings_to_cli_config."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.config import ConfigMutationResult


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


def test_full_capture_cache_failure_keeps_policy_at_actual_config_generation(
    monkeypatch,
):
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 4)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            4,
            {
                "console": {
                    "exchange_capture": True,
                    "exchange_capture_detail": "safe",
                }
            },
        ),
    )

    def mutate(*_args, **kwargs):
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
    assert result.fully_applied is False
    policy = config_module.runtime_capture_policy()
    assert policy.detail is CaptureDetail.FULL
    assert policy.generation == 4


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


def test_legacy_full_capture_never_migrates_to_full_viewer(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 61)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            61,
            {
                "console": {
                    "exchange_capture": True,
                    "exchange_capture_detail": "full",
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.detail is CaptureDetail.FULL
    assert policy.viewer_profile == "safe"
    assert policy.pii_redaction_enabled is False


def test_runtime_capture_policy_projects_independent_trace_rollout_gates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 611)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            611,
            {
                "console": {
                    "exchange_capture": True,
                    "trace_normalized_writes": False,
                    "trace_normalized_reads": True,
                    "trace_legacy_writes": True,
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.normalized_writes_enabled is False
    assert policy.normalized_reads_enabled is True
    assert policy.legacy_writes_enabled is True


def test_runtime_capture_policy_prefers_trace_rollout_environment_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 614)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            614,
            {
                "console": {
                    "trace_normalized_writes": False,
                    "trace_normalized_reads": True,
                    "trace_legacy_writes": False,
                }
            },
        ),
    )
    monkeypatch.setenv("TLDW_CONSOLE_TRACE_NORMALIZED_WRITES", "true")
    monkeypatch.setenv("TLDW_CONSOLE_TRACE_NORMALIZED_READS", "false")
    monkeypatch.setenv("TLDW_CONSOLE_TRACE_LEGACY_WRITES", "true")

    policy = config_module.runtime_capture_policy()

    assert policy.normalized_writes_enabled is True
    assert policy.normalized_reads_enabled is False
    assert policy.legacy_writes_enabled is True


def test_trace_rollout_settings_use_typed_validation_and_field_defaults() -> None:
    from pydantic import BaseModel

    assert issubclass(config_module.TraceRolloutSettings, BaseModel)

    settings = config_module.resolve_trace_rollout_settings(
        {
            "trace_normalized_writes": "invalid",
            "trace_normalized_reads": "false",
            "trace_legacy_writes": "invalid",
        },
        environ={},
    )

    assert settings.normalized_writes_enabled is True
    assert settings.normalized_reads_enabled is False
    assert settings.legacy_writes_enabled is False


def test_runtime_capture_policy_coerces_string_true_capture_and_rollout_gates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 613)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            613,
            {
                "console": {
                    "exchange_capture": "true",
                    "trace_normalized_writes": "true",
                    "trace_normalized_reads": "true",
                    "trace_legacy_writes": "true",
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.enabled is True
    assert policy.normalized_writes_enabled is True
    assert policy.normalized_reads_enabled is True
    assert policy.legacy_writes_enabled is True


def test_runtime_capture_policy_uses_shipping_trace_rollout_defaults(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 612)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            612,
            {"console": {"exchange_capture": True}},
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.normalized_writes_enabled is True
    assert policy.normalized_reads_enabled is True
    assert policy.legacy_writes_enabled is False


def test_versioned_viewer_and_pii_choices_restore_independently(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 62)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            62,
            {
                "console": {
                    "exchange_capture": False,
                    "exchange_capture_detail": "safe",
                    "exchange_capture_pii_redaction": True,
                    "trace_viewer_profile": "full",
                    "trace_viewer_profile_version": 1,
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.enabled is False
    assert policy.pii_redaction_enabled is True
    assert policy.viewer_profile == "full"


def test_runtime_capture_policy_freezes_valid_custom_pii_rules(monkeypatch) -> None:
    secret_pattern = r"private-prefix-\d{8}"
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 621)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            621,
            {
                "console": {
                    "exchange_capture_pii_redaction": True,
                    "trace_custom_pii_rules": {
                        "version": 1,
                        "revision_id": "11111111-1111-4111-8111-111111111111",
                        "rules": [
                            {
                                "id": "customer-id",
                                "label": "Customer ID",
                                "category": "customer_id",
                                "pattern": secret_pattern,
                                "flags": [],
                                "enabled": True,
                                "priority": 10,
                            }
                        ],
                    },
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.custom_pii_ruleset is not None
    assert policy.custom_pii_ruleset.revision_id == (
        "11111111-1111-4111-8111-111111111111"
    )
    assert [rule.rule_id for rule in policy.custom_pii_ruleset.runnable_rules] == [
        "customer-id"
    ]
    assert secret_pattern not in repr(policy)


def test_malformed_privacy_config_falls_back_to_all_safe_defaults(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 63)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            63,
            {
                "console": {
                    "exchange_capture": True,
                    "exchange_capture_detail": "full",
                    "exchange_capture_pii_redaction": "true",
                    "trace_viewer_profile": "full",
                    "trace_viewer_profile_version": "1",
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()

    assert policy.detail is CaptureDetail.FULL
    assert policy.pii_redaction_enabled is False
    assert policy.viewer_profile == "safe"


def test_failed_more_revealing_privacy_change_is_not_published(monkeypatch) -> None:
    config_module._publish_runtime_capture_policy(
        False,
        CaptureDetail.SAFE,
        70,
        pii_redaction_enabled=True,
        viewer_profile="safe",
    )
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 70)

    def mutate(*_args, **kwargs):
        assert kwargs["before_replace"] is None
        assert kwargs["after_replace"] is not None
        return ConfigMutationResult(False, False, "before_replace")

    monkeypatch.setattr(config_module, "apply_settings_mutation_to_cli_config", mutate)

    result = config_module.apply_console_capture_settings(
        enabled=True,
        detail=CaptureDetail.SAFE,
        expected_generation=70,
        pii_redaction_enabled=False,
        viewer_profile="full",
    )

    assert result.failure_phase == "before_replace"
    policy = config_module.runtime_capture_policy()
    assert policy.enabled is False
    assert policy.pii_redaction_enabled is True
    assert policy.viewer_profile == "safe"
