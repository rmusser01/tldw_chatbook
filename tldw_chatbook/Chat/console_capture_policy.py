"""Console exchange-capture and trace rollout policy resolved from configuration.

Moved out of ``tldw_chatbook.config`` (TASK-36945.2). The capture, trace
maintenance and custom-PII modules are imported lazily inside functions, so
importing this module does not load them. Stateful config reads go through the
``config`` module at call time so patches on it still apply.
"""

from __future__ import annotations

import os
import threading as _threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from tldw_chatbook import config as _config
from tldw_chatbook.config import (
    coerce_bool_setting,
    coerce_float_setting,
    coerce_int_setting,
)
from tldw_chatbook.Utils.sensitive_config_keys import validate_trace_privacy_config

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
    from tldw_chatbook.Chat.console_trace_custom_pii import CustomPIIRuleset
    from tldw_chatbook.Chat.console_trace_maintenance import TraceCompactionPolicy
    from tldw_chatbook.config import AtomicConfigSnapshot, ConfigMutationResult


@dataclass(frozen=True, slots=True)
class RuntimeCapturePolicy:
    """Canonical process projection for future Console capture admission.

    Attributes:
        enabled: Whether future Console calls require durable trace capture.
        detail: Admission-frozen trace capture detail.
        generation: Monotonic policy publication generation.
        pii_redaction_enabled: Whether configured PII masking is active.
        viewer_profile: Default redacted viewer profile.
        normalized_writes_enabled: Whether new normalized calls may be written.
        normalized_reads_enabled: Whether normalized calls participate in reads.
        legacy_writes_enabled: Whether compatibility snapshots are also written.
    """

    enabled: bool
    detail: CaptureDetail
    generation: int
    pii_redaction_enabled: bool = False
    viewer_profile: str = "safe"
    normalized_writes_enabled: bool = True
    normalized_reads_enabled: bool = True
    legacy_writes_enabled: bool = False
    custom_pii_ruleset: CustomPIIRuleset | None = field(default=None, repr=False)


_TRACE_ROLLOUT_BOOLEAN_ADAPTER = TypeAdapter(bool)


class TraceRolloutSettings(BaseModel):
    """Validated effective gates for the semantic trace rollout.

    Attributes:
        normalized_writes_enabled: Whether normalized calls may be written.
        normalized_reads_enabled: Whether normalized calls participate in reads.
        legacy_writes_enabled: Whether compatibility snapshots are also written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    normalized_writes_enabled: bool = True
    normalized_reads_enabled: bool = True
    legacy_writes_enabled: bool = False

    @field_validator(
        "normalized_writes_enabled",
        "normalized_reads_enabled",
        "legacy_writes_enabled",
        mode="before",
    )
    @classmethod
    def _coerce_boolean(cls, value: object, info: ValidationInfo) -> bool:
        defaults = {
            "normalized_writes_enabled": True,
            "normalized_reads_enabled": True,
            "legacy_writes_enabled": False,
        }
        try:
            return _TRACE_ROLLOUT_BOOLEAN_ADAPTER.validate_python(value)
        except ValidationError:
            return defaults[info.field_name]


_RUNTIME_CAPTURE_POLICY_LOCK = _threading.RLock()
_RUNTIME_CAPTURE_POLICY: RuntimeCapturePolicy | None = None
_TRACE_ROLLOUT_ENV_NAMES = (
    "TLDW_CONSOLE_TRACE_NORMALIZED_WRITES",
    "TLDW_CONSOLE_TRACE_NORMALIZED_READS",
    "TLDW_CONSOLE_TRACE_LEGACY_WRITES",
)
_RUNTIME_CAPTURE_POLICY_ENV: tuple[str | None, ...] | None = None


def _trace_rollout_environment() -> tuple[str | None, ...]:
    """Return the three rollout overrides as one cache identity.

    Returns:
        Environment values aligned with ``_TRACE_ROLLOUT_ENV_NAMES``.
    """

    return tuple(os.environ.get(name) for name in _TRACE_ROLLOUT_ENV_NAMES)


def _trace_rollout_environment_mapping(
    values: tuple[str | None, ...],
) -> dict[str, str]:
    """Map one captured rollout environment identity back to present values."""

    return {
        name: value
        for name, value in zip(_TRACE_ROLLOUT_ENV_NAMES, values, strict=True)
        if value is not None
    }


def resolve_trace_rollout_settings(
    console: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> TraceRolloutSettings:
    """Validate rollout gates with environment-first precedence.

    Args:
        console: Raw Console configuration mapping.
        environ: Optional environment mapping. Defaults to ``os.environ``.

    Returns:
        The validated effective rollout settings.
    """

    values = console if isinstance(console, Mapping) else {}
    environment = os.environ if environ is None else environ

    def selected(environment_name: str, config_name: str, default: bool) -> object:
        override = environment.get(environment_name)
        return values.get(config_name, default) if override in (None, "") else override

    return TraceRolloutSettings.model_validate(
        {
            "normalized_writes_enabled": selected(
                _TRACE_ROLLOUT_ENV_NAMES[0],
                "trace_normalized_writes",
                True,
            ),
            "normalized_reads_enabled": selected(
                _TRACE_ROLLOUT_ENV_NAMES[1],
                "trace_normalized_reads",
                True,
            ),
            "legacy_writes_enabled": selected(
                _TRACE_ROLLOUT_ENV_NAMES[2],
                "trace_legacy_writes",
                False,
            ),
        }
    )


def resolve_trace_compaction_policy(console: object) -> "TraceCompactionPolicy":
    """Resolve bounded physical trace-maintenance thresholds from Console config."""

    from tldw_chatbook.Chat.console_trace_maintenance import TraceCompactionPolicy

    values = console if isinstance(console, Mapping) else {}
    retry_initial = coerce_float_setting(
        values.get("trace_compaction_retry_initial_seconds"),
        300.0,
        minimum=0.0,
        maximum=3600.0,
    )
    retry_max = coerce_float_setting(
        values.get("trace_compaction_retry_max_seconds"),
        3600.0,
        minimum=retry_initial,
        maximum=86400.0,
    )
    return TraceCompactionPolicy(
        min_database_bytes=coerce_int_setting(
            values.get("trace_compaction_min_database_bytes"),
            64 * 1024 * 1024,
            minimum=0,
        ),
        min_freelist_bytes=coerce_int_setting(
            values.get("trace_compaction_min_freelist_bytes"),
            16 * 1024 * 1024,
            minimum=0,
        ),
        min_freelist_ratio=coerce_float_setting(
            values.get("trace_compaction_min_freelist_ratio"),
            0.20,
            minimum=0.0,
            maximum=1.0,
        ),
        min_idle_seconds=coerce_float_setting(
            values.get("trace_compaction_min_idle_seconds"),
            30.0,
            minimum=0.0,
            maximum=86400.0,
        ),
        retry_initial_seconds=retry_initial,
        retry_max_seconds=retry_max,
        quiesce_timeout_seconds=coerce_float_setting(
            values.get("trace_compaction_quiesce_timeout_seconds"),
            5.0,
            minimum=0.0,
            maximum=60.0,
        ),
        disk_safety_margin_bytes=coerce_int_setting(
            values.get("trace_compaction_disk_safety_margin_bytes"),
            64 * 1024 * 1024,
            minimum=0,
        ),
    )


def _register_runtime_custom_pii_ruleset(
    ruleset: CustomPIIRuleset | None,
) -> CustomPIIRuleset | None:
    """Register valid rules for future masks or disable a reused revision."""

    if ruleset is None or not ruleset.runnable_rules:
        return ruleset
    from tldw_chatbook.Chat.console_trace_custom_pii import (
        register_custom_pii_ruleset,
    )

    if register_custom_pii_ruleset(ruleset):
        return ruleset
    logger.warning("custom_pii_ruleset_revision_conflict")
    return None


def _publish_runtime_capture_policy(
    enabled: bool,
    detail: CaptureDetail,
    generation: int,
    pii_redaction_enabled: bool = False,
    viewer_profile: str = "safe",
    normalized_writes_enabled: bool = True,
    normalized_reads_enabled: bool = True,
    legacy_writes_enabled: bool = False,
    custom_pii_ruleset: CustomPIIRuleset | None = None,
) -> RuntimeCapturePolicy:
    """Publish one validated capture policy without touching general caches."""
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
    from tldw_chatbook.Chat.console_trace_custom_pii import CustomPIIRuleset

    if not isinstance(detail, CaptureDetail):
        raise TypeError("detail must be CaptureDetail")
    if viewer_profile not in {"safe", "full"}:
        raise ValueError("viewer_profile")
    if custom_pii_ruleset is not None and not isinstance(
        custom_pii_ruleset, CustomPIIRuleset
    ):
        raise TypeError("custom_pii_ruleset")
    custom_pii_ruleset = _register_runtime_custom_pii_ruleset(custom_pii_ruleset)
    rollout_environment = _trace_rollout_environment()
    rollout = resolve_trace_rollout_settings(
        {
            "trace_normalized_writes": normalized_writes_enabled,
            "trace_normalized_reads": normalized_reads_enabled,
            "trace_legacy_writes": legacy_writes_enabled,
        },
        environ=_trace_rollout_environment_mapping(rollout_environment),
    )
    policy = RuntimeCapturePolicy(
        bool(enabled),
        detail,
        generation,
        bool(pii_redaction_enabled),
        viewer_profile,
        rollout.normalized_writes_enabled,
        rollout.normalized_reads_enabled,
        rollout.legacy_writes_enabled,
        custom_pii_ruleset,
    )
    global _RUNTIME_CAPTURE_POLICY, _RUNTIME_CAPTURE_POLICY_ENV
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        _RUNTIME_CAPTURE_POLICY = policy
        _RUNTIME_CAPTURE_POLICY_ENV = rollout_environment
    return policy


def runtime_capture_policy() -> RuntimeCapturePolicy:
    """Return the shared runtime capture policy, resolving invalid detail Safe.

    Returns:
        The canonical enabled/detail projection and its config generation.
    """
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    rollout_environment = _trace_rollout_environment()
    global _RUNTIME_CAPTURE_POLICY, _RUNTIME_CAPTURE_POLICY_ENV
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        current = _RUNTIME_CAPTURE_POLICY
        if (
            current is not None
            and current.generation == _config._CONFIG_GENERATION
            and _RUNTIME_CAPTURE_POLICY_ENV == rollout_environment
        ):
            return current
    snapshot = _config._published_runtime_config_snapshot()
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        current = _RUNTIME_CAPTURE_POLICY
        if (
            current is not None
            and current.generation == snapshot.generation
            and _RUNTIME_CAPTURE_POLICY_ENV == rollout_environment
        ):
            return current
        console = snapshot.values.get("console", {})
        if not isinstance(console, Mapping):
            console = {}
        try:
            detail = CaptureDetail(console.get("exchange_capture_detail", "safe"))
        except (TypeError, ValueError):
            detail = CaptureDetail.SAFE
        privacy = validate_trace_privacy_config(console)
        pii_redaction_enabled = privacy.exchange_capture_pii_redaction
        viewer_profile = privacy.effective_viewer_profile
        from tldw_chatbook.Chat.console_trace_custom_pii import (
            validate_custom_pii_rules_config,
        )

        custom_pii_ruleset = validate_custom_pii_rules_config(
            console.get("trace_custom_pii_rules")
        ).ruleset
        custom_pii_ruleset = _register_runtime_custom_pii_ruleset(custom_pii_ruleset)
        rollout = resolve_trace_rollout_settings(
            console,
            environ=_trace_rollout_environment_mapping(rollout_environment),
        )
        current = RuntimeCapturePolicy(
            coerce_bool_setting(console.get("exchange_capture", True), True),
            detail,
            snapshot.generation,
            pii_redaction_enabled,
            viewer_profile,
            rollout.normalized_writes_enabled,
            rollout.normalized_reads_enabled,
            rollout.legacy_writes_enabled,
            custom_pii_ruleset,
        )
        _RUNTIME_CAPTURE_POLICY = current
        _RUNTIME_CAPTURE_POLICY_ENV = rollout_environment
        return current


def apply_console_capture_settings(
    *,
    enabled: bool,
    detail: CaptureDetail,
    expected_generation: int,
    pii_redaction_enabled: bool | None = None,
    viewer_profile: str | None = None,
) -> ConfigMutationResult:
    """Apply the kill switch/detail with privacy-safe publication ordering.

    Args:
        enabled: Future-capture kill-switch state.
        detail: Safe or Full global capture detail.
        expected_generation: Config generation observed by the caller.

    Returns:
        Structured replacement/cache-publication status, including conflicts
        and partial post-replacement cache failures.
    """
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    if (
        type(enabled) is not bool
        or not isinstance(detail, CaptureDetail)
        or (
            pii_redaction_enabled is not None
            and type(pii_redaction_enabled) is not bool
        )
        or (viewer_profile is not None and viewer_profile not in {"safe", "full"})
    ):
        return _config.ConfigMutationResult(False, False, "before_replace")
    current = runtime_capture_policy()
    resolved_pii = (
        current.pii_redaction_enabled
        if pii_redaction_enabled is None
        else pii_redaction_enabled
    )
    resolved_viewer = (
        current.viewer_profile if viewer_profile is None else viewer_profile
    )

    def generation_is_current(snapshot: AtomicConfigSnapshot) -> bool:
        return snapshot.generation == expected_generation

    def publish_before_replace() -> None:
        _publish_runtime_capture_policy(
            enabled,
            detail,
            expected_generation,
            resolved_pii,
            resolved_viewer,
            current.normalized_writes_enabled,
            current.normalized_reads_enabled,
            current.legacy_writes_enabled,
            current.custom_pii_ruleset,
        )

    def publish_after_replace() -> None:
        # The general config generation advances only after cache publication
        # succeeds. Publishing the committed capture owner at the still-current
        # generation keeps it authoritative if that later step fails; on
        # success, the generation bump makes the next read rebuild from the new
        # canonical snapshot. This callback runs while the config write lock is
        # still held, so a newer writer cannot be overwritten afterward.
        _publish_runtime_capture_policy(
            enabled,
            detail,
            expected_generation,
            resolved_pii,
            resolved_viewer,
            current.normalized_writes_enabled,
            current.normalized_reads_enabled,
            current.legacy_writes_enabled,
            current.custom_pii_ruleset,
        )

    more_revealing = (
        (enabled and not current.enabled)
        or (detail is CaptureDetail.FULL and current.detail is CaptureDetail.SAFE)
        or (not resolved_pii and current.pii_redaction_enabled)
        or (resolved_viewer == "full" and current.viewer_profile == "safe")
    )
    privacy_safe = not more_revealing
    result = _config.apply_settings_mutation_to_cli_config(
        {
            "console": {
                "exchange_capture": enabled,
                "exchange_capture_detail": detail.value,
                "exchange_capture_pii_redaction": resolved_pii,
                "trace_viewer_profile": resolved_viewer,
                "trace_viewer_profile_version": 1,
            }
        },
        locked_snapshot_precondition=generation_is_current,
        before_replace=publish_before_replace if privacy_safe else None,
        after_replace=None if privacy_safe else publish_after_replace,
    )
    return result


