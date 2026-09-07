"""Redacted Settings Privacy & Security posture helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_chatbook.Utils.sensitive_config_keys import (
    is_sensitive_config_key,
    validate_trace_privacy_config,
)

SAFE_SKILL_TRUST_STATUSES = frozenset(
    {
        "trusted",
        "trust_uninitialized",
        "trust_locked",
        "quarantined_modified",
        "quarantined_added",
        "quarantined_deleted",
        "quarantined_manifest_error",
        "quarantined_unsupported_path",
        "unavailable",
        "unavailable_error",
    }
)
MAX_SKILL_TRUST_STATUS_CHARS = 80
SAFE_TRACE_COMPACTION_STATUSES = frozenset({"pending", "running", "complete"})
SAFE_TRACE_COMPACTION_REASONS = frozenset(
    {
        "awaiting_gc",
        "running",
        "complete",
        "database_threshold",
        "freelist_threshold",
        "freelist_ratio_threshold",
        "activity_threshold",
        "provider_active",
        "logical_gc_unavailable",
        "connections_busy",
        "active_transaction",
        "wal_checkpoint_failed",
        "lease_lost",
        "insufficient_disk",
        "integrity_check_failed",
        "interrupted",
        "cancelled",
        "vacuum_failed",
        "sqlite_failure",
        "compaction_failure",
    }
)


@dataclass(frozen=True)
class SettingsPrivacyPosture:
    """Redacted privacy posture derived from app configuration.

    Attributes:
        encryption_enabled: Whether config encryption is currently enabled.
        sensitive_config_fields: Count of configured sensitive config fields.
        provider_env_present: Count of configured provider env vars that are set.
        provider_env_missing: Count of configured provider env vars that are missing.
        provider_env_configured: Total configured provider credential env-var references.
        provider_config_secrets: Count of configured provider secrets stored in config.
        redaction_active: Whether visible privacy output redacts raw secret values.
        data_boundary: User-facing local data boundary summary.
        server_boundary: User-facing server token boundary summary.
        skill_trust_enabled: Whether local skill trust service is available.
        skill_trust_status: Redacted local skill trust aggregate status.
        skill_trust_keyring_convenience_enabled: Whether keyring convenience is enabled.
        skill_trust_reduced_rollback_protection: Whether rollback protection is reduced.
        trace_capture_enabled: Whether future Console calls require durable capture.
        trace_pii_masking_enabled: Whether configured trace PII masking is active.
        trace_viewer_profile: Default redacted trace viewer profile.
        trace_normalized_writes_enabled: Whether normalized trace writes are enabled.
        trace_normalized_reads_enabled: Whether normalized trace reads are enabled.
        trace_legacy_writes_enabled: Whether legacy compatibility writes are enabled.
    """

    encryption_enabled: bool
    sensitive_config_fields: int
    provider_env_present: int
    provider_env_missing: int
    provider_env_configured: int
    provider_config_secrets: int
    redaction_active: bool = True
    data_boundary: str = (
        "local data stays local unless explicit server handoff or sync is enabled"
    )
    server_boundary: str = "server tokens are reported as configured/missing only"
    skill_trust_enabled: bool = False
    skill_trust_status: str = "unavailable"
    skill_trust_keyring_convenience_enabled: bool = False
    skill_trust_reduced_rollback_protection: bool = False
    trace_capture_enabled: bool = True
    trace_pii_masking_enabled: bool = False
    trace_viewer_profile: str = "safe"
    trace_normalized_writes_enabled: bool = True
    trace_normalized_reads_enabled: bool = True
    trace_legacy_writes_enabled: bool = False
    trace_custom_pii_enabled_rules: int = 0
    trace_custom_pii_disabled_rules: int = 0
    trace_custom_pii_diagnostics: tuple[str, ...] = ()
    trace_compaction_status: str = "unavailable"
    trace_compaction_reason: str = "unavailable"
    trace_compaction_retry_pending: bool = False
    trace_compaction_progress_basis_points: int = 0
    trace_compaction_allocated_before: int = 0
    trace_compaction_allocated_after: int = 0
    trace_compaction_freelist_before: int = 0
    trace_compaction_freelist_after: int = 0


def build_settings_privacy_posture(
    app_config: object,
    *,
    environ: Mapping[str, str] | None = None,
    skill_trust: Mapping[str, object] | None = None,
    trace_maintenance: Mapping[str, object] | None = None,
) -> SettingsPrivacyPosture:
    """Build a redacted Privacy & Security posture from config and environment.

    Args:
        app_config: The application configuration mapping to inspect.
        environ: Optional environment mapping. Defaults to ``os.environ``.
        skill_trust: Optional redacted local skill trust posture mapping.
        trace_maintenance: Optional content-free physical maintenance status.

    Returns:
        A posture object containing only counts and status booleans.
    """

    env = os.environ if environ is None else environ
    encryption_config = (
        app_config.get("encryption", {}) if isinstance(app_config, Mapping) else {}
    )
    encryption_enabled = (
        bool(encryption_config.get("enabled"))
        if isinstance(encryption_config, Mapping)
        else False
    )
    env_present, env_missing, env_total = _provider_env_var_status_counts(
        app_config,
        env,
    )
    trust = skill_trust if isinstance(skill_trust, Mapping) else {}
    console = app_config.get("console", {}) if isinstance(app_config, Mapping) else {}
    if not isinstance(console, Mapping):
        console = {}
    trace_privacy = validate_trace_privacy_config(console)
    # Keep this first-paint helper aligned with the runtime policy without
    # adding config.py to the eager Settings import closure.
    from tldw_chatbook.config import (
        coerce_bool_setting,
        resolve_trace_rollout_settings,
    )

    rollout = resolve_trace_rollout_settings(console, environ=env)
    from tldw_chatbook.Chat.console_trace_custom_pii import (
        validate_custom_pii_rules_config,
    )

    custom_pii = validate_custom_pii_rules_config(
        console.get("trace_custom_pii_rules")
    )
    custom_rules = () if custom_pii.ruleset is None else custom_pii.ruleset.rules
    maintenance = (
        trace_maintenance if isinstance(trace_maintenance, Mapping) else {}
    )

    return SettingsPrivacyPosture(
        encryption_enabled=encryption_enabled,
        sensitive_config_fields=_sensitive_config_field_count(app_config),
        provider_env_present=env_present,
        provider_env_missing=env_missing,
        provider_env_configured=env_total,
        provider_config_secrets=_provider_config_secret_count(app_config),
        skill_trust_enabled=_safe_bool(trust.get("enabled")),
        skill_trust_status=_safe_skill_trust_status(trust.get("trust_status")),
        skill_trust_keyring_convenience_enabled=_safe_bool(
            trust.get("keyring_convenience_enabled")
        ),
        skill_trust_reduced_rollback_protection=_safe_bool(
            trust.get("reduced_rollback_protection")
        ),
        trace_capture_enabled=coerce_bool_setting(
            console.get("exchange_capture", True),
            True,
        ),
        trace_pii_masking_enabled=trace_privacy.exchange_capture_pii_redaction,
        trace_viewer_profile=trace_privacy.effective_viewer_profile,
        trace_normalized_writes_enabled=rollout.normalized_writes_enabled,
        trace_normalized_reads_enabled=rollout.normalized_reads_enabled,
        trace_legacy_writes_enabled=rollout.legacy_writes_enabled,
        trace_custom_pii_enabled_rules=sum(rule.enabled for rule in custom_rules),
        trace_custom_pii_disabled_rules=sum(not rule.enabled for rule in custom_rules),
        trace_custom_pii_diagnostics=tuple(
            item.display for item in custom_pii.diagnostics
        ),
        trace_compaction_status=_safe_trace_compaction_status(
            maintenance.get("status")
        ),
        trace_compaction_reason=_safe_trace_compaction_reason(
            maintenance.get("reason_code")
        ),
        trace_compaction_retry_pending=maintenance.get("retry_pending") is True,
        trace_compaction_progress_basis_points=_bounded_nonnegative_int(
            maintenance.get("progress_basis_points"), maximum=10000
        ),
        trace_compaction_allocated_before=_bounded_nonnegative_int(
            maintenance.get("allocated_bytes_before")
        ),
        trace_compaction_allocated_after=_bounded_nonnegative_int(
            maintenance.get("allocated_bytes_after")
        ),
        trace_compaction_freelist_before=_bounded_nonnegative_int(
            maintenance.get("freelist_bytes_before")
        ),
        trace_compaction_freelist_after=_bounded_nonnegative_int(
            maintenance.get("freelist_bytes_after")
        ),
    )


def env_var_summary(*, present: int, missing: int, configured: int) -> str:
    """Summarize provider env-var readiness with the counts' relationship.

    "0 present / 19 missing / 19 configured" read as contradictory (rescore
    P3): "configured" meant env-var REFERENCES in config, not set values.

    Args:
        present: Referenced env vars that are set in the environment.
        missing: Referenced env vars that are unset.
        configured: Total env-var references in the provider config.

    Returns:
        A single row stating how many referenced env vars are actually set.
    """
    return (
        f"{present} of {configured} referenced env vars are set ({missing} unset)"
    )


def skill_trust_display(status: str) -> str:
    """Strip the raw enum prefix from a skill-trust status for display.

    Args:
        status: The stored skill-trust status (e.g. "trust_uninitialized").

    Returns:
        The user-facing form without the "trust_" prefix.
    """
    return status.removeprefix("trust_")


def build_privacy_posture_rows(posture: SettingsPrivacyPosture) -> tuple[str, ...]:
    """Return stable redacted rows for visible Privacy & Security status.

    Args:
        posture: Redacted posture values to render as user-facing status rows.

    Returns:
        Tuple of stable, redacted status strings safe to display in Settings.
    """

    return (
        f"Config encryption: {'enabled' if posture.encryption_enabled else 'disabled'}",
        "Redaction: active; raw secret values hidden"
        if posture.redaction_active
        else "Redaction: unavailable",
        f"Sensitive config fields: {posture.sensitive_config_fields} present",
        (
            "Provider env vars: "
            + env_var_summary(
                present=posture.provider_env_present,
                missing=posture.provider_env_missing,
                configured=posture.provider_env_configured,
            )
        ),
        f"Provider config secrets: {posture.provider_config_secrets} present",
        (
            "Skill trust: "
            f"{skill_trust_display(posture.skill_trust_status) if posture.skill_trust_enabled else 'disabled'}"
        ),
        (
            "Skill trust keyring convenience: enabled"
            if posture.skill_trust_keyring_convenience_enabled
            else "Skill trust keyring convenience: disabled"
        ),
        (
            "Skill trust rollback protection: reduced"
            if posture.skill_trust_reduced_rollback_protection
            else "Skill trust rollback protection: full"
        ),
        f"Trace capture: {'On' if posture.trace_capture_enabled else 'Off'}",
        (
            "Trace PII masking: On for future calls; saved conversation unchanged"
            if posture.trace_pii_masking_enabled
            else "Trace PII masking: Off"
        ),
        _custom_pii_rules_row(posture),
        *(
            ("Custom PII diagnostics: " + ", ".join(posture.trace_custom_pii_diagnostics),)
            if posture.trace_custom_pii_diagnostics
            else ()
        ),
        f"Trace viewer: {posture.trace_viewer_profile.title()} disclosure profile",
        _trace_storage_row(posture),
        _trace_maintenance_row(posture),
        (
            "Trace history: compact and legacy traces are readable"
            if posture.trace_normalized_reads_enabled
            else "Trace history: compact traces are retained but temporarily hidden"
        ),
        f"Data boundary: {posture.data_boundary}",
        f"Server boundary: {posture.server_boundary}",
        "Privacy safety: no secret values were printed or written.",
    )


def _trace_storage_row(posture: SettingsPrivacyPosture) -> str:
    if posture.trace_normalized_writes_enabled and not posture.trace_legacy_writes_enabled:
        return "Trace storage: compact ledger for new calls; no transcript copies"
    if posture.trace_normalized_writes_enabled:
        return "Trace storage: compact ledger plus compatibility copies"
    if posture.trace_legacy_writes_enabled:
        return "Trace storage: legacy compatibility copies"
    return "Trace storage: new trace writes paused"


def _custom_pii_rules_row(posture: SettingsPrivacyPosture) -> str:
    enabled = posture.trace_custom_pii_enabled_rules
    disabled = posture.trace_custom_pii_disabled_rules
    invalid = len(posture.trace_custom_pii_diagnostics)
    if enabled == disabled == invalid == 0:
        return "Custom PII rules: none configured"
    return (
        f"Custom PII rules: {enabled} enabled, {disabled} disabled, {invalid} invalid"
    )


def _trace_maintenance_row(posture: SettingsPrivacyPosture) -> str:
    status = posture.trace_compaction_status
    reason = posture.trace_compaction_reason.replace("_", " ")
    if status == "running":
        percent = posture.trace_compaction_progress_basis_points // 100
        return f"Trace physical maintenance: compacting ({percent}%)"
    if status == "complete":
        return (
            "Trace physical maintenance: complete; allocated "
            f"{_format_bytes(posture.trace_compaction_allocated_before)} → "
            f"{_format_bytes(posture.trace_compaction_allocated_after)}, free "
            f"{_format_bytes(posture.trace_compaction_freelist_before)} → "
            f"{_format_bytes(posture.trace_compaction_freelist_after)}"
        )
    if status == "pending":
        retry = "retry pending; " if posture.trace_compaction_retry_pending else ""
        return f"Trace physical maintenance: {retry}{reason}"
    return "Trace physical maintenance: unavailable"


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _safe_trace_compaction_status(value: object) -> str:
    status = str(value or "unavailable")
    return status if status in SAFE_TRACE_COMPACTION_STATUSES else "unavailable"


def _safe_trace_compaction_reason(value: object) -> str:
    reason = str(value or "unavailable")
    return reason if reason in SAFE_TRACE_COMPACTION_REASONS else "unavailable"


def _bounded_nonnegative_int(value: object, *, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int:
        return 0
    return min(maximum, max(0, value))


def _safe_skill_trust_status(value: object) -> str:
    status = str(value or "unavailable").strip()[:MAX_SKILL_TRUST_STATUS_CHARS]
    return status if status in SAFE_SKILL_TRUST_STATUSES else "unavailable"


def _safe_bool(value: object) -> bool:
    return value is True


def _is_configured_secret_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value_text = value.strip()
        if not value_text or value_text in {"None", "null"}:
            return False
        if value_text.startswith("<") and value_text.endswith(">"):
            return False
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    return False


def _iter_config_leaf_values(value: object):
    if not isinstance(value, Mapping):
        return
    for key, child_value in value.items():
        if isinstance(child_value, Mapping):
            yield from _iter_config_leaf_values(child_value)
        else:
            yield key, child_value


def _sensitive_config_field_count(app_config: object) -> int:
    return sum(
        1
        for key, value in _iter_config_leaf_values(app_config)
        if is_sensitive_config_key(key) and _is_configured_secret_value(value)
    )


def _provider_env_var_status_counts(
    app_config: object,
    environ: Mapping[str, str],
) -> tuple[int, int, int]:
    if not isinstance(app_config, Mapping):
        return 0, 0, 0
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return 0, 0, 0
    present = 0
    missing = 0
    for provider_config in api_settings.values():
        if not isinstance(provider_config, Mapping):
            continue
        for key, value in provider_config.items():
            key_text = str(key).strip().lower()
            env_var = str(value or "").strip()
            if not key_text.endswith("_env_var") or not env_var:
                continue
            if environ.get(env_var):
                present += 1
            else:
                missing += 1
    return present, missing, present + missing


def _provider_config_secret_count(app_config: object) -> int:
    if not isinstance(app_config, Mapping):
        return 0
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return 0
    count = 0
    for provider_config in api_settings.values():
        if not isinstance(provider_config, Mapping):
            continue
        count += sum(
            1
            for key, value in _iter_config_leaf_values(provider_config)
            if is_sensitive_config_key(key) and _is_configured_secret_value(value)
        )
    return count
