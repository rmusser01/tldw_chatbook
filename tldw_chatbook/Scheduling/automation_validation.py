"""Validators for Scheduled Tasks automation-definition authoring.

Ported from tldw_server `scheduled_task_automation_service.py` @
e3c198224 -- byte-parity on behavior/codes/messages, not on docstrings
(spec §7.1 drift rule; regenerate the parity tests below if the server
module changes). Only the four validators this module's
callers need are ported here, with the leading underscore dropped:
`validate_schedule`, `validate_recurring_question_config`,
`normalize_finding_policy`, `normalize_retention_policy`. Each accumulates
field-addressed errors shaped ``{"field": str, "code": str, "message":
str}``, exactly as the server emits them.

`_validate_agent_task_config` and `_normalize_visibility_policy` (the
`agent_task` family's validator, and preview-response shaping) are NOT
ported here -- out of this task's scope; see `automation_preview.py` for
the local `family: unsupported` handling of `agent_task` payloads.

Reuses the already-ported `normalize_recurring_question_scope`
(`recurring_question_scope.py`) rather than duplicating it.

`validate_recurring_question_config`'s `mode` keyword (task-31414) is a
LOCAL addition on top of the port: the server always backfills
`scope`/`finding_policy`/`retention_policy`/`generation_mode` with
concrete defaults, which this module's `mode="create"` default still
does byte-for-byte. `mode="update"` is the deliberate divergence -- an
edit that never supplied one of those four leaves it absent in the
normalized `config` instead of inventing a value, so re-syncing this
function from a future server change must preserve that branch, not
flatten it back to unconditional backfill.
"""

from __future__ import annotations

from typing import Any

from .recurring_question_scope import normalize_recurring_question_scope

# Inlined from tldw_server `recurring_question_models.py` (only the three
# constants these validators need -- same precedent as
# `recurring_question_scope.py`'s own constant inlining).
FINDING_POLICY_PRESETS = {"balanced_findings", "high_confidence_only"}
GENERATION_MODES = {"disabled", "optional", "required"}
RETENTION_POLICY_MODES = {"default", "custom"}

_SUPPORTED_SCHEDULE_KINDS = {"one_time", "interval", "daily", "weekly", "cron"}


def field_error(field: str, code: str, message: str) -> dict[str, str]:
    """Build a field-addressed validation error dict.

    Args:
        field: Dotted field path the error applies to (e.g. ``"schedule.kind"``).
        code: Machine-readable error code.
        message: Human-readable message.

    Returns:
        A ``{"field", "code", "message"}`` dict, matching the server's shape.
    """
    return {"field": field, "code": code, "message": message}


def validate_schedule(schedule: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Validate and normalize a raw ``schedule`` payload field.

    Port of the server's ``_validate_schedule``.

    Args:
        schedule: The raw ``schedule`` value from a preview/definition
            payload -- any value; only a dict is accepted.

    Returns:
        A ``(normalized, errors, warnings)`` tuple. ``normalized`` is a
        shallow copy of ``schedule`` (or ``{}`` when it was not a dict).
        ``errors`` reports a missing or unsupported ``kind`` (or a
        non-dict ``schedule`` itself). ``warnings`` is always empty --
        reserved, matches the server's signature.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not isinstance(schedule, dict):
        return {}, [field_error("schedule", "invalid_type", "Schedule must be an object.")], warnings

    normalized = dict(schedule)
    kind = normalized.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        errors.append(field_error("schedule.kind", "required", "Schedule kind is required."))
    elif kind not in _SUPPORTED_SCHEDULE_KINDS:
        errors.append(field_error("schedule.kind", "unsupported", f"Unsupported schedule kind: {kind}"))
    else:
        normalized["kind"] = kind
    return normalized, errors, warnings


def normalize_finding_policy(value: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a ``config.finding_policy`` value, appending errors in place.

    Port of the server's ``_normalize_finding_policy``.

    Args:
        value: The raw ``finding_policy`` value -- any value; only a dict
            contributes fields, everything else normalizes to the default
            preset.
        errors: The caller's error accumulator; an unsupported preset is
            appended to it (not returned separately), matching the
            server's in-place accumulation style.

    Returns:
        The policy dict with its ``preset`` normalized (defaulting to
        ``"balanced_findings"``); other keys of ``value`` pass through
        unchanged.
    """
    policy = dict(value) if isinstance(value, dict) else {}
    preset = str(policy.get("preset") or "balanced_findings").strip() or "balanced_findings"
    if preset not in FINDING_POLICY_PRESETS:
        errors.append(
            field_error(
                "config.finding_policy.preset",
                "unsupported",
                f"Unsupported finding policy preset: {preset}",
            )
        )
    return {**policy, "preset": preset}


def normalize_retention_policy(value: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a ``config.retention_policy`` value, appending errors in place.

    Port of the server's ``_normalize_retention_policy``.

    Args:
        value: The raw ``retention_policy`` value -- any value; only a
            dict contributes fields, everything else normalizes to the
            default mode.
        errors: The caller's error accumulator; an unsupported mode is
            appended to it (not returned separately), matching the
            server's in-place accumulation style.

    Returns:
        The policy dict with its ``mode`` normalized (defaulting to
        ``"default"``); other keys of ``value`` pass through unchanged.
    """
    policy = dict(value) if isinstance(value, dict) else {}
    mode = str(policy.get("mode") or "default").strip() or "default"
    if mode not in RETENTION_POLICY_MODES:
        errors.append(
            field_error(
                "config.retention_policy.mode",
                "unsupported",
                f"Unsupported retention policy mode: {mode}",
            )
        )
    return {**policy, "mode": mode}


def validate_recurring_question_config(
    config: dict[str, Any],
    *,
    mode: str = "create",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Validate and normalize a ``recurring_question`` preview's base config.

    Port of the server's ``_validate_recurring_question_config``. ``config``
    is the whole preview-request base dict (``name``, ``description``,
    ``config``, ``input``, ``schedule``, ...), not just its own ``config``
    sub-field -- matching the server's calling convention in
    ``_normalize_preview``.

    Args:
        config: The preview-request base dict. ``config["config"]`` holds
            the recurring-question option config (``scope``,
            ``finding_policy``, ``retention_policy``, ``generation_mode``);
            ``config["input"]`` holds ``question``.
        mode: ``"create"`` (default) or ``"update"``. A local, non-ported
            addition (task-31414): the server's own port always backfills
            ``scope``/``finding_policy``/``retention_policy``/
            ``generation_mode`` with concrete defaults, which is correct
            for a brand-new definition but wrong for an edit that never
            touched those fields -- an EDIT payload that omits one of the
            four leaves it absent in ``normalized["config"]`` instead of
            inventing a value the caller never supplied. A key that IS
            present is validated/normalized identically in both modes
            (strictness never relaxes, only the backfill does).

    Returns:
        A ``(normalized, errors, warnings)`` tuple. ``normalized`` is
        ``config`` with ``name``, ``input``, and ``config`` replaced by
        their validated/normalized forms (``schedule`` and other keys
        pass through untouched -- the caller is expected to overwrite
        those itself, matching the server's ``_normalize_preview``).
        ``errors`` covers a missing ``name``/``input.question``, an
        unsupported ``generation_mode``, unsupported finding/retention
        policy values, and any scope errors. ``warnings`` holds only the
        scope warnings' ``code`` strings (the scope warning's ``source``
        is dropped here -- a real server behavior, ported as-is).
    """
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    name = str(config.get("name") or "").strip()
    option_config = dict(config.get("config") or {})
    input_config = dict(config.get("input") or {})
    question = str(input_config.get("question") or "").strip()

    if not name:
        errors.append(field_error("name", "required", "Name is required."))
    if not question:
        errors.append(field_error("input.question", "required", "Question is required."))

    backfill = mode != "update"
    normalized_config = dict(option_config)

    if backfill or "scope" in option_config:
        scope, scope_errors, scope_warnings = normalize_recurring_question_scope(option_config.get("scope"))
        normalized_config["scope"] = scope
        errors.extend(scope_errors)
        warnings.extend(warning["code"] for warning in scope_warnings)

    if backfill or "finding_policy" in option_config:
        normalized_config["finding_policy"] = normalize_finding_policy(
            option_config.get("finding_policy"), errors
        )

    if backfill or "retention_policy" in option_config:
        normalized_config["retention_policy"] = normalize_retention_policy(
            option_config.get("retention_policy"), errors
        )

    if backfill or "generation_mode" in option_config:
        generation_mode = str(option_config.get("generation_mode") or "optional").strip() or "optional"
        if generation_mode not in GENERATION_MODES:
            errors.append(
                field_error(
                    "config.generation_mode",
                    "unsupported",
                    f"Unsupported generation mode: {generation_mode}",
                )
            )
        normalized_config["generation_mode"] = generation_mode

    normalized = dict(config)
    normalized["name"] = name
    normalized["input"] = {**input_config, "question": question}
    normalized["config"] = normalized_config
    return normalized, errors, warnings
