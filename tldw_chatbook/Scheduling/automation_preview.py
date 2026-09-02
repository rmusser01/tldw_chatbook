"""Pure local preview service for Scheduled Tasks automation authoring.

Runs the ported validators (`automation_validation.py`) against a payload
shaped like the server's `ScheduledTaskPreviewCreateRequest`
(`Tests/Scheduling/fixtures/server_responses/automation_endpoints.md`) and
fills the existing `AutomationPreview` model -- no I/O, no server round
trip. Mirrors the server's preview assembly (`_normalize_preview` /
`_create_preview` in `scheduled_task_automation_service.py`): same
mode-required/not-allowed errors, the same `normalized_config` shape, the
same `warnings` shape (``[{"message": str}, ...]``), the same
`visibility_policy` wrap (``{"mode": str}``), and the same
``status = "invalid" if validation_errors else "valid"`` derivation.

Only the `recurring_question` family is implemented. `agent_task`'s
validator (`_validate_agent_task_config`, message redaction) is not
ported in this task's scope; a payload requesting that family gets a
single `family: unsupported` validation error rather than a crash. A
`family` value that is not a recognized `AutomationFamily` member at all
(not even `agent_task`) raises `ValueError` -- `family` is a UI-controlled
selector, not free-form user input, so that is a caller-contract
violation rather than a field error.

`schedule_preview` is a local addition beyond the server's response: the
server's response schema declares it as a bare `dict[str, Any]` and the
service just puts the normalized `schedule` dict there (no next-run
computation). Since this local preview never round-trips to a server, it
adds a `next_occurrences` key -- up to three upcoming run times computed
via `schedule_compute.compute_next_run_at` -- so an authoring modal has
something concrete to show the user for "when will this run".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .automation_validation import (
    field_error,
    validate_recurring_question_config,
    validate_schedule,
)
from .models import AutomationFamily, AutomationPreview, PreviewStatus
from .schedule_compute import compute_next_run_at

_OCCURRENCE_COUNT = 3


def _normalize_visibility_policy(family: str, value: Any) -> str:
    """Port of the server's `_normalize_visibility_policy`.

    Args:
        family: The automation family value (``"recurring_question"`` or
            ``"agent_task"``), used only for the fallback default.
        value: The raw ``visibility_policy`` payload value.

    Returns:
        A visibility-policy mode string: ``value`` itself when it is a
        non-blank string; the ``"mode"``/``"visibility"``/``"policy"`` key
        of ``value`` when it is a dict and one of those is a non-blank
        string; otherwise the family's default (``"metadata_only"`` for
        ``agent_task``, ``"findings_only"`` otherwise).
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        mode = value.get("mode") or value.get("visibility") or value.get("policy")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    return "metadata_only" if family == "agent_task" else "findings_only"


def _next_occurrences(schedule: dict[str, Any], *, now: datetime, count: int = _OCCURRENCE_COUNT) -> list[str]:
    """Up to `count` sequential upcoming run times as UTC ISO-8601 strings.

    Each occurrence is computed by advancing `now` to the previous
    occurrence and calling `compute_next_run_at` again -- this naturally
    terminates a spent `one_time` schedule at one entry and never raises,
    since `compute_next_run_at` itself never raises on a malformed or
    incomplete schedule (it just returns `None`, which stops the loop).

    Args:
        schedule: A normalized schedule dict (`validate_schedule`'s
            output).
        now: The time to compute occurrences from.
        count: The maximum number of occurrences to return.

    Returns:
        Up to `count` ISO-8601 UTC datetime strings, in ascending order.
    """
    occurrences: list[str] = []
    current = now
    for _ in range(count):
        next_run = compute_next_run_at(schedule, now=current)
        if next_run is None:
            break
        occurrences.append(next_run.isoformat())
        current = next_run
    return occurrences


def preview_automation_definition(payload: dict[str, Any], *, now: datetime | None = None) -> AutomationPreview:
    """Validate and preview an automation-definition authoring payload.

    Pure and synchronous -- no I/O. Runs the ported server validators
    against a payload shaped like the server's
    `ScheduledTaskPreviewCreateRequest` and returns a filled
    `AutomationPreview`, matching the shape of the server's
    `ScheduledTaskPreviewResponse`. Fields that are only meaningful after
    a real server round trip (`id`, `payload_hash`, `risk_class`,
    `expires_at`, `created_by`, `consumed_at`, `created_definition_id`)
    are left at the model's defaults.

    Args:
        payload: A dict with `mode` (``"create"``/``"update"``, default
            ``"create"``), `family` (required -- `"recurring_question"` or
            `"agent_task"`), `definition_id`, `definition_version`,
            `name`, `description`, `config`, `input`, `schedule`,
            `visibility_policy`, `notification_policy`, `approval_policy`
            -- see
            `Tests/Scheduling/fixtures/server_responses/automation_endpoints.md`
            (`ScheduledTaskPreviewCreateRequest`).
        now: The current time, used to compute `schedule_preview`'s
            upcoming occurrences. Defaults to `datetime.now(timezone.utc)`.

    Returns:
        An `AutomationPreview` with `status` `"valid"` or `"invalid"`,
        `validation_errors` and `warnings` from every validator that ran,
        and a `schedule_preview` dict (the normalized schedule plus a
        `next_occurrences` list).

    Raises:
        ValueError: If `family` is missing or not a recognized
            `AutomationFamily` member.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    family = AutomationFamily(payload.get("family"))
    mode = payload.get("mode") or "create"
    definition_id = payload.get("definition_id")
    definition_version = payload.get("definition_version")

    mode_errors: list[dict[str, Any]] = []
    if mode == "update":
        if not definition_id:
            mode_errors.append(
                field_error(
                    "definition_id", "required_for_update", "Definition id is required for update previews."
                )
            )
        if definition_version is None:
            mode_errors.append(
                field_error(
                    "definition_version",
                    "required_for_update",
                    "Definition version is required for update previews.",
                )
            )
    else:
        if definition_id is not None:
            mode_errors.append(
                field_error(
                    "definition_id", "not_allowed_for_create", "Definition id is not allowed for create previews."
                )
            )
        if definition_version is not None:
            mode_errors.append(
                field_error(
                    "definition_version",
                    "not_allowed_for_create",
                    "Definition version is not allowed for create previews.",
                )
            )

    schedule, schedule_errors, schedule_warnings = validate_schedule(payload.get("schedule") or {})
    visibility_policy_str = _normalize_visibility_policy(family.value, payload.get("visibility_policy"))

    base: dict[str, Any] = {
        "name": payload.get("name"),
        "description": payload.get("description"),
        "config": payload.get("config") or {},
        "input": payload.get("input") or {},
        "schedule": schedule,
        "visibility_policy": visibility_policy_str,
        "notification_policy": payload.get("notification_policy") or {},
        "approval_policy": payload.get("approval_policy") or {},
    }

    family_errors: list[dict[str, Any]] = []
    if family is AutomationFamily.RECURRING_QUESTION:
        normalized, errors, warnings = validate_recurring_question_config(base)
    else:
        normalized = dict(base)
        errors = []
        warnings = []
        family_errors.append(
            field_error("family", "unsupported", f"Unsupported automation family: {family.value}")
        )
    redaction_policy: dict[str, Any] = {"mode": "none", "fields": []}

    normalized["family"] = family.value
    normalized["schedule"] = schedule
    normalized["visibility_policy"] = visibility_policy_str

    validation_errors = [*mode_errors, *family_errors, *errors, *schedule_errors]
    combined_warnings = [*warnings, *schedule_warnings]
    warnings_out = [{"message": warning} for warning in combined_warnings]

    schedule_preview = {**schedule, "next_occurrences": _next_occurrences(schedule, now=now)}
    status = PreviewStatus.INVALID if validation_errors else PreviewStatus.VALID

    return AutomationPreview(
        mode=mode,
        family=family,
        definition_id=definition_id,
        definition_version=definition_version,
        status=status,
        normalized_config=normalized,
        validation_errors=validation_errors,
        warnings=warnings_out,
        visibility_policy={"mode": visibility_policy_str},
        schedule_preview=schedule_preview,
        redaction_policy=redaction_policy,
    )
