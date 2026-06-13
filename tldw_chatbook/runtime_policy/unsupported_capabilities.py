"""Validation helpers for source-scoped unsupported capability reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .registry import CAPABILITY_REGISTRY


class UnsupportedCapabilityReportError(ValueError):
    """Raised when an unsupported-capability report cannot be safely consumed."""


_REQUIRED_KEYS = frozenset(
    {
        "operation_id",
        "source",
        "supported",
        "reason_code",
        "user_message",
        "affected_action_ids",
    }
)
_ALLOWED_SOURCES = frozenset({"local", "server", "workspace"})


def _reject(message: str) -> None:
    raise UnsupportedCapabilityReportError(message)


def _require_non_empty_string(value: Any, *, field_name: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        _reject(f"Unsupported capability report item {index} has invalid {field_name!r}.")
    return value


def _normalize_affected_action_ids(value: Any, *, index: int, registry: Mapping[str, Any]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        _reject(f"Unsupported capability report item {index} has invalid 'affected_action_ids'.")

    action_ids: list[str] = []
    for action_index, action_id in enumerate(value):
        if not isinstance(action_id, str) or not action_id.strip():
            _reject(
                f"Unsupported capability report item {index} has invalid action_id at "
                f"affected_action_ids[{action_index}]."
            )
        if action_id not in registry:
            _reject(
                f"Unsupported capability report item {index} references unknown action_id: {action_id}"
            )
        action_ids.append(action_id)
    return action_ids


def validate_unsupported_capability_report(
    report: Iterable[Mapping[str, Any]],
    *,
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate and copy an unsupported-capability report.

    Scope services intentionally expose plain dictionaries so UI code can render
    them without importing each domain. This helper is the central contract that
    keeps those dictionaries source-safe and aligned to runtime-policy action IDs.
    """

    if isinstance(report, (str, bytes)) or not isinstance(report, Iterable):
        _reject("Unsupported capability report must be an iterable of mappings.")

    action_registry = registry or CAPABILITY_REGISTRY
    normalized_report: list[dict[str, Any]] = []

    for index, item in enumerate(report):
        if not isinstance(item, Mapping):
            _reject(f"Unsupported capability report item {index} is not a mapping.")

        missing_keys = _REQUIRED_KEYS.difference(item)
        if missing_keys:
            _reject(
                f"Unsupported capability report item {index} is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        operation_id = _require_non_empty_string(item["operation_id"], field_name="operation_id", index=index)
        source = _require_non_empty_string(item["source"], field_name="source", index=index)
        if source not in _ALLOWED_SOURCES:
            _reject(f"Unsupported capability report item {index} has unsupported source: {source}")
        if item["supported"] is not False:
            _reject(f"Unsupported capability report item {index} must set supported=False.")
        _require_non_empty_string(item["reason_code"], field_name="reason_code", index=index)
        _require_non_empty_string(item["user_message"], field_name="user_message", index=index)

        normalized_item = deepcopy(dict(item))
        normalized_item["operation_id"] = operation_id
        normalized_item["source"] = source
        normalized_item["affected_action_ids"] = _normalize_affected_action_ids(
            item["affected_action_ids"],
            index=index,
            registry=action_registry,
        )
        normalized_report.append(normalized_item)

    return normalized_report


def collect_unsupported_capability_reports(
    reports_by_scope: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate reports from multiple panes/services and return a flat labeled list."""

    if not isinstance(reports_by_scope, Mapping):
        _reject("Unsupported capability report collection must be a mapping.")

    collected: list[dict[str, Any]] = []
    for report_scope, report in reports_by_scope.items():
        if not isinstance(report_scope, str) or not report_scope.strip():
            _reject("Unsupported capability report collection contains an invalid scope label.")
        for item in validate_unsupported_capability_report(report, registry=registry):
            record = dict(item)
            record["report_scope"] = report_scope
            collected.append(record)
    return collected
