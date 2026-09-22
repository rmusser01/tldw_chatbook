"""Shared request-validation helpers (ADR-176).

The bound/coercion checkers and the passthrough-allowlist lookup were
byte-identical (modulo issue types) between the image and video validation
modules; they live here once. The ``issue`` factory and the
extra-params allowlist enforcement stay per modality -- they construct
modality-specific validation-issue records.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

IssueFactory = Callable[[str, str], Any]


def positive_int_attr(config: Any, attr: str, default: int) -> int:
    """Read a positive int attribute from a config object.

    Args:
        config: The modality config.
        attr: Attribute name.
        default: Fallback when absent or non-positive.

    Returns:
        The attribute value when positive, else ``default``.
    """
    try:
        value = int(getattr(config, attr, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def validate_int_bound(
    issues: list,
    value: Any,
    *,
    path: str,
    max_value: int,
    issue: IssueFactory,
) -> bool:
    """Validate an optional int in ``[1, max_value]``.

    Args:
        issues: List collecting validation issues.
        value: Raw request value (None skips).
        path: Issue path.
        max_value: Inclusive upper bound.
        issue: Factory building issues.

    Returns:
        True when the value is valid or None.
    """
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(issue(f"{path} must be an integer", path))
        return False
    if value <= 0 or value > max_value:
        issues.append(issue(f"{path} out of range", path))
        return False
    return True


def validate_positive_finite_float(
    issues: list,
    value: Any,
    *,
    path: str,
    issue: IssueFactory,
) -> None:
    """Validate an optional finite positive float.

    Args:
        issues: List collecting validation issues.
        value: Raw request value (None skips).
        path: Issue path.
        issue: Factory building issues.
    """
    if value is None:
        return
    if isinstance(value, bool):
        issues.append(issue(f"{path} must be a finite positive number", path))
        return
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        issues.append(issue(f"{path} must be a finite positive number", path))
        return
    if not math.isfinite(candidate) or candidate <= 0:
        issues.append(issue(f"{path} must be a finite positive number", path))


def allowed_extra_params_for_backend(
    backend: str,
    config: Any,
    attr_by_backend: Mapping[str, str],
) -> set[str]:
    """Return configured passthrough allowlist keys for a backend.

    Args:
        backend: Backend name as spelled in requests.
        config: The modality's generation config.
        attr_by_backend: Backend name -> config attribute holding its
            ``allowed_extra_params`` list.

    Returns:
        The allowlisted keys (empty for unknown backends).
    """
    backend_name = str(backend or "").strip().lower()
    attr = attr_by_backend.get(backend_name)
    if not attr:
        return set()
    return {str(item).strip() for item in getattr(config, attr, []) or [] if str(item).strip()}
