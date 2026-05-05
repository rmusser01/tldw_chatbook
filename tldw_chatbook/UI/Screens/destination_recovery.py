"""Shared recovery copy helpers for destination shell blocked states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DestinationRecoveryState:
    """Taxonomy-aligned recovery state for a disabled destination action.

    Args:
        status_label: Short user-facing state label.
        unavailable_what: Specific workflow or control that cannot run.
        why: Immediate reason in user language.
        next_action: Concrete user step that can unblock or recover the workflow.
        recovery_action: Target route, retry action, setup action, or selection action.
        authority_owner: Owner of the capability or blocker.
        stable_selector: Stable widget selector used to expose and test this state.
        disabled_tooltip: Tooltip copy for the disabled control.
    """

    status_label: str
    unavailable_what: str
    why: str
    next_action: str
    recovery_action: str
    authority_owner: str
    stable_selector: str
    disabled_tooltip: str

    @staticmethod
    def _sentence(value: str) -> str:
        text = value.strip()
        if not text or text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    @property
    def visible_copy(self) -> str:
        """Render visible multi-line recovery copy."""

        return "\n".join(
            (
                self.status_label,
                f"Unavailable: {self._sentence(self.unavailable_what)}",
                f"Why: {self._sentence(self.why)}",
                f"Next: {self._sentence(self.next_action)}",
                f"Recovery: {self._sentence(self.recovery_action)}",
                f"Owner: {self._sentence(self.authority_owner)}",
            )
        )


def _clause(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _policy_recovery_for_reason(reason_code: str | None) -> tuple[str, str, str]:
    normalized_reason = str(reason_code or "").strip().lower()
    if normalized_reason == "wrong_source":
        return (
            "Wrong source",
            "Switch to the required source, then retry this workflow.",
            "Source switch or Settings",
        )
    if normalized_reason in {"server_not_configured", "server_profile_missing"}:
        return (
            "Server not configured",
            "Add an active server profile in Settings before retrying.",
            "Settings",
        )
    if normalized_reason in {"server_unreachable", "server_unavailable"}:
        return (
            "Server unavailable",
            "Check server availability, then retry this workflow.",
            "Retry",
        )
    if normalized_reason in {
        "server_auth_required",
        "auth_required",
        "credential_store_unavailable",
        "server_credentials_unavailable",
    }:
        return (
            "Server sign-in required",
            "Reconnect or configure server credentials in Settings before retrying.",
            "Settings",
        )
    if normalized_reason in {
        "server_session_invalid",
        "stale_authorization",
        "profile_no_longer_authorized",
    }:
        return (
            "Server session expired",
            "Re-authenticate the active server profile before retrying.",
            "Settings",
        )
    if normalized_reason == "capability_disabled":
        return (
            "Capability disabled",
            "Enable this capability in Settings or the governing policy before retrying.",
            "Settings or governing policy",
        )
    return (
        "Policy denied",
        "Review workspace policy or ask the authority owner to allow this action.",
        "Workspace policy",
    )


def policy_denied_recovery_state(
    exc: Any,
    *,
    unavailable_what: str,
    stable_selector: str,
    policy_message: str | None = None,
) -> DestinationRecoveryState:
    """Map a runtime-policy denial into visible destination recovery copy.

    Args:
        exc: Runtime-policy denial object with reason, message, and owner fields.
        unavailable_what: Specific workflow or control blocked by the denial.
        stable_selector: Stable widget selector for the rendered recovery state.
        policy_message: Optional sanitized policy message to prefer over `exc`.

    Returns:
        Destination recovery state with taxonomy-aligned visible copy and tooltip.
    """

    status_label, next_action, recovery_action = _policy_recovery_for_reason(
        getattr(exc, "reason_code", None)
    )
    why = _clause(
        policy_message if policy_message is not None else getattr(exc, "user_message", None),
        "Runtime policy blocked this action",
    )
    authority_owner = _clause(getattr(exc, "authority_owner", None), "runtime policy")
    disabled_tooltip = " ".join(
        (
            DestinationRecoveryState._sentence(why),
            DestinationRecoveryState._sentence(next_action),
        )
    )
    return DestinationRecoveryState(
        status_label=status_label,
        unavailable_what=unavailable_what,
        why=why,
        next_action=next_action,
        recovery_action=recovery_action,
        authority_owner=authority_owner,
        stable_selector=stable_selector,
        disabled_tooltip=disabled_tooltip,
    )


def _dependency_names(missing_dependencies: Iterable[str] | str) -> str:
    if isinstance(missing_dependencies, str):
        dependencies = [missing_dependencies]
    else:
        dependencies = [str(dependency).strip() for dependency in missing_dependencies]
    dependencies = [dependency for dependency in dependencies if dependency]
    return ", ".join(dependencies) or "required optional dependency"


def optional_dependency_recovery_state(
    *,
    unavailable_what: str,
    missing_dependencies: Iterable[str] | str,
    install_target: str,
    stable_selector: str,
    recovery_action: str,
    authority_owner: str = "optional dependency",
) -> DestinationRecoveryState:
    """Build recovery copy for a missing optional dependency blocker.

    Args:
        unavailable_what: Specific workflow or control blocked by the missing dependency.
        missing_dependencies: Missing package, extra, or feature names.
        install_target: User-facing install command or setup target.
        stable_selector: Stable widget selector for the rendered recovery state.
        recovery_action: Target setup area or action.
        authority_owner: Owner of the blocker.

    Returns:
        Destination recovery state with dependency-specific visible copy and tooltip.
    """

    dependency_names = _dependency_names(missing_dependencies)
    why = f"Missing optional dependencies: {dependency_names}."
    next_action = f"Install with {install_target} and restart."
    disabled_tooltip = " ".join(
        (
            DestinationRecoveryState._sentence(unavailable_what),
            DestinationRecoveryState._sentence(why),
            DestinationRecoveryState._sentence(next_action),
        )
    )
    return DestinationRecoveryState(
        status_label="Dependency missing",
        unavailable_what=unavailable_what,
        why=why,
        next_action=next_action,
        recovery_action=recovery_action,
        authority_owner=authority_owner,
        stable_selector=stable_selector,
        disabled_tooltip=disabled_tooltip,
    )
