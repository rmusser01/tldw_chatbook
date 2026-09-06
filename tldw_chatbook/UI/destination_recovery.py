"""Shared recovery copy helpers for destination and shell blocked states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


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
        severity: Callout tint: ``warning`` (`$warning`, the `.ds-recovery-callout`
            default every blocked state has always rendered with) or ``error``
            (`$error`, a hard failure).
        retry_id: Id of the Retry button that recovers this state, for callouts
            that carry their own Retry. Empty when recovery is not a retry.
        attempt: How many consecutive times this same failure has repeated
            (task-31632 final review I-1). A caller that dedups a fresh state
            against the previous one by equality (this dataclass's own
            ``==``) would otherwise render nothing when a Retry reproduces a
            byte-identical failure -- bumping this on repeat breaks that
            equality so the repaint, and the reason for it, are both visible.
            Defaults to 1 (a first attempt); ``message`` stays silent about it
            until it actually repeats.
    """

    status_label: str
    unavailable_what: str
    why: str
    next_action: str
    recovery_action: str
    authority_owner: str
    stable_selector: str
    disabled_tooltip: str
    severity: Literal["error", "warning"] = "warning"
    retry_id: str = ""
    attempt: int = 1

    @staticmethod
    def _sentence(value: str) -> str:
        text = value.strip()
        if not text or text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    @property
    def message(self) -> str:
        """Render the one-line callout copy: what failed, then why.

        Silent about ``attempt`` on a first failure; once a caller bumps it
        past 1 (the same failure repeating), the label names the attempt so
        a retry against an unchanged failure still reads as a fresh press.
        """

        base = f"{self.unavailable_what} · {self.why}"
        if self.attempt > 1:
            return f"{base} · attempt {self.attempt}"
        return base

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
        policy_message
        if policy_message is not None
        else getattr(exc, "user_message", None),
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


def load_failure_recovery_state(
    *,
    what: str,
    reason: str,
    retry_id: str,
    stable_selector: str,
    kind: Literal["error", "timeout"] = "error",
) -> DestinationRecoveryState:
    """Build one recovery callout for a load that failed, with its reason.

    Args:
        what: What could not be loaded, as a clause ("Couldn't load page 1").
        reason: Why it failed, in the reader's terms ("database is locked").
        retry_id: Id of the Retry button rendered inside the callout.
        stable_selector: Stable widget selector for the rendered callout.
        kind: ``timeout`` for a deadline a later attempt may beat (tinted
            `$warning`), ``error`` for a hard failure (tinted `$error`).

    Returns:
        Recovery state whose `message` reads "<what> · <reason>".
    """

    state_what = _clause(what, "Couldn't load this view")
    state_reason = _clause(reason, "reason unavailable")
    return DestinationRecoveryState(
        status_label="Timed out" if kind == "timeout" else "Load failed",
        unavailable_what=state_what,
        why=state_reason,
        next_action="Retry",
        recovery_action="Retry",
        # Hardcoded on purpose (task-31632 review): every caller so far is a
        # local read, and the callout does not paint the owner -- only
        # ``visible_copy`` does, which no load-failure surface uses. Make it
        # a parameter when a remote/server load first needs one.
        authority_owner="local data source",
        stable_selector=stable_selector,
        disabled_tooltip=DestinationRecoveryState._sentence(
            f"{state_what} · {state_reason}"
        ),
        severity="warning" if kind == "timeout" else "error",
        retry_id=retry_id,
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
    install_target: str | None = None,
    install_targets: Iterable[str] | None = None,
    stable_selector: str,
    recovery_action: str,
    authority_owner: str = "optional dependency",
) -> DestinationRecoveryState:
    """Build recovery copy for a missing optional dependency blocker.

    Args:
        unavailable_what: Specific workflow or control blocked by the missing dependency.
        missing_dependencies: Missing package, extra, or feature names.
        install_target: User-facing install command or setup target.
        install_targets: Optional source/package install commands. When provided, the
            first command is treated as the source checkout path and the second as the
            packaged install path.
        stable_selector: Stable widget selector for the rendered recovery state.
        recovery_action: Target setup area or action.
        authority_owner: Owner of the blocker.

    Returns:
        Destination recovery state with dependency-specific visible copy and tooltip.

    Raises:
        ValueError: If neither `install_target` nor `install_targets` is provided.
    """

    dependency_names = _dependency_names(missing_dependencies)
    why = f"Missing optional dependencies: {dependency_names}."
    install_target_list = [
        str(target).strip() for target in (install_targets or ()) if str(target).strip()
    ]
    if install_target_list:
        if len(install_target_list) >= 2:
            next_action = (
                f"Install with {install_target_list[0]} for source checkouts or "
                f"{install_target_list[1]} for packaged installs, then restart."
            )
        else:
            next_action = f"Install with {install_target_list[0]} and restart."
    elif install_target:
        next_action = f"Install with {install_target} and restart."
    else:
        raise ValueError("install_target or install_targets is required")
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
