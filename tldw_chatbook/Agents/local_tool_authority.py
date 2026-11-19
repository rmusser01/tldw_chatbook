"""Native local-tool authority rechecks and review-source read admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_tool_provider import RunAdmittedWorkspaceRoot


def authorize_review_source(
    self, name: str, args: dict, *, expected_binding_id: str
) -> str | None:
    """Reuse native read permission/approval without dispatching a second reader.

    Called outside review-store/DB locks. This does not grant write access,
    execute a local tool, or change the selected binding.
    """
    from tldw_chatbook.Agents import local_tool_provider as provider

    if (
        name not in {"fs_read", "fs_list", "fs_grep", "git_diff"}
        or name not in self._specs
    ):
        return "source_permission_unavailable"
    try:
        authority, _clean_args = self._select_admitted_root(name, args)
        if (
            authority is None
            or authority.binding_id != expected_binding_id
            or not self._authority_is_valid(authority, write=False)
            or self._kill_switch_engaged()
        ):
            return "source_authority_unavailable"
        gate = self._verdict_for(name, args, provider.current_run_id())
        if gate.verdict != "allow":
            return "source_permission_" + gate.verdict
        # Recheck both the binding and the current deny floor after a
        # potentially blocking native approval callback.
        state = self._resolve_state(self.hub_tool_for(name))
        if state.state == "deny" or not self._authority_is_valid(
            authority, write=False
        ):
            return "source_permission_denied"
        return None
    except Exception:  # noqa: BLE001 - review-source admission must fail closed
        return "source_permission_unavailable"


def authority_is_valid(
    self,
    authority: RunAdmittedWorkspaceRoot | None,
    *,
    write: bool,
) -> bool:
    """Fail closed while revalidating legacy or run-admitted authority."""
    if authority is None:
        return self._root_is_valid()
    if write and not authority.allow_write:
        return False
    try:
        return bool(authority.guard(write))
    except Exception:  # noqa: BLE001 - invocation must fail closed
        return False


def kill_switch_engaged(self) -> bool:
    """Never-raise kill-switch read.

    The fail-open/fail-closed POLICY for store read errors lives in the
    injected callable, not here: the controller's composition closure
    swallows `get_kill_switch` errors and returns False (fail open --
    deliberate MCPToolProvider._kill_switch_engaged parity), so in
    production this guard never sees a raise. What remains here is
    only the protocol boundary: if the injected callable ITSELF
    propagates (a test double, or a future composition without its own
    guard), invoke() still cannot raise, and that case fails closed
    (treated as engaged).
    """
    from tldw_chatbook.Agents import local_tool_provider as provider

    try:
        return bool(self._kill_switch())
    except Exception as exc:  # noqa: BLE001 — invoke() must never raise
        provider.logger.warning(f"LocalToolProvider: kill_switch read failed: {exc}")
        return True
