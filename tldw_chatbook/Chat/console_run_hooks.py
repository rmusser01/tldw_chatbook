"""Console Hook bodies; state and replaceable names stay on the live controller."""

from __future__ import annotations

from typing import Any


def _notify_run_hook_approval(
    self, kind: str, payload: dict[str, Any], state: dict[str, Any]
) -> None:
    """Publish one successfully admitted permission round, including headless runs."""
    engine = self._run_hooks_engine()
    if engine is None:
        return
    from tldw_chatbook.Agents.run_hooks import summarize_hook_arguments

    session_id = payload.get("session_id") or state.get("session_id")
    run_id = (
        state.get("run_id")
        if kind == "worktree_merge"
        else payload.get("run_id") or state.get("run_id")
    )
    if kind == "approval":
        calls = [
            {
                "name": row.get("llm_name") or row.get("tool_name") or "",
                "args_summary": summarize_hook_arguments(row.get("arguments") or {}),
            }
            for row in payload.get("calls", ())
        ]
    elif kind == "worktree_merge":
        action = payload.get("action") or payload.get("mode")
        arguments = {
            key: payload[key]
            for key in (
                "handle_id",
                "run_id",
                "action",
                "mode",
                "branch",
                "worktree",
                "source",
                "destination",
            )
            if key in payload
        }
        calls = [
            {
                "name": (
                    "discard_agent_worktree"
                    if action == "discard"
                    else "merge_agent_worktree"
                ),
                "args_summary": summarize_hook_arguments(arguments),
            }
        ]
    else:
        arguments = (
            {"url": payload.get("url", "")}
            if kind == "skill_install"
            else {
                key: payload[key]
                for key in ("skill_name", "script_path", "mechanism", "args")
                if key in payload
            }
        )
        calls = [
            {
                "name": "install_skill"
                if kind == "skill_install"
                else "run_skill_script",
                "args_summary": summarize_hook_arguments(arguments),
            }
        ]
    engine.notify(
        "ApprovalRequested",
        session_id=session_id,
        run_id=run_id,
        data={
            "calls": calls,
            "session_active": bool(
                session_id == self.store.active_session_id
                and self._interrupt_host.view_visible is not False
                and (
                    self.set_pending_decision is not None
                    or self._interrupt_host._setter(kind) is not None
                )
            ),
        },
    )


def _run_hooks_engine(self):
    """Resolve the app-owned run-hooks engine for this send, or ``None``.

    Spec 2026-09-11 (Task 7): the submit path reaches the engine through
    the optional ``ensure_run_hooks`` accessor (the Task 5 bridge seam --
    a bound ``ConsoleRuntime.ensure_run_hooks``, or a test double).
    ``None`` -- no accessor wired, or no ``[hooks]`` configured -- means
    the fire site skips entirely; the accessor is consulted per send, so
    an engine built after the first-ever ``[hooks]`` entry appears is
    picked up without rebuilding this controller.
    """
    accessor = self._ensure_run_hooks
    return accessor() if accessor is not None else None


def _hook_admission_reason(self) -> str | None:
    from tldw_chatbook.Agents.run_hooks import inspect_hooks_config
    from tldw_chatbook.config import read_hooks_config_snapshot

    try:
        if self._hook_permissions_accessor is not None:
            return self._hook_permissions_accessor().snapshot().blocked_reason
        saved = read_hooks_config_snapshot()
        inventory = inspect_hooks_config(
            {"hooks": saved.section} if saved.section_present else {}
        )
        if not inventory.requires_authority:
            return None
        return "Hook review required; permission owner unavailable."
    except Exception:  # noqa: BLE001 -- unavailable authority must refuse admission
        return "Hooks unavailable; review or disable hooks before sending."
