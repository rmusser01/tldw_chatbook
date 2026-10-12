"""Native tool result observations; never store raw result bodies as rule facts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Chat.console_chat_controller import _WATCHLISTS_RECEIPT_ID_RE


def observe_watchlists_operation_result(
    self,
    _run_id: str,
    _call_key: str,
    tool_name: str,
    result: Any,
) -> None:
    """WORKER THREAD: retain only canonical local receipt identities."""
    operation_ids = watchlists_operation_receipt_ids(tool_name, result)
    if not operation_ids:
        return
    with self._watchlists_receipt_lock:
        followed = list(self._followed_watchlists_operation_ids)
        for operation_id in operation_ids:
            if operation_id not in followed:
                followed.append(operation_id)
        self._followed_watchlists_operation_ids = tuple(followed)
    app = getattr(self, "app", None)
    marshal = getattr(app, "call_from_thread", None)
    if callable(marshal):
        marshal(self.remount_watchlists_operation_receipts)


def observe_response_rule_tool_result(
    self,
    session_id: str,
    message_id: str,
    run_id: str,
    call_key: str,
    tool_name: str,
    result: Any,
) -> None:
    """Retain definitive state without exposing raw sensitive result bodies."""
    self.observe_watchlists_operation_result(run_id, call_key, tool_name, result)
    from tldw_chatbook.Agents.agent_models import ToolResult

    if not isinstance(result, ToolResult):
        return
    if message_id not in self.store.active_path_message_ids(session_id):
        return
    self.store.record_response_rule_tool_result(
        message_id,
        call_key,
        state=result.dispatch_state or "uncertain",
        outcome="blocked"
        if result.outcome == "blocked"
        else "succeeded"
        if result.ok
        else "failed",
        tool_name=tool_name,
    )


def remount_watchlists_operation_receipts(self) -> None:
    """UI THREAD: publish the process-local canonical receipt snapshot."""
    callback = self.follow_watchlists_operations
    if callback is None:
        return
    with self._watchlists_receipt_lock:
        operation_ids = self._followed_watchlists_operation_ids
    callback(operation_ids)


def unfollow_watchlists_operation(self, operation_id: str) -> bool:
    """Forget one canonical receipt without touching its domain work."""
    if not _WATCHLISTS_RECEIPT_ID_RE.fullmatch(operation_id):
        return False
    with self._watchlists_receipt_lock:
        followed = self._followed_watchlists_operation_ids
        if operation_id not in followed:
            return False
        self._followed_watchlists_operation_ids = tuple(
            receipt_id for receipt_id in followed if receipt_id != operation_id
        )
    return True


def watchlists_operation_receipt_ids(
    tool_name: str,
    result: Any,
) -> tuple[str, ...]:
    """Extract canonical receipt identity from a structured local result."""
    if (
        tool_name
        not in {
            "watchlists_check_sources",
            "watchlists_generate_briefing",
        }
        or getattr(result, "ok", False) is not True
    ):
        return ()
    content = getattr(result, "content", None)
    if not isinstance(content, str):
        return ()
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return ()
    if not isinstance(payload, Mapping) or payload.get("status") != "accepted":
        return ()
    if tool_name == "watchlists_generate_briefing":
        candidates = (payload.get("operation_id"),)
    else:
        operations = payload.get("operations")
        if not isinstance(operations, list):
            return ()
        candidates = tuple(
            row.get("operation_id")
            for row in operations[:50]
            if isinstance(row, Mapping)
        )
    return tuple(
        operation_id
        for operation_id in candidates
        if isinstance(operation_id, str)
        and _WATCHLISTS_RECEIPT_ID_RE.fullmatch(operation_id)
    )
