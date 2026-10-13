"""Rule fences and evidence held by the native Chat store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

from .models import RuleEvidence
from .store import ResponseRuleStore


def subscribe_response_rule_invalidation(
    self, callback: Callable[[str, str], None]
) -> Callable[[], None]:
    """Fence native rule sources before user-owned source mutations."""
    self._response_rule_invalidations.append(callback)

    def unsubscribe() -> None:
        if callback in self._response_rule_invalidations:
            self._response_rule_invalidations.remove(callback)

    return unsubscribe


def invalidate_response_rule_source(self, session_id: str, reason: str) -> None:
    for callback in tuple(self._response_rule_invalidations):
        callback(session_id, reason)


def hold_response_rule_completion(self, message_id: str) -> None:
    self._message_or_raise(message_id)
    if message_id in self._response_rule_completion_holds:
        raise RuntimeError("response_rule_completion_already_held")
    self._response_rule_completion_holds[message_id] = None


def release_response_rule_completion(self, message_id: str) -> None:
    """Publish only the unchanged successful answer's once-owned signal."""
    pending = self._response_rule_completion_holds.pop(message_id, None)
    if pending is None:
        return
    try:
        message = self._message_or_raise(message_id)
        session_id = self.session_id_for_message(message_id)
        if (
            message.status == "complete"
            and self.active_leaf(session_id) == message_id
            and pending
            == (
                self.message_completion_generation(message_id),
                self.response_rule_source_version(message_id),
            )
        ):
            self._publish_message_completed(session_id, message_id)
    except KeyError:
        return


def response_rule_source_version(self, message_id: str) -> int:
    """Read an exact durable version or native semantic-content fence."""
    message = self._message_or_raise(message_id)
    if message.persisted_message_id is not None:
        try:
            version = self._persisted_message_version_or_reject(message)
        except Exception:  # noqa: BLE001 - unavailable durable fencing must refuse the source.
            raise ValueError("rule_source_version_unavailable") from None
        if version is None:
            raise ValueError("rule_source_version_unavailable")
        return version
    return self._message_speech_revisions[message_id] + 1


def bind_response_rule_task_root(self, message_id: str, user_message_id: str) -> None:
    """Pin a host-admitted correction to its original task on this branch."""
    session_id = self._message_session_index[message_id]
    path = self.active_path_message_ids(session_id)
    if (
        user_message_id not in path
        or self._message_or_raise(user_message_id).role is not ConsoleMessageRole.USER
    ):
        raise ValueError("rule_source_task_unavailable")
    with self._response_rule_evidence_lock:
        self._response_rule_task_roots[message_id] = user_message_id


def response_rule_task_root(self, message_id: str) -> str | None:
    """Resolve live or inert durable host ancestry on the actual active branch."""

    with self._response_rule_evidence_lock:
        root = self._response_rule_task_roots.get(message_id)
    if root is not None:
        return root
    session_id = self._message_session_index[message_id]
    session = self._session_or_raise(session_id)
    if self.response_rule_store is None or session.persisted_conversation_id is None:
        return None
    path = self.active_path_message_ids(session_id)
    if message_id not in path:
        return None
    prefix = path[: path.index(message_id) + 1]
    nodes = {node_id: self._message_or_raise(node_id) for node_id in prefix}
    persisted = {
        m.persisted_message_id: m.id for m in nodes.values() if m.persisted_message_id
    }
    current = message_id
    visited = set()
    while current not in visited:
        visited.add(current)
        message = nodes[current]
        parent = (
            self.response_rule_store.repository.historical_machine_parent(
                session.persisted_conversation_id, message.persisted_message_id
            )
            if message.persisted_message_id
            else None
        )
        if parent is None:
            return next(
                (
                    node_id
                    for node_id in reversed(prefix[: prefix.index(current)])
                    if nodes[node_id].role is ConsoleMessageRole.USER
                ),
                None,
            )
        parent_native = persisted.get(parent)
        if (
            parent_native is None
            or prefix.index(parent_native) >= prefix.index(current)
            or nodes[parent_native].role is not ConsoleMessageRole.ASSISTANT
        ):
            raise ValueError("rule_source_task_unavailable")
        current = parent_native
    raise ValueError("rule_source_task_unavailable")


def record_response_rule_tool_result(
    self, message_id: str, call_id: str, *, state: str, outcome: str, tool_name: str
) -> None:
    """Retain body-free facts from the real definitive tool callback."""
    from .models import RuleEvidence

    if state not in {"settled", "not_started", "uncertain"} or outcome not in {
        "succeeded",
        "failed",
        "blocked",
    }:
        raise ValueError("invalid_rule_dispatch_fact")
    self._message_or_raise(message_id)
    ref = f"{message_id}:{call_id}"
    fact = RuleEvidence(ref, state, outcome, None, f"{tool_name[:128]}: {outcome}")
    with self._response_rule_evidence_lock:
        previous = self._response_rule_results.setdefault(message_id, {}).get(ref)
        if previous is not None and previous != fact:
            fact = replace(fact, state="uncertain", outcome="unknown")
        self._response_rule_results[message_id][ref] = fact


def response_rule_tool_results(self, message_id: str) -> tuple[RuleEvidence, ...]:
    """Read frozen dispatch facts for one exact native response owner."""
    with self._response_rule_evidence_lock:
        return tuple(self._response_rule_results.get(message_id, {}).values())


def bind_response_rule_store(self, rules: ResponseRuleStore) -> None:
    """Bind one app-owned rules service sharing the exact persistence DB."""
    from .store import ResponseRuleStore

    binder = getattr(self.persistence, "bind_response_rule_store", None)
    if (
        not isinstance(rules, ResponseRuleStore)
        or rules.repository.db is not getattr(self.persistence, "db", None)
        or not callable(binder)
    ):
        raise ValueError("response_rules_persistence_owner_mismatch")
    if self.response_rule_store is not None and self.response_rule_store is not rules:
        raise ValueError("response_rules_owner_already_bound")
    binder(rules)
    self.response_rule_store = rules
    for session in getattr(self, "_sessions", {}).values():
        if session.ephemeral:
            rules.register_temporary(session.id)
