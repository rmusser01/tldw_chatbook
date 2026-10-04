"""Freeze only the current task's existing model-bound execution evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

from .models import RuleEvidence, RuleInput, RuleSource

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

MAX_CAPTURE_BYTES = 64 * 1024


def capture_rule_input(store: ConsoleChatStore, source: RuleSource) -> RuleInput:
    """Capture an eligible response and task-chain facts without new retrieval.

    Result bodies come only from already model-bound continuation records.
    Definitive callbacks contribute body-free dispatch facts. Missing provenance
    and unavailable work revisions stay unknown; display previews never fill them.
    """
    try:
        session = next((s for s in store.sessions() if s.id == source.session_id), None)
        if (
            session is None
            or session.persisted_conversation_id != source.conversation_id
        ):
            raise ValueError("rule_source_owner_changed")
        path = store.active_path_message_ids(source.session_id)
        messages = {
            m.id: m for m in store.read_only_messages_for_session(source.session_id)
        }
        target = next(
            (
                m
                for m in messages.values()
                if source.message_id in (m.id, m.persisted_message_id)
            ),
            None,
        )
        if target is None or target.id not in path:
            raise ValueError("rule_source_unavailable")
        leaf = messages[path[-1]]
        if (
            source.branch_id not in (leaf.id, leaf.persisted_message_id)
            or target.role is not ConsoleMessageRole.ASSISTANT
            or target.status != "complete"
            or target.generation_metadata
        ):
            raise ValueError("rule_source_ineligible")
        if store.response_rule_source_version(target.id) != source.message_version:
            raise ValueError("rule_source_stale")
        prefix = path[: path.index(target.id) + 1]
        task_root = store.response_rule_task_root(target.id)
        if task_root is None:
            task_root = next(
                (
                    node_id
                    for node_id in reversed(prefix)
                    if messages[node_id].role is ConsoleMessageRole.USER
                ),
                None,
            )
        if (
            task_root not in prefix
            or messages[task_root].role is not ConsoleMessageRole.USER
        ):
            raise ValueError("rule_source_task_unavailable")
        chain = prefix[prefix.index(task_root) :]
        evidence = []
        complete = True
        remaining = MAX_CAPTURE_BYTES
        for node_id in chain:
            message = messages[node_id]
            facts = {
                fact.ref: fact for fact in store.response_rule_tool_results(node_id)
            }
            checkpoint = message.provider_continuation
            calls = (
                tuple(call for round_ in checkpoint.rounds for call in round_.calls)
                if checkpoint is not None
                else ()
            )
            for call in calls:
                ref = f"{node_id}:{call.call_id}"
                fact = facts.pop(ref, None)
                if fact is None:
                    fact = RuleEvidence(
                        ref,
                        "not_started" if call.state == "pending" else "uncertain",
                        "unknown",
                        None,
                        "",
                    )
                text = call.result.value if call.result is not None else ""
                encoded = text.encode("utf-8")
                if len(encoded) > remaining:
                    text = encoded[:remaining].decode("utf-8", errors="ignore")
                    complete = False
                remaining = max(0, remaining - len(text.encode("utf-8")))
                evidence.append(replace(fact, text=text))
                complete &= fact.state != "uncertain" and call.result is not None
            # Legacy adapters provide definitive facts but no exact provider-bound
            # body at this seam. Do not substitute raw callback or display text.
            if facts:
                evidence.extend(facts.values())
                complete = False
        if store.response_rule_source_version(target.id) != source.message_version:
            raise ValueError("rule_source_stale")
        return RuleInput(
            messages[task_root].content,
            target.content,
            tuple(evidence),
            bool(evidence) and complete,
            None,
        )
    except KeyError:
        raise ValueError("rule_source_unavailable") from None
