"""Rule input pinned to an already accepted Goal iteration, never inferred from text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_goal_runs import GoalIterationAuthorization


@dataclass(frozen=True)
class GoalRuleInput:
    authorization: GoalIterationAuthorization = field(repr=False)
    request: Any = field(repr=False)
    task_message_id: str
    task_version: int

    def validate(self, store) -> str:
        """Recheck the actual authority and its unchanged admitted native handoff."""
        authority = self.authorization
        if (
            type(authority) is not GoalIterationAuthorization
            or not authority.accepted
            or not authority._coordinator.authorizes(authority, self.request.session_id)
        ):
            raise ValueError("rule_goal_authority_unavailable")
        try:
            authority.context.check()
        except AutomaticWorkRefused as exc:
            raise ValueError("rule_goal_authority_unavailable") from exc
        message = store.get_message(self.task_message_id)
        if (
            message.role is not ConsoleMessageRole.SYSTEM
            or getattr(message.metadata, "origin", None) != "goal_iteration"
            or message.content != self.request.draft.strip()
            or store.response_rule_source_version(message.id) != self.task_version
            or message.id not in store.active_path_message_ids(self.request.session_id)
        ):
            raise ValueError("rule_goal_handoff_changed")
        return message.id


def pin_goal_input(
    controller, message_id: str, configuration, authorization
) -> GoalRuleInput | None:
    """Bind the real capability, runtime custody request and immediate parent row."""
    if type(authorization) is not GoalIterationAuthorization:
        return None
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    turn_id = configuration.skill_context_maximum.get("plugin_turn_id")
    record = getattr(runtime, "_turn_custody", {}).get(turn_id)
    request = getattr(record, "request", None)
    store = controller.store
    session_id = store.session_id_for_message(message_id)
    if (
        request is None
        or request.session_id != session_id
        or authorization.session_id != session_id
    ):
        raise ValueError("rule_goal_handoff_unavailable")
    path = store.active_path_message_ids(session_id)
    position = path.index(message_id)
    if position == 0:
        raise ValueError("rule_goal_handoff_unavailable")
    root = path[position - 1]
    pinned = GoalRuleInput(
        authorization, request, root, store.response_rule_source_version(root)
    )
    pinned.validate(store)
    return pinned
