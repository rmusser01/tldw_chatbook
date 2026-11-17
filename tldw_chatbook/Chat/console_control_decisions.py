"""Fixed local decisions forwarded to the exact original pending owners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .console_hook_review import HookReviewResult


@dataclass(frozen=True, slots=True)
class ApprovalRoundDecision:
    round_id: str
    call_ids: tuple[str, ...]
    decisions: dict[str, str]


@dataclass(frozen=True, slots=True)
class ProjectBindingDecision:
    decision_id: str
    action: str
    binding_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectDispatchDecision:
    decision_id: str
    action: str


@dataclass(frozen=True, slots=True)
class InitialHookDecision:
    review_id: str
    generation: int
    result: HookReviewResult
    presentation_token: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class ChatCreateDecision:
    request_id: str
    allow: bool
    remember: bool


@dataclass(frozen=True, slots=True)
class QuestionDecision:
    request_id: str
    answers: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SkillInstallDecision:
    request_id: str
    allow: bool


@dataclass(frozen=True, slots=True)
class SkillScriptDecision:
    request_id: str
    allow: bool
    remember: bool


@dataclass(frozen=True, slots=True)
class WorktreeMergeDecision:
    request_id: str
    allow: bool


Decision = (
    ApprovalRoundDecision
    | ProjectBindingDecision
    | ProjectDispatchDecision
    | InitialHookDecision
    | ChatCreateDecision
    | QuestionDecision
    | SkillInstallDecision
    | SkillScriptDecision
    | WorktreeMergeDecision
)


def resolve_original_decision(controls, identity, decision):
    runtime, host = controls._runtime, controls._host
    if type(decision) in (ProjectBindingDecision, ProjectDispatchDecision):
        with runtime._project_decision_lock:
            pending = (
                runtime._project_binding_decisions
                if type(decision) is ProjectBindingDecision
                else runtime._project_dispatch_decisions
            ).get(decision.decision_id)
            if pending is None or pending.session_id != identity.session_id:
                return False
        if type(decision) is ProjectBindingDecision:
            return runtime.resolve_project_instruction_binding(
                decision.decision_id, decision.action, decision.binding_id
            )
        return runtime.resolve_project_instruction_dispatch(
            decision.decision_id, decision.action
        )

    kinds = {
        ApprovalRoundDecision: "approval",
        InitialHookDecision: "hook_review",
        ChatCreateDecision: "chat_create",
        QuestionDecision: "question",
        SkillInstallDecision: "skill_install",
        SkillScriptDecision: "skill_script",
        WorktreeMergeDecision: "worktree_merge",
    }
    if type(decision) not in kinds:
        raise TypeError("Unsupported local Console decision")
    key = (
        decision.round_id
        if type(decision) is ApprovalRoundDecision
        else decision.review_id
        if type(decision) is InitialHookDecision
        else decision.request_id
    )
    with host.lock:
        state = host.registries[kinds[type(decision)]].get(key)
        if state is None or state.get("session_id") != identity.session_id:
            return False
        if (
            type(decision) is ApprovalRoundDecision
            and tuple(state["names"]) != decision.call_ids
        ):
            return False
    if type(decision) is ApprovalRoundDecision:
        controls._permissions.resolve_approval_round(
            decision.decisions, round_id=decision.round_id
        )
    elif type(decision) is InitialHookDecision:
        return runtime.resolve_initial_hook_review(
            decision.review_id,
            decision.generation,
            decision.result,
            presentation_token=decision.presentation_token,
        )
    elif type(decision) is ChatCreateDecision:
        host.resolve_pending_chat_create(
            decision.allow, decision.remember, decision.request_id
        )
    elif type(decision) is QuestionDecision:
        host.resolve_pending_question(decision.answers, decision.request_id)
    elif type(decision) is SkillInstallDecision:
        host.resolve_pending_skill_install(
            decision.allow, request_id=decision.request_id
        )
    elif type(decision) is SkillScriptDecision:
        host.resolve_pending_skill_script(
            decision.allow, decision.remember, decision.request_id
        )
    elif type(decision) is WorktreeMergeDecision:
        host.resolve_pending_worktree_merge(
            decision.allow, request_id=decision.request_id
        )
    return True
