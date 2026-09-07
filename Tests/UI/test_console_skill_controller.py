"""Isolated policy tests for the Console skill controller (TASK-3070.6)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from tldw_chatbook.Chat.console_command_grammar import CommandParse, KIND_COMMAND
from tldw_chatbook.Chat.console_skill_resolver import SKILL_UNTRUSTED_REFUSE
from tldw_chatbook.UI.Console_Modules.skill import (
    CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE,
    ConsoleSkillController,
)
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState


class _SkillsService:
    def __init__(self, contexts: list[object]) -> None:
        self._contexts = list(contexts)
        self.calls: list[str | None] = []

    async def get_context(self, *, mode: str | None = None) -> object:
        self.calls.append(mode)
        value = self._contexts.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class _ChatController:
    def __init__(self) -> None:
        self.installs: list[tuple[bool, str | None]] = []
        self.scripts: list[tuple[bool, bool, str | None]] = []

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None
    ) -> None:
        self.installs.append((allow, request_id))

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, *, request_id: str | None
    ) -> None:
        self.scripts.append((allow, remember, request_id))


def _controller(
    contexts: list[object] | None = None,
) -> tuple[
    ConsoleSkillController,
    list[str],
    list[TaskResumeState],
    _SkillsService,
    _ChatController,
    list[str],
]:
    service = _SkillsService(contexts or [{}])
    app = SimpleNamespace(skills_scope_service=service)
    messages: list[str] = []
    states = [
        TaskResumeState(
            summary="keep",
            pending_approval={"round": "approval"},
            diff_summary="unchanged",
        )
    ]
    chat = _ChatController()
    syncs: list[str] = []

    async def _append(message: str) -> None:
        messages.append(message)

    def _set_state(state: TaskResumeState) -> None:
        states.append(state)

    controller = ConsoleSkillController(
        app_instance=app,
        append_native_console_system_message=_append,
        sync_console_command_popup=lambda: syncs.append("sync"),
        task_resume_state=lambda: states[-1],
        set_task_resume_state=_set_state,
        current_chat_controller=lambda: chat,
    )
    return controller, messages, states, service, chat, syncs


def _skill(
    name: str,
    *,
    description: str = "",
    user_invocable: bool = True,
    trust_blocked: bool = False,
) -> Mapping[str, Any]:
    return {
        "name": name,
        "description": description,
        "user_invocable": user_invocable,
        "trust_blocked": trust_blocked,
    }


@pytest.mark.asyncio
async def test_context_fetch_is_fresh_and_fails_closed() -> None:
    controller, _, _, service, _, _ = _controller(
        [
            {"available_skills": [_skill("first")]},
            RuntimeError("private provider detail"),
            ["not", "a", "mapping"],
        ]
    )

    first = await controller._fetch_console_skill_context()
    failed = await controller._fetch_console_skill_context()
    malformed = await controller._fetch_console_skill_context()

    assert first["available_skills"][0]["name"] == "first"
    assert failed == {}
    assert malformed == {}
    assert service.calls == ["local", "local", "local"]


def test_trusted_and_blocked_projections_are_filtered_and_stable() -> None:
    context = {
        "available_skills": [
            _skill("Zulu", description="last"),
            _skill("alpha", description="first"),
            _skill("hidden", user_invocable=False),
            _skill("blocked", trust_blocked=True),
            {"description": "missing name"},
        ],
        "blocked_skills": [
            {"name": "review-me", "trust_reason_code": "skill_modified"},
            {"description": "missing name"},
        ],
    }

    candidates = ConsoleSkillController._console_skill_trusted_candidates_from_context(
        context
    )
    blocked = ConsoleSkillController._console_skill_blocked_summaries(context)

    assert [(item.name, item.description) for item in candidates] == [
        ("alpha", "first"),
        ("Zulu", "last"),
    ]
    assert [item["name"] for item in blocked] == ["review-me"]


@pytest.mark.asyncio
async def test_refresh_replaces_candidates_and_syncs_popup() -> None:
    controller, _, _, _, _, syncs = _controller(
        [{"available_skills": [_skill("beta"), _skill("Alpha")]}]
    )
    assert controller._console_skill_candidates == ()

    await controller._refresh_console_skill_candidates()

    assert [item.name for item in controller._console_skill_candidates] == [
        "Alpha",
        "beta",
    ]
    assert syncs == ["sync"]


@pytest.mark.asyncio
async def test_refresh_re_reads_the_authoritative_context() -> None:
    controller, _, _, service, _, _ = _controller(
        [
            {"available_skills": [_skill("first")]},
            {"available_skills": [_skill("second")]},
        ]
    )

    await controller._refresh_console_skill_candidates()
    assert [item.name for item in controller._console_skill_candidates] == ["first"]
    await controller._refresh_console_skill_candidates()

    assert [item.name for item in controller._console_skill_candidates] == ["second"]
    assert service.calls == ["local", "local"]


@pytest.mark.asyncio
async def test_blocked_exact_and_prefix_responses_preserve_copy() -> None:
    controller, messages, _, _, _, _ = _controller()
    blocked = (
        {"name": "code-review", "trust_reason_code": "skill_modified"},
        {"name": "code-scan", "trust_status": "needs review"},
    )

    assert await controller._console_skill_blocked_match_response(
        "code-review", blocked
    )
    assert messages[-1] == SKILL_UNTRUSTED_REFUSE.format(
        name="code-review", reason="skill_modified"
    )

    assert await controller._console_skill_blocked_match_response("code", blocked)
    assert messages[-1] == CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE.format(count=2)
    assert not await controller._console_skill_blocked_match_response("other", blocked)


@pytest.mark.asyncio
async def test_skills_command_lists_or_emits_static_run_hint() -> None:
    controller, messages, _, service, _, _ = _controller(
        [{"available_skills": [_skill("review", description="Review a diff.")]}]
    )

    await controller._console_command_skills(
        CommandParse(kind=KIND_COMMAND, name="skills")
    )
    await controller._console_command_skills(
        CommandParse(kind=KIND_COMMAND, name="skills", args="review now")
    )

    assert "$review — Review a diff." in messages[0]
    assert messages[1] == "Run skills by typing $review — /skills only lists them."
    assert service.calls == ["local"]


def test_pending_state_updates_only_the_named_field() -> None:
    controller, _, states, _, _, _ = _controller()
    install = {"request_id": "install-1"}
    script = {"request_id": "script-1"}

    controller._set_console_pending_skill_install(install)
    after_install = states[-1]
    assert after_install.pending_skill_install == install
    assert after_install.pending_skill_script is None
    assert after_install.pending_approval == {"round": "approval"}
    assert after_install.summary == "keep"
    assert after_install.diff_summary == "unchanged"

    controller._set_console_pending_skill_script(script)
    after_script = states[-1]
    assert after_script.pending_skill_install == install
    assert after_script.pending_skill_script == script
    assert after_script.pending_approval == {"round": "approval"}


def test_plain_decision_forwarders_preserve_values_and_request_ids() -> None:
    controller, _, _, _, chat, _ = _controller()

    controller.handle_console_skill_install_decided(True, request_id="install-1")
    controller.handle_console_skill_script_decided(False, True, request_id="script-1")

    assert chat.installs == [(True, "install-1")]
    assert chat.scripts == [(False, True, "script-1")]
