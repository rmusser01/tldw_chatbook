"""Task 9: `/skills` registered command + bare `/skill-name` fallback dispatch
(list / run / refuse) on the native Console composer.

Mirrors ``Tests/UI/test_console_command_composer.py``'s harness style (real
``TldwCli`` app via ``_build_test_app``, real ``ChatScreen`` via
``ConsoleHarness``) but swaps in a fake ``skills_scope_service`` so candidate
population is fully scripted rather than depending on the real local skills
store.
"""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILL_UNTRUSTED_REFUSE,
    SKILLS_EMPTY_LIST_ROW,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE,
    CONSOLE_SKILL_RUN_MARKER_TEMPLATE,
    CONSOLE_SKILL_RUN_REFUSE_REASON_ABSENT,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def _skill(name: str, description: str = "Does the thing.") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "user_invocable": True,
        "trust_blocked": False,
    }


def _blocked_skill(name: str, *, reason: str = "skill_modified") -> dict[str, Any]:
    return {
        "name": name,
        "description": "Needs review.",
        "user_invocable": True,
        "trust_blocked": True,
        "trust_reason_code": reason,
    }


class FakeSkillsScopeService:
    """Minimal stand-in for ``SkillsScopeService.get_context``."""

    def __init__(
        self,
        *,
        available_skills: list[dict[str, Any]] | None = None,
        blocked_skills: list[dict[str, Any]] | None = None,
    ) -> None:
        self.available_skills = available_skills or []
        self.blocked_skills = blocked_skills or []
        self.calls: list[str | None] = []

    async def get_context(self, *, mode: str | None = None) -> Mapping[str, Any]:
        self.calls.append(mode)
        return {
            "available_skills": list(self.available_skills),
            "blocked_skills": list(self.blocked_skills),
        }


def _console_message_contents(console, role: ConsoleMessageRole) -> list[str]:
    store = console._ensure_console_chat_store()
    if store.active_session_id is None:
        return []
    messages = store.messages_for_session(store.active_session_id)
    return [message.content for message in messages if message.role is role]


@pytest.mark.asyncio
async def test_bare_skills_command_lists_trusted_skills():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[
            _skill("code-review", "Reviews a diff."),
            _skill("release-notes", "Drafts release notes."),
        ]
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        system_rows = _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert any(
            "/code-review — Reviews a diff." in row and "/release-notes — Drafts release notes." in row
            for row in system_rows
        )


@pytest.mark.asyncio
async def test_bare_skills_command_with_no_skills_shows_empty_row():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert SKILLS_EMPTY_LIST_ROW in _console_message_contents(console, ConsoleMessageRole.SYSTEM)


@pytest.mark.asyncio
async def test_skills_command_unknown_name_shows_absent_refuse_row():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("code-review"), _skill("release-notes")]
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills unknownskill")
        submit_spy = AsyncMock()
        console._submit_console_native_draft = submit_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = SKILL_UNTRUSTED_REFUSE.format(
            name="unknownskill", reason=CONSOLE_SKILL_RUN_REFUSE_REASON_ABSENT
        )
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()
        # The draft is preserved -- a refusal never clears the composer.
        assert composer.draft_text() == "/skills unknownskill"


@pytest.mark.asyncio
async def test_skills_command_named_blocked_skill_shows_untrusted_refuse_row():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("code-review")],
        blocked_skills=[_blocked_skill("sketchy-skill", reason="skill_modified")],
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills sketchy-skill")
        submit_spy = AsyncMock()
        console._submit_console_native_draft = submit_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = SKILL_UNTRUSTED_REFUSE.format(name="sketchy-skill", reason="skill_modified")
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()


@pytest.mark.asyncio
async def test_bare_fallback_trusted_skill_submits_raw_command_and_appends_marker():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        # Let the mount-time cache refresh populate the fallback resolver's
        # candidate snapshot before typing a bare `/code-review` draft.
        for _ in range(40):
            if console._console_skill_candidates:
                break
            await pilot.pause(0.05)
        assert console._console_skill_candidates, "candidate snapshot never populated"

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/code-review fix it")
        submit_spy = AsyncMock()
        console._submit_console_native_draft = submit_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        submit_spy.assert_called_once_with("/code-review fix it")
        marker_expected = CONSOLE_SKILL_RUN_MARKER_TEMPLATE.format(name="code-review")
        assert marker_expected in _console_message_contents(console, ConsoleMessageRole.TOOL)


@pytest.mark.asyncio
async def test_run_skill_prefix_matching_only_blocked_skills_shows_needs_review_hint():
    """Review-mandated addition: a typed prefix that matches ONLY needs-review
    skills must not silently open an (always-empty) picker or otherwise read
    like the generic empty state -- it gets a distinguishing count hint."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("release-notes")],
        blocked_skills=[_blocked_skill("review-helper"), _blocked_skill("review-tool")],
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills review")

        baseline_depth = len(host.screen_stack)
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE.format(count=2)
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert len(host.screen_stack) == baseline_depth, "no picker should have opened"


@pytest.mark.asyncio
async def test_skills_command_ambiguous_prefix_opens_picker_prefilled():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[
            _skill("code-review", "Reviews a diff."),
            _skill("code-format", "Formats code."),
        ]
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills code")

        baseline_depth = len(host.screen_stack)
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1, "the picker must have opened"

        from tldw_chatbook.Widgets.Console.console_skill_picker_modal import FILTER_INPUT_ID
        from textual.widgets import Input

        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == "code"
        # The draft is left exactly as typed -- the picker is a detour.
        assert composer.draft_text() == "/skills code"
