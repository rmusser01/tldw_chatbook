"""Task 9: `/skills` registered command + bare `/skill-name` fallback dispatch
(list / run / refuse) on the native Console composer.

Mirrors ``Tests/UI/test_console_command_composer.py``'s harness style (real
``TldwCli`` app via ``_build_test_app``, real ``ChatScreen`` via
``ConsoleHarness``) but swaps in a fake ``skills_scope_service`` so candidate
population is fully scripted rather than depending on the real local skills
store.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_command_composer import _spy_submit_draft
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILL_UNTRUSTED_REFUSE,
    SKILLS_EMPTY_LIST_ROW,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
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
    """Minimal stand-in for ``SkillsScopeService.get_context``/``execute_skill``.

    ``execute_skill`` exists because Task 10's provider-payload substitution
    rule renders the triggering `/skill-name` turn at build time through
    the controller's injected skills service (the same object staged on
    ``app.skills_scope_service`` here) -- a raw-command submit in these
    tests therefore reaches ``execute_skill`` even though the assertions
    below are about dispatch (raw command stored + marker order), not the
    rendered payload.
    """

    def __init__(
        self,
        *,
        available_skills: list[dict[str, Any]] | None = None,
        blocked_skills: list[dict[str, Any]] | None = None,
    ) -> None:
        self.available_skills = available_skills or []
        self.blocked_skills = blocked_skills or []
        self.calls: list[str | None] = []
        self.executions: list[tuple[str, str | None]] = []

    async def get_context(self, *, mode: str | None = None) -> Mapping[str, Any]:
        self.calls.append(mode)
        return {
            "available_skills": list(self.available_skills),
            "blocked_skills": list(self.blocked_skills),
        }

    async def execute_skill(
        self, name: str, *, mode: str | None = None, args: str | None = None
    ) -> Mapping[str, Any]:
        self.executions.append((name, args))
        return {
            "skill_name": name,
            "rendered_prompt": f"RENDERED[{name}:{args}]",
            "allowed_tools": None,
            "execution_mode": "inline",
            "fork_output": None,
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
    gateway = CapturingGateway(chunks=("accepted",))
    app.console_provider_gateway_factory = lambda: gateway
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
        # `wraps=` keeps the real submit path running (needed for the
        # accepted-hook that now appends the marker -- see below) while
        # still letting us assert what was actually submitted.
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_called_once_with("/code-review fix it")
        marker_expected = CONSOLE_SKILL_RUN_MARKER_TEMPLATE.format(name="code-review")
        assert marker_expected in _console_message_contents(console, ConsoleMessageRole.TOOL)


@pytest.mark.asyncio
async def test_bare_fallback_skill_marker_lands_after_user_message_not_before():
    """Reviewer repro (Task 9 fix-wave): appending the TOOL "driving this
    turn" marker right after `_dispatch_console_draft_send` merely
    *schedules* the real submit via `run_worker` -- not after it actually
    runs -- so the marker could land BEFORE the user turn it is meant to
    follow. Store order must always be [USER, TOOL, ASSISTANT]."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    gateway = CapturingGateway(chunks=("accepted",))
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        for _ in range(40):
            if console._console_skill_candidates:
                break
            await pilot.pause(0.05)
        assert console._console_skill_candidates, "candidate snapshot never populated"

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/code-review fix it")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        messages = store.messages_for_session(session_id)

        marker_expected = CONSOLE_SKILL_RUN_MARKER_TEMPLATE.format(name="code-review")
        assert [message.role for message in messages] == [
            ConsoleMessageRole.USER,
            ConsoleMessageRole.TOOL,
            ConsoleMessageRole.ASSISTANT,
        ]
        assert messages[0].content == "/code-review fix it"
        assert messages[1].content == marker_expected


class _RaceConditionSkillsScopeService(FakeSkillsScopeService):
    """Reports a skill as trusted at candidate-listing time, but refuses it
    (``SkillTrustBlockedError``) at actual execute-time -- simulating trust
    being revoked in the window between the composer's dispatch-time
    resolution (``_console_command_run_skill``, which staged the "driving
    this turn" marker) and the controller's own build-time re-verification
    (``ConsoleChatController._apply_skill_substitution``)."""

    async def execute_skill(self, name, *, mode=None, args=None):
        self.executions.append((name, args))
        raise SkillTrustBlockedError(
            skill_name=name, reason_code="skill_modified",
            trust_status="quarantined_modified",
        )


@pytest.mark.asyncio
async def test_skill_trust_revoked_between_resolve_and_submit_refuses_without_marker():
    """Qodo finding 3 (PR #636 bot review) repro: a skill resolved (and
    staged for the "driving this turn" marker) at dispatch time can still
    be refused by the controller's own build-time trust re-check. The
    refusal must produce NO TOOL marker (the skill never actually ran) and
    must not leak the staged marker name onto the next, unrelated send."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = _RaceConditionSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    gateway = CapturingGateway(chunks=("accepted",))
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        for _ in range(40):
            if console._console_skill_candidates:
                break
            await pilot.pause(0.05)
        assert console._console_skill_candidates, "candidate snapshot never populated"

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/code-review fix it")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "isn't trusted (skill_modified)")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        messages = store.messages_for_session(session_id)

        # The raw command still persists (honest record), followed by the
        # refuse row -- but NO TOOL marker: the skill never actually ran.
        assert not [m for m in messages if m.role is ConsoleMessageRole.TOOL]
        expected = SKILL_UNTRUSTED_REFUSE.format(name="code-review", reason="skill_modified")
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert console._console_pending_skill_marker_name is None

        # The staged marker must not leak onto the NEXT, unrelated send.
        composer.load_draft("just a normal message")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")
        assert not [
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.TOOL
        ]


class _ToggleReadinessGateway:
    """First ``resolve_for_send`` call is blocked; every later call is ready.

    Lets a single test drive "the skill run's submit is refused inside
    `submit_draft`" followed immediately by a normal successful send,
    without needing to tear down/rebuild the Console controller (whose
    ``provider_gateway`` is cached at construction) mid test.
    """

    def __init__(self) -> None:
        self._calls = 0

    async def resolve_for_send(self, selection):
        self._calls += 1
        ready = self._calls > 1
        return SimpleNamespace(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=selection.explicit_model or selection.configured_model or "test-model",
            ready=ready,
            visible_copy="" if ready else "Provider blocked: test setup incomplete.",
        )

    async def stream_chat(self, resolution, messages):
        yield "accepted"


@pytest.mark.asyncio
async def test_blocked_skill_run_does_not_leak_marker_into_next_send():
    """Leak test: a refused/blocked submit must never leave its staged skill
    marker name to attach itself to the NEXT, unrelated accepted send."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.skills_scope_service = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    gateway = _ToggleReadinessGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        for _ in range(40):
            if console._console_skill_candidates:
                break
            await pilot.pause(0.05)
        assert console._console_skill_candidates, "candidate snapshot never populated"

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/code-review fix it")
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        # The blocked first attempt must have cleared the staged marker name
        # rather than leaving it consumed by whatever sends next.
        assert console._console_pending_skill_marker_name is None

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        assert not [
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.TOOL
        ]

        composer.load_draft("just a normal message")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        tool_rows = [
            message.content
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.TOOL
        ]
        assert tool_rows == []


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
async def test_bare_fallback_prefix_matching_only_blocked_skills_shows_needs_review_hint():
    """Fold-in (Task 9 fix-wave review): a bare `/name` draft that the
    fallback resolver leaves as KIND_UNKNOWN (its cached candidate snapshot
    is trusted-only, so a needs-review-only match never gets claimed) must
    surface the same distinguishing hint `/skills <name>` shows, not the
    generic "Unknown command" hint."""
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
        composer.load_draft("/review")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE.format(count=2)
        system_rows = _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert expected in system_rows
        assert not any(row.startswith("Unknown command") for row in system_rows)
        assert console._console_unknown_send_armed is None


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
