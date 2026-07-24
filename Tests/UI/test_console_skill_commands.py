"""`/skills` registered command (list / static `$name` run-form hint) on
the native Console composer (Task 4: the OLD bare `/skill-name` fallback
dispatch this file used to also cover was hard-removed -- invocation is now
exclusively the controller-side `$name` mention, tested in
``Tests/Chat/test_console_skill_substitution.py``).

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

from Tests.UI.test_console_command_composer import _spy_submit_draft
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_skill_resolver import SKILLS_EMPTY_LIST_ROW
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE,
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

    ``execute_skill`` exists because the provider-payload substitution rule
    renders the triggering `$skill-name` turn at build time through the
    controller's injected skills service (the same object staged on
    ``app.skills_scope_service`` here) -- a raw-command submit in these
    tests therefore reaches ``execute_skill`` even though some assertions
    below are about dispatch (raw command stored, no skill run) rather than
    the rendered payload.
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
            "$code-review — Reviews a diff." in row
            and "$release-notes — Drafts release notes." in row
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

        assert SKILLS_EMPTY_LIST_ROW in _console_message_contents(
            console, ConsoleMessageRole.SYSTEM
        )


@pytest.mark.asyncio
async def test_skills_command_named_run_form_shows_dollar_hint_for_unknown_name():
    """Hard removal (Task 4): `/skills <name>` never resolves or runs a
    skill anymore -- it always shows the same "run it with $name" hint,
    regardless of whether the typed name matches anything at all."""
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

        expected = "Run skills by typing $unknownskill — /skills only lists them."
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()
        # The draft is preserved -- the hint never clears the composer.
        assert composer.draft_text() == "/skills unknownskill"


@pytest.mark.asyncio
async def test_skills_command_named_run_form_shows_dollar_hint_for_blocked_name():
    """Same hint even when the typed name matches a trust-blocked skill --
    `/skills <name>` no longer distinguishes blocked/absent/ambiguous, it
    never resolves anything at all."""
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

        expected = "Run skills by typing $sketchy-skill — /skills only lists them."
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()


@pytest.mark.asyncio
async def test_skills_command_exact_trusted_name_shows_hint_and_never_runs():
    """The highest-value hard-removal regression: `/skills code-review`
    where "code-review" EXACTLY matches a real, trusted skill -- the
    previously-working run form. It must show the same static `$name` hint,
    execute nothing, and submit nothing."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    skills = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    app.skills_scope_service = skills
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/skills code-review fix it")
        submit_spy = AsyncMock()
        console._submit_console_native_draft = submit_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = "Run skills by typing $code-review — /skills only lists them."
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()
        assert skills.executions == []
        # The draft is preserved for correction into the `$` form.
        assert composer.draft_text() == "/skills code-review fix it"


@pytest.mark.asyncio
async def test_leading_dollar_skill_mention_executes_through_normal_send():
    """Hard removal (Task 4): typing `$code-review fix it` directly is no
    longer intercepted by any composer-level command dispatch -- it is a
    plain user send. The skill still actually runs because the CONTROLLER
    (Tasks 2/3) resolves and splices the leading `$name` mention at
    provider-payload build time; the stored transcript keeps the raw
    `$`-prefixed text untouched (ephemeral substitution only). The
    composer-staged TOOL "driving this turn" marker machinery was deleted
    with the bare `/name` dispatch it served (fix-wave branch (a)), so no
    TOOL row of any kind appears for a `$name` send."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    skills = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    app.skills_scope_service = skills
    gateway = CapturingGateway(chunks=("accepted",))
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("$code-review fix it")
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        # The raw `$`-prefixed draft is submitted verbatim -- no composer
        # command dispatch ever intercepts it.
        submit_spy.assert_called_once_with("$code-review fix it")
        # The skill actually ran (controller-side substitution).
        assert skills.executions == [("code-review", "fix it")]
        # The stored transcript keeps the raw mention -- only the ephemeral
        # provider payload is rendered.
        user_rows = [
            message.content
            for message in console._ensure_console_chat_store().messages_for_session(
                console._ensure_console_chat_store().active_session_id
            )
            if message.role is ConsoleMessageRole.USER
        ]
        assert "$code-review fix it" in user_rows
        # No TOOL rows at all -- the composer-staged "driving this turn"
        # marker machinery is deleted (the visible `$name` user row already
        # documents which skill drove the turn).
        assert _console_message_contents(console, ConsoleMessageRole.TOOL) == []


@pytest.mark.asyncio
async def test_bare_slash_skill_name_no_longer_auto_runs_shows_unknown_command_hint():
    """Hard removal (Task 4): a bare `/code-review fix it` draft -- the OLD
    invocation form -- is no longer claimed by any fallback resolver, even
    though "code-review" is a real, trusted skill. It parses exactly like
    any other unrecognized word: the generic "Unknown command" hint, armed
    for a literal send on a second Enter. The skill never runs."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    skills = FakeSkillsScopeService(
        available_skills=[_skill("code-review", "Reviews a diff.")]
    )
    app.skills_scope_service = skills
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/code-review fix it")
        submit_spy = AsyncMock()
        console._submit_console_native_draft = submit_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        expected = (
            "Unknown command /code-review — available: "
            "/prompt, /system, /skills, /prefill. Press Enter again to send as text."
        )
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        submit_spy.assert_not_called()
        assert skills.executions == []
        assert composer.draft_text() == "/code-review fix it"
        assert console._console_unknown_send_armed == "/code-review fix it"


@pytest.mark.asyncio
async def test_run_resolved_console_skill_composes_dollar_command():
    """The skill-picker submit path (`_run_resolved_console_skill`, the sole
    remaining consumer of a resolved skill name) composes a `$name [args]`
    draft, not `/name [args]` -- verified directly since the picker is no
    longer reachable through any live `/skills` or bare-word composer
    input (both were hard-removed above)."""
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
        submit_spy = await _spy_submit_draft(console)

        await console._run_resolved_console_skill("code-review", "fix it")
        await _wait_for_text(console, pilot, "accepted")
        submit_spy.assert_called_once_with("$code-review fix it")

        submit_spy.reset_mock()
        await console._run_resolved_console_skill("code-review", "")
        await pilot.pause(0.2)
        submit_spy.assert_called_once_with("$code-review")


@pytest.mark.asyncio
async def test_skills_command_named_run_form_shows_dollar_hint_for_blocked_prefix():
    """Same static hint even when the typed name is a prefix matching only
    needs-review skills -- `/skills <name>` no longer distinguishes any
    resolution outcome, and never opens the (now-unreachable) picker."""
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

        expected = "Run skills by typing $review — /skills only lists them."
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert len(host.screen_stack) == baseline_depth, "no picker should have opened"


@pytest.mark.asyncio
async def test_bare_unknown_word_needs_review_prefix_shows_hint_not_unknown_command():
    """The blocked-hint path itself STAYS (Task 4 brief): a bare `/name`
    draft that matches no registered command reaches KIND_UNKNOWN directly
    (there is no fallback resolver at all anymore), and a needs-review-only
    match there still surfaces the distinguishing count hint instead of the
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
async def test_skills_command_named_run_form_shows_dollar_hint_for_ambiguous_prefix():
    """Same static hint even when the typed name is an ambiguous prefix
    (multiple trusted matches) -- `/skills <name>` never opens the picker
    anymore, for any resolution outcome."""
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

        expected = "Run skills by typing $code — /skills only lists them."
        assert expected in _console_message_contents(console, ConsoleMessageRole.SYSTEM)
        assert len(host.screen_stack) == baseline_depth, "no picker should have opened"
        # The draft is left exactly as typed.
        assert composer.draft_text() == "/skills code"
