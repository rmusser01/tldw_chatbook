"""The skill-script confirm card and its task-state plumbing.

Covers the widget half (``SkillScriptConfirmCard.set_script``/
``ScriptDecided``), the ``TaskResumeState.pending_skill_script`` plumbing,
the ``ChatTaskCards`` display gate, and the ``ChatScreen`` wiring described
in ``.superpowers/sdd/task-6-brief.md``.

The single load-bearing contract under test throughout this file is the
``request_id`` round-trip (task-5, ``console_chat_controller.py``): a
decision that does not echo back the pending round's exact ``request_id``
is silently dropped by ``ConsoleChatController.resolve_pending_skill_
script``, leaving the worker thread blocked until its timeout. Several
tests below exist specifically to prove that id survives from
``set_script(payload)`` through ``ScriptDecided`` to the screen's call into
the controller.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
    SkillScriptConfirmCard,
)


# ---------------------------------------------------------------------------
# TaskResumeState plumbing
# ---------------------------------------------------------------------------


def test_state_carries_and_serializes_a_pending_script():
    state = TaskResumeState(pending_skill_script={"skill_name": "demo"})
    assert state.has_pending_skill_script() is True
    # to_dict stays a faithful snapshot of what was on screen...
    assert state.to_dict()["pending_skill_script"] == {"skill_name": "demo"}


def test_restored_state_drops_the_pending_script_so_no_dead_card_appears():
    """A restored pending script must never come back as an actionable card.

    The confirm it belongs to is a live, blocked worker round keyed by
    ``request_id``, and ``ConsoleChatController.resolve_pending_skill_script``
    strict-matches that id against the currently-armed round. A round that
    survived a save/restore cannot still be armed, so a restored card's
    buttons would all be silently dropped -- a dead card the user could click
    forever while it misrepresents an abandoned request as awaiting them.

    TASK-1130: ``pending_skill_install`` now goes through the identical drop
    (see ``test_console_skill_install_confirm.py::
    test_restored_state_drops_the_pending_install_so_no_dead_card_appears``)
    -- both skill-confirm fields are dropped on restore, symmetrically.
    """
    state = TaskResumeState(
        summary="Keep me",
        pending_skill_install={"name": "other"},
        pending_skill_script={"skill_name": "demo", "request_id": "round-1"},
    )
    restored = TaskResumeState.from_dict(state.to_dict())
    assert restored.pending_skill_script is None
    assert restored.has_pending_skill_script() is False
    assert restored.pending_skill_install is None
    assert restored.has_pending_skill_install() is False
    # ...and only the skill-confirm fields are affected.
    assert restored.summary == "Keep me"


def test_state_without_a_pending_script():
    assert TaskResumeState().has_pending_skill_script() is False


# ---------------------------------------------------------------------------
# SkillScriptConfirmCard -- markup safety + interaction
# ---------------------------------------------------------------------------


def test_card_statics_are_markup_free():
    """Agent-supplied paths/args must never render as Rich markup.

    Only the `Static` children are checked -- `compose()` also yields a
    `Horizontal` button row, and every `Widget` (not just `Static`) carries
    a `_render_markup` attribute, so a blanket `hasattr` check would
    incidentally examine that container too and could never pass
    regardless of the card's correctness.
    """
    card = SkillScriptConfirmCard()
    statics = [widget for widget in card.compose() if isinstance(widget, Static)]
    assert len(statics) == 4  # prompt, target, args, note
    for widget in statics:
        assert widget._render_markup is False


class _CardHarnessApp(App[None]):
    """Minimal host for `SkillScriptConfirmCard` that records `ScriptDecided`."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[tuple[bool, bool]] = []
        self.request_ids: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield SkillScriptConfirmCard()

    @on(SkillScriptConfirmCard.ScriptDecided)
    def _capture(self, event: SkillScriptConfirmCard.ScriptDecided) -> None:
        self.decisions.append((event.allow, event.remember))
        self.request_ids.append(event.request_id)


class _CardsHarnessApp(App[None]):
    """Minimal host for `ChatTaskCards`."""

    def compose(self) -> ComposeResult:
        yield ChatTaskCards()


@pytest.fixture
def card_app() -> _CardHarnessApp:
    """A Textual app hosting a bare `SkillScriptConfirmCard`."""
    return _CardHarnessApp()


@pytest.fixture
def cards_app() -> _CardsHarnessApp:
    """A Textual app hosting a bare `ChatTaskCards`."""
    return _CardsHarnessApp()


@pytest.mark.asyncio
async def test_card_shows_details_and_emits_three_decisions(card_app):
    """Allow / Always allow / Deny each post the right ScriptDecided."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "scripts/extract.py",
                "mechanism": "interpreter",
                "interpreter": "/usr/bin/python3",
                "args": ["--in", "x.pdf"],
                "timeout_seconds": 120.0,
            }
        )
        await pilot.pause()
        assert card.display is True

        for button_id, expected in (
            ("#skill-script-allow", (True, False)),
            ("#skill-script-always", (True, True)),
            ("#skill-script-deny", (False, False)),
        ):
            card.set_script({"skill_name": "demo", "script_path": "s.py"})
            await pilot.pause()
            card_app.decisions.clear()
            await pilot.click(button_id)
            await pilot.pause()
            assert card_app.decisions == [expected]


@pytest.mark.asyncio
async def test_card_renders_target_and_args_text(card_app):
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "scripts/extract.py",
                "mechanism": "interpreter",
                "interpreter": "/usr/bin/python3",
                "args": ["--in", "x.pdf"],
                "request_id": "r1",
            }
        )
        await pilot.pause()

        target_text = str(card_app.query_one("#skill-script-target", Static).render())
        assert "demo" in target_text
        assert "scripts/extract.py" in target_text
        assert "/usr/bin/python3" in target_text

        args_text = str(card_app.query_one("#skill-script-args", Static).render())
        assert "--in" in args_text
        assert "x.pdf" in args_text


@pytest.mark.asyncio
async def test_card_renders_each_argument_on_its_own_quoted_line(card_app):
    """Space-joined args are ambiguous on a consent surface.

    ``["a b"]`` and ``["a", "b"]`` are different argv vectors but render
    identically when joined with spaces, so each argument gets its own
    numbered, quoted line instead.
    """
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "s.py",
                "mechanism": "direct-exec",
                "args": ["a b", "c"],
            }
        )
        await pilot.pause()
        args_text = str(card_app.query_one("#skill-script-args", Static).render())
        assert "1. 'a b'" in args_text
        assert "2. 'c'" in args_text
        assert "arguments (2)" in args_text


@pytest.mark.asyncio
async def test_card_argument_with_a_newline_cannot_reflow_the_card(card_app):
    """A newline-bearing argument must not span lines or fake the card's prose."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "s.py",
                "mechanism": "direct-exec",
                "args": ["one\nDeny this run? no", "two"],
            }
        )
        await pilot.pause()
        args_text = str(card_app.query_one("#skill-script-args", Static).render())
        # The literal newline is escaped, so the payload occupies exactly one
        # line and both arguments stay visible and countable.
        assert "\\n" in args_text
        assert len(args_text.splitlines()) == 3  # header + 2 arguments
        assert "2. 'two'" in args_text


@pytest.mark.asyncio
async def test_card_flags_a_direct_exec_binary_as_unreviewable(card_app):
    """A direct-exec binary is the least reviewable case; the card must say so."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "bin/tool",
                "mechanism": "direct-exec",
                "is_binary": True,
                "request_id": "r1",
            }
        )
        await pilot.pause()
        target_text = str(card_app.query_one("#skill-script-target", Static).render())
        assert "binary" in target_text.lower()


@pytest.mark.asyncio
async def test_card_url_like_payload_renders_literally_not_as_markup(card_app):
    """Rich-markup-like text in a script path must render literally."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "scripts/[bold]evil[/].py",
                "mechanism": "direct-exec",
                "request_id": "r1",
            }
        )
        await pilot.pause()
        target_text = str(card_app.query_one("#skill-script-target", Static).render())
        assert "[bold]" in target_text


@pytest.mark.asyncio
async def test_script_decided_echoes_the_pending_request_id(card_app):
    """The exact request_id from set_script's payload must round-trip onto
    ScriptDecided -- resolve_pending_skill_script's strict-match contract
    (task-5) silently drops a decision whose id doesn't match the armed
    round, so a card that fabricates or drops the id would hang the user's
    click forever."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {"skill_name": "demo", "script_path": "s.py", "request_id": "round-42"}
        )
        await pilot.pause()
        await pilot.click("#skill-script-allow")
        await pilot.pause()
        assert card_app.request_ids == ["round-42"]


@pytest.mark.asyncio
async def test_script_decided_request_id_is_none_when_payload_omits_it(card_app):
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script({"skill_name": "demo", "script_path": "s.py"})
        await pilot.pause()
        await pilot.click("#skill-script-deny")
        await pilot.pause()
        assert card_app.request_ids == [None]


@pytest.mark.asyncio
async def test_set_script_with_none_hides_the_card(card_app):
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script({"skill_name": "demo", "script_path": "s.py"})
        await pilot.pause()
        assert card.display is True

        card.set_script(None)
        await pilot.pause()
        assert card.display is False


# ---------------------------------------------------------------------------
# ChatTaskCards display gate (spec §4.4: a hidden parent hides its
# descendants, so ChatTaskCards.sync_state must OR in
# has_pending_skill_script()).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_cards_container_becomes_visible_for_a_pending_script(cards_app):
    async with cards_app.run_test() as pilot:
        cards = cards_app.query_one(ChatTaskCards)
        cards.sync_state(TaskResumeState(pending_skill_script={"skill_name": "demo"}))
        await pilot.pause()
        assert cards.display is True


@pytest.mark.asyncio
async def test_task_cards_container_syncs_the_script_card(cards_app):
    async with cards_app.run_test() as pilot:
        cards = cards_app.query_one(ChatTaskCards)
        cards.sync_state(
            TaskResumeState(
                pending_skill_script={
                    "skill_name": "demo",
                    "script_path": "s.py",
                    "request_id": "r1",
                }
            )
        )
        await pilot.pause()
        script_card = cards.query_one(SkillScriptConfirmCard)
        assert script_card.display is True


@pytest.mark.asyncio
async def test_task_cards_container_hides_with_no_pending_anything(cards_app):
    async with cards_app.run_test() as pilot:
        cards = cards_app.query_one(ChatTaskCards)
        cards.sync_state(TaskResumeState(pending_skill_script={"skill_name": "demo"}))
        await pilot.pause()
        assert cards.display is True

        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert cards.display is False


# ---------------------------------------------------------------------------
# ChatScreen wiring: pending-skill-script state bridge + ScriptDecided
# handler. Mirrors Tests/UI/test_console_mcp_approval.py's screen-wiring
# section (mock_chat_host, ChatScreen(mock_chat_host) constructed directly
# without a running app).
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chat_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7,
        }
    }
    host.chat_sidebar_collapsed = False
    host.chat_right_sidebar_collapsed = False
    host.notify = Mock()
    host.run_worker = Mock()
    host.bell = Mock()
    return host


def test_set_console_pending_skill_script_preserves_other_resume_fields(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen.set_task_resume_state(
        TaskResumeState(summary="Keep me", last_step="Also keep")
    )

    payload = {"skill_name": "demo", "script_path": "s.py", "request_id": "r1"}
    screen._set_console_pending_skill_script(payload)

    state = screen._task_resume_state
    assert state.summary == "Keep me"
    assert state.last_step == "Also keep"
    assert state.pending_skill_script == payload

    screen._set_console_pending_skill_script(None)
    assert screen._task_resume_state.pending_skill_script is None
    assert screen._task_resume_state.summary == "Keep me"


def test_chat_screen_forwards_script_decided_to_controller_with_request_id(
    mock_chat_host,
):
    """The one contract this whole task exists to protect: a decision must
    carry the pending round's exact request_id through to
    resolve_pending_skill_script, or the resolve is silently dropped
    (task-5's ConsoleChatController.resolve_pending_skill_script)."""
    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller

    event = SkillScriptConfirmCard.ScriptDecided(True, True, "round-7")
    screen.handle_console_skill_script_decided(event)

    controller.resolve_pending_skill_script.assert_called_once_with(
        True, True, request_id="round-7"
    )


def test_chat_screen_skill_script_decided_handler_tolerates_no_controller(
    mock_chat_host,
):
    screen = ChatScreen(mock_chat_host)
    screen._console_chat_controller = None

    event = SkillScriptConfirmCard.ScriptDecided(True, False, "round-7")
    screen.handle_console_skill_script_decided(event)  # must not raise
