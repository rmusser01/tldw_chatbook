"""PRD A8 / AC-A6: a composer send answers a mounted question instead of starting a turn."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _Card:
    def __init__(self, answers, request_id="round-1", display=True):
        self.display = display
        self._request_id = request_id
        self._answers = answers
        self.cleared = False

    def collect_answers(self):
        return [dict(a) for a in self._answers]

    def set_questions(self, payload):
        assert payload is None
        self.cleared = True


def _answers():
    return [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": None, "unanswered": True},
    ]


def _screen(card, *, image=None, launch=None, controller=None):
    cleared = []
    return SimpleNamespace(
        _console_chat_controller=controller if controller is not None else Mock(),
        query=lambda selector: [card] if card is not None else [],
        _console_pending_image_attachment=lambda: image,
        _retrieval=SimpleNamespace(_pending_launch=lambda: launch),
        _clear_console_composer_draft=lambda: cleared.append(True),
        _cleared=cleared,
    )


def test_typed_text_fills_every_unanswered_question_and_resolves_the_round():
    card = _Card(_answers())
    screen = _screen(card)
    assert ChatScreen._answer_pending_question_with_draft(screen, "  apac only  ") is True
    screen._console_chat_controller.resolve_pending_question.assert_called_once()
    answers, kwargs = screen._console_chat_controller.resolve_pending_question.call_args
    assert kwargs == {"request_id": "round-1"}
    assert answers[0] == [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": "apac only", "unanswered": False},
    ]
    assert card.cleared is True and screen._cleared == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"card": None},
        {"card": _Card(_answers(), display=False)},
        {"card": _Card(_answers(), request_id=None)},
        {"card": _Card(_answers()), "image": object()},
        {"card": _Card(_answers()), "launch": object()},
        {"card": _Card(_answers()), "controller": None},
    ],
    ids=["no-card", "hidden-card", "no-round-id", "staged-attachment", "staged-rag", "no-controller"],
)
def test_no_interception_without_a_live_card_or_with_staged_context(kwargs):
    kwargs = dict(kwargs)
    controller = kwargs.pop("controller", Mock())
    card = kwargs.pop("card")
    screen = _screen(card, controller=controller, **kwargs)
    if controller is None:
        screen._console_chat_controller = None
    assert ChatScreen._answer_pending_question_with_draft(screen, "text") is False
    if controller is not None:
        controller.resolve_pending_question.assert_not_called()
    assert screen._cleared == []


def test_blank_draft_never_intercepts():
    screen = _screen(_Card(_answers()))
    assert ChatScreen._answer_pending_question_with_draft(screen, "   ") is False
    screen._console_chat_controller.resolve_pending_question.assert_not_called()


@pytest.mark.asyncio
async def test_visible_send_resolves_the_question_and_does_not_dispatch():
    """End to end through the real send action: a plain message with a live
    card answers it; the draft is never queued as a turn."""
    from tldw_chatbook.Chat.console_command_grammar import (
        KIND_NOT_COMMAND,
        CommandParse,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = ConsoleComposerBar()
    composer.insert_text("use apac only")
    dispatched = []

    async def dispatch(draft, *, stash=None):
        dispatched.append(draft)
        return True

    card = _Card(_answers())
    controller = Mock()
    screen = SimpleNamespace(
        _console_pending_send_stash=None,
        _raw_cli=SimpleNamespace(start_user_command=Mock()),
        _console_composer_or_none=lambda: composer,
        query_one=lambda *_a, **_k: composer,
        query=lambda selector: [card],
        _console_pending_image_attachment=lambda: None,
        _focus_console_composer_if_needed=lambda **_k: None,
        _dismiss_console_guidance=lambda: None,
        _console_command_registry=SimpleNamespace(
            parse=lambda draft: CommandParse(kind=KIND_NOT_COMMAND)
        ),
        _console_unknown_send_armed=None,
        _dispatch_console_draft_send=dispatch,
        _console_chat_controller=controller,
        _retrieval=SimpleNamespace(_pending_launch=lambda: None),
        _clear_console_composer_draft=lambda: composer.clear_draft(),
    )
    screen._answer_pending_question_with_draft = (
        lambda draft: ChatScreen._answer_pending_question_with_draft(screen, draft)
    )
    assert await ChatScreen._send_console_message_from_visible_action(screen) is False
    assert dispatched == []
    controller.resolve_pending_question.assert_called_once()
    (answers,), _ = controller.resolve_pending_question.call_args
    assert answers[1]["other_text"] == "use apac only"


@pytest.mark.asyncio
async def test_visible_send_without_a_card_dispatches_as_before():
    from tldw_chatbook.Chat.console_command_grammar import (
        KIND_NOT_COMMAND,
        CommandParse,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = ConsoleComposerBar()
    composer.insert_text("hello")
    dispatched = []

    async def dispatch(draft, *, stash=None):
        dispatched.append(draft)
        return True

    controller = Mock()
    screen = SimpleNamespace(
        _console_pending_send_stash=None,
        _raw_cli=SimpleNamespace(start_user_command=Mock()),
        _console_composer_or_none=lambda: composer,
        query_one=lambda *_a, **_k: composer,
        query=lambda selector: [],
        _console_pending_image_attachment=lambda: None,
        _focus_console_composer_if_needed=lambda **_k: None,
        _dismiss_console_guidance=lambda: None,
        _console_command_registry=SimpleNamespace(
            parse=lambda draft: CommandParse(kind=KIND_NOT_COMMAND)
        ),
        _console_unknown_send_armed=None,
        _dispatch_console_draft_send=dispatch,
        _console_chat_controller=controller,
        _retrieval=SimpleNamespace(_pending_launch=lambda: None),
        _clear_console_composer_draft=lambda: composer.clear_draft(),
    )
    screen._answer_pending_question_with_draft = (
        lambda draft: ChatScreen._answer_pending_question_with_draft(screen, draft)
    )
    assert await ChatScreen._send_console_message_from_visible_action(screen) is True
    assert dispatched == ["hello"]
    controller.resolve_pending_question.assert_not_called()
