"""PRD A8 / AC-A6: a composer send answers a mounted question instead of starting a turn."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _Card:
    def __init__(self, answers, request_id="round-1", display=True, session_id="s1"):
        self.display = display
        self._request_id = request_id
        self._payload = {"request_id": request_id, "session_id": session_id}
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


def _controller(active_session_id="s1"):
    controller = Mock()
    controller.store.active_session_id = active_session_id
    return controller


def _screen(card, *, image=None, launch=None, controller=None):
    cleared = []
    return SimpleNamespace(
        _console_chat_controller=controller if controller is not None else _controller(),
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
        {"card": _Card(_answers(), session_id="other-session")},
    ],
    ids=["no-card", "hidden-card", "no-round-id", "staged-attachment", "staged-rag", "no-controller", "stale-card-for-another-session"],
)
def test_no_interception_without_a_live_card_or_with_staged_context(kwargs):
    kwargs = dict(kwargs)
    controller = kwargs.pop("controller", _controller())
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
    controller = _controller()
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

    controller = _controller()
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


# --- Qodo #2380 round 1 ------------------------------------------------------


@pytest.mark.parametrize("draft", ["/foo", "/nope x", "  /help  "])
def test_slash_text_is_never_intercepted_even_when_confirmed_as_plain_text(draft):
    screen = _screen(_Card(_answers()))
    assert ChatScreen._answer_pending_question_with_draft(screen, draft) is False
    screen._console_chat_controller.resolve_pending_question.assert_not_called()
    assert screen._cleared == []


def test_typed_text_is_cleaned_and_bounded_like_the_cards_other_box():
    screen = _screen(_Card(_answers()))
    assert ChatScreen._answer_pending_question_with_draft(screen, "  a\nb\x07 " + "x" * 600) is True
    (answers,), _ = screen._console_chat_controller.resolve_pending_question.call_args
    assert answers[1]["other_text"].startswith("a b x") and len(answers[1]["other_text"]) == 500


def test_malformed_card_answers_leave_the_draft_and_card_in_place():
    bad = [{"question": "q", "selected": "not-a-list", "other_text": None, "unanswered": True}]
    card = _Card(bad)
    screen = _screen(card)
    assert ChatScreen._answer_pending_question_with_draft(screen, "text") is False
    assert card.cleared is False and screen._cleared == []
    screen._console_chat_controller.resolve_pending_question.assert_not_called()


@pytest.mark.asyncio
async def test_typed_answer_resolves_a_real_round_through_the_real_card_and_composer():
    """Integration: a real ConsoleChatController round waiting on a worker
    thread, the real ChatTaskCards/ChatQuestionCard under the consolidated
    CSS, the real ConsoleComposerBar, and the real visible send action. Only
    the screen object is a stand-in: constructing ``ChatScreen`` with the
    shared mock host trips a real-path check on current dev."""
    import threading

    from textual.app import ComposeResult

    from Tests.console_provider_doubles import persisted_console_store
    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_command_grammar import (
        KIND_NOT_COMMAND,
        CommandParse,
    )
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    class _Harness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield ChatTaskCards(id="chat-task-cards")

    questions = [
        {"question": "Which DB?", "header": "DB", "multiSelect": False,
         "options": [{"label": "Postgres", "description": ""}, {"label": "SQLite", "description": ""}]},
        {"question": "Regions?", "header": "Region", "multiSelect": True,
         "options": [{"label": "eu", "description": ""}, {"label": "us", "description": ""}]},
    ]
    app = _Harness()
    controller = ConsoleChatController(store=persisted_console_store(), provider_gateway=object())
    session = controller.new_session(title="s")
    box = {}
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            cards = app.query_one(ChatTaskCards)
            controller.app = app
            controller.set_pending_question = lambda payload: cards.sync_state(
                TaskResumeState(pending_question=payload)
            )
            thread = threading.Thread(
                target=lambda: box.update(
                    result=controller.request_user_questions(questions, session_id=session.id)
                )
            )
            thread.start()
            for _ in range(40):
                await pilot.pause(0.05)
                if list(cards.query("#chat-question-card")):
                    break
            card = cards.query_one("#chat-question-card")
            await pilot.pause()
            assert card.display is True

            composer = ConsoleComposerBar()
            composer.insert_text("apac only")
            dispatched = []

            async def dispatch(draft, *, stash=None):
                dispatched.append(draft)
                return True

            screen = SimpleNamespace(
                _console_pending_send_stash=None,
                _raw_cli=SimpleNamespace(start_user_command=Mock()),
                _console_composer_or_none=lambda: composer,
                query_one=lambda *_a, **_k: composer,
                query=app.query,
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
            assert dispatched == [], "the draft was an answer, not a turn"
            # The worker's teardown marshals its card re-derive through
            # `call_from_thread`, which needs THIS loop: wait asynchronously.
            for _ in range(100):
                await pilot.pause(0.05)
                if "result" in box:
                    break
            thread.join(timeout=5)
            assert box["result"]["answered"] is True
            assert [a["other_text"] for a in box["result"]["answers"]] == ["apac only", "apac only"]
            assert composer.draft_text() == ""
            await pilot.pause()
            assert card.display is False
            assert controller.pending_question_ids() == []
    finally:
        controller.begin_shutdown()
