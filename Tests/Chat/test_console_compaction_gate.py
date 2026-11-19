"""TASK-33621.4 AC#2: turns that cannot show the hold card stop before commit.

Queued and Buddy sends are refused by ``refuse_without_card``, and a send
with no preparation by ``refuse_unprepared`` (the real screen flows are in
Tests/UI/test_console_compaction_hold_cardless_sends.py).

Continue, Regenerate and Retry over the Ask trigger used to commit an
assistant row, mark it Failed and explain afterwards ("Use Compact now in
Chat settings ..."). The gate asks the same side-effect-free preflight probe
the composer hold uses, and refuses before anything is committed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
from tldw_chatbook.Chat.console_compaction_gate import (
    refuse_at_ask_trigger,
    refuse_unprepared,
    refuse_without_card,
)
from tldw_chatbook.Chat.console_context_budget_copy import ask_trigger_refusal_copy
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_turn_preparation import ContextCompactionHold

_HOLD = ContextCompactionHold(
    session_id="session-1",
    used_tokens=1_656,
    trigger_tokens=1_200,
    budget_tokens=1_500,
    estimated=False,
)


class _Controller:
    def __init__(self, *, mode, hold) -> None:
        self.mode = mode
        self.hold = hold
        self.blocked: list[tuple[str, str]] = []
        self.assessed: list[dict] = []
        self.store = SimpleNamespace(sessions=lambda: [SimpleNamespace(id="session-1")])

    def _compaction_mode_for(self, _session):
        return self.mode

    async def _assess_context_compaction(self, **kwargs):
        self.assessed.append(kwargs)
        return self.hold, None

    def _block(self, session_id: str, copy: str):
        self.blocked.append((session_id, copy))
        return SimpleNamespace(accepted=False, visible_copy=copy)


def _run(controller):
    return asyncio.run(
        refuse_at_ask_trigger(
            controller,
            "session-1",
            object(),
            [{"role": "user", "content": "x"}],
            "continued",
        )
    )


def test_over_the_ask_trigger_the_turn_is_refused_before_commit() -> None:
    controller = _Controller(mode=ContextCompactionMode.ASK, hold=_HOLD)

    result = _run(controller)

    assert result is not None and not result.accepted
    assert controller.assessed[0]["uncommitted_user_message_id"] is None
    ((session_id, copy),) = controller.blocked
    assert session_id == "session-1"
    assert copy.startswith("Not continued:")
    assert "1,656 of 1,500 tokens" in copy and "1,200" in copy
    assert "/compact" in copy


@pytest.mark.parametrize(
    ("mode", "hold"),
    [
        (ContextCompactionMode.ASK, None),  # below the trigger
        (ContextCompactionMode.AUTOMATIC, _HOLD),  # compacts on its own
        (ContextCompactionMode.OFF, _HOLD),
    ],
)
def test_otherwise_the_turn_goes_on_unprobed_or_unblocked(mode, hold) -> None:
    controller = _Controller(mode=mode, hold=hold)

    assert _run(controller) is None
    assert controller.blocked == []
    if mode is not ContextCompactionMode.ASK:
        assert controller.assessed == []  # no probe cost outside Ask


def test_the_refusal_copy_flags_an_assumed_window() -> None:
    copy = ask_trigger_refusal_copy(
        action="regenerated",
        used_tokens=30_000,
        budget_tokens=32_000,
        trigger_tokens=25_600,
        estimated=True,
    )

    assert copy.startswith("Not regenerated:")
    assert "assumed" in copy


class _CardlessController:
    """Store-draft and preparation seams ``refuse_without_card`` drives."""

    def __init__(self) -> None:
        self.draft = "my own console draft"
        self.abandoned: list[str] = []
        self.blocked: list[tuple[str, str]] = []
        self.store = SimpleNamespace(
            session_draft=lambda _session_id: self.draft,
            set_session_draft=self._set_draft,
        )

    def _set_draft(self, _session_id: str, text: str) -> None:
        self.draft = text

    def _abandon_preparation(self, preparation_id: str) -> None:
        # Like store.cancel_preparation for a manual origin: the cancelled
        # send's text replaces the composer draft.
        self.abandoned.append(preparation_id)
        self.draft = "buddy reply text"

    def _block(self, session_id: str, copy: str):
        self.blocked.append((session_id, copy))
        return ConsoleSubmitResult(False, False, copy)


_PREPARATION = SimpleNamespace(session_id="session-1", preparation_id="prep-1")


def test_a_queued_send_over_the_trigger_is_refused_back_to_its_queue() -> None:
    controller = _CardlessController()

    result = refuse_without_card(
        controller,
        _PREPARATION,
        _HOLD,
        origin=ConsoleSubmissionOrigin.QUEUED,
        queue_entry_id="entry-1",
        preserve_composer=False,
    )

    assert controller.abandoned == ["prep-1"]
    assert not result.accepted
    assert result.visible_copy.startswith("Not sent from the queue:")
    assert "/compact" in result.visible_copy
    assert (result.session_id, result.origin, result.queue_entry_id) == (
        "session-1",
        ConsoleSubmissionOrigin.QUEUED,
        "entry-1",
    )


def test_a_buddy_reply_is_refused_and_the_console_draft_is_kept() -> None:
    controller = _CardlessController()

    result = refuse_without_card(
        controller,
        _PREPARATION,
        _HOLD,
        origin=ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id=None,
        preserve_composer=True,
    )

    assert controller.abandoned == ["prep-1"]
    assert result.visible_copy.startswith("Not sent:")
    assert controller.draft == "my own console draft"


def test_below_the_trigger_a_cardless_send_goes_on_untouched() -> None:
    controller = _CardlessController()

    assert (
        refuse_without_card(
            controller,
            _PREPARATION,
            None,
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id="entry-1",
            preserve_composer=False,
        )
        is None
    )
    assert controller.abandoned == [] and controller.blocked == []


class _UnpreparedController:
    """The echo and block seams ``refuse_unprepared`` drives."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.blocked: list[tuple[str, str]] = []
        self.store = SimpleNamespace(delete_message=self.deleted.append)

    def _block(self, session_id: str, copy: str):
        self.blocked.append((session_id, copy))
        return ConsoleSubmitResult(False, False, copy)


def test_an_unprepared_send_over_the_trigger_drops_its_echo_and_is_refused() -> None:
    controller = _UnpreparedController()

    result = refuse_unprepared(
        controller,
        "session-1",
        _HOLD,
        echoed_user=SimpleNamespace(id="echo-1"),
        origin=ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id=None,
    )

    assert controller.deleted == ["echo-1"]
    assert not result.accepted
    assert result.visible_copy.startswith("Not sent:")
    assert "/compact" in result.visible_copy
    assert controller.blocked == [("session-1", result.visible_copy)]
    assert (result.session_id, result.origin) == (
        "session-1",
        ConsoleSubmissionOrigin.MANUAL,
    )


def test_below_the_trigger_an_unprepared_send_goes_on_untouched() -> None:
    controller = _UnpreparedController()

    assert (
        refuse_unprepared(
            controller,
            "session-1",
            None,
            echoed_user=SimpleNamespace(id="echo-1"),
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id="entry-1",
        )
        is None
    )
    assert controller.deleted == [] and controller.blocked == []
