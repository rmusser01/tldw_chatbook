"""Ask once whether to keep citation records (TASK-33620.8.3).

Owner decision 2026-10-30: an install whose config.toml still carries the old
shipped ``[rag_citations] canonical_writes_enabled = false`` is asked, after its
first reply sent with Library evidence, "Keep a record of which sources replies
cite?". The answer is saved beside config.toml
(``Chat.citation_records_choice``); config.toml is never rewritten. The command
palette's "Console: Citation records…" changes the answer later.

The question opens with no button focused, so typing meant for the composer
cannot answer it, and Escape closes it unanswered (it comes back after a
restart). Imported only when a question is due or the palette entry is used.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Label, Static

from tldw_chatbook.Chat import citation_records_choice as choice
from tldw_chatbook.Chat.citation_records_choice import (
    CitationRecordsAnswer,
    ConfiguredSwitch,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

QUESTION_TITLE = "Keep a record of which sources replies cite?"
KEEP_LABEL = "Keep records"
DONT_KEEP_LABEL = "Don't keep"
PALETTE_LABEL = "Console: Citation records…"

_ABOUT = (
    "Chatbook can keep a record of the Library sources each reply cites, so "
    "you can open them later from Cited sources, also after a restart."
)
_WHERE = (
    "Your answer is saved for this profile; config.toml is not changed. You "
    f"can change it later from the command palette: {PALETTE_LABEL}"
)
AFTER_REPLY_MESSAGE = (
    f"{_ABOUT} This profile still has the older setting that keeps these "
    "records off, so the sources of the reply you just got were not saved."
    f"\n\n{_WHERE}"
)
KEPT_NOTICE = (
    "Citation records are on. Replies that cite your Library now show Cited sources."
)
NOT_KEPT_NOTICE = (
    "Citation records stay off. To change this later, use "
    f"{PALETTE_LABEL} in the command palette."
)
SAVE_FAILED_NOTICE = (
    "Your answer couldn't be saved, so Chatbook will ask again after a restart."
)
CONFIG_ON_NOTICE = (
    "Citation records are on: config.toml turns them on ([rag_citations] "
    "canonical_writes_enabled). To turn them off, set it to false and restart "
    "Chatbook."
)
CONFIG_OFF_NOTICE = (
    "Citation records are off because of config.toml ([rag_citations] "
    "canonical_writes_enabled). To turn them on, set it to true and restart "
    "Chatbook."
)
UNAVAILABLE_NOTICE = "Citation records aren't available in this session."


def palette_message(current: CitationRecordsAnswer | None) -> str:
    """The question's text when opened from the command palette.

    Args:
        current: The saved answer, if any.

    Returns:
        The current state, what records are for, and where the answer lives.
    """
    state = (
        "Citation records are on for this profile (you chose Keep records)."
        if current is CitationRecordsAnswer.KEEP
        else "Citation records are off for this profile."
    )
    return f"{state}\n\n{_ABOUT}\n\n{_WHERE}"


class CitationRecordsQuestionDialog(ConfirmationDialog):
    """Keep records / Don't keep; Escape or a backdrop click leaves it unanswered.

    Neither button takes focus when the dialog opens: it can appear right after
    a reply, while the user is typing the next message, and a typed Space or
    Enter must not answer it.
    """

    AUTO_FOCUS = "#confirmation-dialog"

    def __init__(self, message: str) -> None:
        super().__init__(
            title=QUESTION_TITLE,
            message=message,
            confirm_label=KEEP_LABEL,
            cancel_label=DONT_KEEP_LABEL,
        )

    def compose(self) -> ComposeResult:
        """The confirmation layout; neither answer is styled as destructive."""
        with VerticalScroll(id="confirmation-dialog", classes="confirmation-scroll"):
            yield Static(self.title, classes="dialog-title", markup=False)
            yield Label(self.message, classes="dialog-message", markup=False)
            with Horizontal(classes="button-container"):
                yield Button(self.cancel_label, id="cancel-button")
                yield Button(self.confirm_label, id="confirm-button", variant="primary")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Don't keep is an answer; Escape and the backdrop are not."""
        self.result = False if source == "button" else None
        self.dismiss_safe_once(self.result)


def ask_pending_question(screen: Any, repository: Any, session_id: Any) -> bool:
    """Ask the waiting question after this session's reply has settled.

    Asked only while Console itself is the top screen; otherwise the question
    keeps waiting for the session's next settled reply.

    Args:
        screen: The Console screen.
        repository: The citation repository the reply was captured with.
        session_id: The session whose reply count discovery just resolved.

    Returns:
        True when the question was opened.
    """
    if screen.app.screen is not screen:
        return False
    question = choice.question_for(repository)
    if not question.take_pending(session_id):
        return False
    logger.info("Citation records question opened after a Library reply")
    screen.app.push_screen(
        CitationRecordsQuestionDialog(AFTER_REPLY_MESSAGE),
        callback=partial(_answered, screen, repository),
    )
    return True


def open_citation_records_choice(screen: Any) -> None:
    """Command palette: change the answer, or say what config.toml decides.

    Args:
        screen: The Console screen.
    """
    _token, repository = screen._console_citation_repository_readiness()
    if repository is None:
        screen.app.notify(UNAVAILABLE_NOTICE, severity="warning", markup=False)
        return
    switch = choice.configured_switch()
    if switch is ConfiguredSwitch.ON:
        screen.app.notify(CONFIG_ON_NOTICE, markup=False)
        return
    if switch is ConfiguredSwitch.OFF:
        screen.app.notify(CONFIG_OFF_NOTICE, markup=False)
        return
    screen.run_worker(
        _open_from_palette(screen, repository),
        group="console-citation-records-question",
        exclusive=True,
    )


async def _open_from_palette(screen: Any, repository: Any) -> None:
    read = await asyncio.to_thread(choice.load_answer)
    screen.app.push_screen(
        CitationRecordsQuestionDialog(palette_message(read.answer)),
        callback=partial(_answered, screen, repository),
    )


def _answered(screen: Any, repository: Any, result: bool | None) -> None:
    if result is None:
        logger.info("Citation records question closed without an answer")
        return
    answer = CitationRecordsAnswer.KEEP if result else CitationRecordsAnswer.OFF
    screen.run_worker(
        _save_and_apply(screen, repository, answer),
        group="console-citation-records-answer",
        exclusive=True,
    )


async def _save_and_apply(
    screen: Any, repository: Any, answer: CitationRecordsAnswer
) -> None:
    # The session follows the answer at once, saved or not.
    enabled = choice.apply_answer(repository, answer)
    try:
        await asyncio.to_thread(choice.save_answer, answer)
        saved = True
    except Exception as exc:  # noqa: BLE001 - the session still gets its answer
        logger.warning(
            "Citation records answer not saved; exception_type={}",
            type(exc).__name__,
        )
        saved = False
    logger.info(
        "Citation records answer applied; answer={}; saved={}; records_on={}",
        answer.value,
        saved,
        enabled,
    )
    notice = KEPT_NOTICE if enabled else NOT_KEPT_NOTICE
    if not saved:
        notice = f"{notice} {SAVE_FAILED_NOTICE}"
    screen.app.notify(
        notice, severity="information" if saved else "warning", markup=False
    )


__all__ = [
    "AFTER_REPLY_MESSAGE",
    "CitationRecordsQuestionDialog",
    "PALETTE_LABEL",
    "QUESTION_TITLE",
    "ask_pending_question",
    "open_citation_records_choice",
    "palette_message",
]
