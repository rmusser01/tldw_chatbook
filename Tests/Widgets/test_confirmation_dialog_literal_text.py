"""A destructive confirmation must name the thing it is about to destroy.

TASK-32802.2. ``ConfirmationDialog`` rendered its title and message with
markup on, and 47 modules feed it prose with a user-supplied name spliced
in. On Textual 8 that means:

    '[TODO] Q3 plan'  named  ' Q3 plan'     -- the wrong subject
    '[IMPORTANT]'     named  ''             -- no subject at all
    '[/b] plan'       raised MarkupError inside compose

The third is the serious one: the irreversible "Delete stored Full
captures" confirmation never appeared, so the user could not see what they
were about to lose. Measured before the fix with this same harness.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield Label("host")


# The shapes users put in a conversation, watchlist, preset or file name.
# `[bold]` is the one the old escape happened to survive, and is kept as the
# control that this is not a regression in the ordinary case.
SUBJECTS = [
    "[TODO] Q3 plan",
    "[IMPORTANT]",
    "[/b] plan",
    "[WIP] draft",
    "[bold]not actually bold[/bold]",
    "plain name",
    "R&D report",
]


async def _rendered(subject: str) -> tuple[str, str]:
    message = f'Delete stored Full captures for "{subject}"? This cannot be undone.'
    title = f"Delete captures: {subject}"
    app = _Harness()
    async with app.run_test() as pilot:
        await app.push_screen(
            ConfirmationDialog(title=title, message=message, confirm_label="Delete")
        )
        await pilot.pause()
        return (
            app.screen.query_one(".dialog-title").render().plain,
            app.screen.query_one(".dialog-message").render().plain,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", SUBJECTS)
async def test_the_confirmation_names_its_subject_verbatim(subject):
    """Both the title and the message must show the name the user gave."""
    rendered_title, rendered_message = await _rendered(subject)
    assert subject in rendered_message, (
        f"the confirmation named the wrong subject: {rendered_message!r}"
    )
    assert subject in rendered_title, (
        f"the dialog title named the wrong subject: {rendered_title!r}"
    )


@pytest.mark.asyncio
async def test_a_closing_tag_does_not_stop_the_dialog_appearing():
    """The failure that hid an irreversible action behind an exception."""
    _title, message = await _rendered("[/b] plan")
    assert "[/b] plan" in message


@pytest.mark.asyncio
async def test_callers_must_not_pre_escape():
    """The dialog renders literally, so an escaped caller shows a backslash.

    This is the contract the 11 pre-escaping call sites were changed to
    match; it fails loudly if someone re-adds an escape at a caller.
    """
    _title, message = await _rendered("\\[TODO] pre-escaped")
    assert "\\[TODO]" in message
