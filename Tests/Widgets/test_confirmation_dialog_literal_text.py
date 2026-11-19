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
from textual.color import Color
from textual.containers import VerticalScroll
from textual.widgets import Button, Label

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Widgets.cancel_confirmation_dialog import CancelConfirmationDialog
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


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("dismiss_key", ("enter", "escape"))
async def test_cancel_confirmation_scroll_body_keeps_primary_border_and_safe_default(
    dismiss_key: str,
) -> None:
    """Keep cancellation's border, literal prose and safe default together.

    Args:
        dismiss_key: The safe default action or the dialog's Escape binding.
    """
    app = ConsolidatedCSSApp(css_path=BUNDLED_STYLESHEET)
    results: list[bool] = []
    title = "Cancel [TODO] queued prompt?"
    message = "Keep [/b] unsent prompt?"
    dialog = CancelConfirmationDialog(title=title, message=message)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog, results.append)
        await pilot.pause()
        body = dialog.query_one("#confirmation-dialog", VerticalScroll)
        assert dialog.query_one(".dialog-title").render().plain == title
        assert dialog.query_one(".dialog-message").render().plain == message
        primary = Color.parse(str(app.get_css_variables()["primary"]))
        accent = Color.parse(str(app.get_css_variables()["accent"]))
        assert primary != accent, "the fixture must distinguish the two borders"
        for edge in (
            body.styles.border_top,
            body.styles.border_right,
            body.styles.border_bottom,
            body.styles.border_left,
        ):
            assert edge == ("thick", primary), (
                f"cancellation inherited the base accent border: {edge!r}"
            )
        keep_processing = dialog.query_one("#cancel-button", Button)
        assert app.focused is keep_processing
        region, clip = dialog._compositor.visible_widgets[keep_processing]
        assert region.width > 0
        assert region.height > 0
        assert region.intersection(clip) == region
        center = region.x + region.width // 2, region.y + region.height // 2
        assert dialog.get_widget_at(*center)[0] is keep_processing
        await pilot.press(dismiss_key)
        await pilot.pause()
        assert dialog not in app.screen_stack
        assert results == [False]
