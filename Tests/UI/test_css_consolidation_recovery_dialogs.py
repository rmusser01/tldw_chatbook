"""Mounted state coverage for the less frequent consolidated recovery dialogs."""

from pathlib import Path

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp


def make_dialog(kind, root: Path):
    if kind == "markdown":
        from tldw_chatbook.Widgets.Console.console_save_markdown_modal import (
            ConsoleSaveMarkdownModal,
        )

        return ConsoleSaveMarkdownModal(default_path=str(root / "chat.md"))
    if kind == "roleplay":
        from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
            RoleplayDraftRecoveryDialog,
        )

        return RoleplayDraftRecoveryDialog(("conversation", "character"))
    if kind == "notes":
        from tldw_chatbook.Notes.recovery_review import NotesRecoveryReview
        from tldw_chatbook.Widgets.Library.notes_recovery_dialog import (
            NotesRecoveryDialog,
        )

        async def approve(_review):
            pytest.fail("Style review must not approve a recovered pairing")

        review = NotesRecoveryReview(
            "notes.sync_bindings",
            "fixture",
            root,
            (("note.md", "retained"),),
            (),
            (),
        )
        return NotesRecoveryDialog(review, current=lambda: True, approve=approve)
    from tldw_chatbook.Skills_Interop.recovery_activation import RecoveryReview
    from tldw_chatbook.UI.Screens.skills_screen import SkillRecoveryReviewModal

    review = RecoveryReview("fixture", (), (str(root), 0, 0), (), (), (), "fixture")
    return SkillRecoveryReviewModal(review, root, lambda: True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["markdown", "roleplay", "notes", "skills"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_consolidated_dialog_actions_keep_focus_and_cancel(
    request, tmp_path, kind, theme, size
):
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    app.theme = theme
    result = []
    async with app.run_test(size=size) as pilot:
        modal = make_dialog(kind, tmp_path)
        app.push_screen(modal, result.append)
        await pilot.pause()
        controls = list(modal.query("Button, Input"))
        assert controls and any(isinstance(control, Button) for control in controls)
        for control in controls:
            assert isinstance(control, (Button, Input))
            control.scroll_visible(animate=False)
            control.focus()
            await pilot.pause()
            assert control.has_focus
            region, clip = modal._compositor.visible_widgets[control]
            assert region.intersection(clip) == region
            await pilot.hover(control)
            await pilot.pause()
            control.disabled = True
            await pilot.pause()
            control.disabled = False
            await pilot.pause()
        if kind == "markdown":
            # This existing dialog exposes Cancel but has no Escape binding.
            await pilot.click("#console-save-markdown-cancel")
        else:
            await pilot.press("escape")
        await pilot.pause()
        assert modal not in app.screen_stack
        assert result == [None]
