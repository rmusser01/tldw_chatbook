"""Composer paste controls own local events without rebuilding their inputs."""

import pytest
from textual import on
from textual.containers import Vertical
from textual.widgets import Checkbox, Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Settings_Widgets.composer_paste_panel import (
    ComposerPasteSettingsPanel,
)


class PastePanelHarness(ConsolidatedCSSApp):
    CSS_PATH = list(APP_STYLESHEETS)

    def __init__(self):
        super().__init__()
        self.edits = []

    def compose(self):
        with Vertical(id="settings-console-behavior-card"):
            yield ComposerPasteSettingsPanel(
                enabled=True, threshold="50", default_threshold=1000
            )

    @on(ComposerPasteSettingsPanel.Edited)
    def record_edit(self, event):
        self.edits.append((event.key, event.value))


@pytest.mark.parametrize("size", [(90, 30), (180, 50)])
@private_profile_test
async def test_sync_preserves_input_identity_selection_and_user_edits(request, size):
    app = PastePanelHarness()
    async with app.run_test(size=size) as pilot:
        panel = app.query_one(ComposerPasteSettingsPanel)
        threshold = panel.query_one(Input)
        await pilot.pause()
        assert app.edits == []
        for selector in (
            "#settings-console-collapse-large-pastes-label",
            "#settings-console-collapse-large-pastes-help",
        ):
            assert panel.query_one(selector).styles.padding.left == 1
            assert panel.query_one(selector).styles.padding.right == 1
        threshold.focus()
        await pilot.press("end", "shift+left")
        selection = threshold.selection
        panel.sync_values(enabled=False, threshold="50")
        await pilot.pause()
        assert panel.query_one(Input) is threshold
        assert app.focused is threshold
        assert threshold.selection == selection
        assert app.edits == []
        panel.sync_values(enabled=False, threshold="120")
        await pilot.pause()
        assert app.edits == []
        threshold.value = "50"
        await pilot.pause()
        toggle = panel.query_one(Checkbox)
        toggle.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.edits == [
            ("paste_collapse_threshold", "50"),
            ("collapse_large_pastes", True),
        ]
        assert panel.query_one(Input) is threshold
