"""Re-keying wide-modal selectors must preserve real consumer geometry."""

import pytest
from textual.containers import Vertical

from Tests.private_profile import private_profile_test
from Tests.UI.test_modal_wide_tier import WideTierHarness
from tldw_chatbook.Utils.file_extraction import ExtractedFile
from tldw_chatbook.Widgets.file_extraction_dialog import FileExtractionDialog
from tldw_chatbook.Widgets.Persona_Widgets.conversation_attach_picker import (
    ConversationAttachPicker,
)
from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import (
    DictionaryAttachPicker,
)


@pytest.mark.parametrize("kind", ["extraction", "dictionary", "conversation"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.asyncio
@private_profile_test
async def test_wide_modal_preserves_geometry_through_threshold_and_resize(
    kind, theme, request
):
    app = WideTierHarness()
    app.theme = theme
    rows = [{"conversation_id": "review", "title": "Review conversation"}]
    if kind == "extraction":
        modal = FileExtractionDialog(
            [
                ExtractedFile(
                    filename="review.txt",
                    content="Review only",
                    language="text",
                    start_pos=0,
                    end_pos=11,
                )
            ]
        )
        base_percent, base_cap, wide_cap = 80, 100, 150
    else:
        modal = (
            DictionaryAttachPicker(rows)
            if kind == "dictionary"
            else ConversationAttachPicker(rows)
        )
        base_percent, base_cap, wide_cap = 60, 80, 120
    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal)
        for width in (120, 149, 150, 170, 200, 120):
            await pilot.resize_terminal(width, 48)
            await pilot.pause()
            await pilot.pause()
            body = modal.query_one(f"{type(modal).__name__} > Vertical", Vertical)
            expected = (
                min(wide_cap, width * 85 // 100)
                if width >= 150
                else min(base_cap, width * base_percent // 100)
            )
            assert body.region.width == expected
            assert body.region.x == (width - expected) // 2
            assert 0 < body.region.height <= app.size.height
