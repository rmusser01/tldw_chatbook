"""Settings ▸ Console Behavior: the two free-text Side chat fields take the row.

#3057 review R3 #10. The card's one 32-cell control column fits every Select
and number on it, and it also caught "Side chat model" and "Prompt template":
a model id and a prompt were cut to 27 cells of text, and the built-in
template shown as the placeholder ("Give me more details about:
{selection}", 39 cells) was never whole. Mounted with the real application
stylesheet at 211x44.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_console_behavior_grammar import _open_card
from Tests.UI.test_settings_console_fallback_rows import _SIZE, _app
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.config import DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE

_FREE_TEXT = (
    "#settings-console-sidechat-model",
    "#settings-console-sidechat-prompt-template",
)


@pytest.mark.asyncio
@private_profile_test
async def test_side_chat_free_text_fields_take_the_row_past_the_control_column(
    request,
):
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        column = card.query_one("#settings-console-max-parallel-runs", Input)
        template = card.query_one(_FREE_TEXT[1], Input)

        for selector in _FREE_TEXT:
            field = card.query_one(selector, Input)
            assert field.region.width > column.region.width, (selector, field.region)
            assert field.region.x == column.region.x, (selector, field.region)
            assert field.region.right <= card.region.right, (selector, field.region)
        assert template.placeholder == DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE
        assert template.content_region.width > cell_len(template.placeholder)
