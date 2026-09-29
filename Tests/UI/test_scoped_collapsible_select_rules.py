"""TASK-33003.1: the retired CCP/evals sheets no longer style every Collapsible/Select.

`features/_conversations.tcss` and `features/_evaluation_unified.tcss` shipped
bare `Collapsible`, `Collapsible > CollapsibleTitle` and `Select` type rules.
Neither sheet's owning root is composed anywhere any more, so the rules only
reached other screens: a 3-row title, a tall border, and a blank row plus a
full-row width under every Select. These probes mount the two
model-configuration surfaces at 211x44 under the production stylesheets.
"""

from typing import ClassVar

import pytest
from textual.widgets import Collapsible, Select
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp, app_css_text
from Tests.UI.test_console_mcp_approval import _sample_calls
from Tests.UI.test_console_session_settings import StyledModalHarness
from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
from Tests.UI.test_non_obscuring_focus_contract import css_blocks
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsContextEstimate,
    ConsoleSettingsModal,
)

SIZE = (211, 44)


class _StyledSettingsHarness(DestinationHarness):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]


def test_app_stylesheets_carry_no_unscoped_collapsible_or_select_geometry():
    text = app_css_text()
    assert css_blocks(text, "Select") == []
    assert css_blocks(text, "Collapsible > Container") == []
    assert css_blocks(text, "Collapsible.-collapsed") == []
    assert not any(
        "height" in block for block in css_blocks(text, "Collapsible > CollapsibleTitle")
    )
    assert not any("tall" in block for block in css_blocks(text, "Collapsible"))


def _assert_no_leaked_geometry(screen) -> None:
    titles = list(screen.query(CollapsibleTitle))
    selects = list(screen.query(Select))
    assert titles and selects, "probe found nothing to measure"
    for title in titles:
        assert str(title.styles.height) != "3", f"{title.parent.id} title forced to 3"
        if title.region.area:
            assert title.region.height == 1, (title.parent.id, title.region)
    for collapsible in screen.query(Collapsible):
        assert "tall" not in {edge[0] for edge in collapsible.styles.border}, (
            f"{collapsible.id} took the tall border"
        )
    for select in selects:
        assert select.styles.margin.bottom == 0, f"{select.id} kept a blank row"
        assert str(select.styles.width) != "100%", f"{select.id} claims 100%"


@pytest.mark.asyncio
async def test_chat_settings_collapsed_section_costs_two_rows_under_production_css():
    app = StyledModalHarness()
    async with app.run_test(size=SIZE) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(
                    provider="llama_cpp",
                    model="model-a",
                    base_url="http://127.0.0.1:9099",
                ),
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        screen = app.screen
        _assert_no_leaked_geometry(screen)

        sections = [c for c in screen.query(Collapsible) if c.region.area]
        assert len(sections) == 3
        for section in sections:
            assert section.collapsed
            margin = section.styles.margin
            cost = section.region.height + margin.top + margin.bottom
            assert cost <= 2, f"{section.id} costs {cost} rows ({section.region})"

        # Focus must not add a row either: the retired `height: 3` used to
        # absorb the app-wide focus rule's border-bottom (review round 1).
        for section in sections:
            section.query_one(CollapsibleTitle).focus()
            await pilot.pause()
            assert section.region.height == 1, f"focused {section.id}: {section.region}"

        screen.query_one("#console-settings-view-context").press()
        await pilot.pause()
        shown = [s for s in screen.query(Select) if s.region.area]
        assert shown, "Context view shows no Select to measure"
        for select in shown:
            row = select.parent
            assert select.region.right <= row.content_region.right, select.id


@pytest.mark.asyncio
@private_profile_test
async def test_settings_providers_models_has_no_leaked_geometry(request):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"}
    }
    host = _StyledSettingsHarness(app, "settings")
    async with host.run_test(size=SIZE) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = host.screen_stack[-1]
        assert screen.query_one("#settings-generation-defaults", Collapsible)
        _assert_no_leaked_geometry(screen)


class _StyledApprovalCardHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def compose(self):
        yield ChatApprovalCard()


@pytest.mark.asyncio
async def test_approval_decision_select_keeps_its_bounded_share_of_the_row():
    """The Select in a Horizontal row must not claim the row once the bare rule is gone.

    Card-level twin of test_console_mcp_approval's Console-mounted guard,
    which needs the full production Console to settle before it measures.
    """
    app = _StyledApprovalCardHarness()
    async with app.run_test(size=(200, 40)) as pilot:
        app.query_one(ChatApprovalCard).set_batch(_sample_calls(), timeout_seconds=45.0)
        await pilot.pause()
        rows = list(app.query(".approval-row"))
        assert len(rows) == 2
        for row in rows:
            select = row.query_one(".approval-row-decision", Select)
            assert select.styles.margin.bottom == 0
            assert 0 < select.size.width < row.size.width
            assert select.region.right <= row.region.right
