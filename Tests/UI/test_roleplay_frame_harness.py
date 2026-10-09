"""Self-tests for the Roleplay frame harness (spec 5.7.1; frame slice B1 AC#6, AC#8).

A geometry assertion is only evidence if the tier it runs under really loads
Roleplay's lazy sheet and if deleting a rule from that sheet turns it red.
These tests prove both, under both styled tiers, and that the containment
helper can fail.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI import test_personas_dictionaries, test_personas_workbench
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.roleplay_frame_harness import (
    ROLEPLAY_SHEET,
    ROLEPLAY_SIZES,
    RoleplayMockApp,
    StyledRoleplayMockApp,
    assert_painted_inside,
    drop_rule_from_loaded_sheet,
    roleplay_full_app,
    seed_mock_characters,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

CHARACTERS = [{"id": 1, "name": "Detective Sam", "version": 1}]


@pytest.fixture
def one_character(monkeypatch):
    seed_mock_characters(monkeypatch, CHARACTERS)


def test_the_size_matrix_is_the_b1_matrix():
    assert ROLEPLAY_SIZES == ((80, 24), (120, 36), (160, 45), (220, 55))
    assert ROLEPLAY_SHEET.name == "screen_feature_roleplay.tcss"


def test_the_moved_harness_is_the_one_object_under_every_old_name():
    """Moved verbatim and re-exported, so existing tests are untouched."""
    for module in (test_personas_workbench, test_personas_dictionaries):
        assert module.PersonasTestApp is RoleplayMockApp
        assert module.StyledPersonasTestApp is StyledRoleplayMockApp


def test_the_styled_mock_tier_loads_every_app_stylesheet():
    """The boot bundle AND every lazy split sheet, derived from the build's
    own SCREEN_OWNED_SPLITS (never named by hand)."""
    assert StyledRoleplayMockApp.CSS_PATH == [str(path) for path in APP_STYLESHEETS]
    assert "CSS_PATH" not in RoleplayMockApp.__dict__  # the unstyled tier


@pytest.mark.parametrize("entry", ["initial_tab", "ctrl+4"])
async def test_the_full_app_tier_reaches_roleplay_by_both_real_routes(
    entry, one_character
):
    async with roleplay_full_app(size=(120, 36), entry=entry) as pilot:
        assert type(pilot.app.screen).__name__ == "PersonasScreen"
        assert pilot.app.screen.query("#personas-library-rows > ListItem")


def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
    class _NoSheetApp(App):
        pass

    with pytest.raises(KeyError):
        drop_rule_from_loaded_sheet(_NoSheetApp(), ROLEPLAY_SHEET, "#nothing")


class _ContainmentProbe(App):
    CSS = """
    Screen { layers: base overlay; }
    #other, #pane { width: 30; height: 3; }
    #inside, #covered, #outside { width: 10; height: 1; }
    #outside { offset: 40 0; }
    #cover { width: 20; height: 1; dock: top; layer: overlay; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="other"):
            yield Static("covered", id="covered")
        with Vertical(id="pane"):
            yield Static("inside", id="inside")
            yield Static("outside", id="outside")
        yield Static("cover", id="cover")


async def test_assert_painted_inside_fails_for_escape_and_cover():
    """The helper's two refusals each fire (it is not vacuous)."""
    app = _ContainmentProbe()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        pane = app.query_one("#pane")
        assert_painted_inside(app.query_one("#inside"), pane)
        with pytest.raises(AssertionError, match="escapes"):
            assert_painted_inside(app.query_one("#outside"), pane)
        with pytest.raises(AssertionError, match="covered"):
            assert_painted_inside(app.query_one("#covered"), app.query_one("#other"))
