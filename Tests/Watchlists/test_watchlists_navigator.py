from textual.widgets import Button

from tldw_chatbook.UI.Watchlists_Modules.watchlists_navigator import WatchlistsNavigator


def test_navigator_has_all_section_buttons():
    navigator = WatchlistsNavigator()
    buttons = list(navigator.compose())
    assert len(buttons) == 5
    assert [button.id for button in buttons] == ["nav-overview", "nav-sources", "nav-items", "nav-runs", "nav-rules"]
