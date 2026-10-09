"""A partially mounted Console has not applied its responsive width band."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.css.query import NoMatches

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen, console_rail_width_band

pytestmark = pytest.mark.bootstrap_profile

_REQUIRED = (
    "#console-workspace-grid",
    "#console-left-rail",
    "#console-right-rail",
    "#console-context-rail-handle",
    "#console-inspector-rail-handle",
)


@pytest.mark.parametrize("missing", _REQUIRED)
def test_partial_mount_resize_defers_band_until_all_rails_exist(missing):
    widgets = {selector: SimpleNamespace(display=True) for selector in _REQUIRED}
    withheld = widgets.pop(missing)

    def query(selector, *_args):
        if selector not in widgets:
            raise NoMatches(selector)
        return widgets[selector]

    rail_state = SimpleNamespace(left_open=True, right_open=True)
    screen = SimpleNamespace(
        _last_console_workspace_width_band=None,
        query_one=query,
        app=SimpleNamespace(focused=None),
        _is_descendant_or_self=lambda *_: False,
        _current_console_rail_state=Mock(return_value=rail_state),
        _sync_console_rail_visibility_if_changed=Mock(),
        _notify_console_responsive_rail_collapse=Mock(),
        _request_console_control_bar_sync=Mock(),
    )
    event = SimpleNamespace(size=SimpleNamespace(width=160))
    ChatScreen._adapt_console_workspace_to_width(screen, event)
    assert screen._last_console_workspace_width_band is None
    screen._current_console_rail_state.assert_not_called()
    screen._request_console_control_bar_sync.assert_not_called()

    # The normal later resize/retry must still apply the exact same band.
    widgets[missing] = withheld
    ChatScreen._adapt_console_workspace_to_width(screen, event)
    assert screen._last_console_workspace_width_band == console_rail_width_band(160)
    screen._current_console_rail_state.assert_called_once_with(available_columns=160)
    screen._sync_console_rail_visibility_if_changed.assert_called_once_with(rail_state)
    screen._request_console_control_bar_sync.assert_called_once()
    ChatScreen._adapt_console_workspace_to_width(screen, event)
    assert screen._current_console_rail_state.call_count == 1
    event.size.width = 80
    ChatScreen._adapt_console_workspace_to_width(screen, event)
    assert screen._last_console_workspace_width_band == console_rail_width_band(80)
    assert screen._current_console_rail_state.call_count == 2
