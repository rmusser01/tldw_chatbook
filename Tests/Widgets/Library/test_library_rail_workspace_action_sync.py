"""Retained workspace actions track the same policy used on first mount."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from Tests.Widgets.Library.test_library_rail import _make_shell
from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail


@pytest.mark.asyncio
async def test_retained_handoff_action_tracks_recovery_without_replacing_button(
    widget_pilot,  # noqa: F811
):
    state = SimpleNamespace(
        source_rows=(),
        context_handoff_enabled=True,
        context_handoff_tooltip="Eligible workspace context",
    )
    policy_owner = SimpleNamespace(
        _library_lookup_error=None,
        _library_lookup_recovery_state=None,
        _has_local_sources=lambda: True,
    )

    def action_state():
        return LibraryScreen._workspace_handoff_action_state(policy_owner, state)

    def action_widgets():
        blocked, tooltip = action_state()
        return LibraryScreen._workspace_action_widgets(
            policy_owner, state, handoff_disabled=blocked, handoff_tooltip=tooltip
        )

    shell = _make_shell()
    preferences = LibraryRailPreferences()
    async with widget_pilot(
        LibraryRail,
        shell=shell,
        preferences=preferences,
        workspaces_body_factory=action_widgets,
    ) as pilot:
        rail = pilot.app.test_widget
        button = rail.query_one("#library-use-in-console", Button)
        assert button.tooltip == "Stage Library source context in Console."
        assert not button.has_class("library-source-action-blocked")
        for error, recovery, expected in (
            (
                "Unavailable",
                None,
                "Library source services are unavailable; retry Library later.",
            ),
            (
                "Policy denied",
                SimpleNamespace(disabled_tooltip="Custom policy tooltip."),
                "Custom policy tooltip.",
            ),
            (None, None, "Stage Library source context in Console."),
        ):
            policy_owner._library_lookup_error = error
            policy_owner._library_lookup_recovery_state = recovery
            rail.sync_state(shell, preferences, workspace_handoff_action=action_state())
            await pilot.pause()
            assert rail.query_one("#library-use-in-console", Button) is button
            assert button.tooltip == expected
            assert button.has_class("library-source-action-blocked") is bool(error)
            assert button.disabled is False
