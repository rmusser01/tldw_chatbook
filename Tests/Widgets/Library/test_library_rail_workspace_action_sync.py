"""Retained workspace actions track the same policy used on first mount."""

from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widget import Widget
from textual.widgets import Button, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from Tests.Widgets.Library.test_library_rail import _make_shell
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_rail import (
    LibraryRail,
    library_dim_label_text,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.display_state import build_library_workspace_depth_state


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_mode", ["current", "omitted", "empty"])
@pytest.mark.parametrize("include_summary", [True, False])
async def test_retained_handoff_summary_tracks_owner_without_remount(
    widget_pilot: Callable[..., Any],  # noqa: F811
    tmp_path: Path,
    summary_mode: str,
    include_summary: bool,
) -> None:
    """Patch supplied summaries while preserving optional rows and live state.

    Args:
        widget_pilot: Mount a standalone rail in the existing Textual harness.
        tmp_path: Private directory for the caller-owned workspace database.
        summary_mode: Supply current text, omit the argument, or supply empty text.
        include_summary: Whether the optional Handoff Static exists at mount.
    """
    with closing(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="rail-test")
    ) as db:
        registry = LocalWorkspaceRegistryService(db)
        registry.create_workspace(workspace_id="ws-a", name="Workspace A")
        registry.create_workspace(workspace_id="ws-b", name="Workspace B")
        registry.set_active_workspace("ws-a")
        for workspace_id, note_id in (("ws-a", "local"), ("ws-b", "cross")):
            registry.link_membership(
                workspace_id, item_type="note", item_id=note_id, title=note_id
            )
        state = build_library_workspace_depth_state(
            registry_service=registry, source_records={}
        )
        policy_owner = SimpleNamespace(
            _library_lookup_error=None,
            _library_lookup_recovery_state=None,
            _has_local_sources=lambda: bool(state.source_rows),
        )
        initial_summary = LibraryScreen._workspace_handoff_summary_label(
            policy_owner, state
        )
        body_calls = 0

        def action_state() -> tuple[bool, str]:
            """Read the current action from its existing policy owner.

            Returns:
                Current blocked flag and tooltip.
            """
            return LibraryScreen._workspace_handoff_action_state(policy_owner, state)

        def body_widgets() -> list[Widget]:
            """Compose real summary and action widgets once, with overflow.

            Returns:
                Summary, actions, and enough content to exercise rail scrolling.
            """
            nonlocal body_calls
            body_calls += 1
            blocked, tooltip = action_state()
            widgets = (
                [
                    Static(
                        library_dim_label_text("Handoff", initial_summary),
                        id="library-workspaces-handoff",
                    )
                ]
                if include_summary
                else []
            )
            widgets.extend(
                LibraryScreen._workspace_action_widgets(
                    policy_owner,
                    state,
                    handoff_disabled=blocked,
                    handoff_tooltip=tooltip,
                )
            )
            widgets.append(Static("\n".join(["Overflow"] * 40)))
            return widgets

        shell = _make_shell()
        preferences = LibraryRailPreferences(details_open=True)
        async with widget_pilot(
            LibraryRail,
            shell=shell,
            preferences=preferences,
            workspaces_body_factory=body_widgets,
        ) as pilot:
            rail = pilot.app.test_widget
            rail.styles.height = 12
            await pilot.pause()
            button = rail.query_one("#library-use-in-console", Button)
            summary = (
                rail.query_one("#library-workspaces-handoff", Static)
                if include_summary
                else None
            )
            if summary is not None:
                assert summary.renderable == library_dim_label_text(
                    "Handoff", initial_summary
                )
            button.focus()
            await pilot.pause()
            rail.scroll_to(y=2, animate=False, force=True)
            await pilot.pause()
            scroll_y = rail.scroll_y
            assert scroll_y > 0
            assert pilot.app.focused is button
            for notes in (({"id": "local"},), ({"id": "local"}, {"id": "cross"}), ()):
                state = build_library_workspace_depth_state(
                    registry_service=registry, source_records={"notes": notes}
                )
                current_summary = LibraryScreen._workspace_handoff_summary_label(
                    policy_owner, state
                )
                kwargs = {}
                if summary_mode != "omitted":
                    kwargs["workspace_handoff_summary"] = (
                        current_summary if summary_mode == "current" else ""
                    )
                rail.sync_state(
                    shell,
                    preferences,
                    workspace_handoff_action=action_state(),
                    **kwargs,
                )
                await pilot.pause()
                assert rail.query_one("#library-use-in-console", Button) is button
                blocked, tooltip = action_state()
                assert button.tooltip == tooltip
                assert button.has_class("library-source-action-blocked") is blocked
                assert button.disabled is False
                if summary is not None:
                    expected_summary = {
                        "current": current_summary,
                        "omitted": initial_summary,
                        "empty": "",
                    }[summary_mode]
                    assert (
                        rail.query_one("#library-workspaces-handoff", Static) is summary
                    )
                    assert summary.renderable == library_dim_label_text(
                        "Handoff", expected_summary
                    )
                else:
                    assert not pilot.app.query("#library-workspaces-handoff")
                assert pilot.app.focused is button
                assert button.is_attached
                assert rail.scroll_y == scroll_y
                assert body_calls == 1


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
