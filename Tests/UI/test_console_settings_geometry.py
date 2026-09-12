"""Responsive geometry contracts for the Console Conversation Settings modal."""

from __future__ import annotations

import pytest
from textual import events
from textual.containers import ScrollableContainer
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    MODEL_DISCOVER_BUTTON_ID,
    MODEL_DISCOVER_BUTTON_LABEL,
    ConsoleSettingsModal,
    _settings_screen_region,
)


GEOMETRY_SIZES = ((80, 24), (100, 30), (160, 40))
LONG_CONNECTION_STATUS = (
    "Testing connection to a configured private endpoint by listing models; "
    "this does not generate text or verify model generation."
)


class GeometryHarness(ConsolidatedCSSApp):
    """Isolated app that loads the same consolidated CSS as production."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
            }
        }


def build_geometry_modal(app: GeometryHarness, *, ready: bool) -> ConsoleSettingsModal:
    """Build either a blocked first-use or configured power-user draft."""
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a" if ready else None,
            base_url="http://127.0.0.1:9099",
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a", "model-b"] if ready else []},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )


@pytest.mark.parametrize("size", GEOMETRY_SIZES)
@pytest.mark.parametrize("ready", [False, True], ids=["blocked", "ready"])
@pytest.mark.asyncio
async def test_conversation_settings_size_matrix_has_bounded_fluid_geometry(
    size: tuple[int, int],
    ready: bool,
) -> None:
    """Every supported viewport stays bounded and derives compact mode from width."""
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=ready)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        container = modal.query_one("#console-settings-modal")
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        compact = container.size.width < 100

        assert modal.has_class("-conversation-settings-compact") is compact
        assert 0 < container.region.width <= app.size.width
        assert 0 < container.region.height <= app.size.height
        assert body.virtual_size.width <= body.container_size.width
        assert container.styles.overflow_y == "hidden"
        assert body.styles.overflow_y == "auto"
        assert [
            widget
            for widget in modal.query(ScrollableContainer)
            if type(widget) is ScrollableContainer and widget.display
        ] == [body]

        for widget in modal.query("*"):
            if not widget.display:
                continue
            assert widget.region.width >= 0
            assert widget.region.height >= 0
            screen_region = _settings_screen_region(widget)
            assert screen_region.x >= 0
            assert screen_region.right <= app.size.width, (
                widget.id,
                widget.styles.width,
                widget.styles.max_width,
                widget.parent.region if widget.parent is not None else None,
                container.region,
                body.region,
            )

        connection = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        cancel = modal.query_one("#console-settings-cancel", Button)
        save_default = modal.query_one("#console-settings-save-default", Button)
        save = modal.query_one("#console-settings-save", Button)
        assert str(connection.label) == MODEL_DISCOVER_BUTTON_LABEL
        assert str(save_default.label) == "Save as provider defaults"
        assert str(save.label) == "Use for this conversation"
        painted = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )
        assert "Use for this conversation" in painted
        assert "Use in this conversation" not in painted
        for action in (connection, cancel, save_default, save):
            assert action.region.width >= len(str(action.label))

        connection_row = modal.query_one("#console-settings-connection-actions")
        footer = modal.query_one("#console-settings-actions")
        if compact:
            assert connection_row.layout.name == "vertical"
            assert footer.layout.name == "vertical"
            assert connection.region.height == 1
            assert len({action.region.y for action in (cancel, save_default, save)}) == 3
            assert {
                action.region.width for action in (cancel, save_default, save)
            } == {footer.content_region.width}
        else:
            assert connection_row.layout.name == "horizontal"
            assert footer.layout.name == "horizontal"
            assert len({action.region.y for action in (cancel, save_default, save)}) == 1

        status = modal.query_one("#console-settings-model-discover-status", Static)
        status.display = True
        status.update(LONG_CONNECTION_STATUS)
        await pilot.pause()
        assert status.region.height >= 2

        readiness_text = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        if ready:
            assert "Ready to send" in readiness_text
        else:
            assert "Not ready" in readiness_text
            assert "Ready to send" not in readiness_text


@pytest.mark.parametrize("size", GEOMETRY_SIZES)
@pytest.mark.asyncio
async def test_connection_and_completion_actions_are_keyboard_reachable(
    size: tuple[int, int],
) -> None:
    """Body scrolling reaches Connection while completion actions remain reachable."""
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        body = modal.query_one("#console-settings-body", ScrollableContainer)
        connection = modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        connection.focus()
        await pilot.pause(0.12)
        await pilot.pause()
        assert app.focused is connection
        focus_chain = []
        current = connection
        while current is not None:
            focus_chain.append(
                (
                    current.id,
                    current.virtual_region.y,
                    current.virtual_region.height,
                )
            )
            if current is body:
                break
            current = current.parent
        assert 0 <= body.scroll_y <= body.max_scroll_y, focus_chain
        connection_region = _settings_screen_region(connection)
        assert connection_region.bottom > body.content_region.y, focus_chain
        assert connection_region.y < body.content_region.bottom, focus_chain

        save = modal.query_one("#console-settings-save", Button)
        save.focus()
        save.scroll_visible(animate=False)
        await pilot.pause()
        assert app.focused is save
        save_region = _settings_screen_region(save)
        assert save_region.y >= 0
        assert save_region.bottom <= app.size.height


@pytest.mark.asyncio
async def test_compact_tab_traversal_reveals_every_body_focus_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every default Tab target inside the compact body intersects its viewport."""
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        body = modal.query_one("#console-settings-body", ScrollableContainer)
        reveal_records = []
        original_reveal = modal._reveal_focused_control

        def record_reveal(widget, generation):
            reveal_records.append((widget, modal.focused))
            original_reveal(widget, generation)

        monkeypatch.setattr(modal, "_reveal_focused_control", record_reveal)
        targets = modal._settings_focus_targets()
        assert targets
        targets[0].focus()
        await pilot.pause(0.12)
        await pilot.pause()

        for index, expected in enumerate(targets):
            assert app.focused is expected
            if index:
                assert reveal_records[-1] == (expected, expected)
            if body in expected.ancestors:
                target_region = _settings_screen_region(expected)
                viewport = body.content_region
                assert target_region.right > viewport.x
                assert target_region.x < viewport.right
                if target_region.height <= viewport.height:
                    assert target_region.y >= viewport.y
                    assert target_region.bottom <= viewport.bottom
                else:
                    assert target_region.bottom > viewport.y
                    assert target_region.y < viewport.bottom
                ancestor = expected.parent
                while ancestor is not None and ancestor is not body:
                    assert float(ancestor.scroll_y) == 0, (
                        expected.id,
                        ancestor.id,
                        ancestor.scroll_y,
                    )
                    ancestor = ancestor.parent
                assert ancestor is body
            if index < len(targets) - 1:
                reveal_generation = modal._focus_reveal_generation
                await pilot.press("tab")
                await pilot.pause(0.12)
                await pilot.pause()
                assert modal._focus_reveal_generation > reveal_generation


@pytest.mark.asyncio
async def test_new_focus_supersedes_pending_suspended_anchor_and_is_revealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new target clears restore fencing and resumes ordinary ancestor reveal."""
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        body = modal.query_one("#console-settings-body", ScrollableContainer)
        modal.query_one("#model-search-picker-input", Input).focus()
        body.scroll_to(y=7, animate=False)
        await pilot.pause()
        snapshot = modal.capture_suspended_draft()
        assert snapshot.scroll_anchor > 0

        modal.query_one("#console-settings-provider-picker-input", Input).focus()
        await pilot.pause()
        modal._restore_suspended_scroll_and_focus(snapshot)
        assert modal._pending_suspended_scroll_restore is not None

        reveal_records = []
        original_reveal = modal._reveal_focused_control

        def record_reveal(widget, generation):
            reveal_records.append((widget, modal.focused))
            original_reveal(widget, generation)

        monkeypatch.setattr(modal, "_reveal_focused_control", record_reveal)
        new_target = modal.query_one("#console-settings-base-url", Input)
        new_target.focus()
        await pilot.pause(0.12)
        await pilot.pause()

        assert app.focused is new_target
        assert modal._pending_suspended_scroll_restore is None
        assert reveal_records[-1] == (new_target, new_target)
        target_region = _settings_screen_region(new_target)
        viewport = body.content_region
        assert target_region.y >= viewport.y
        assert target_region.bottom <= viewport.bottom
        ancestor = new_target.parent
        while ancestor is not None and ancestor is not body:
            assert float(ancestor.scroll_y) == 0
            ancestor = ancestor.parent
        assert ancestor is body

        reveal_count = len(reveal_records)
        stale_target = modal.query_one("#model-search-picker-input", Input)
        reveal_generation = modal._focus_reveal_generation
        modal.on_descendant_focus(events.DescendantFocus(stale_target))
        assert modal._focus_reveal_generation == reveal_generation
        modal._queue_focus_reveal_after_layout(
            stale_target,
            reveal_generation,
        )
        await pilot.pause()
        assert len(reveal_records) == reveal_count
