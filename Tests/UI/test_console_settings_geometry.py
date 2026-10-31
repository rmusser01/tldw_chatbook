"""Responsive geometry contracts for the Console Chat settings modal."""

from __future__ import annotations

import time

import pytest
from rich.cells import cell_len
from textual import events
from textual.containers import ScrollableContainer
from textual.widgets import Button, Collapsible, Input, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_settings_model_change import pick_mode_opener, settle
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CONNECTION_DISCLOSURE_ID,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    MODEL_DISCOVER_BUTTON_ID,
    MODEL_DISCOVER_BUTTON_LABEL,
    ConsoleSettingsModal,
    _settings_screen_region,
)

# TASK-33006.24: Tests/UI/conftest.py imports tldw_chatbook.app per test, which
# fails closed with RecoveryRequired("raw_source_selection_changed") under the
# per-test sandbox, so this file ran only when another file imported the app
# first. No test here writes config: the shared bootstrap profile is safe.
pytestmark = pytest.mark.bootstrap_profile

GEOMETRY_SIZES = ((80, 24), (100, 30), (160, 40), (200, 50))
#: The two full-screen sizes the app is used at (ThinkPad, MacBook).
FULL_SCREEN_SIZES = ((211, 44), (235, 52))
#: The view-tab row as painted: each tab is its label and one cell either side.
MODEL_VIEW_TAB_ROW = " ● Model and generation  Context and memory "
CONTEXT_VIEW_TAB_ROW = " Model and generation  ● Context and memory "
#: TASK-33006.1 (spec §6, mock (b)): $ds-chat-settings-width by $ds-size-22,
#: each capped at $ds-percent-95 of the viewport. The 196-column wide tier is
#: gone. The width is 176, not mock (b)'s 150, since TASK-33007 capture
#: finding 15 (150 cut every help line over 95 cells).
MODAL_WIDTH, MODAL_HEIGHT, VIEWPORT_CAP_PERCENT = 176, 22, 95


def expected_modal_size(viewport: tuple[int, int]) -> tuple[int, int]:
    """The modal's width and height at ``viewport`` under the production CSS."""
    width, height = viewport
    return (
        min(MODAL_WIDTH, int(width * VIEWPORT_CAP_PERCENT / 100)),
        min(MODAL_HEIGHT, int(height * VIEWPORT_CAP_PERCENT / 100)),
    )
LONG_CONNECTION_STATUS = (
    "Testing connection to a configured private endpoint by listing models; "
    "this does not generate text or verify model generation."
)


class GeometryHarness(ConsolidatedCSSApp):
    """Isolated app that loads the same consolidated CSS as production."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
            }
        }


def build_geometry_modal(
    app: GeometryHarness, *, ready: bool, **kwargs
) -> ConsoleSettingsModal:
    """Build either a blocked first-use or configured power-user draft.

    The ready draft holds a Temperature the saved chain lacks, so the footer
    offers every action: Save as model default shows only while a save would
    change the defaults (Phase 6 final review I5).

    Args:
        app: The harness whose configuration the modal reads.
        ready: Whether the draft names a model and is ready to send.
        **kwargs: Further ``ConsoleSettingsModal`` arguments.
    """
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a" if ready else None,
            base_url="http://127.0.0.1:9099",
            temperature=0.9 if ready else None,
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a", "model-b"] if ready else []},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        **kwargs,
    )


def painted_tab_row(app: GeometryHarness, modal: ConsoleSettingsModal) -> str:
    """The view-tab row as the compositor paints it.

    A tab's label string holds its '● ' mark even while the tab is too narrow
    to paint it, so the label alone cannot show a crop.
    """
    row = modal.query_one("#console-settings-view-tabs").region.y
    return app.screen._compositor.render_strips()[row].text


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_view_tabs_paint_their_whole_labels_after_each_view_switch(
    size: tuple[int, int],
) -> None:
    """TASK-33003.11: a view tab is laid out again when its label changes.

    ``Button.label`` repaints without a layout, so a tab could keep the width
    of its old label and crop its '● ' mark. A click or Enter on a tab painted
    whole labels before the fix too. Alt+M from the Context view, which shows
    the Model view with no tab pressed, painted the Model tab without its mark.

    Compose yields both tabs unmarked; the view switch on_mount runs marks
    the open view's tab before the first paint, whichever view that is.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
    """
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True, focus_context=True)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await settle(pilot, app)
        assert CONTEXT_VIEW_TAB_ROW in painted_tab_row(app, modal)

    app = GeometryHarness()
    modal = build_geometry_modal(
        app, ready=True, model_picker=pick_mode_opener(app, app.app_config)
    )

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await settle(pilot, app)
        assert MODEL_VIEW_TAB_ROW in painted_tab_row(app, modal)

        await pilot.click("#console-settings-view-context")
        await settle(pilot, app)
        assert CONTEXT_VIEW_TAB_ROW in painted_tab_row(app, modal)

        await pilot.click("#console-settings-view-model")
        await settle(pilot, app)
        assert MODEL_VIEW_TAB_ROW in painted_tab_row(app, modal)

        await pilot.click("#console-settings-view-context")
        await settle(pilot, app)
        await pilot.press("alt+m")  # Change, from a Context view field
        await settle(pilot, app)
        assert isinstance(app.screen, ConsoleModelPopover)
        await pilot.press("escape")  # no pick: the Model view shows, Change focused
        await settle(pilot, app)
        assert app.screen is modal
        assert app.focused is modal.query_one("#console-settings-model-change")
        assert MODEL_VIEW_TAB_ROW in painted_tab_row(app, modal)


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_a_selected_view_tab_paints_its_mark_while_it_holds_focus(
    size: tuple[int, int],
) -> None:
    """TASK-33006.23: a selected tab that holds focus paints its '●' mark.

    Shift+Tab reaches the selected tab with no view switch. Enter on the
    other tab switches by keyboard and moves focus into the view, and
    Shift+Tab then returns to the tab that is now selected.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
    """
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=size) as pilot:

        async def focus_tab(view: str) -> None:
            """Shift+Tab until the ``view`` tab holds focus."""
            tab = modal.query_one(f"#console-settings-view-{view}")
            for _ in range(40):  # more than one lap of the focus order
                if app.focused is tab:
                    break
                await pilot.press("shift+tab")
            await settle(pilot, app)
            assert app.focused is tab

        await app.push_screen(modal)
        await settle(pilot, app)

        await focus_tab("model")  # the open view's tab: no switch
        assert MODEL_VIEW_TAB_ROW in painted_tab_row(app, modal)

        await pilot.press("tab", "enter")  # the Context tab
        await settle(pilot, app)
        assert app.focused is not modal.query_one("#console-settings-view-context")
        await focus_tab("context")
        assert CONTEXT_VIEW_TAB_ROW in painted_tab_row(app, modal)

        await pilot.press("shift+tab", "enter")  # the Model tab
        await settle(pilot, app)
        await focus_tab("model")
        assert MODEL_VIEW_TAB_ROW in painted_tab_row(app, modal)


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
        assert not container.has_class("-conversation-settings-wide")
        assert (container.region.width, container.region.height) == (
            expected_modal_size(size)
        )
        # A ready chat opens Connection closed (TASK-33006.1); open it to
        # measure its actions. A blocked chat already opens it.
        connection_disclosure = modal.query_one(
            f"#{CONNECTION_DISCLOSURE_ID}", Collapsible
        )
        assert connection_disclosure.collapsed is ready
        connection_disclosure.collapsed = False
        await pilot.pause()
        await pilot.pause()
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
        # Cancel shows only with the recovery actions (TASK-33007
        # capture finding 20); the Model view's footer leads with Use saved
        # defaults (rewritten on purpose).
        adopt = modal.query_one("#console-settings-use-saved-defaults", Button)
        save_default = modal.query_one("#console-settings-save-default", Button)
        save = modal.query_one("#console-settings-save", Button)
        assert not modal.query_one("#console-settings-cancel", Button).display
        assert str(connection.label) == MODEL_DISCOVER_BUTTON_LABEL
        assert str(save_default.label) == "Save as model default"
        assert str(save.label) == "Apply to this chat (Ctrl+Enter)"
        # A blocked chat has no model, so there is no model default to save.
        assert save_default.display is ready
        footer_actions = (adopt, save_default, save) if ready else (adopt, save)
        painted = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )
        assert "Apply to this chat (Ctrl+Enter)" in painted
        assert "Use for this conversation" not in painted
        for action in (connection, *footer_actions):
            assert action.region.width >= len(str(action.label))

        connection_row = modal.query_one("#console-settings-connection-actions")
        footer = modal.query_one("#console-settings-actions")
        if compact:
            assert connection_row.layout.name == "vertical"
            assert footer.layout.name == "vertical"
            assert connection.region.height == 1
            assert len({action.region.y for action in footer_actions}) == len(footer_actions)
            assert {
                action.region.width for action in footer_actions
            } == {footer.content_region.width}
        else:
            assert connection_row.layout.name == "horizontal"
            assert footer.layout.name == "horizontal"
            assert len({action.region.y for action in footer_actions}) == 1

        status = modal.query_one("#console-settings-model-discover-status", Static)
        status.display = True
        status.update(LONG_CONNECTION_STATUS)
        await pilot.pause()
        # Enough wrapped lines to hold the text at its measured width.
        assert status.region.height >= -(-len(LONG_CONNECTION_STATUS) // status.region.width)

        readiness_text = str(
            modal.query_one("#console-settings-readiness", Static).renderable
        )
        # TASK-33005.3 (AC#11): the spec §5 words replace "Ready to send".
        if ready:
            assert readiness_text.startswith("Ready · not tested\n")
        else:
            assert readiness_text.startswith("Not ready · ")


@pytest.mark.parametrize(
    "size",
    ((120, 24), (150, 40), (211, 44), (235, 52)),
    ids=["narrow", "below-token", "full-screen", "wide-full-screen"],
)
@pytest.mark.asyncio
async def test_chat_settings_size_is_its_width_token_by_22_capped_to_the_viewport(
    size: tuple[int, int],
) -> None:
    """The modal is 176x22, capped at 95% of the viewport; no wide tier.

    Rewritten on purpose by TASK-33006.1 (AC#13): it pinned the 85%/196
    wide tier, which mock (b)'s fixed frame replaces. Restoring the tier
    (or a percentage width) fails the full-screen cases: at 211 and 235
    columns the frame must stay exactly 176 wide and 22 tall. Rewritten on
    purpose again by TASK-33007 capture finding 15: 150 became 176.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
    """
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        container = modal.query_one("#console-settings-modal")
        assert not container.has_class("-conversation-settings-wide")
        assert (container.region.width, container.region.height) == (
            expected_modal_size(size)
        )
        assert not modal.has_class("-conversation-settings-compact")


@pytest.mark.parametrize(
    ("start_size", "end_size"),
    (((140, 40), (211, 44)), ((235, 52), (120, 40))),
    ids=["grow-to-full-screen", "shrink-below-the-token"],
)
@pytest.mark.asyncio
async def test_chat_settings_size_tracks_live_resize(
    start_size: tuple[int, int],
    end_size: tuple[int, int],
) -> None:
    """An open modal re-derives its capped 176x22 frame on resize.

    Rewritten on purpose by TASK-33006.1 (AC#13): it followed the retired
    wide-tier class across the 150-column boundary. Now the frame must follow
    the viewport cap in both directions with no tier class at all.

    Args:
        start_size: Terminal size (columns, rows) the modal is mounted at.
        end_size: Terminal size (columns, rows) resized to while it stays open.
    """
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=start_size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        container = modal.query_one("#console-settings-modal")
        assert (container.region.width, container.region.height) == (
            expected_modal_size(start_size)
        )

        await pilot.resize_terminal(*end_size)
        await pilot.pause()
        await pilot.pause()

        assert app.size.width == end_size[0]
        assert not container.has_class("-conversation-settings-wide")
        assert (container.region.width, container.region.height) == (
            expected_modal_size(end_size)
        )


@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=["211x44", "235x52"])
@pytest.mark.parametrize(
    ("provider", "model"),
    (("openai", "gpt-5.6-terra"), ("anthropic", "claude-unknown-model")),
    ids=["reasoning-rows", "thinking-rows"],
)
@pytest.mark.asyncio
async def test_every_field_help_line_is_whole_at_full_screen(
    size: tuple[int, int], provider: str, model: str
) -> None:
    """Capture finding 15 (08): at 150 columns the Reasoning rows' help
    ("Support not verified for this model. blank = provider default · How
    much of its reasoning the model summarizes for you.", 119 cells) was cut
    with an ellipsis at both full-screen sizes. Every shown row's help line
    is whole at 211x44 and 235x52, whose support is unknown and value blank.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
        provider: The draft's provider.
        model: A model whose reasoning or thinking support is unknown.
    """
    app = GeometryHarness()
    app.app_config = {
        "api_settings": {provider: {"api_key": "sk-made-up-key-for-a-layout-test"}}
    }
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider=provider, model=model),
        app_config=app.app_config,
        providers_models={provider: [model]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        rows = [
            row
            for row in modal.query(".console-settings-field-row")
            if row.display and row.region.width
        ]
        helps = [line for row in rows for line in row.query(".console-settings-help-line")]
        assert any("not verified" in str(line.content) for line in helps)
        cut = {
            str(line.id): (str(line.content), line.region.width)
            for line in helps
            if cell_len(str(line.content)) > line.region.width
        }
        assert not cut


@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=["211x44", "235x52"])
@pytest.mark.parametrize("view", ["model", "context"])
@pytest.mark.asyncio
async def test_footer_hint_and_buttons_are_set_apart(
    size: tuple[int, int], view: str
) -> None:
    """Capture finding 16 (08): the footer ran together, "Esc close Matches
    saved defaults  Default for new chats (Ctrl+N)  Apply to this chat
    (Ctrl+Enter)": the hint touched the first button and each button the
    next. A blank panel cell now sits before every footer button.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
        view: The view whose footer is measured.
    """
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._show_settings_view(view)
        await pilot.pause()
        await pilot.pause()

        footer = modal.query_one("#console-settings-actions")
        items = sorted(
            (
                widget
                for widget in (
                    modal.query_one("#console-settings-esc-hint"),
                    *footer.query(Button),
                )
                if widget.display and widget.region.width
            ),
            key=lambda widget: widget.region.x,
        )
        # The Context view's footer is "Esc close" and Apply since capture
        # finding 20 (round 2) took its Cancel away.
        assert len(items) >= (3 if view == "model" else 2)
        assert len({widget.region.y for widget in items}) == 1
        touching = [
            (str(left.id), str(right.id))
            for left, right in zip(items, items[1:])
            if right.region.x <= left.region.right
        ]
        assert not touching


#: The Console's navigation bar: the active tab's box top, the tab row and
#: the rule under it.
NAV_BAR_ROWS = 3


@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=["211x44", "235x52"])
@pytest.mark.asyncio
async def test_context_view_leaves_the_nav_bar_visible(size: tuple[int, int]) -> None:
    """Capture finding 21 (round 2, 08b): the Context view grew to 42 of 44
    rows and covered the navigation bar. Centred, it now keeps the bar's
    three rows clear above it (and as many below), and stays taller than the
    Model view so it still scrolls less."""
    app = GeometryHarness()
    modal = build_geometry_modal(app, ready=True)

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._show_settings_view("context")
        await pilot.pause()
        await pilot.pause()

        frame = modal.query_one("#console-settings-modal").region
        assert frame.y >= NAV_BAR_ROWS, frame
        assert frame.bottom <= size[1] - NAV_BAR_ROWS, frame
        assert frame.height > MODAL_HEIGHT, frame


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
        reveal_records = []
        original_reveal = modal._reveal_focused_control

        def record_reveal(widget, generation):
            reveal_records.append((widget, modal.focused))
            original_reveal(widget, generation)

        monkeypatch.setattr(modal, "_reveal_focused_control", record_reveal)

        async def wait_for_reveal(widget) -> None:
            """Poll, bounded by wall clock, until the modal reveals ``widget``."""
            deadline = time.monotonic() + 10
            while (widget, widget) not in reveal_records:
                assert time.monotonic() < deadline, f"no reveal for #{widget.id}"
                await pilot.pause(0.01)

        # TASK-33006.4: Change and Temperature replace the deleted pickers
        # as two distinct restorable targets.
        change = modal.query_one("#console-settings-model-change")
        change.focus()
        # TASK-33006.24: focusing arms Change's own reveal (a 0.06 s timer).
        # Scrolling away before it ran raced the pause below: whenever the
        # pause outlasted the timer, the reveal scrolled Change back into view
        # and the anchor was 0 (5 of 20 runs). Wait for the reveal itself; a
        # fixed pause only narrows the race.
        await wait_for_reveal(change)
        body.scroll_to(y=7, animate=False)
        await pilot.pause()
        snapshot = modal.capture_suspended_draft()
        assert snapshot.scroll_anchor > 0

        modal.query_one("#console-settings-temperature", Input).focus()
        await pilot.pause()
        modal._restore_suspended_scroll_and_focus(snapshot)
        assert modal._pending_suspended_scroll_restore is not None

        new_target = modal.query_one("#console-settings-base-url", Input)
        new_target.focus()
        await wait_for_reveal(new_target)
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
        stale_target = modal.query_one("#console-settings-model-change")
        reveal_generation = modal._focus_reveal_generation
        modal.on_descendant_focus(events.DescendantFocus(stale_target))
        assert modal._focus_reveal_generation == reveal_generation
        modal._queue_focus_reveal_after_layout(
            stale_target,
            reveal_generation,
        )
        await pilot.pause()
        assert len(reveal_records) == reveal_count
