"""Mounted Console persistent rail first-start contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat import console_chat_store as console_chat_store_module
from tldw_chatbook.Chat.console_chat_models import ConsoleWorkspaceContext
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS,
    build_console_rail_preference_key,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TabState
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


def _assert_selector_hidden_or_absent(screen, selector: str) -> None:
    matches = list(screen.query(selector))
    assert not matches or all(not _is_displayed(widget) for widget in matches)


def _static_text(widget) -> str:
    renderable = getattr(widget, "renderable", "")
    return getattr(renderable, "plain", str(renderable))


def _button_text(widget) -> str:
    label = getattr(widget, "label", "")
    return str(label) if label is not None else ""


def _css_block(css: str, selector: str) -> str:
    start = css.index(selector)
    end = css.index("}", start)
    return css[start:end]


def _assert_handle_visible_text_fits(handle) -> None:
    handle_width = handle.region.width
    visible_chunks = [
        _button_text(widget)
        for widget in handle.query(Button)
        if _is_displayed(widget) and _button_text(widget)
    ]
    visible_chunks.extend(
        _static_text(widget)
        for widget in handle.query("Static")
        if _is_displayed(widget) and _static_text(widget)
    )
    assert visible_chunks
    for text in visible_chunks:
        assert len(text) <= handle_width, (
            f"handle text {text!r} exceeds handle width {handle_width}"
        )


def _assert_handle_aligned_with_workbench_frame(screen, handle_selector: str) -> None:
    handle = screen.query_one(handle_selector)
    main_column = screen.query_one("#console-main-column")
    workspace_grid = screen.query_one("#console-workspace-grid")

    assert handle.region.y == main_column.region.y
    assert handle.region.height == main_column.region.height
    assert handle.region.y >= workspace_grid.region.y
    assert handle.region.y + handle.region.height <= (
        workspace_grid.region.y + workspace_grid.region.height
    )


def _assert_handle_button_contained(handle) -> None:
    button = handle.query_one(Button)
    button_label = _button_text(button)
    handle_right = handle.region.x + handle.region.width
    button_right = button.region.x + button.region.width
    handle_bottom = handle.region.y + handle.region.height
    button_bottom = button.region.y + button.region.height
    usable_x = handle.region.x + 1
    usable_y = handle.region.y + 1
    usable_right = handle_right - 1
    usable_bottom = handle_bottom - 1
    usable_width = max(1, handle.region.width - 2)
    usable_height = max(1, handle.region.height - 2)

    assert button.region.x >= usable_x
    assert button_right <= usable_right
    assert button.region.width <= usable_width
    assert button.region.width >= len(button_label)
    assert button.region.y == usable_y
    assert button_bottom == usable_bottom
    assert button.region.height == usable_height


def _assert_right_handle_lightweight(screen) -> None:
    handle = screen.query_one("#console-inspector-rail-handle")
    main_column = screen.query_one("#console-main-column")
    workspace_grid = screen.query_one("#console-workspace-grid")
    button = handle.query_one(Button)
    button_label = _button_text(button)

    assert handle.region.y == main_column.region.y
    assert handle.region.height == main_column.region.height
    assert handle.has_class("console-frame-quiet")
    assert handle.region.y >= workspace_grid.region.y
    assert handle.region.y + handle.region.height <= (
        workspace_grid.region.y + workspace_grid.region.height
    )
    assert button.region.x >= handle.region.x
    assert button.region.x + button.region.width <= handle.region.x + handle.region.width
    assert button.region.width >= len(button_label)
    assert button.region.width >= len(button_label) + 2
    assert button.region.y >= handle.region.y
    assert button.region.y + button.region.height <= handle.region.y + handle.region.height
    assert 1 <= button.region.height <= 4


async def _wait_for_badge(screen, pilot, selector: str, expected: str) -> str:
    for _ in range(20):
        matches = list(screen.query(selector))
        for widget in matches:
            if _is_displayed(widget):
                text = _static_text(widget)
                if expected in text:
                    return text
        await pilot.pause(0.05)
    visible_text = " ".join(
        _static_text(widget) for widget in screen.query(selector) if _is_displayed(widget)
    )
    raise AssertionError(
        f"{selector} did not include {expected!r}; visible badge text={visible_text!r}"
    )


async def _wait_for_displayed(screen, pilot, selector: str):
    for _ in range(20):
        widget = screen.query_one(selector)
        if _is_displayed(widget):
            return widget
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} was not displayed")


async def _wait_for_hidden(screen, pilot, selector: str) -> None:
    for _ in range(20):
        if not any(_is_displayed(widget) for widget in screen.query(selector)):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} was still displayed")


async def _wait_for_main_column_width_change(
    screen,
    pilot,
    *,
    original_width: int,
    direction: str,
) -> int:
    for _ in range(20):
        width = screen.query_one("#console-main-column").region.width
        if direction == "increase" and width > original_width:
            return width
        if direction == "decrease" and width < original_width:
            return width
        await pilot.pause(0.05)
    raise AssertionError(
        f"main column width did not {direction}; original={original_width}, "
        f"current={screen.query_one('#console-main-column').region.width}"
    )


async def _wait_for_native_console_session(screen, pilot):
    for _ in range(20):
        store = getattr(screen, "_console_chat_store", None)
        if store is not None and store.active_session_id is not None:
            return store.ensure_session(workspace_id=store.workspace_context.active_workspace_id)
        await pilot.pause(0.05)
    raise AssertionError("native Console store did not expose an active session")


async def _wait_for_saved_settings(pilot, saved_settings, expected_count: int) -> None:
    for _ in range(40):
        if len(saved_settings) >= expected_count:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"expected at least {expected_count} saved settings, saw {len(saved_settings)}"
    )


def _rail_prefs(
    *,
    left_open: bool,
    right_open: bool,
    session_open: bool = True,
    context_open: bool = True,
    model_open: bool = True,
    details_open: bool = False,
) -> dict[str, bool]:
    """Full serialized rail-preference shape (left/right rails + four sections).

    The persisted shape now carries the four collapsible left-rail section
    states alongside the left/right rail openness. Section states default to
    the first-run layout (Session/Context/Model open, Details collapsed).
    """
    return {
        "left_open": left_open,
        "right_open": right_open,
        "session_open": session_open,
        "context_open": context_open,
        "model_open": model_open,
        "details_open": details_open,
    }


class _FixedUuid:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


def test_generated_console_stylesheet_includes_rail_rules():
    stylesheets = (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
    )

    for stylesheet in stylesheets:
        css = stylesheet.read_text(encoding="utf-8")
        for selector in (
            "#console-right-rail",
            ".console-rail-handle",
            ".console-rail-header",
            ".console-rail-title",
            ".console-rail-collapse-button",
        ):
            assert selector in css
        assert "content-align: left middle;" in css
        composer_focus = _css_block(css, "#console-native-composer.console-composer-focused")
        assert "border: heavy $ds-action-focus;" not in composer_focus
        assert "border: thick $ds-action-focus;" not in css
        for selector in (
            "#console-left-rail:focus",
            "#console-left-rail-body:focus",
            "#console-workspace-context:focus",
            "#console-workspace-conversations:focus",
        ):
            focus_block = _css_block(css, selector)
            assert "border: solid $ds-action-focus;" not in focus_block
            assert "border: thick $ds-action-focus;" not in focus_block
            assert "outline: none;" in focus_block

        right_handle = _css_block(css, ".console-rail-handle-right")
        right_button = _css_block(css, ".console-rail-handle-button-right")
        assert "width: 11;" in right_handle
        assert "min-width: 11;" in right_handle
        assert "max-width: 11;" in right_handle
        assert "width: 11;" in right_button
        assert "max-width: 11;" in right_button


def test_generated_console_stylesheet_includes_rail_section_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        ".console-rail-section-header",
        ".console-rail-section-title",
        ".console-rail-section-toggle",
        ".console-rail-section-body",
        ".console-model-section-line",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector


@pytest.mark.asyncio
async def test_console_first_start_renders_left_rail_and_right_handle():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert _is_displayed(console.query_one("#console-left-rail"))
        assert _is_displayed(console.query_one("#console-staged-context-tray"))
        assert _is_displayed(console.query_one("#console-workspace-context"))
        _assert_selector_hidden_or_absent(console, "#console-context-rail-handle")
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        _assert_selector_hidden_or_absent(console, "#console-run-inspector-state")
        _assert_selector_hidden_or_absent(
            console,
            "#console-live-work-source-readiness",
        )
        assert _is_displayed(console.query_one("#console-inspector-rail-handle"))
        right_handle = console.query_one("#console-inspector-rail-handle")
        assert right_handle.has_class("console-rail-handle-right")
        assert right_handle.region.width == 11
        _assert_right_handle_lightweight(console)
        open_button = console.query_one("#console-inspector-rail-open", Button)
        assert str(open_button.label) == "Inspector"
        assert open_button.tooltip == "Open Inspector rail"


@pytest.mark.asyncio
async def test_console_first_start_does_not_create_rail_state_config_on_read():
    app = _build_test_app()
    console_config = app.app_config.setdefault("console", {})
    console_config.pop("rail_state", None)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

    assert "rail_state" not in console_config


@pytest.mark.asyncio
async def test_console_first_start_right_handle_is_focusable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        for _ in range(80):
            focused = console.focused
            if getattr(focused, "id", None) == "console-inspector-rail-open":
                assert isinstance(focused, Button)
                return
            await pilot.press("tab")

        focused_id = getattr(console.focused, "id", None)
        raise AssertionError(
            "console-inspector-rail-open was not reachable by tab; "
            f"focused={focused_id!r}"
        )


@pytest.mark.asyncio
async def test_console_context_rail_collapse_hides_left_rail_and_expands_main_column():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")
        first_start_width = console.query_one("#console-main-column").region.width

        await pilot.click("#console-context-rail-collapse")

        await _wait_for_hidden(console, pilot, "#console-left-rail")
        await _wait_for_hidden(console, pilot, "#console-staged-context-tray")
        await _wait_for_hidden(console, pilot, "#console-workspace-context")
        assert _is_displayed(console.query_one("#console-context-rail-handle"))
        _assert_handle_aligned_with_workbench_frame(
            console,
            "#console-context-rail-handle",
        )
        left_handle = console.query_one("#console-context-rail-handle")
        assert left_handle.has_class("console-rail-handle-left")
        _assert_handle_button_contained(left_handle)
        open_button = console.query_one("#console-context-rail-open", Button)
        assert str(open_button.label) == "Context >"
        assert open_button.tooltip == "Open Context rail"
        assert (
            await _wait_for_main_column_width_change(
                console,
                pilot,
                original_width=first_start_width,
                direction="increase",
            )
        ) > first_start_width


@pytest.mark.asyncio
async def test_console_visible_rail_headers_are_left_aligned_and_collapse_buttons_signal_direction():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")

        context_title = console.query_one("#console-context-rail-title", Static)
        context_collapse = console.query_one("#console-context-rail-collapse", Button)
        assert context_title.has_class("console-rail-title")
        assert str(context_collapse.label) == "<"
        assert context_collapse.tooltip == "Collapse Context rail"
        assert context_collapse.region.width >= 3
        assert context_title.region.x < context_collapse.region.x

        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")

        inspector_title = console.query_one("#console-inspector-rail-title", Static)
        inspector_collapse = console.query_one("#console-inspector-rail-collapse", Button)
        assert inspector_title.has_class("console-rail-title")
        assert str(inspector_collapse.label) == ">"
        assert inspector_collapse.tooltip == "Collapse Inspector rail"
        assert inspector_collapse.region.width >= 3
        assert inspector_title.region.x < inspector_collapse.region.x


@pytest.mark.asyncio
async def test_console_main_column_keeps_priority_width_when_both_rails_are_open():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")
        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")

        left_width = console.query_one("#console-left-rail").region.width
        main_width = console.query_one("#console-main-column").region.width
        right_width = console.query_one("#console-right-rail").region.width

        assert main_width >= left_width + right_width + 16


@pytest.mark.asyncio
async def test_console_inspector_rail_open_restores_right_rail_and_narrows_main_column():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")
        right_collapsed_width = console.query_one("#console-main-column").region.width

        await pilot.click("#console-inspector-rail-open")

        await _wait_for_displayed(console, pilot, "#console-right-rail")
        assert _is_displayed(console.query_one("#console-run-inspector-state"))
        assert _is_displayed(console.query_one("#console-live-work-source-readiness"))
        await _wait_for_hidden(console, pilot, "#console-inspector-rail-handle")
        assert (
            await _wait_for_main_column_width_change(
                console,
                pilot,
                original_width=right_collapsed_width,
                direction="decrease",
            )
        ) < right_collapsed_width


@pytest.mark.asyncio
async def test_console_inspector_rail_body_scrolls_below_fixed_header():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")

        right_rail = console.query_one("#console-right-rail")
        header = right_rail.query_one(".console-rail-header")
        body = console.query_one("#console-inspector-rail-body")
        run_inspector = console.query_one("#console-run-inspector")
        source_readiness = console.query_one("#console-live-work-source-readiness")

        assert body.parent is right_rail
        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= right_rail.region.height - header.region.height
        assert run_inspector.parent is body
        assert source_readiness.parent is body
        assert run_inspector.region.x >= body.region.x
        assert source_readiness.region.x >= body.region.x
        assert run_inspector.region.x + run_inspector.region.width <= (
            body.region.x + body.region.width
        )
        assert source_readiness.region.x + source_readiness.region.width <= (
            body.region.x + body.region.width
        )


@pytest.mark.asyncio
async def test_console_rail_state_persists_by_workspace_session_key(monkeypatch):
    app = _build_test_app()
    app.app_config = {"console": {"rail_state": {}}}
    saved_settings = []

    def fake_save_setting(section, key, value):
        saved_settings.append((section, key, value))
        return True

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        fake_save_setting,
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")
        await pilot.click("#console-context-rail-collapse")
        await _wait_for_hidden(console, pilot, "#console-left-rail")
        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")
        await _wait_for_saved_settings(pilot, saved_settings, 2)

    rail_state = app.app_config["console"]["rail_state"]
    expected_key = f"console_rail_state:{DEFAULT_WORKSPACE_ID}:{session.id}"
    assert rail_state[expected_key] == _rail_prefs(left_open=False, right_open=True)
    assert saved_settings[-1] == (
        "console.rail_state",
        expected_key,
        _rail_prefs(left_open=False, right_open=True),
    )

    app.console_rail_session_id = session.id
    remounted_host = ConsoleHarness(app)
    async with remounted_host.run_test(size=(180, 48)) as pilot:
        console = remounted_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        assert _is_displayed(console.query_one("#console-context-rail-handle"))
        assert _is_displayed(console.query_one("#console-right-rail"))
        _assert_selector_hidden_or_absent(console, "#console-inspector-rail-handle")


@pytest.mark.asyncio
async def test_console_rail_preference_save_failure_notifies_without_worker_crash(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config = {"console": {"rail_state": {}}}
    notifications = []

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: False,
        raising=False,
    )
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
        raising=False,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-collapse")

        await pilot.click("#console-context-rail-collapse")

        await _wait_for_hidden(console, pilot, "#console-left-rail")
        for _ in range(40):
            if notifications:
                break
            await pilot.pause(0.05)

    assert notifications == [
        (
            "Console rail preference is saved for this session only.",
            {"severity": "warning"},
        )
    ]


@pytest.mark.asyncio
async def test_console_rail_state_uses_workspace_session_specific_keys(monkeypatch):
    app = _build_test_app()
    app.app_config = {
        "console": {
            "rail_state": {
                "console_rail_state:workspace-a:session-a": {
                    "left_open": False,
                    "right_open": True,
                }
            }
        }
    }

    def workspace_a(self):
        return ConsoleWorkspaceContext(active_workspace_id="workspace-a")

    def workspace_b(self):
        return ConsoleWorkspaceContext(active_workspace_id="workspace-b")

    monkeypatch.setattr(ChatScreen, "_current_console_workspace_context", workspace_a)
    monkeypatch.setattr(
        ChatScreen,
        "_current_console_session_id",
        lambda self: "session-a",
        raising=False,
    )
    host_a = ConsoleHarness(app)
    async with host_a.run_test(size=(180, 48)) as pilot:
        console = host_a.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        assert _is_displayed(console.query_one("#console-right-rail"))

    monkeypatch.setattr(ChatScreen, "_current_console_workspace_context", workspace_b)
    monkeypatch.setattr(
        ChatScreen,
        "_current_console_session_id",
        lambda self: "session-b",
        raising=False,
    )
    host_b = ConsoleHarness(app)
    async with host_b.run_test(size=(180, 48)) as pilot:
        console = host_b.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert _is_displayed(console.query_one("#console-left-rail"))
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert _is_displayed(console.query_one("#console-inspector-rail-handle"))


@pytest.mark.asyncio
async def test_console_session_preference_copies_to_durable_conversation_key(monkeypatch):
    app = _build_test_app()
    app.app_config = {
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:session-1": {
                    "left_open": False,
                    "right_open": True,
                }
            }
        }
    }

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: True,
        raising=False,
    )
    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-1"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        assert session.id == "session-1"
        session.persisted_conversation_id = "conv-1"
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

    rail_state = app.app_config["console"]["rail_state"]
    # The seeded session fallback entry is left untouched (raw two-key shape);
    # the durable conversation copy is written through the full serialized shape.
    assert rail_state[f"console_rail_state:{DEFAULT_WORKSPACE_ID}:session-1"] == {
        "left_open": False,
        "right_open": True,
    }
    assert rail_state[
        f"console_rail_state:{DEFAULT_WORKSPACE_ID}:conv-1"
    ] == _rail_prefs(left_open=False, right_open=True)


@pytest.mark.asyncio
async def test_console_rail_key_prefers_native_session_over_legacy_conversation(monkeypatch):
    app = _build_test_app()
    app.app_config = {"console": {"rail_state": {}}}

    monkeypatch.setattr(
        chat_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: True,
        raising=False,
    )
    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-native"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        console.chat_state.add_tab(
            TabState(
                tab_id="legacy",
                title="Legacy",
                conversation_id="legacy-conv",
                is_active=True,
            )
        )
        console.chat_state.active_tab_id = "legacy"
        session = await _wait_for_native_console_session(console, pilot)
        assert session.persisted_conversation_id is None

        await pilot.click("#console-context-rail-collapse")
        await _wait_for_hidden(console, pilot, "#console-left-rail")

        rail_state = app.app_config["console"]["rail_state"]
        assert rail_state[
            f"console_rail_state:{DEFAULT_WORKSPACE_ID}:session-native"
        ] == _rail_prefs(left_open=False, right_open=False)
        assert f"console_rail_state:{DEFAULT_WORKSPACE_ID}:legacy-conv" not in rail_state

        session.persisted_conversation_id = "native-conv"
        console._sync_console_rail_visibility(console._current_console_rail_state())

    rail_state = app.app_config["console"]["rail_state"]
    assert rail_state[
        f"console_rail_state:{DEFAULT_WORKSPACE_ID}:native-conv"
    ] == _rail_prefs(left_open=False, right_open=False)
    assert f"console_rail_state:{DEFAULT_WORKSPACE_ID}:legacy-conv" not in rail_state


@pytest.mark.asyncio
async def test_console_rail_fallback_migration_read_path_does_not_create_empty_state(
    monkeypatch,
):
    app = _build_test_app()
    console_config = app.app_config.setdefault("console", {})
    console_config.pop("rail_state", None)

    monkeypatch.setattr(
        console_chat_store_module,
        "uuid4",
        lambda: _FixedUuid("session-no-fallback"),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        session = await _wait_for_native_console_session(console, pilot)
        session.persisted_conversation_id = "conv-no-fallback"
        console._current_console_rail_state()

    assert "rail_state" not in console_config


@pytest.mark.asyncio
async def test_console_provider_blocked_badge_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert "setup" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "setup",
        )
        # The shared Workbench recovery banner must stay hidden — the blocking
        # setup modal's own action button carries this guidance instead (Phase 2
        # spec, section 2 revised).
        recovery = console.query_one("#workbench-recovery-callout")
        assert not _is_displayed(recovery)
        settings_button = console.query_one("#console-setup-modal-action", Button)
        assert _is_displayed(settings_button)
        assert settings_button.disabled is False


@pytest.mark.asyncio
async def test_console_provider_ready_with_missing_model_uses_model_recovery_copy(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": ""},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = ""
    monkeypatch.setattr(
        ChatScreen,
        "_effective_console_provider_model",
        lambda self: ("llama_cpp", ""),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        inspector_state = console._build_console_inspector_state(None)
        provider_row = next(
            row for row in inspector_state.rows if row.label == "Provider"
        )
        assert provider_row.status == "blocked"
        assert "Select a model before sending." in provider_row.recovery
        assert "is ready" not in provider_row.recovery


@pytest.mark.asyncio
async def test_console_failed_badge_takes_priority_over_provider_blocked():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    app.console_run_status_override = "failed"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        badge = await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "failed",
        )
        assert "blocked" not in badge
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_pending_approval_badge_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.console_pending_approval_count = 1
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert "1 appr" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "1 appr",
        )
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_tool_badge_when_no_higher_priority_inspector_badge():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.console_tool_count = 2
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert "tools" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "tools",
        )
        _assert_selector_hidden_or_absent(console, "#console-right-rail")


@pytest.mark.asyncio
async def test_console_left_staged_context_badge_does_not_auto_open_context():
    app = _build_test_app()
    app.console_rail_session_id = "badge-session"
    app.app_config = {
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:badge-session": {
                    "left_open": False,
                    "right_open": False,
                }
            }
        }
    }
    app.pending_console_launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="RAG result",
        payload={"query": "badge sync"},
        status="ready",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")

        _assert_selector_hidden_or_absent(console, "#console-left-rail")
        badge = await _wait_for_badge(
            console,
            pilot,
            "#console-context-rail-badge",
            "staged",
        )
        assert badge in {"1 staged", "staged"}


@pytest.mark.asyncio
async def test_console_badge_state_update_after_mount_does_not_auto_open_inspector():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "console": {
            "rail_state": {
                f"console_rail_state:{DEFAULT_WORKSPACE_ID}:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        _assert_selector_hidden_or_absent(console, "#console-inspector-rail-badge")

        app.console_pending_approval_count = 1
        console._sync_console_control_bar()
        await pilot.pause(0.05)

        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert "1 appr" in await _wait_for_badge(
            console,
            pilot,
            "#console-inspector-rail-badge",
            "1 appr",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [100, 120, 140])
async def test_console_compact_width_preserves_main_column_and_forces_right_collapse(
    columns: int,
):
    assert columns < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
    session_id = f"compact-{columns}"
    preference_key = f"console_rail_state:{DEFAULT_WORKSPACE_ID}:{session_id}"
    app = _build_test_app()
    app.console_rail_session_id = session_id
    app.app_config = {
        "console": {
            "rail_state": {
                preference_key: {
                    "left_open": False,
                    "right_open": True,
                }
            }
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(columns, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        await pilot.pause(0.05)

        main_column = console.query_one("#console-main-column")
        workspace_grid = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer")
        right_handle = console.query_one("#console-inspector-rail-handle")

        assert main_column.region.width >= 52
        assert composer.region.width >= workspace_grid.region.width - 2
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        assert _is_displayed(right_handle)
        assert right_handle.region.width == 11
        _assert_right_handle_lightweight(console)
        _assert_handle_visible_text_fits(right_handle)

    assert app.app_config["console"]["rail_state"][preference_key]["right_open"] is True


@pytest.mark.asyncio
async def test_console_desktop_composer_span_ignores_rail_width_changes():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        await pilot.pause(0.05)

        workspace_grid = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer")
        first_start_composer_width = composer.region.width
        first_start_main_width = console.query_one("#console-main-column").region.width

        assert first_start_composer_width >= workspace_grid.region.width - 2

        await pilot.click("#console-context-rail-collapse")
        await _wait_for_hidden(console, pilot, "#console-left-rail")
        left_collapsed_main_width = await _wait_for_main_column_width_change(
            console,
            pilot,
            original_width=first_start_main_width,
            direction="increase",
        )
        assert composer.region.width == first_start_composer_width

        await pilot.click("#console-inspector-rail-open")
        await _wait_for_displayed(console, pilot, "#console-right-rail")
        right_open_main_width = await _wait_for_main_column_width_change(
            console,
            pilot,
            original_width=left_collapsed_main_width,
            direction="decrease",
        )

        assert right_open_main_width < left_collapsed_main_width
        assert composer.region.width == first_start_composer_width


@pytest.mark.asyncio
async def test_console_left_rail_renders_four_sections_with_details_collapsed():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-details")

        for section_id in ("session", "context", "model", "details"):
            assert _is_displayed(
                console.query_one(f"#console-rail-section-header-{section_id}")
            )
        assert _is_displayed(console.query_one("#console-rail-section-body-session"))
        assert _is_displayed(console.query_one("#console-rail-section-body-context"))
        assert _is_displayed(console.query_one("#console-rail-section-body-model"))
        assert not _is_displayed(console.query_one("#console-rail-section-body-details"))
        # Session content: workspace context tray without duplicate heading.
        assert _is_displayed(console.query_one("#console-workspace-context"))
        _assert_selector_hidden_or_absent(console, "#console-workspace-context-title")
        # Details content exists but is hidden.
        assert list(console.query("#console-workspace-details"))
        # Model section content.
        assert _is_displayed(console.query_one("#console-model-section-line1"))
        assert _is_displayed(console.query_one("#console-model-section-configure"))


@pytest.mark.asyncio
async def test_console_details_toggle_expands_and_persists():
    app = _build_test_app()
    # Ready console so the first-run setup modal is not blocking the rail.
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"}
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-details")

        await pilot.click("#console-rail-section-toggle-details")
        await pilot.pause(0.1)
        assert _is_displayed(console.query_one("#console-rail-section-body-details"))
        assert _is_displayed(console.query_one("#console-workspace-authority-label"))

    rail_state_config = app.app_config.get("console", {}).get("rail_state", {})
    assert any(
        isinstance(value, dict) and value.get("details_open") is True
        for value in rail_state_config.values()
    )


@pytest.mark.asyncio
async def test_console_rail_section_sync_applies_stored_scope_preferences():
    """Runtime rail syncs re-apply stored section prefs for the current scope.

    This is the resume path: when the preference scope switches to a
    conversation whose stored prefs differ from the composed defaults (for
    example Details was expanded there before a relaunch), the next rail
    sync must apply those flags to the section bodies and headers.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-details")
        assert not _is_displayed(console.query_one("#console-rail-section-body-details"))
        assert _is_displayed(console.query_one("#console-rail-section-body-model"))

        workspace_context = console._current_console_workspace_context()
        key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            conversation_id=console._current_console_rail_conversation_id(),
            session_id=console._current_console_session_id(),
        )
        rail_config = app.app_config.setdefault("console", {}).setdefault(
            "rail_state", {}
        )
        rail_config[key.value] = {"details_open": True, "model_open": False}

        console._sync_console_rail_visibility_if_changed(
            console._current_console_rail_state()
        )
        await pilot.pause(0.1)

        assert _is_displayed(console.query_one("#console-rail-section-body-details"))
        assert not _is_displayed(console.query_one("#console-rail-section-body-model"))
        details_toggle = console.query_one(
            "#console-rail-section-toggle-details", Button
        )
        model_toggle = console.query_one("#console-rail-section-toggle-model", Button)
        assert _button_text(details_toggle) == "-"
        assert _button_text(model_toggle) == "+"


def test_generated_console_stylesheet_includes_setup_card_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        ".console-setup-step",
        ".console-setup-step-done",
        ".console-setup-step-active",
        ".console-setup-step-pending",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    for stale in (".console-provider-recovery-strip", ".console-provider-blocker"):
        assert stale not in component_css, stale
        assert stale not in generated_css, stale


def test_generated_console_stylesheet_includes_setup_modal_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#console-setup-modal",
        ".console-setup-modal-card",
        ".console-setup-modal-title",
        ".console-setup-modal-action",
        "layer: console-setup-overlay",
        "layers: console-workbench console-setup-overlay",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
