"""Fleet-UX expert review F2 (task-1232): parallel-agents discoverability.

At rest, nothing communicated that each Console tab runs its own agent, in
parallel, under a cap: F1 Help covered panes/transcript/composer shortcuts
only (zero mentions of "agent"/"parallel"/"approval"/"workspace"), the
footer omitted Alt+W/Alt+1..9, and the capability only taught itself after
the user had already, accidentally, run two agents at once.

This file covers the three fixes:
  1. F1 Help gains an "Agents" section (tabs=agents, live cap, approval
     flow, marker legend, hotkeys, screen-scope caveat).
  2. A one-time dismissible coach-mark on the first second-tab creation.
  3. Alt+W/Alt+1..9 reachable from Help (the footer is a single-line,
     non-wrapping Static already ~120 chars for its 7 existing hints at
     the narrowest width this suite tests Console against -- see the
     "Agents & fleet" shortcut group's docstring in chat_screen.py for the
     full crowding judgment).
"""

from __future__ import annotations

import re
from html import unescape

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_FLEET_MARKER_LEGEND,
    CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    _console_workbench_agents_notes,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _compositor_text(svg: str) -> str:
    """Rejoin an exported-screenshot SVG's per-segment `<text>` nodes into
    plain text, honoring the real compositor render (scroll-clipped/off-
    screen content simply never becomes a `<text>` node at all).

    Mirrors ``test_workbench_visual_snapshots.py``'s established SVG-
    assertion idiom (`_assert_command_palette_evidence`): style boundaries
    can split one contiguous phrase across adjacent nodes, so a raw
    substring search on the whole SVG is unreliable -- the per-node
    rejoin fixes that while still only ever seeing what was ACTUALLY
    painted on screen, unlike reading a widget's `.renderable` (which
    always holds the full, unclipped text regardless of scroll position).
    """
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


async def _scan_scroll_checkpoints(pilot, scroll: VerticalScroll) -> str:
    """Union of compositor-rendered text across a handful of scroll offsets.

    A full row-by-row scan would work but is wasteful; content only ever
    moves monotonically as `scroll_y` increases, so sampling home, end, and
    a few evenly-spaced offsets between them is enough to prove a line
    somewhere in the middle is reachable by scrolling, without hardcoding
    exactly which offset it lands at (a multi-row wrap anywhere above a
    line shifts its position by a variable amount -- see the "Leaving
    Console cancels" line, which needs a 1-row scroll at 80x24 but none at
    160x40).
    """
    max_y = scroll.max_scroll_y
    checkpoints = sorted(
        {round(max_y * fraction) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)}
    )
    chunks: list[str] = []
    for y in checkpoints:
        scroll.scroll_to(y=y, animate=False, immediate=True)
        await pilot.pause()
        chunks.append(_compositor_text(pilot.app.export_screenshot(simplify=True)))
    # Scroll back to the top so callers see a known, at-rest starting point.
    scroll.scroll_to(y=0, animate=False, immediate=True)
    await pilot.pause()
    return "\n".join(chunks)


async def _wait_for_active_session_change(
    store, pilot, previous_session_id, *, attempts: int = 40
) -> str:
    """Wait for the Console store to activate a session other than
    `previous_session_id` and return its id."""
    for _ in range(attempts):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not change. "
        f"previous={previous_session_id!r}; active={store.active_session_id!r}"
    )


# --- AC#1: F1 Help gains an "Agents" section -------------------------------


def test_console_workbench_agents_notes_covers_required_content():
    """The pure content builder covers every AC#1 bullet."""
    notes = _console_workbench_agents_notes(7)
    joined = " ".join(notes)

    assert "own agent" in joined
    assert "background" in joined
    assert "7 runs in parallel" in joined
    assert "Settings > Console Behavior" in joined
    assert "ask before running" in joined
    assert "◆" in joined
    assert CONSOLE_FLEET_MARKER_LEGEND in notes
    assert "● running" in CONSOLE_FLEET_MARKER_LEGEND
    assert "◆ needs approval" in CONSOLE_FLEET_MARKER_LEGEND
    assert "✓ finished" in CONSOLE_FLEET_MARKER_LEGEND
    assert "✗ failed" in CONSOLE_FLEET_MARKER_LEGEND
    assert "Leaving Console cancels" in joined


def test_console_help_map_includes_agents_and_fleet_hotkeys():
    """Alt+W and Alt+1..9 (task-1232 AC#1/#3) are reachable from F1 Help."""
    groups = dict(CONSOLE_WORKBENCH_SHORTCUT_GROUPS)
    assert "Agents & fleet" in groups
    fleet_shortcuts = dict(groups["Agents & fleet"])
    assert fleet_shortcuts["Alt+W"] == "switch workspace"
    assert fleet_shortcuts["Alt+1..9"] == "jump to tab 1-9"
    assert "Ctrl+T" in fleet_shortcuts
    assert "Ctrl+K" in fleet_shortcuts


def test_console_workbench_agents_notes_pluralizes_a_cap_of_one():
    """Minor (b): cap=1 is a supported floored value (MIN_CONSOLE_MAX_
    PARALLEL_RUNS) -- "1 run", never "1 runs"."""
    notes = _console_workbench_agents_notes(1)
    joined = " ".join(notes)
    assert "1 run in parallel" in joined
    assert "1 runs" not in joined


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (160, 40)])
async def test_console_f1_help_is_scrollable_and_reachable_at_realistic_sizes(
    monkeypatch, size
):
    """Fleet-UX review round 1 (Critical fix): `#workbench-help-panel` used
    to be a plain, unstyled `Vertical` -- Textual's own defaults
    (`height: 1fr`, `overflow: hidden hidden`) HARD-CLIPPED anything past
    the fold with no scrollbar, so the new Agents section and the
    Alt+W/Alt+1..9 hotkeys (AC#3's sole mechanism) were unreachable at
    every realistic terminal size; only the previous test's exact 160x48
    happened to fit all ~44 lines.

    Driven with real SVG compositor captures (`_compositor_text`), not
    `Static.renderable` (which always holds the full unclipped text
    regardless of scroll position -- exactly the blind spot that let the
    clipping bug ship unnoticed) at the reviewer's specified realistic
    sizes: 80x24 and 160x40.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        controller = console._ensure_console_chat_controller()
        # A distinctive, non-default cap proves the help copy is read live
        # (the default is 3 -- see CONSOLE_DEFAULT_MAX_PARALLEL_RUNS).
        monkeypatch.setattr(
            type(controller), "max_parallel_runs", property(lambda self: 7)
        )
        await console.action_show_workbench_help()
        await pilot.pause()

        panel = host.screen_stack[-1]
        assert isinstance(panel, WorkbenchHelpPanel)
        scroll = panel.query_one("#workbench-help-scroll", VerticalScroll)
        close_button = panel.query_one("#workbench-help-close", Button)

        # The fix's core claim: real overflow, with a visible scrollbar, at
        # both realistic sizes -- not just the generous 160x48 the original
        # (pre-review) test happened to use.
        assert scroll.max_scroll_y > 0
        assert scroll.show_vertical_scrollbar is True

        # Close is pinned OUTSIDE the scroll body (a sibling row below it),
        # so it is on-screen and compositor-visible before any scrolling,
        # regardless of how tall the scrollable content is.
        assert (
            close_button.region.y + close_button.region.height
            <= pilot.app.size.height
        )
        at_rest = _compositor_text(pilot.app.export_screenshot(simplify=True))
        assert "Close" in at_rest

        # The Agents section renders right after Actions, ahead of
        # Shortcuts -- reachable AT REST, no scrolling required, at both
        # sizes.
        assert "Agents:" in at_rest
        assert "Each Console tab runs its own agent" in at_rest
        assert "7 runs in parallel" in at_rest
        assert "Settings > Console Behavior" in at_rest
        assert "Built-in tools ask before running" in at_rest
        # The full CONSOLE_FLEET_MARKER_LEGEND line is ~95 chars -- at these
        # widths the Static wraps it across two rows, and a hard line-wrap
        # swallows the space at the break (observed: "...finished · ✗" /
        # "failed — clears..."), so checking the exact single-line constant
        # against wrapped, compositor-rendered text is itself unreliable.
        # Assert its components instead (still proves the legend rendered
        # and is reachable at rest, without overfitting to one wrap point).
        assert "Status markers:" in at_rest
        assert "● running" in at_rest
        assert "◆ needs approval" in at_rest
        assert "✓ finished" in at_rest

        # AC#3's sole mechanism -- Alt+W/Alt+1..9 -- lives in the LAST
        # shortcut group and is genuinely below the fold at rest at these
        # sizes (this is the exact clipping the Critical finding is about).
        assert "Alt+W" not in at_rest
        assert "Alt+1..9" not in at_rest

        # "Leaving Console cancels..." (the last Agents note, right before
        # Shortcuts) sits close enough to the fold that its exact reach
        # varies by a row or two between these two sizes (multi-row wraps
        # above it shift everything below by a variable amount) -- sample a
        # few scroll offsets across the range (compositor-honest each time)
        # rather than asserting one specific position, which is what this
        # line's own wrap-sensitivity argues against.
        reachable_text = await _scan_scroll_checkpoints(pilot, scroll)
        assert "Leaving Console cancels" in reachable_text

        # The reviewer's literal reachability check: after scroll_end, the
        # hotkeys (and the still-pinned Close button) are genuinely on
        # screen.
        scroll.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()
        after_scroll_end = _compositor_text(
            pilot.app.export_screenshot(simplify=True)
        )
        assert "Alt+W" in after_scroll_end
        assert "switch workspace" in after_scroll_end
        assert "Alt+1..9" in after_scroll_end
        assert "jump to tab 1-9" in after_scroll_end
        assert "Close" in after_scroll_end


# --- AC#2: one-time coach-mark on first second-tab creation ----------------


@pytest.mark.asyncio
async def test_fleet_coachmark_shows_on_first_second_tab_creation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is False

        # Real "new tab" action -- mirrors how a user opens a second tab.
        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_a_id)
        await pilot.pause()

        assert banner.display is True
        text = _static_text(console.query_one("#console-fleet-coachmark-text", Static))
        assert "own agent" in text
        assert "3 in parallel" in text  # CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        assert "Settings > Console Behavior" in text


@pytest.mark.asyncio
async def test_fleet_coachmark_dismiss_hides_banner_and_persists_flag_in_memory():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_a_id)
        await pilot.pause()
        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is True

        await pilot.click("#console-fleet-coachmark-dismiss")
        await pilot.pause()

        assert banner.display is False
        onboarding = app.app_config.get("console", {}).get("onboarding", {})
        assert onboarding.get("fleet_coachmark_seen") is True
        assert console._console_fleet_coachmark_seen() is True


@pytest.mark.asyncio
async def test_fleet_coachmark_does_not_reappear_for_a_third_tab_after_dismiss():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        session_b_id = await _wait_for_active_session_change(
            store, pilot, session_a_id
        )
        await pilot.pause()
        await pilot.click("#console-fleet-coachmark-dismiss")
        await pilot.pause()
        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is False

        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_b_id)
        await pilot.pause()

        assert banner.display is False


@pytest.mark.asyncio
async def test_fleet_coachmark_seen_flag_survives_restart_via_real_config_seam(
    monkeypatch, tmp_path
):
    """Drives the REAL config seam (not a mocked in-memory dict): dismiss
    persists to an actual TOML file via `save_setting_to_cli_config`, and a
    fresh "restarted" screen reading that same file never shows the banner.
    """
    from tldw_chatbook import config as config_module

    config_path = tmp_path / "fleet-coachmark-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _build_test_app()
        host = ConsoleHarness(app)

        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            session_a_id = store.active_session_id

            await pilot.click("#console-new-chat-tab")
            await _wait_for_active_session_change(store, pilot, session_a_id)
            await pilot.pause()
            assert console.query_one("#console-fleet-coachmark").display is True

            await pilot.click("#console-fleet-coachmark-dismiss")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()

        # Real disk read (force_reload bypasses the in-process cache) --
        # proves the write actually landed on the tmp config file, not just
        # in the in-memory app_config dict.
        fresh_config = config_module.load_settings(force_reload=True)
        onboarding = fresh_config.get("console", {}).get("onboarding", {})
        assert onboarding.get("fleet_coachmark_seen") is True

        # "Restart": a brand-new app/screen instance whose app_config is
        # seeded from the persisted disk config, exactly like real boot.
        restarted_app = _build_test_app()
        restarted_app.app_config = fresh_config
        restarted_host = ConsoleHarness(restarted_app)
        async with restarted_host.run_test(size=(160, 48)) as pilot:
            console = restarted_host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            session_a_id = store.active_session_id

            await pilot.click("#console-new-chat-tab")
            await _wait_for_active_session_change(store, pilot, session_a_id)
            await pilot.pause()

            assert console.query_one("#console-fleet-coachmark").display is False
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
