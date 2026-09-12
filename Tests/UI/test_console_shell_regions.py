"""Painted-geometry contract for the Console shell's production regions.

The expectation table pins the approved shell policy at each size, including
regions that are legitimately hidden in compact mode. It mounts the production
hierarchy with the shipped stylesheet so simplified widget geometry cannot
stand in for application behavior.

Three sizes are pinned:

- 160x45 and 235x52 both sit ABOVE the shell's own ``-console-compact``
  height threshold (``CONSOLE_COMPACT_HEIGHT_ROWS = 35``, see
  ``chat_screen.py``) and above the right-rail width-forced-collapse
  threshold (``CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150``, see
  ``Chat/console_rail_state.py``). Against a freshly built harness (no
  stored rail preferences, no active run, right rail closed by default),
  every region is hidden/hittable identically at both -- an observed fact
  about this id list and these two sizes, not an assumption.
- 120x30 crosses BOTH thresholds (30 < 35 rows, 120 < 150 columns) AND
  lands inside a third, narrower, deliberate rule:
  ``_should_open_standard_width_inspector`` auto-opens the Inspector rail
  whenever available columns fall in ``118..128`` and the Inspector already
  has a "Run recipe" row plus a companion row (Blocked impact / Next
  action / Sources / Tools / Approvals / Artifacts) -- the fresh harness's
  setup-blocked state supplies exactly that, so at 120x30 the Inspector is
  OPEN by default where it is closed at the other two sizes. Inspector-first
  compact priority then hides the Context rail, exposes its reveal handle,
  and grants the Inspector compact-override authority. The Transcript's
  minimum-width waiver keeps all displayed workspace-grid children inside
  both the grid and viewport. The Inspector reveal handle stays hidden, and
  ``#console-run-inspector`` is newly present in the DOM with ``display=True``
  -- but see the "clipped" state below before assuming that means visible.

``#console-mode-bar`` is hidden unconditionally at any size: it is a legacy
compatibility seam retained only for older selectors and is composed via
``_hidden_static`` regardless of geometry (see the comment directly above
its ``compose()`` yield in chat_screen.py).

A third expectation state, ``"clipped"``, exists alongside "hittable" and
"hidden": at 120x30 narrow-layout overflow can leave a mounted region with
positive virtual geometry whose reported center is either outside the screen
or painted by an unrelated widget. The auto-opened Inspector's
scrollable body (``#console-inspector-rail-body``) has a real viewport only
3 rows tall against ~28 rows of virtual content. Textual still reports a
non-empty, ``display=True`` ``.region`` for ``#console-run-inspector`` (a
child scrolled below that viewport), but its *unclipped* center is either
outside the screen or resolves to an unrelated painted widget. So this region
is neither cleanly "hidden" (``display`` is True) nor cleanly "hittable" (its
own reported center never resolves to itself or a descendant) -- pinning it
as a fabricated "hittable" or "hidden" would misrepresent what the shell
does today. "clipped" asserts both halves of that observed reality.
"""

from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from textual.errors import NoWidget
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_rail_state import build_console_rail_preference_key
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle

# (id, expected_at_160x45, expected_at_235x52, expected_at_120x30) where
# expected is "hittable" | "hidden" | "clipped" -- pinned against the
# production hierarchy and shipped stylesheet.
# TASK-23197 changed the 120x30 column. The Inspector used to open ITSELF
# between 118 and 128 columns, which tripped priority resolution and
# force-collapsed Context to a stub -- so a one-column resize from 117 to 118
# swapped which sidebar the user had. The automatic open is now declined when
# it would evict a visible Context rail, so at 120x30 the default is Context
# open with the Inspector collapsed to its handle: the same two-pane shape as
# 117 columns, which is what makes the boundary stop being a cliff.
_REGIONS: list[tuple[str, str, str, str]] = [
    ("#console-shell", "hittable", "hittable", "hittable"),
    ("#console-left-rail", "hittable", "hittable", "hittable"),
    ("#console-left-rail-body", "hittable", "hittable", "hittable"),
    ("#console-main-column", "hittable", "hittable", "hittable"),
    ("#console-context-rail-handle", "hidden", "hidden", "hidden"),
    ("#console-inspector-rail-handle", "hittable", "hittable", "hittable"),
    ("#console-control-bar", "hittable", "hittable", "hittable"),
    ("#console-mode-bar", "hidden", "hidden", "hidden"),
    ("#console-native-composer", "hittable", "hittable", "hittable"),
    ("#console-run-inspector", "hidden", "hidden", "hidden"),
]

_EXPECTED_BY_SIZE = {
    (160, 45): 0,
    (235, 52): 1,
    (120, 30): 2,
}


class ProductionCSSConsoleHarness(ConsoleHarness):
    """Console harness with the exact production stylesheet stack and order."""

    CSS_PATH = TldwCli.CSS_PATH


@asynccontextmanager
async def make_console_pilot(*, size, app=None, hide_setup_overlay=True):
    """Mount a fresh Console (ChatScreen) at ``size`` via the production harness.

    Build a fresh ``TldwCli`` with every real I/O seam faked out
    (``_build_test_app``), push its real ``ChatScreen`` onto a
    ``ConsoleHarness`` carrying the exact production CSS stack, and wait for
    the composer -- the same "the shell is up" signal used elsewhere -- before
    handing control to the caller.
    """
    app = app or _build_test_app()
    host = ProductionCSSConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        # The setup-blocked state supplies the Inspector rows that trigger its
        # production auto-open. Hide only its covering overlay afterward so
        # hit-tests can inspect the underlying shell geometry.
        if hide_setup_overlay:
            console.query_one("#console-setup-modal").display = False
            await pilot.pause()
        yield pilot


def _assert_workspace_state_is_contained(
    screen,
    *,
    expected_displayed: set[str],
    default_context_only: bool,
) -> None:
    """Assert displayed workspace panes have real, contained compositor regions."""
    grid = screen.query_one("#console-workspace-grid")
    children = tuple(grid.children)

    assert {child.id for child in children} == {
        "console-context-rail-handle",
        "console-left-rail",
        "console-main-column",
        "console-right-rail",
        "console-inspector-rail-handle",
    }
    displayed = tuple(child for child in children if child.display)
    assert {child.id for child in displayed} == expected_displayed

    for child in displayed:
        assert child.region.width > 0 and child.region.height > 0, (
            f"{child.id} has no painted geometry: {child.region}"
        )
        assert grid.content_region.contains_region(child.region), (
            f"{child.id} escapes workspace content: "
            f"child={child.region}, grid={grid.content_region}"
        )
        assert screen.region.contains_region(child.region), (
            f"{child.id} escapes the {screen.region.width}x"
            f"{screen.region.height} viewport: "
            f"child={child.region}, screen={screen.region}"
        )
        point = (
            child.region.x + child.region.width // 2,
            child.region.y + (child.region.height - 1) // 2,
        )
        hit = screen.get_widget_at(*point)[0]
        # What this guards is OCCLUSION: no sibling workspace pane may paint
        # over another's centre. Asserting DOM ancestry instead was a proxy
        # that also runs at FIRST PAINT, where a freshly mounted descendant
        # can already be painting while its `ancestors` list is still empty
        # and its region not yet settled. TASK-23199 surfaced that by
        # removing the Sessions section, which moved the Conversations search
        # box onto the rail's centre point. Naming the real property makes
        # the check independent of mount ordering without weakening it.
        occluder = next(
            (
                sibling
                for sibling in displayed
                if sibling is not child
                and (hit is sibling or sibling in hit.ancestors)
            ),
            None,
        )
        assert occluder is None, (
            f"{child.id} centre {point} is painted over by sibling pane "
            f"{occluder.id if occluder else None}: hit={hit!r}"
        )

    main = screen.query_one("#console-main-column")
    if default_context_only:
        assert main.region.width >= 55

    transcript = screen.query_one("#console-native-transcript")
    assert transcript.content_region.width >= 40
    transcript_point = (
        transcript.content_region.x + transcript.content_region.width // 2,
        transcript.content_region.y + (transcript.content_region.height - 1) // 2,
    )
    transcript_hit = screen.get_widget_at(*transcript_point)[0]
    assert transcript_hit is transcript or transcript in transcript_hit.ancestors, (
        f"native transcript is not painted at {transcript_point}: "
        f"hit={transcript_hit!r}"
    )


def _transcript_anchor_state(transcript: ConsoleTranscript) -> tuple[bool, bool]:
    """Return Textual's raw anchor flags for continuity assertions."""
    return (
        bool(transcript.is_anchored),
        bool(transcript._anchor_released),
    )


def _transcript_is_following_tail(transcript: ConsoleTranscript) -> bool:
    """Return the transcript's semantic tail-follow state."""
    return transcript._is_following_tail()


async def _seed_resize_transcript(screen, pilot):
    """Create a selected, detached real transcript at a stable reading offset."""
    store = screen._ensure_console_chat_store()
    selected_message_id = ""
    for index in range(24):
        message = store.append_message(
            store.active_session_id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content="\n".join(
                f"resize message {index} line {line}" for line in range(3)
            ),
        )
        selected_message_id = message.id
    await screen._sync_native_console_chat_ui()
    transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
    for _ in range(40):
        if transcript.max_scroll_y > 0:
            break
        await pilot.pause(0.05)
    assert transcript.max_scroll_y > 0

    transcript.select_message(selected_message_id)
    transcript.release_anchor()
    transcript.scroll_to(y=2, animate=False)
    await pilot.pause()
    assert transcript.selected_message_id == selected_message_id
    assert _transcript_is_following_tail(transcript) is False
    assert transcript.scroll_y == 2
    return (
        selected_message_id,
        _transcript_is_following_tail(transcript),
        _transcript_anchor_state(transcript),
        transcript.scroll_y,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stacked", "left_width", "right_width", "left_label", "right_label"),
    [
        (False, 13, 11, "Context ▸", "◂ Inspect"),
        (True, 3, 3, "C\no\nn\nt\ne\nx\nt", "I\nn\ns\np\ne\nc\nt\no\nr"),
    ],
)
async def test_fresh_console_composes_saved_rail_label_style(
    stacked, left_width, right_width, left_label, right_label
):
    """A fresh Console reads the saved style for both collapsed handles."""
    app = _build_test_app()
    app.app_config.setdefault("console", {})["stack_collapsed_rail_labels"] = stacked
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        left = console.query_one("#console-context-rail-handle", ConsoleRailHandle)
        right = console.query_one("#console-inspector-rail-handle", ConsoleRailHandle)
        left_button = console.query_one("#console-context-rail-open", Button)
        right_button = console.query_one("#console-inspector-rail-open", Button)
        right_badge = console.query_one("#console-inspector-rail-badge", Static)

        assert left.styles.width.value == left_width
        assert right.styles.width.value == right_width
        assert left._display_label() == left_label
        assert right._display_label() == right_label
        assert left_button.tooltip == "Open Context rail"
        assert right_button.tooltip == "Open Inspector rail"
        assert str(right_badge.renderable) == right._display_badge()
        assert right_badge.tooltip == right.badge

        console.query_one("#console-context-rail-collapse", Button).press()
        await pilot.pause()
        assert left.display is True
        assert console.query_one("#console-left-rail").display is False
        console.query_one("#console-context-rail-open", Button).press()
        await pilot.pause()
        assert left.display is False
        assert console.query_one("#console-left-rail").display is True

        assert right.display is True
        console.query_one("#console-inspector-rail-open", Button).press()
        await pilot.pause()
        assert right.display is False
        assert console.query_one("#console-right-rail").display is True
        console.query_one("#console-inspector-rail-collapse", Button).press()
        await pilot.pause()
        assert right.display is True
        assert console.query_one("#console-right-rail").display is False


@pytest.mark.asyncio
async def test_console_pilot_uses_the_exact_production_css_stack() -> None:
    """Geometry evidence loads every production stylesheet in production order."""
    async with make_console_pilot(size=(120, 30)) as pilot:
        assert pilot.app.CSS_PATH == pilot.app.app_instance.CSS_PATH


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52), (120, 30)])
@pytest.mark.parametrize(
    "region_id,expect_160x45,expect_235x52,expect_120x30", _REGIONS
)
async def test_region_geometry_is_stable(
    region_id, expect_160x45, expect_235x52, expect_120x30, size
):
    expected = (expect_160x45, expect_235x52, expect_120x30)[_EXPECTED_BY_SIZE[size]]
    async with make_console_pilot(size=size) as pilot:
        nodes = pilot.app.screen.query(region_id)
        if expected == "hidden":
            # Tightened per review: a query returning zero nodes is NOT the
            # same fact as a mounted-but-display:none node, and Tasks 3/4
            # extract exactly the blocks holding these ids -- a silently
            # dropped/renamed id must fail this baseline, not sail through
            # the same branch as "legitimately hidden".
            assert len(nodes) == 1 and not nodes[0].display
            return
        if expected == "clipped":
            # See the module docstring's "clipped" paragraph: mounted and
            # displayed with a purported region, but scrolled below its
            # container's real viewport so nothing is actually painted at
            # its own reported center -- pin both halves of that fact.
            assert len(nodes) == 1
            node = nodes[0]
            assert node.display and node.region.width > 0
            center = node.region.center
            if not pilot.app.screen.region.contains(*center):
                return
            try:
                hit = pilot.app.screen.get_widget_at(*center)[0]
            except NoWidget:
                return
            assert not (
                hit is node or node in hit.ancestors or hit in node.walk_children()
            )
            return
        node = nodes[0]
        assert node.display and node.region.width > 0
        hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
        assert hit is node or node in hit.ancestors or hit in node.walk_children()


@pytest.mark.asyncio
async def test_compact_workspace_grid_children_are_contained() -> None:
    """The real 120x30 workspace keeps every displayed pane horizontally in bounds."""
    async with make_console_pilot(size=(120, 30)) as pilot:
        screen = pilot.app.screen
        grid = screen.query_one("#console-workspace-grid")
        children = tuple(grid.children)

        assert {child.id for child in children} == {
            "console-context-rail-handle",
            "console-left-rail",
            "console-main-column",
            "console-right-rail",
            "console-inspector-rail-handle",
        }
        displayed = tuple(child for child in children if child.display)
        assert len(displayed) == 3

        for child in displayed:
            child_id = child.id
            assert child.region.width > 0 and child.region.height > 0, (
                f"{child_id} has no painted geometry: child={child.region}"
            )
            assert grid.content_region.x <= child.region.x, (
                f"{child_id} starts before workspace grid content: "
                f"child={child.region}, grid={grid.content_region}"
            )
            assert child.region.right <= grid.content_region.right, (
                f"{child_id} ends after workspace grid content: "
                f"child={child.region}, grid={grid.content_region}"
            )
            assert screen.region.x <= child.region.x, (
                f"{child_id} starts before the 120x30 viewport: "
                f"child={child.region}, screen={screen.region}"
            )
            assert child.region.right <= screen.region.right, (
                f"{child_id} ends after the 120x30 viewport: "
                f"child={child.region}, screen={screen.region}"
            )

        # TASK-23197: Context stays, the Inspector collapses to its handle.
        assert {child.id for child in displayed} == {
            "console-left-rail",
            "console-main-column",
            "console-inspector-rail-handle",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "expected_displayed"),
    [
        pytest.param(
            (120, 30),
            {
                "console-left-rail",
                "console-main-column",
                "console-inspector-rail-handle",
            },
            id="120x30-default-context-visible",
        ),
        pytest.param(
            (80, 24),
            {"console-main-column"},
            id="80x24-default-single-pane",
        ),
    ],
)
async def test_bounded_rail_default_shell_matrix_is_compositor_contained(
    size: tuple[int, int],
    expected_displayed: set[str],
) -> None:
    """Default compact states keep hidden rails out of the hit-test plane."""

    async with make_console_pilot(size=size) as pilot:
        screen = pilot.app.screen
        _assert_workspace_state_is_contained(
            screen,
            expected_displayed=expected_displayed,
            default_context_only=False,
        )
        left = screen.query_one("#console-left-rail")
        right = screen.query_one("#console-right-rail")
        if size == (80, 24):
            assert left.display is False
            assert right.display is False
            assert screen.query_one("#console-context-rail-handle").display is False
            assert screen.query_one("#console-inspector-rail-handle").display is False
        else:
            # TASK-23197: at 120x30 the default now keeps Context and
            # collapses the Inspector, instead of the Inspector opening
            # itself and evicting Context.
            assert left.display is True
            assert right.display is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "stored_left_open",
        "stored_right_open",
        "effective_left_open",
        "effective_right_open",
        "expected_displayed",
    ),
    [
        pytest.param(
            True,
            False,
            True,
            False,
            {
                "console-left-rail",
                "console-main-column",
                "console-inspector-rail-handle",
            },
            id="context-open-inspector-closed",
        ),
        pytest.param(
            False,
            False,
            False,
            False,
            {
                "console-context-rail-handle",
                "console-main-column",
                "console-inspector-rail-handle",
            },
            id="both-closed",
        ),
        pytest.param(
            False,
            True,
            False,
            True,
            {
                "console-context-rail-handle",
                "console-main-column",
                "console-right-rail",
            },
            id="context-closed-inspector-open",
        ),
        pytest.param(
            True,
            True,
            False,
            True,
            {
                "console-context-rail-handle",
                "console-main-column",
                "console-right-rail",
            },
            id="inspector-wins-open-conflict",
        ),
    ],
)
async def test_exact_100_workspace_state_matrix_is_contained(
    monkeypatch,
    stored_left_open,
    stored_right_open,
    effective_left_open,
    effective_right_open,
    expected_displayed,
) -> None:
    """Compose-time and settled rail states both have usable 100x30 geometry."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    stored_preferences = {
        "left_open": stored_left_open,
        "right_open": stored_right_open,
    }
    shared_key = build_console_rail_preference_key(layout_scope="global")
    app.app_config.setdefault("console", {})["rail_state"] = {
        shared_key.value: stored_preferences
    }
    save_spy = Mock()
    pre_sync_observations = []
    queued_first_sync_states = []
    original_sync = ChatScreen._sync_console_rail_visibility_if_changed

    def sync_after_first_paint_assertions(screen, rail_state):
        """Inspect the first compositor geometry, then delegate normal sync."""
        if pre_sync_observations:
            return original_sync(screen, rail_state)

        queued_first_sync_states.append(rail_state)
        if len(queued_first_sync_states) == 1:

            def assert_first_paint_then_delegate_sync():
                """Run after layout but before any queued visibility sync."""
                setup_modal = screen.query_one("#console-setup-modal")
                assert setup_modal.display is False
                _assert_workspace_state_is_contained(
                    screen,
                    expected_displayed=expected_displayed,
                    default_context_only=stored_left_open and not stored_right_open,
                )
                pre_sync_observations.append(True)
                for queued_state in queued_first_sync_states:
                    original_sync(screen, queued_state)
                queued_first_sync_states.clear()

            screen.call_after_refresh(assert_first_paint_then_delegate_sync)
        return None

    monkeypatch.setattr(ChatScreen, "_save_console_rail_preferences", save_spy)
    monkeypatch.setattr(
        ChatScreen,
        "_sync_console_rail_visibility_if_changed",
        sync_after_first_paint_assertions,
    )

    async with make_console_pilot(
        size=(100, 30),
        app=app,
        hide_setup_overlay=False,
    ) as pilot:
        screen = pilot.app.screen
        assert pre_sync_observations == [True]
        assert screen.query_one("#console-setup-modal").display is False

        # `_last_console_rail_state` is a settled-state oracle only; compose-time
        # geometry was asserted by the first-sync wrapper before delegation.
        rail_state = screen._last_console_rail_state
        assert rail_state is not None
        assert rail_state.preferred_left_open is stored_left_open
        assert rail_state.preferred_right_open is stored_right_open
        assert rail_state.left_open is effective_left_open
        assert rail_state.right_open is effective_right_open

        _assert_workspace_state_is_contained(
            screen,
            expected_displayed=expected_displayed,
            default_context_only=stored_left_open and not stored_right_open,
        )
        save_spy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_width", "destination_width"),
    [
        pytest.param(101, 100, id="101-to-100"),
        pytest.param(100, 101, id="100-to-101"),
        pytest.param(99, 100, id="99-to-100"),
        pytest.param(100, 99, id="100-to-99"),
    ],
)
async def test_exact_100_live_resize_preserves_workspace_and_interaction_state(
    monkeypatch,
    start_width,
    destination_width,
) -> None:
    """All adjacent exact-100 crossings preserve reading and focus continuity."""
    app = _build_test_app()
    _configure_native_ready_console(app)

    async with make_console_pilot(size=(start_width, 30), app=app) as pilot:
        screen = pilot.app.screen
        (
            selected_message_id,
            was_following_tail,
            anchor_state,
            reading_y,
        ) = await _seed_resize_transcript(screen, pilot)
        initial_rail_state = screen._last_console_rail_state
        assert initial_rail_state is not None
        assert initial_rail_state.preferred_left_open is True
        assert initial_rail_state.preferred_right_open is False

        starts_open = start_width >= 100
        initial_focus_selector = (
            "#console-context-rail-collapse"
            if starts_open
            else "#console-context-rail-open"
        )
        initial_focus = screen.query_one(initial_focus_selector, Button)
        initial_focus.focus()
        await pilot.pause()
        assert pilot.app.focused is initial_focus

        save_spy = Mock()
        monkeypatch.setattr(screen, "_save_console_rail_preferences", save_spy)
        await pilot.resize_terminal(destination_width, 30)
        await pilot.pause(0.2)
        await pilot.pause()

        destination_open = destination_width >= 100
        expected_displayed = {
            "console-left-rail" if destination_open else "console-context-rail-handle",
            "console-main-column",
            "console-inspector-rail-handle",
        }
        _assert_workspace_state_is_contained(
            screen,
            expected_displayed=expected_displayed,
            default_context_only=destination_open,
        )

        rail_state = screen._last_console_rail_state
        assert rail_state is not None
        assert rail_state.preferred_left_open is initial_rail_state.preferred_left_open
        assert (
            rail_state.preferred_right_open is initial_rail_state.preferred_right_open
        )
        assert rail_state.left_open is destination_open
        assert rail_state.right_open is False
        assert rail_state.left_forced_collapsed is (destination_width < 100)
        assert rail_state.left_compact_override is (destination_width == 100)
        assert rail_state.compact_override is (destination_width == 100)

        expected_focus_selector = (
            "#console-context-rail-collapse"
            if destination_open
            else "#console-context-rail-open"
        )
        expected_focus = screen.query_one(expected_focus_selector, Button)
        assert expected_focus.display is True
        assert pilot.app.focused is expected_focus

        current_transcript = screen.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        assert current_transcript.max_scroll_y >= reading_y
        selected_row = current_transcript.query_one(
            f"#console-message-{selected_message_id}"
        )
        assert selected_row.is_mounted
        assert current_transcript.selected_message_id == selected_message_id
        assert _transcript_is_following_tail(current_transcript) is was_following_tail
        assert _transcript_anchor_state(current_transcript) == anchor_state
        assert current_transcript.scroll_y == min(
            reading_y, current_transcript.max_scroll_y
        )
        save_spy.assert_not_called()
