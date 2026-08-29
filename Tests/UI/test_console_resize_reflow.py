"""TASK-361: live-resize reflow convergence + stale-overlay dismissal.

The review saw a live browser-viewport resize (900x620 -> 700x480) leave the
rail full-width with the transcript/inspector gone and a nav tooltip stuck over
the header, whereas a cold start at the same size was fine. On a native resize
the pane reflow converges to the cold-start layout (locked here); the resize now
also dismisses any visible tooltip so a mounted overlay can't survive the repaint.
"""

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Static, Tooltip

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
)
from tldw_chatbook.UI.Console_Modules.left_rail import (
    CONTEXT_SECTION_DESCRIPTORS,
    ConsoleLeftRail,
)
from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
from tldw_chatbook.Widgets.Console.console_bounded_section import ConsoleBoundedSection
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.app import TldwCli

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app

_PANES = (
    "#console-left-rail",
    "#console-transcript-surface",
    "#console-native-composer",
)


def _ready_console_host() -> ConsoleHarness:
    """Build a Console whose setup modal cannot steal resize-test focus."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


class _ProductionResizeConsoleHarness(ConsoleHarness):
    """Real ChatScreen hierarchy with the exact application stylesheet stack."""

    CSS_PATH = TldwCli.CSS_PATH


def _ready_production_console_host() -> _ProductionResizeConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return _ProductionResizeConsoleHarness(app)


def _resize_popover() -> ConsoleModelPopover:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    origin = ConsoleSettingsOrigin("session-a", None, 0)
    draft = ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=origin.session_id,
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )

    return ConsoleModelPopover(
        origin=origin,
        app_config={"api_settings": {"llama_cpp": {}}},
        initial_draft=draft,
        providers_models={"llama_cpp": ["model-a"]},
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=lambda state, **_kwargs: state,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: (
            ConsoleSettingsReadiness("Ready", "Ready.", True)
        ),
    )


def _resize_full_settings() -> ConsoleSettingsModal:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    return ConsoleSettingsModal(
        settings=settings,
        app_config={
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings": {"llama_cpp": {}},
        },
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        default_readiness_resolver=lambda _provider, _model: (
            ConsoleSettingsReadiness("Ready", "Ready.", True)
        ),
    )


def _assert_non_overlapping_regions(buttons: list[Button]) -> None:
    assert all(button.region.width > 0 and button.region.height > 0 for button in buttons)
    for index, button in enumerate(buttons):
        assert all(
            not button.region.overlaps(other.region)
            for other in buttons[index + 1 :]
        )


def _pane_layout(console) -> dict:
    """Return the display state of the required Console panes plus compact.

    Queries every pane directly (no swallowing): a missing selector raises and
    fails the test loudly rather than degrading to ``None`` and passing.
    """
    layout = {
        selector: bool(console.query_one(selector).display) for selector in _PANES
    }
    layout["compact"] = console.query_one("#console-shell").has_class(
        "-console-compact"
    )
    return layout


async def _wait_for_context_condition(
    pilot,
    condition,
    *,
    attempts: int = 20,
) -> None:
    """Wait through bounded refresh turns until one Context condition is stable."""

    stable_passes = 0
    for _ in range(attempts):
        await pilot.pause()
        if condition():
            stable_passes += 1
            if stable_passes == 2:
                return
        else:
            stable_passes = 0
    pytest.fail("Context condition did not stabilize within the refresh bound")


def _context_allocation_idle(rail: ConsoleLeftRail) -> bool:
    return not rail._allocation_reconcile_scheduled and all(
        not section._reconcile_scheduled
        for section in rail.query(ConsoleBoundedSection)
    )


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.asyncio
async def test_popover_actions_remain_reachable_and_ordered_at_narrow_width(
    width: int,
) -> None:
    app = ConsolidatedCSSApp()
    modal = _resize_popover()

    async with app.run_test(size=(width, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        panel = modal.query_one("#console-model-popover")
        main = list(modal.query("#console-popover-main-actions Button"))
        assert [str(button.label) for button in main] == [
            "Cancel",
            "Full settings…",
            "Defaults…",
            "Apply to this chat",
        ]
        assert panel.region.x >= 0
        assert panel.region.right <= width
        assert panel.region.bottom <= 24
        assert all(panel.region.contains_region(button.region) for button in main)
        assert all(button.can_focus and not button.disabled for button in main)
        _assert_non_overlapping_regions(main)
        main[0].focus()
        await pilot.pause()
        main_focus_order: list[str] = []
        for _ in main:
            focused = app.focused
            main_focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert main_focus_order == [
            "console-popover-cancel",
            "console-popover-full-settings",
            "console-popover-defaults",
            "console-popover-apply",
        ]

        await pilot.click("#console-popover-defaults")
        await pilot.pause()
        await pilot.pause()
        defaults = list(modal.query("#console-popover-default-actions Button"))
        assert [str(button.label) for button in defaults] == [
            "Save as model default",
            "Make default for new chats",
            "Back",
        ]
        assert all(panel.region.contains_region(button.region) for button in defaults)
        assert all(button.can_focus and not button.disabled for button in defaults)
        _assert_non_overlapping_regions(defaults)
        defaults_focus_order: list[str] = []
        for _ in defaults:
            focused = app.focused
            defaults_focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert defaults_focus_order == [
            "console-popover-save-model-default",
            "console-popover-make-new-chat-default",
            "console-popover-defaults-back",
        ]
        defaults[0].focus()
        await pilot.pause()
        assert app.focused is defaults[0]


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.asyncio
async def test_full_settings_actions_remain_mouse_reachable_at_narrow_width(
    width: int,
) -> None:
    """Every full-Settings action stays painted inside the production modal."""

    app = ConsolidatedCSSApp()
    modal = _resize_full_settings()
    async with app.run_test(size=(width, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        panel = modal.query_one("#console-settings-modal")
        actions = list(modal.query("#console-settings-actions Button"))
        assert [str(button.label) for button in actions] == [
            "Cancel",
            "Save as model default",
            "Make default for new chats",
            "Apply to this chat",
        ]
        assert panel.region.x >= 0
        assert panel.region.right <= width
        assert panel.region.bottom <= 24
        assert all(button.display for button in actions)
        assert all(button.can_focus and not button.disabled for button in actions)
        assert all(panel.region.contains_region(button.region) for button in actions), (
            panel.region,
            [(button.id, button.region) for button in actions],
        )
        _assert_non_overlapping_regions(actions)

        actions[0].focus()
        await pilot.pause()
        focus_order: list[str] = []
        for _ in actions:
            focused = app.focused
            focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert focus_order == [
            "console-settings-cancel",
            "console-settings-save-default",
            "console-settings-make-default",
            "console-settings-save",
        ]


@pytest.mark.asyncio
async def test_resize_priority_hands_context_focus_to_reveal_button() -> None:
    """Crossing 117-to-118 hides focused Context and focuses its handle."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()
        assert pilot.app.focused is collapse

        await pilot.resize_terminal(118, 40)
        await pilot.pause(0.2)

        reveal = console.query_one("#console-context-rail-open")
        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-right-rail").display is True
        assert reveal.display is True
        assert pilot.app.focused is reveal


@pytest.mark.asyncio
async def test_consecutive_resize_keeps_focus_on_reopened_context_rail() -> None:
    """A focused Context handle hands focus back when Context reopens."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()

        await pilot.resize_terminal(118, 40)
        await pilot.pause(0.2)
        reveal = console.query_one("#console-context-rail-open")
        assert pilot.app.focused is reveal

        await pilot.resize_terminal(129, 40)
        await pilot.pause(0.2)

        collapse = console.query_one("#console-context-rail-collapse")
        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is False
        assert collapse.display is True
        assert pilot.app.focused is collapse


@pytest.mark.asyncio
async def test_resize_priority_hands_inspector_focus_to_reveal_button() -> None:
    """Crossing 128-to-129 hides focused Inspector and focuses its handle."""
    host = _ready_console_host()

    async with host.run_test(size=(128, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-inspector-rail-collapse")
        collapse.focus()
        await pilot.pause()
        assert pilot.app.focused is collapse

        await pilot.resize_terminal(129, 40)
        await pilot.pause(0.2)

        reveal = console.query_one("#console-inspector-rail-open")
        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert reveal.display is True
        assert pilot.app.focused is reveal


@pytest.mark.asyncio
async def test_resize_event_width_drives_priority_and_focus(monkeypatch) -> None:
    """The Resize width wins over a stale screen-width lookup."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()
        monkeypatch.setattr(console, "_console_rail_available_columns", lambda: 160)

        await pilot.resize_terminal(120, 40)
        await pilot.pause(0.2)

        rail_state = console._last_console_rail_state
        reveal = console.query_one("#console-context-rail-open")
        assert rail_state is not None
        assert rail_state.left_open is False
        assert rail_state.right_open is True
        assert rail_state.right_compact_override is True
        assert rail_state.compact_override is True
        assert reveal.display is True
        assert pilot.app.focused is reveal


@pytest.mark.asyncio
async def test_console_live_resize_converges_to_cold_start_layout() -> None:
    """A live resize converges to the cold-start layout at that size.

    TASK-361 AC#1: after resizing down, the panes are all present and the header
    is compacted -- the same layout a cold start produces -- not the review's
    rail-full-width / panes-gone divergence.

    TASK-2154.1 (LY-08) changed WHAT the cold-start layout is at 90 cols: the
    left rail now force-collapses below 100 columns (rendering override)
    instead of overflowing the grid. The convergence contract itself --
    ``live == cold`` -- is unchanged.
    """
    cold_host = _ready_console_host()
    async with cold_host.run_test(size=(90, 30)) as pilot:
        cold_console = cold_host.screen_stack[-1]
        await pilot.pause()
        await pilot.pause()
        cold = _pane_layout(cold_console)

    live_host = _ready_console_host()
    async with live_host.run_test(size=(160, 48)) as pilot:
        live_console = live_host.screen_stack[-1]
        await pilot.pause()
        await pilot.resize_terminal(90, 30)
        await pilot.pause()
        await pilot.pause()
        live = _pane_layout(live_console)

    assert live == cold
    # The transcript and composer stay present and the header is compacted at
    # 30 rows; at 90 cols the left rail is force-collapsed by the TASK-2154.1
    # narrow-width rule, not eaten by / eating the grid.
    assert cold["#console-left-rail"] is False
    assert cold["#console-transcript-surface"] is True
    assert cold["#console-native-composer"] is True
    assert cold["compact"] is True


@pytest.mark.asyncio
async def test_console_resize_dismisses_stale_tooltip() -> None:
    """A live resize dismisses a visible tooltip overlay.

    TASK-361 AC#2: the review saw a nav tooltip stick over the header across
    reflows. With a tooltip shown, a resize must hide the real overlay widget so
    it cannot survive the repaint.
    """
    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await pilot.pause()

        # _clear_tooltip() (Textual Screen) hides the screen's Tooltip child;
        # ensure one exists and is shown, exactly as a hover would leave it.
        try:
            tooltip = console.get_child_by_type(Tooltip)
        except NoMatches:
            tooltip = Tooltip(id="textual-tooltip")
            await console.mount(tooltip)
            await pilot.pause()
        tooltip.display = True
        assert tooltip.display is True

        await pilot.resize_terminal(120, 40)
        await pilot.pause()

        assert tooltip.display is False


@pytest.mark.asyncio
async def test_model_summary_sync_invalidates_mounted_context_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production Model mutation seam refreshes bounded demand afterward."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        readiness = {"label": "Ready"}
        monkeypatch.setattr(
            console,
            "_build_console_settings_summary_state",
            lambda: ConsoleSettingsSummaryState(
                provider_row="Provider: test",
                model_row="Model: test",
                context_row="Context: 0",
                sampling_row="T 0.7 · max_tokens 100",
                identity_row="Identity: character",
                readiness_label=readiness["label"],
            ),
        )
        console._sync_console_settings_summary()
        rail.activate_section("model")
        await _wait_for_context_condition(
            pilot,
            lambda: (
                rail._active_section_id == "model"
                and not rail.query_one("#console-model-section-recovery").display
                and _context_allocation_idle(rail)
            ),
        )
        before_demand = model.desired_content_lines
        before_allocation = model.allocation
        recovery = rail.query_one("#console-model-section-recovery")
        reconcile_runs = 0
        original_reconcile = rail._run_allocation_reconcile

        def reconcile_spy() -> None:
            nonlocal reconcile_runs
            reconcile_runs += 1
            original_reconcile()

        monkeypatch.setattr(rail, "_run_allocation_reconcile", reconcile_spy)
        readiness["label"] = "Provider recovery required"
        console._sync_console_settings_summary()
        await _wait_for_context_condition(
            pilot,
            lambda: recovery.display and _context_allocation_idle(rail),
        )

        assert recovery.display is True
        assert model.desired_content_lines >= before_demand
        assert model.allocation == before_allocation
        assert reconcile_runs >= 1
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_height_resize_requests_one_coalesced_context_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public height-adaptation lifecycle invalidates after compact mutation."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        for descriptor in CONTEXT_SECTION_DESCRIPTORS:
            rail.apply_section_open(descriptor.section_id, True)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        sections = list(rail.query(ConsoleBoundedSection))
        before_allocations = tuple(section.allocation for section in sections)
        before_viewport_height = rail.query_one(
            "#console-left-rail-body"
        ).content_region.height

        helper_calls = 0
        reconcile_runs = 0
        original_helper = console._request_console_context_allocation_reconcile
        original_reconcile = rail._run_allocation_reconcile

        def helper_spy() -> None:
            nonlocal helper_calls
            helper_calls += 1
            original_helper()

        def reconcile_spy() -> None:
            nonlocal reconcile_runs
            reconcile_runs += 1
            original_reconcile()

        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            helper_spy,
        )
        monkeypatch.setattr(rail, "_run_allocation_reconcile", reconcile_spy)
        await pilot.resize_terminal(160, 30)
        await _wait_for_context_condition(
            pilot,
            lambda: (
                console.query_one("#console-shell").has_class("-console-compact")
                and _context_allocation_idle(rail)
            ),
        )
        after_allocations = tuple(section.allocation for section in sections)
        outer = rail.query_one("#console-left-rail-body")
        cue = rail.query_one("#console-left-rail-outer-hint", Static)
        assert console.query_one("#console-shell").has_class("-console-compact")
        assert helper_calls >= 1
        assert reconcile_runs == 1
        assert outer.content_region.height < before_viewport_height
        assert (
            before_allocations
            == after_allocations
            == tuple(None for _section in sections)
        )
        assert [section.max_content_lines for section in sections] == [
            descriptor.max_content_lines for descriptor in CONTEXT_SECTION_DESCRIPTORS
        ]
        assert all(
            sum(
                child.virtual_region_with_margin.height
                for child in section.children
                if child not in {section.viewport, section.hint} and child.display
            )
            + section.viewport.content_region.height
            == min(section.desired_content_lines, section.max_content_lines)
            for section in sections
        )
        assert str(outer.styles.overflow_y) == "auto"
        assert outer.max_scroll_y > 0
        assert cue.display is True
        last_header = rail.query_one("#console-rail-section-header-character")
        outer.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert last_header.region.overlaps(outer.content_region)
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_production_bounded_rail_resize_reconciles_geometry_and_focus() -> None:
    """Resize, recompose, and shrink preserve honest local geometry and focus."""

    host = _ready_production_console_host()
    async with host.run_test(size=(160, 52)) as pilot:
        screen = host.screen_stack[-1]
        if not screen.query_one("#console-right-rail").display:
            assert await pilot.click("#console-inspector-rail-open")
        inspector = screen.query_one("#console-right-rail", ConsoleInspectorRail)
        sources = inspector.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        target = Button("source action", id="production-resize-source-action")
        content = Static("\n".join(f"resize source {row}" for row in range(29)))
        await sources.viewport.remove_children()
        await sources.viewport.mount(content, target)
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 30
                and sources.viewport.content_region.height == 20
                and sources.viewport.max_scroll_y == 10
                and sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )

        target.focus()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                pilot.app.focused is target
                and sources.viewport.scroll_y == sources.viewport.max_scroll_y
            ),
        )
        hit = screen.get_widget_at(target.region.x + 1, target.region.y)[0]
        assert hit is target or target in hit.ancestors

        await pilot.resize_terminal(160, 45)
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.viewport.content_region.height == 20
                and sources.viewport.scroll_y <= sources.viewport.max_scroll_y
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert pilot.app.focused is target
        assert sources.region.contains_region(sources.viewport.region)
        assert sources.region.contains_region(sources.hint.region)

        original_section = sources
        content.update("\n".join(f"recomposed source {row}" for row in range(24)))
        await sources.recompose()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 25
                and sources.viewport.content_region.height == 20
                and sources.viewport.max_scroll_y == 5
                and sources.viewport.scroll_y == 5
                and sources.hint.display
                and not sources._reconcile_scheduled
            ),
        )
        assert (
            inspector.query_one(
                "#console-bounded-section-sources", ConsoleBoundedSection
            )
            is original_section
        )

        await content.remove()
        replacement = Button(
            "replacement source action", id="production-resize-source-replacement"
        )
        await sources.viewport.mount(
            Static("\n".join(f"shrunk source {row}" for row in range(9))),
            replacement,
        )
        await target.remove()
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 10
                and sources.viewport.content_region.height == 10
                and sources.viewport.scroll_y == 0
                and not sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert pilot.app.focused is replacement


@pytest.mark.asyncio
async def test_public_close_active_falls_back_and_rail_reopen_keeps_local_state() -> (
    None
):
    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        for _ in range(5):
            await pilot.pause()

        session_toggle = rail.query_one("#console-rail-section-toggle-session", Button)
        session_toggle.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(session_toggle)
        for _ in range(4):
            await pilot.pause()
        assert rail._active_section_id == "workspace"

        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        model_body = rail.query_one("#console-rail-section-body-model")
        overflow = Static("\n".join(f"line {index}" for index in range(30)))
        await model_body.mount(overflow)
        rail.activate_section("model")
        for _ in range(6):
            await pilot.pause()
        model.viewport.scroll_to(y=3, animate=False, immediate=True)
        await pilot.pause()
        retained_offset = model.viewport.scroll_y
        assert retained_offset > 0

        assert await pilot.click("#console-context-rail-collapse")
        for _ in range(4):
            await pilot.pause()
        assert rail.display is False
        assert await pilot.click("#console-context-rail-open")
        for _ in range(6):
            await pilot.pause()

        assert rail.display is True
        assert rail._active_section_id == "model"
        assert model.viewport.scroll_y == retained_offset


@pytest.mark.asyncio
async def test_all_named_context_mutation_seams_request_the_mounted_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every production Context mutation seam delegates to the rail helper."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        requests: list[str] = []
        current_seam = ""

        def request_spy() -> None:
            requests.append(current_seam)

        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            request_spy,
        )

        def assert_requested(label: str, mutation) -> None:
            nonlocal current_seam
            current_seam = label
            before = len(requests)
            mutation()
            assert requests[before:] == [label]

        console._console_rail_system_line_last = ("stale", False)
        monkeypatch.setattr(
            console,
            "_console_rail_system_line_state",
            lambda: ("System: changed", False),
        )
        assert_requested("session settings", console._sync_console_rail_system_line)

        monkeypatch.setattr(console, "_sync_console_rail_system_line", lambda: None)
        monkeypatch.setattr(console, "_sync_console_agent_section", lambda: None)
        assert_requested("model rows", console._sync_console_settings_summary)

        monkeypatch.undo()
        requests.clear()
        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            request_spy,
        )
        console._console_agent_section_last = object()
        assert_requested(
            "agent status steps actions steering fleet and pinned summary",
            console._sync_console_agent_section,
        )
        assert_requested(
            "workspace conversations and details",
            console._sync_console_workspace_context,
        )

        current_seam = "character remount and reaction"
        before = len(requests)
        await console._render_character_avatar_into_section(
            spec=None,
            name="Changed character",
            manual_label="happy",
            is_current=lambda: True,
        )
        assert requests[before:] == [current_seam]

        rail_state = console._current_console_rail_state()
        assert_requested(
            "rail collapse and reopen",
            lambda: console._sync_console_rail_visibility(rail_state),
        )

        monkeypatch.setattr(console, "_set_console_rail_preference", lambda **_: None)
        assert_requested(
            "section toggles and open state",
            lambda: console._toggle_console_rail_section(
                "details",
                next_open=not rail_state.details_open,
            ),
        )

        # Let the alias sync queued by the workspace mutation finish before
        # exercising the alias-mount seam directly.
        for _ in range(3):
            await pilot.pause()
        aliases = list(console.query("#console-new-workspace-conversation"))
        if aliases and isinstance(aliases[0], Button):
            await aliases[0].remove()
        current_seam = "conversation alias mount"
        before = len(requests)
        await console._sync_console_legacy_workspace_context_aliases()
        assert requests[before:] == [current_seam]
