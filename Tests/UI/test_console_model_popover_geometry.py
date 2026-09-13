"""Responsive geometry contracts for the Console Alt+M quick model popover."""

from __future__ import annotations

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

#: Viewport width where the wide tier engages, mirrored from the
#: Conversation settings modal (PR #2670).
POPOVER_WIDE_VIEWPORT_COLUMNS = 150
#: The wide tier's fixed cap, deliberately subordinate to the settings
#: modal's 196 so the quick surface stays visually lighter.
POPOVER_WIDE_MAX_WIDTH = 170
#: Base tier: the fixed compact width the popover has always used.
POPOVER_BASE_WIDTH = 60


class PopoverGeometryHarness(ConsolidatedCSSApp):
    """Isolated app that loads the same consolidated CSS as production."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


def build_geometry_popover() -> ConsoleModelPopover:
    """Build one ready-to-mount quick popover with a minimal draft."""

    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
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

    def rebase(
        state: ConsoleSettingsDraftState, **_kwargs: object
    ) -> ConsoleSettingsDraftState:
        return state

    def commit(_submission: ConsoleSettingsSubmission):  # type: ignore[no-untyped-def]
        raise AssertionError("geometry test never submits")

    return ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session-a", None, 0),
        app_config={
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}},
        },
        initial_draft=draft,
        providers_models={"llama_cpp": ["model-a", "model-b"]},
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=rebase,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: (
            ConsoleSettingsReadiness("Ready", "Ready.", True)
        ),
    )


@pytest.mark.parametrize(
    ("size", "expect_wide"),
    (
        ((120, 40), False),
        ((150, 40), True),
        ((160, 50), True),
        ((200, 50), True),
        ((235, 50), True),
    ),
    ids=["narrow", "at-threshold", "wide", "wide-uncapped-edge", "capped"],
)
@pytest.mark.asyncio
async def test_console_model_popover_wide_tier_engages_at_150_viewport_columns(
    size: tuple[int, int],
    expect_wide: bool,
) -> None:
    """The wide tier keys off the viewport width, not the popover's own width.

    Removing the ``-console-popover-wide`` toggle (or the tier's CSS rule in
    ``ConsoleModelPopover.DEFAULT_CSS``) fails this test: at >= 150 columns
    the container must outgrow the fixed 60-column base width (85% of the
    viewport, capped at 170), while below the threshold the base geometry
    (``width: 60``) is unchanged and the tier class must be absent.

    Args:
        size: Terminal size (columns, rows) the harness app runs at.
        expect_wide: Whether ``#console-model-popover`` must carry
            ``-console-popover-wide``: True at >= 150 viewport columns,
            False below, where the fixed base width must hold.
    """
    app = PopoverGeometryHarness()
    popover = build_geometry_popover()

    async with app.run_test(size=size) as pilot:
        await app.push_screen(popover)
        await pilot.pause()
        await pilot.pause()

        container = popover.query_one("#console-model-popover")
        assert container.has_class("-console-popover-wide") is expect_wide
        if expect_wide:
            assert container.region.width == min(
                POPOVER_WIDE_MAX_WIDTH, int(app.size.width * 85 / 100)
            )
        else:
            assert container.region.width == POPOVER_BASE_WIDTH
        assert 0 < container.region.width <= app.size.width


@pytest.mark.parametrize(
    ("start_size", "end_size", "expect_wide_after_resize"),
    (
        ((140, 40), (200, 50), True),
        ((200, 50), (120, 40), False),
    ),
    ids=["grow-past-threshold", "shrink-below-threshold"],
)
@pytest.mark.asyncio
async def test_console_model_popover_wide_tier_tracks_live_resize_across_threshold(
    start_size: tuple[int, int],
    end_size: tuple[int, int],
    expect_wide_after_resize: bool,
) -> None:
    """An open popover re-syncs its width tier when the terminal is resized.

    Skipping the responsive width sync in ``on_resize`` fails this test: the
    ``-console-popover-wide`` class (and the 85%-width geometry it drives)
    would stay frozen at the mount-time tier instead of following the
    viewport across the 150-column boundary in either direction.

    Args:
        start_size: Terminal size (columns, rows) the popover is mounted at.
        end_size: Terminal size (columns, rows) resized to while the
            popover stays open.
        expect_wide_after_resize: Whether ``#console-model-popover`` must
            carry ``-console-popover-wide`` once the resize settles; the
            container width must follow the matching tier's formula.
    """
    app = PopoverGeometryHarness()
    popover = build_geometry_popover()

    async with app.run_test(size=start_size) as pilot:
        await app.push_screen(popover)
        await pilot.pause()
        await pilot.pause()

        container = popover.query_one("#console-model-popover")
        assert container.has_class("-console-popover-wide") is (
            start_size[0] >= POPOVER_WIDE_VIEWPORT_COLUMNS
        )

        await pilot.resize_terminal(*end_size)
        await pilot.pause()
        await pilot.pause()

        assert app.size.width == end_size[0]
        assert container.has_class("-console-popover-wide") is (
            expect_wide_after_resize
        )
        if expect_wide_after_resize:
            assert container.region.width == min(
                POPOVER_WIDE_MAX_WIDTH, int(end_size[0] * 85 / 100)
            )
        else:
            assert container.region.width == POPOVER_BASE_WIDTH
