"""Settings model-catalog group layout regression tests (task-1367).

The ``Vertical#settings-model-catalog-group`` (Providers & Models pane) used
to inherit Textual's default ``height: 1fr``, which never resolves inside the
auto-height providers card -- the group rendered as a 2-row empty bordered
box with the "Automatic refresh" header, instant-apply hint, auto-refresh
Checkbox, stale-hours Input and per-provider toggles all clipped.

These tests load the REAL application stylesheet (``tldw_cli_modular.tcss``)
via ``_SettingsCssHarness`` and assert, at 120x35 / 100x30 / 80x24, that the
group sizes to its content and that its header and hint copy actually render
when the detail pane is scroll-stepped.
"""

import pytest

from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import (
    _SettingsCssHarness,
    _scrolled_region_rows,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (100, 30), (80, 24)])
async def test_model_catalog_group_renders_all_controls(size):
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)

        group = screen.query_one("#settings-model-catalog-group")
        assert str(group.styles.height) == "auto", (
            f"group must size to content inside the auto-height card, got "
            f"{group.styles.height!r}"
        )
        assert group.outer_size.height > 6, (
            f"group clipped at {size}: outer_size={group.outer_size}"
        )
        for selector in (
            "#settings-model-catalog-instant-hint",
            "#settings-model-catalog-auto-refresh",
            "#settings-model-catalog-stale-hours",
            "#settings-mc-auto-openai",
            "#settings-mc-auto-anthropic",
        ):
            control = screen.query_one(selector)
            assert control.outer_size.height >= 1, f"{selector} clipped at {size}"

        # Scroll-step the detail pane and confirm the header and hint copy
        # actually render on screen (border glyphs between wrapped words are
        # normalized away before phrase matching).
        # Upstream task-1716 split the detail pane into a pinned header and
        # the scrollable -body; scroll the body.
        detail = screen.query_one("#settings-detail-pane-body")
        rows = await _scrolled_region_rows(pilot, detail)
        text = " ".join(
            "".join(ch if ch.isalnum() else " " for ch in " ".join(rows)).split()
        )
        assert "Automatic refresh" in text, "section header never rendered"
        assert "applies immediately" in text, "instant-apply hint copy never rendered"
