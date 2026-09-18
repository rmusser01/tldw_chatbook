"""TASK-32593: compact Settings keeps field content readable with its sidebar.

Uses the production stylesheet and actual Providers/Network composition. Removing
compact row adaptation must fail the 20-cell and painted-policy assertions.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Select, Static

from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness, _settle

THEMES = ("textual-dark", "textual-light")
PROVIDER_FIELDS = (
    ("#settings-model-value", "Model", "audit-model-with-context"),
    ("#settings-provider-endpoint-value", "Endpoint", "https://example.invalid/v1"),
)
NETWORK_FIELD = (
    "#settings-network-ca-path",
    "CA bundle path",
    "/missing/audit-corp-ca.pem",
)
POLICIES = (
    ("verify", "Verify certificates (default)"),
    ("off", "Disable verification"),
    ("custom-ca", "Custom CA bundle"),
)


def _paint(widget) -> str:
    region = widget.region
    strips = widget.screen._compositor.render_strips()
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(len(strips), region.bottom))
    )


async def _reveal(pilot, widget) -> None:
    widget.parent.scroll_visible(animate=False)
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("category", ("providers-models", "network"))
async def test_compact_fields_keep_labels_focus_and_values_across_resize(
    theme, category
):
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 45)) as pilot:
        host.theme = theme
        await _settle(pilot)
        screen = _active_destination_screen(host)
        screen._select_category(category, restore_focus=True)
        await pilot.pause()
        fields = PROVIDER_FIELDS if category == "providers-models" else (NETWORK_FIELD,)
        if category == "network":
            screen.query_one("#settings-network-ssl-mode", Select).value = "custom-ca"
        for selector, _label, value in fields:
            field = screen.query_one(selector, Input)
            field.focus()
            await pilot.pause()
            await pilot.press("home", "shift+end", *value)
            await pilot.pause()
            assert field.value == value

        for size in ((120, 45), (80, 24), (120, 45)):
            focused_before_resize = host.focused
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert host.focused is focused_before_resize
            sidebar = screen.query_one("#settings-category-pane")
            assert sidebar.display and sidebar.region.width >= 24
            for selector, label, value in fields:
                field = screen.query_one(selector, Input)
                row = field.parent
                field_label = row.query_one(".settings-input-label", Static)
                host.set_focus(None)
                await _reveal(pilot, field)
                rest_edge = field.styles.border_left
                field.focus()
                await pilot.pause()
                await pilot.press("end")
                await pilot.pause()
                await _reveal(pilot, field)
                assert field.has_focus
                assert field.value == value
                assert field.content_region.height == 1
                assert field.styles.border_left != rest_edge
                assert field.styles.border_left[0] == "thick"
                assert field.region.right <= row.content_region.right
                assert field.content_region.width >= 20, (selector, size, field.region)
                assert label in _paint(row), (label, size, _paint(row))
                assert field.value[-8:] in _paint(field), (
                    selector,
                    size,
                    _paint(field),
                )
                if size[0] == 120:
                    assert field.region.y == field_label.region.y
                    assert field_label.region.width == 24
                else:
                    assert field.region.y > field_label.region.y

        if category == "network":
            host.set_focus(None)
            await pilot.pause()
            screen.action_settings_save_category()
            await pilot.pause()
            assert any(
                "CA bundle path invalid: Path does not exist" in str(n.message)
                for n in host._notifications
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
async def test_compact_network_closed_select_paints_every_full_policy(theme):
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        host.theme = theme
        await _settle(pilot)
        screen = _active_destination_screen(host)
        screen._select_category("network", restore_focus=True)
        await pilot.pause()
        select = screen.query_one("#settings-network-ssl-mode", Select)
        for value, label in POLICIES:
            select.value = value
            select.focus()
            await pilot.pause()
            await _reveal(pilot, select)
            assert select.has_focus
            assert label in _paint(select), (theme, label, _paint(select))
            assert "Certificate verification" in _paint(select.parent)
            assert select.region.right <= select.parent.content_region.right
