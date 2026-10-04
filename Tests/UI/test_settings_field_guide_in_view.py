"""A focused Settings field's whole guide stays inside the inspector view.

TASK-33003.9. The Phase 2 capture of Console Behavior's Temperature
fallback at 211x44 showed the Focused field guide down to "Saved as", with
"Validation" cut off below the Scope Inspector's fold. The route: focus the
field at 235x52, then resize the terminal to 211x44. Focus pins the guide
(``_scroll_impact_pane_to_field_guide``), but only on the focus event; the
resize re-wraps the inspector's prose and kept the old scroll offset, so the
guide slid down past the fold. Providers & Models fails the other way:
211x44 to 235x52 un-wraps the rows above its guide and pushes "Focused
setting" above the top edge.

Every test mounts the real Settings screen under the production bundle and
moves focus with real clicks, Tab presses and "/" searches, then checks the
guide's rows from "Focused setting" to "Validation" against the inspector
body's visible region.
"""

import pytest
from textual.containers import VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Collapsible, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen, _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _open_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.Chat.console_provider_support import MODEL_CONFIG_FIELDS
from tldw_chatbook.UI.Screens.settings_screen import (
    CONSOLE_DEFAULT_FIELD_NAMES,
    PROVIDER_MODEL_PROFILE_FIELD_KEYS,
)

SIZES = ((211, 44), (235, 52))
CONSOLE_FALLBACKS = CONSOLE_DEFAULT_FIELD_NAMES
MODEL_DEFAULTS = {
    f"settings-{key.replace('_', '-')}": name
    for key, name in PROVIDER_MODEL_PROFILE_FIELD_KEYS.items()
}


async def _settle(pilot, hops: int = 3) -> None:
    for _ in range(hops):
        await pilot.pause()


def _focused_id(pilot) -> str:
    return str(getattr(pilot.app.focused, "id", "") or "")


def _assert_guide_in_view(screen, prefix: str, fields: dict[str, str], route: str):
    """Every guide row, "Focused setting" to "Validation", lies in the view."""
    focused = str(getattr(screen.app.focused, "id", "") or "")
    label = MODEL_CONFIG_FIELDS[fields[focused]].label
    texts = _assert_rows_in_view(screen, prefix, route)
    assert texts[0] == f"Focused setting: {label}", (route, texts)
    assert texts[-1].startswith("Validation: "), (route, texts)


def _assert_rows_in_view(screen, prefix: str, route: str) -> list[str]:
    """Every row of the category's guide lies inside the inspector view."""
    focused = str(getattr(screen.app.focused, "id", "") or "")
    body = screen.query_one("#settings-impact-pane-body", VerticalScroll)
    view = body.scrollable_content_region
    rows = []
    while True:
        try:
            rows.append(
                screen.query_one(f"#settings-{prefix}-field-guide-{len(rows)}", Static)
            )
        except QueryError:
            break
    texts = [str(row.renderable) for row in rows]
    assert texts, (route, prefix)
    clipped = [
        (text.split(":")[0], row.region.y, row.region.bottom)
        for row, text in zip(rows, texts)
        if not (view.y <= row.region.y and row.region.bottom <= view.bottom)
    ]
    assert not clipped, (
        f"{route}: {focused}'s guide rows outside the inspector view "
        f"{view.y}..{view.bottom} (scroll_y={body.scroll_y}): {clipped}"
    )
    return texts


def _usable(screen, field_id: str) -> bool:
    try:
        widget = screen.query_one(f"#{field_id}")
    except QueryError:
        return False
    return not widget.disabled and all(
        node.display and not getattr(node, "disabled", False)
        for node in widget.ancestors_with_self
        if node is not screen
    )


async def _click_each(pilot, screen, prefix, fields, route):
    detail = screen.query_one("#settings-detail-pane-body")
    clicked = 0
    for field_id in fields:
        if not _usable(screen, field_id):
            continue
        detail.scroll_to_widget(screen.query_one(f"#{field_id}"), animate=False)
        await _settle(pilot)
        await pilot.click(f"#{field_id}")
        await _settle(pilot)
        if _focused_id(pilot) != field_id:
            # A clicked Select opens its option list; Esc closes it and
            # hands focus back to the Select.
            await pilot.press("escape")
            await _settle(pilot)
        assert _focused_id(pilot) == field_id, (route, field_id, _focused_id(pilot))
        _assert_guide_in_view(screen, prefix, fields, f"{route} click")
        clicked += 1
    return clicked


async def _tab_through(pilot, screen, prefix, fields, route, start_id):
    screen.query_one(f"#{start_id}").focus()
    await _settle(pilot)
    seen = {start_id}
    _assert_guide_in_view(screen, prefix, fields, f"{route} tab")
    for _ in range(3 * len(fields)):
        await pilot.press("tab")
        await _settle(pilot, 2)
        field_id = _focused_id(pilot)
        if field_id not in fields:
            continue
        seen.add(field_id)
        _assert_guide_in_view(screen, prefix, fields, f"{route} tab")
    return seen


async def _resize_both_ways(pilot, screen, prefix, fields, route, size):
    other = SIZES[1] if size == SIZES[0] else SIZES[0]
    for step in (other, size):
        await pilot.resize_terminal(*step)
        await _settle(pilot, 5)
        _assert_guide_in_view(
            screen, prefix, fields, f"{route} resized to {step[0]}x{step[1]}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@private_profile_test
async def test_console_behavior_fallback_guide_stays_in_the_inspector_view(
    request, size
):
    """AC#2/AC#4: click, Tab and search landing, then a resize (the capture)."""
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle_settings_mount_storm(pilot)
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        await _settle(pilot)
        screen = _active_destination_screen(host)
        route = f"console {size[0]}x{size[1]}"
        # TASK-33007.7, rewritten on purpose: the six samplers sit in a closed
        # Sampling disclosure (opened here so every fallback is clickable),
        # and the rows now start at Temperature, not Streaming.
        screen.query_one("#settings-console-sampling", Collapsible).collapsed = False
        await _settle(pilot)

        assert await _click_each(
            pilot, screen, "console-behavior", CONSOLE_FALLBACKS, route
        ) == len(CONSOLE_FALLBACKS)
        seen = await _tab_through(
            pilot,
            screen,
            "console-behavior",
            CONSOLE_FALLBACKS,
            route,
            "settings-console-default-temperature",
        )
        assert seen == set(CONSOLE_FALLBACKS), set(CONSOLE_FALLBACKS) - seen
        # Every generation label is also a Providers & Models field, and that
        # category ranks first, so "/" lands on P&M (the next test). This is
        # the landing that "/" + Enter runs for a Console Behavior match.
        for field_id, name in CONSOLE_FALLBACKS.items():
            screen.query_one("#settings-category-console-behavior").focus()
            await _settle(pilot)
            screen._land_search_focus_on_field(
                field_id, MODEL_CONFIG_FIELDS[name].label
            )
            await _settle(pilot)
            assert _focused_id(pilot) == field_id
            _assert_guide_in_view(
                screen, "console-behavior", CONSOLE_FALLBACKS, f"{route} landing"
            )

        screen._land_search_focus_on_field(
            "settings-console-default-temperature", "Temperature"
        )
        await _settle(pilot)
        await _resize_both_ways(
            pilot, screen, "console-behavior", CONSOLE_FALLBACKS, route, size
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@private_profile_test
async def test_model_default_guide_stays_in_the_inspector_view(request, size):
    """AC#3/AC#4: "/" search, click and Tab, then a resize."""
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle_settings_mount_storm(pilot)
        route = f"model defaults {size[0]}x{size[1]}"

        searched = []
        for field_id, name in MODEL_DEFAULTS.items():
            # Esc releases the previous landing's Input first; "/" typed
            # into an Input is a literal slash.
            await pilot.press("escape", "slash")
            search = pilot.app.focused
            assert isinstance(search, Input) and search.id == "settings-category-search"
            search.value = MODEL_CONFIG_FIELDS[name].label.lower()
            await pilot.press("enter")
            await _settle(pilot, 4)
            screen = _active_destination_screen(host)
            if _focused_id(pilot) != field_id:
                # Hidden for the test provider's model: the search status
                # says so and focus stays put.
                assert not _usable(screen, field_id), (
                    route,
                    field_id,
                    _focused_id(pilot),
                )
                continue
            _assert_guide_in_view(screen, "provider", MODEL_DEFAULTS, f"{route} search")
            searched.append(field_id)
        assert "settings-model-profile-temperature" in searched, searched

        screen = _active_destination_screen(host)
        clicked = await _click_each(pilot, screen, "provider", MODEL_DEFAULTS, route)
        assert clicked == len(searched)
        seen = await _tab_through(
            pilot, screen, "provider", MODEL_DEFAULTS, route, searched[0]
        )
        assert seen == set(searched), set(searched) - seen

        screen.query_one("#settings-model-profile-temperature").focus()
        await _settle(pilot)
        await _resize_both_ways(pilot, screen, "provider", MODEL_DEFAULTS, route, size)


# One guided field per category whose guide the fix reaches only through the
# shared hook (``_reveal_settings_focus_after_refresh``): (category, field,
# the guide's first row for that field).
SHARED_HOOK_CASES = (
    (
        "appearance",
        "settings-appearance-palette-theme-limit",
        "Focused setting: Palette limit",
    ),
    (
        "storage",
        "settings-storage-chachanotes-db-path",
        "Focused setting: ChaChaNotes DB",
    ),
    (
        # A fresh profile's built-in RAG profile is read-only, so its
        # retrieval fields are disabled; the profile picker stays usable.
        "library-rag",
        "settings-library-rag-profile-select",
        "Focused group: Profiles",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize(
    "category, field_id, first_row",
    SHARED_HOOK_CASES,
    ids=[case[0] for case in SHARED_HOOK_CASES],
)
@private_profile_test
async def test_other_guided_categories_keep_the_guide_in_view_on_resize(
    request, category, field_id, first_row, size
):
    """Appearance, Storage and Library/RAG: focus a guided field, resize both ways."""
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle_settings_mount_storm(pilot)
        await _open_settings_category(pilot, f"#settings-category-{category}")
        await _settle(pilot)
        screen = _active_destination_screen(host)
        assert _usable(screen, field_id), field_id
        screen.query_one(f"#{field_id}").focus()
        await _settle(pilot)
        assert _focused_id(pilot) == field_id
        assert screen._active_settings_field_id == field_id
        texts = _assert_rows_in_view(screen, category, f"{category} focus")
        assert texts[0].startswith(first_row), texts
        body = screen.query_one("#settings-impact-pane-body", VerticalScroll)
        first = screen.query_one(f"#settings-{category}-field-guide-0")
        other = SIZES[1] if size == SIZES[0] else SIZES[0]
        for step in (other, size):
            await pilot.resize_terminal(*step)
            await _settle(pilot, 5)
            route = f"{category} resized to {step[0]}x{step[1]}"
            _assert_rows_in_view(screen, category, route)
            # These guides stay in view without the re-pin; this shows the
            # shared hook reaches them: the guide heads the view again, as
            # focus left it, unless the pane cannot scroll that far.
            view_top = body.scrollable_content_region.y
            assert first.region.y == view_top or body.scroll_y == body.max_scroll_y, (
                route,
                first.region.y,
                view_top,
                body.scroll_y,
                body.max_scroll_y,
            )
