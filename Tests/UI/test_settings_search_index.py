"""Drift guard for the Settings field-level search index (TASK-23109).

The index in ``tldw_chatbook/UI/Screens/settings_search_index.py`` is
hand-maintained (with derived sections), so this suite mounts every
category in the real harness and fails when a rendered setting control is
missing from it -- without this, a new setting silently becomes unfindable
by "/" search (exactly how "Reduce motion" went missing between task-1715
and TASK-23109).

A "rendered setting control" is:

* any value-editing widget (Input/Select/Checkbox/Switch/TextArea/
  SelectionList) inside the detail-pane body, or
* a Button that is the ONLY interactive widget in a labeled
  ``settings-input-row`` (the screen's toggle-button pattern -- "Reduce
  motion", "Context LLM", boolean flips rendered as Buttons).

Non-setting utility controls are excluded through the explicit, commented
allowlist below -- every exclusion says why it is not a searchable setting.
"""

import pytest
from textual.containers import Horizontal
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Select,
    SelectionList,
    Static,
    Switch,
    TextArea,
)

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    ALL_CATEGORY_IDS,
    _click_settings_category,
    _settle_settings,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_search_index import FIELD_SEARCH_INDEX

#: Widget types that hold a user-editable value.
VALUE_WIDGET_TYPES = (Input, Select, Checkbox, Switch, TextArea, SelectionList)

#: (category value, widget id) pairs that render a value widget which is NOT
#: a searchable persisted setting. Every entry needs a justification.
NON_SETTING_CONTROLS: frozenset[tuple[str, str]] = frozenset(
    {
        # The raw TOML editor IS the whole Advanced Config category; the
        # category itself is already findable by name and owned keys.
        ("advanced-config", "settings-advanced-config-editor"),
        # Search boxes filter content; they are not settings.
        ("internal-prompts", "internal-prompts-search"),
        ("providers-models", "settings-provider-search"),
        # Transient discovery output picker, not a persisted setting.
        ("providers-models", "settings-discovered-models-list"),
        # Manual-provider entry is the fallback leg of the already-indexed
        # Provider select ("Other"); it has no stable label of its own.
        ("providers-models", "settings-provider-manual-value"),
        # Session view filter for the workspace list, not persisted config.
        ("workspaces", "settings-workspaces-show-archived"),
        # One-shot action buttons operating on the adjacent, already-indexed
        # setting (clear the saved API key / reset the context window) --
        # actions, not settings.
        ("providers-models", "settings-provider-api-key-clear"),
        ("providers-models", "settings-model-context-window-reset"),
    }
)


def _enclosing_input_row(widget) -> Horizontal | None:
    parent = widget.parent
    while parent is not None and not isinstance(parent, Horizontal):
        parent = parent.parent
    if parent is not None and "settings-input-row" in parent.classes:
        return parent
    return None


def _is_labeled_row_toggle_button(widget: Button) -> bool:
    """The screen's Button-as-boolean-toggle pattern (e.g. Reduce motion)."""
    row = _enclosing_input_row(widget)
    if row is None:
        return False
    has_label = any(
        isinstance(child, Static) and "settings-input-label" in child.classes
        for child in row.children
    )
    if not has_label:
        return False
    interactive = [
        child
        for child in row.children
        if isinstance(child, VALUE_WIDGET_TYPES + (Button,))
    ]
    return len(interactive) == 1


def _indexed_field_ids(category: SettingsCategoryId) -> set[str]:
    return {field_id for field_id, _label in FIELD_SEARCH_INDEX.get(category, ())}


@pytest.mark.asyncio
async def test_every_rendered_setting_is_in_the_search_index():
    """Every mounted setting control is searchable or explicitly excluded."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    missing: list[tuple[str, str]] = []
    seen_setting_widgets = 0

    async with host.run_test(size=(190, 55)) as pilot:
        await _settle_settings(pilot)
        for category_value in ALL_CATEGORY_IDS:
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)
            category = SettingsCategoryId(category_value)
            indexed = _indexed_field_ids(category)
            try:
                body = screen.query_one("#settings-detail-pane-body")
            except Exception:
                continue
            for widget in body.query("*"):
                is_setting = isinstance(widget, VALUE_WIDGET_TYPES) or (
                    isinstance(widget, Button)
                    and _is_labeled_row_toggle_button(widget)
                )
                if not is_setting:
                    continue
                widget_id = widget.id or ""
                if not widget_id:
                    # A setting without an id cannot be focused by search.
                    missing.append((category_value, f"<no id: {widget!r}>"))
                    continue
                seen_setting_widgets += 1
                if (category_value, widget_id) in NON_SETTING_CONTROLS:
                    continue
                if widget_id not in indexed:
                    missing.append((category_value, widget_id))

    # The sweep must actually have seen the forms, or the guard is vacuous.
    assert seen_setting_widgets >= 100, seen_setting_widgets
    assert not missing, (
        "Rendered settings missing from FIELD_SEARCH_INDEX "
        "(add them to settings_search_index.py, or justify an exclusion in "
        f"NON_SETTING_CONTROLS): {sorted(set(missing))}"
    )


def test_non_setting_allowlist_has_no_stale_rows():
    """Every allowlist row names a category that exists and is not indexed."""
    known_categories = set(ALL_CATEGORY_IDS)
    for category_value, widget_id in NON_SETTING_CONTROLS:
        assert category_value in known_categories, category_value
        category = SettingsCategoryId(category_value)
        assert widget_id not in _indexed_field_ids(category), (
            f"{widget_id} is both allowlisted and indexed"
        )
