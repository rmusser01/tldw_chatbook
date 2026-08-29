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

Declared blindness (review finding 5/12): coverage gaps the harness cannot
see are named in ``HARNESS_BLIND_CATEGORIES`` / ``DECLARED_UNMOUNTED_IDS``
with the reason, instead of being silently absorbed.
"""

import pytest
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
from tldw_chatbook.UI.Screens.settings_search_index import (
    FIELD_SEARCH_INDEX,
    SPEECH_TTS_PROVIDER_FORM_FIELDS,
)

#: Widget types that hold a user-editable value.
VALUE_WIDGET_TYPES = (Input, Select, Checkbox, Switch, TextArea, SelectionList)

#: (category value, widget id) pairs that render a value widget which is NOT
#: a searchable persisted setting. Every entry needs a justification, and
#: every entry must actually render (asserted below).
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
        # "Use Official OpenAI" preset button: a one-shot action that fills
        # the adjacent (indexed) Base URL field -- an action, not a setting.
        ("speech-tts", "settings-speech-openai-official-preset"),
    }
)

#: Categories whose setting controls the harness CANNOT compose, with the
#: reason. The forward sweep does not require coverage here and the reverse
#: sweep does not require their indexed ids to resolve (review finding 12:
#: blindness must be declared, not silent).
HARNESS_BLIND_CATEGORIES: dict[str, str] = {
    "agents": (
        "the harness builds the app with chachanotes_db=None, so "
        "AgentsSettingsPanel composes only its no-database notice; the "
        "AGENTS index rows are hand-pinned against "
        "Widgets/settings_agents_panel.py"
    ),
}

#: Indexed ids that never mount in the harness, with the reason. The
#: Speech & TTS panel composes only the DEFAULT provider's configure form
#: (openai in this harness), so the other providers' form fields resolve
#: only after the user switches provider; their index rows are pinned
#: against the panel SOURCE by
#: test_speech_provider_form_fields_match_the_panel_source instead.
DECLARED_UNMOUNTED_IDS: frozenset[str] = frozenset(
    f"settings-speech-{provider_id}-{field_key.replace('_', '-')}"
    for provider_id, fields in SPEECH_TTS_PROVIDER_FORM_FIELDS.items()
    if provider_id != "openai"
    for field_key, _label in fields
)

#: Per-category minimum number of setting controls the sweep must see --
#: a floor per formful category rather than one global count, so one
#: category's form silently failing to compose cannot hide behind the
#: others (review finding 12).
PER_CATEGORY_MIN_SETTINGS: dict[str, int] = {
    "providers-models": 20,
    "speech-tts": 15,
    "appearance": 15,
    "theme": 10,
    "splash_screen": 5,
    "storage": 5,
    "console-behavior": 30,
    "library-rag": 20,
    "image_generation": 40,
    "video_generation": 15,
    "privacy-security": 1,
}


def _enclosing_input_row(widget):
    """Nearest ancestor container carrying settings-input-row, ANY depth.

    Review finding 12: the old version stopped at the first Horizontal, so
    a control nested one container deeper than its row was invisible.
    """
    node = widget.parent
    while node is not None:
        classes = getattr(node, "classes", ())
        if "settings-input-row" in classes:
            return node
        node = node.parent
    return None


def _row_label_text(widget) -> str:
    row = _enclosing_input_row(widget)
    if row is None:
        return ""
    for child in row.query(Static):
        if "settings-input-label" in child.classes:
            return str(child.renderable)
    return ""


def _is_labeled_row_toggle_button(widget: Button) -> bool:
    """The screen's Button-as-boolean-toggle pattern (e.g. Reduce motion)."""
    row = _enclosing_input_row(widget)
    if row is None:
        return False
    if not any(
        "settings-input-label" in child.classes for child in row.query(Static)
    ):
        return False
    interactive = [
        child
        for child in row.query("*")
        if isinstance(child, VALUE_WIDGET_TYPES + (Button,))
    ]
    return len(interactive) == 1


def _indexed_field_ids(category: SettingsCategoryId) -> set[str]:
    return {field_id for field_id, _label in FIELD_SEARCH_INDEX.get(category, ())}


def _normalized_label(text: str) -> str:
    return text.replace("⚠", "").strip().rstrip(":").strip().lower()


@pytest.mark.asyncio
async def test_every_rendered_setting_is_in_the_search_index():
    """Forward, reverse, allowlist, and label sweeps in one mounted pass."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    missing: list[tuple[str, str]] = []
    label_drift: list[tuple[str, str, str]] = []
    seen_by_category: dict[str, int] = {}
    all_ids_by_category: dict[str, set[str]] = {}

    async with host.run_test(size=(190, 55)) as pilot:
        await _settle_settings(pilot)
        for category_value in ALL_CATEGORY_IDS:
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)
            category = SettingsCategoryId(category_value)
            indexed = FIELD_SEARCH_INDEX.get(category, ())
            indexed_ids = {fid for fid, _ in indexed}
            body = screen.query_one("#settings-detail-pane-body")
            all_ids_by_category[category_value] = {
                w.id for w in body.query("*") if w.id
            }
            for widget in body.query("*"):
                is_setting = isinstance(widget, VALUE_WIDGET_TYPES) or (
                    isinstance(widget, Button)
                    and _is_labeled_row_toggle_button(widget)
                )
                if not is_setting:
                    continue
                # Review finding 12: id-less settings count toward the
                # floor AND are flagged -- they cannot be focused by search.
                seen_by_category[category_value] = (
                    seen_by_category.get(category_value, 0) + 1
                )
                widget_id = widget.id or ""
                if not widget_id:
                    missing.append((category_value, f"<no id: {widget!r}>"))
                    continue
                if (category_value, widget_id) in NON_SETTING_CONTROLS:
                    continue
                if widget_id not in indexed_ids:
                    missing.append((category_value, widget_id))
                    continue
                # Label sweep (review finding 8): search is containment-
                # based, so the guard is bidirectional containment -- some
                # indexed label must contain the rendered row label (typing
                # the visible words finds it) or be contained by it (an
                # indexed short form of a long parenthetical row label).
                # True drift ("hybrid alpha" vs "Hybrid balance",
                # "Preferred Library rail width" vs "Preferred rail width")
                # fails both directions.
                rendered = _normalized_label(_row_label_text(widget))
                if rendered:
                    labels = {
                        _normalized_label(label)
                        for fid, label in indexed
                        if fid == widget_id
                    }
                    if not any(
                        rendered in label or label in rendered
                        for label in labels
                    ):
                        label_drift.append(
                            (category_value, widget_id, rendered)
                        )

    for category_value, minimum in PER_CATEGORY_MIN_SETTINGS.items():
        assert category_value not in HARNESS_BLIND_CATEGORIES
        seen = seen_by_category.get(category_value, 0)
        assert seen >= minimum, (
            f"{category_value}: saw only {seen} setting controls "
            f"(floor {minimum}) -- did its form fail to compose?"
        )

    assert not missing, (
        "Rendered settings missing from FIELD_SEARCH_INDEX "
        "(add them to settings_search_index.py, or justify an exclusion in "
        f"NON_SETTING_CONTROLS): {sorted(set(missing))}"
    )
    assert not label_drift, (
        "Indexed labels drifted from the rendered row labels "
        f"(fix settings_search_index.py): {sorted(set(label_drift))}"
    )

    # Allowlist rows must actually render (review finding 12).
    for category_value, widget_id in NON_SETTING_CONTROLS:
        assert widget_id in all_ids_by_category.get(category_value, set()), (
            f"stale allowlist row: {category_value}/{widget_id} never rendered"
        )

    # Reverse direction (review finding 12): every indexed id resolves in a
    # mounted app, unless its absence is declared with a reason.
    unresolved: list[tuple[str, str]] = []
    for category in SettingsCategoryId:
        if category.value in HARNESS_BLIND_CATEGORIES:
            continue
        mounted = all_ids_by_category.get(category.value, set())
        for field_id in _indexed_field_ids(category):
            if field_id in DECLARED_UNMOUNTED_IDS:
                continue
            if field_id not in mounted:
                unresolved.append((category.value, field_id))
    assert not unresolved, (
        "Indexed ids that never resolved in the mounted app (fix the id, or "
        f"declare the gap with a reason): {sorted(unresolved)}"
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


def test_speech_provider_form_fields_match_the_panel_source():
    """Source pin for the six provider forms the harness cannot mount.

    The Speech panel composes only the default provider's configure form,
    so SPEECH_TTS_PROVIDER_FORM_FIELDS (which the index derives its rows
    from) is compared against the panel's _compose_provider_form source:
    every self._input/_select/_switch(provider_id, "key", "Label") call
    must have a matching table row and vice versa.
    """
    import re

    import tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel as panel

    source = open(panel.__file__, encoding="utf-8").read()
    start = source.find("def _compose_provider_form")
    assert start != -1
    end = source.find("\n    def ", start + 10)
    body_lines = source[start:end].split("\n")

    provider = None
    from_source: set[tuple[str, str, str]] = set()
    for i, line in enumerate(body_lines):
        branch = re.search(r'provider_id == "([a-z_0-9]+)"', line)
        if branch:
            provider = branch.group(1)
        if re.search(r"self\._(input|select|switch)\(", line):
            chunk = "\n".join(body_lines[i : i + 6])
            args = re.search(
                r'provider_id,\s*\n?\s*"([a-z_0-9]+)",\s*\n?\s*"([^"]+)"', chunk
            )
            if args:
                assert provider is not None, chunk
                from_source.add((provider, args.group(1), args.group(2)))

    from_table = {
        (provider_id, key, label)
        for provider_id, fields in SPEECH_TTS_PROVIDER_FORM_FIELDS.items()
        for key, label in fields
    }
    assert from_source, "panel source extraction found no provider fields"
    assert from_table == from_source, (
        f"only in table: {sorted(from_table - from_source)}; "
        f"only in panel: {sorted(from_source - from_table)}"
    )
