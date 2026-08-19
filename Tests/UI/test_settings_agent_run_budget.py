# Tests/UI/test_settings_agent_run_budget.py
"""TASK-18600: the Agent run budget section in Settings > Console Behavior.

These exercise the screen's own staging/validation logic directly rather
than through a mounted app: the five fields are table-driven, so what is
worth pinning is the table's contract (every field reaches config, floors
are refused rather than clamped, the derived step-floor warning fires) and
not five copies of a widget-mounting test.
"""

from __future__ import annotations

import pytest

from textual.widgets import Input

from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    AGENT_BUDGET_FIELDS,
    AGENT_BUDGET_FIELDS_BY_KEY,
    AGENT_BUDGET_KEYS,
    CONSOLE_BEHAVIOR_CONSOLE_KEYS,
    CONSOLE_BEHAVIOR_SAVE_ORDER,
    SettingsScreen,
)


class _Screen:
    """The budget methods under test, bound to a bare object.

    They touch only `_console_settings()` and `_settings_drafts`, so this
    avoids mounting a Textual app for what is pure staging logic.
    """

    def __init__(self, saved: dict | None = None):
        self._saved = saved or {}
        self._settings_drafts: dict = {}

    def _console_settings(self) -> dict:
        return self._saved

    # bind the real implementations
    _loaded_agent_budget_value = SettingsScreen._loaded_agent_budget_value
    _agent_budget_value = SettingsScreen._agent_budget_value
    _normalise_agent_budget_value = SettingsScreen._normalise_agent_budget_value
    # These two are @staticmethod on the real screen; re-wrapping keeps them
    # static here, otherwise copying them onto this class rebinds them as
    # instance methods and `self` arrives as the first argument.
    _format_agent_budget_number = staticmethod(
        SettingsScreen._format_agent_budget_number
    )
    _stage_agent_budget_value = SettingsScreen._stage_agent_budget_value
    _humanise_seconds = staticmethod(SettingsScreen._humanise_seconds)
    _agent_budget_hint = SettingsScreen._agent_budget_hint
    _agent_budget_help_text = SettingsScreen._agent_budget_help_text
    _agent_budget_step_floor_warning = SettingsScreen._agent_budget_step_floor_warning


# -- the table's contract ---------------------------------------------------


def test_all_five_limits_are_exposed():
    """AC#1: the section covers every limit the owner asked for, and the
    per-tool-call ceiling that makes a long wall budget actually usable."""
    assert set(AGENT_BUDGET_KEYS) == {
        "agent_max_model_turns",
        "agent_max_steps",
        "agent_max_wall_seconds",
        "agent_max_total_tokens",
        "agent_max_tool_call_seconds",
    }


def test_every_field_is_wired_into_the_console_save_path():
    """A field in the table but absent from these two collections would
    render, stage, and then silently fail to save."""
    for key in AGENT_BUDGET_KEYS:
        assert key in CONSOLE_BEHAVIOR_CONSOLE_KEYS, key
        assert key in CONSOLE_BEHAVIOR_SAVE_ORDER, key


def test_field_ids_and_keys_are_unique():
    assert len({f.widget_id for f in AGENT_BUDGET_FIELDS}) == len(AGENT_BUDGET_FIELDS)
    assert len({f.key for f in AGENT_BUDGET_FIELDS}) == len(AGENT_BUDGET_FIELDS)


def test_the_token_budget_is_presented_first():
    """It is the limit that actually stops a run, so it leads the section
    rather than sitting under two backstops nobody will hit."""
    assert AGENT_BUDGET_FIELDS[0].key == "agent_max_total_tokens"


# -- loading and staging ----------------------------------------------------


def test_saved_values_are_loaded():
    screen = _Screen({"agent_max_model_turns": 55, "agent_max_wall_seconds": 12.5})
    assert (
        screen._loaded_agent_budget_value(
            AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
        )
        == 55
    )
    assert (
        screen._loaded_agent_budget_value(
            AGENT_BUDGET_FIELDS_BY_KEY["agent_max_wall_seconds"]
        )
        == 12.5
    )


def test_unset_values_fall_back_to_the_shipped_default():
    screen = _Screen({})
    for field in AGENT_BUDGET_FIELDS:
        assert screen._loaded_agent_budget_value(field) == field.default


def test_staging_a_value_marks_the_draft_dirty():
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
    screen._stage_agent_budget_value(field, "77")
    draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
    assert draft.values["agent_max_model_turns"] == 77
    assert draft.is_dirty


def test_staging_back_to_the_saved_value_clears_the_draft():
    """Otherwise Save stays enabled after a user types a change and undoes
    it, and 'no changes to save' never fires."""
    screen = _Screen({"agent_max_model_turns": 40})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
    screen._stage_agent_budget_value(field, "41")
    assert SettingsCategoryId.CONSOLE_BEHAVIOR in screen._settings_drafts
    screen._stage_agent_budget_value(field, "40")
    assert SettingsCategoryId.CONSOLE_BEHAVIOR not in screen._settings_drafts


def test_invalid_text_is_kept_staged_rather_than_discarded():
    """Mid-edit text ('1', on the way to '1000') must survive a re-render;
    the save path re-normalises and reports the error."""
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
    screen._stage_agent_budget_value(field, "not-a-number")
    draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
    assert draft.values["agent_max_model_turns"] == "not-a-number"


# -- validation -------------------------------------------------------------


def test_below_floor_values_are_refused_not_clamped():
    """AC#4: silently clamping is how a 2000-turn budget quietly becomes
    something the user never chose."""
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
    with pytest.raises(ValueError) as exc:
        screen._normalise_agent_budget_value(field, "0")
    assert "at least" in str(exc.value)


def test_zero_is_valid_where_it_means_unlimited():
    screen = _Screen({})
    for key in ("agent_max_total_tokens", "agent_max_tool_call_seconds"):
        field = AGENT_BUDGET_FIELDS_BY_KEY[key]
        assert screen._normalise_agent_budget_value(field, "0") == 0


def test_non_finite_input_is_refused():
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_wall_seconds"]
    for bad in ("inf", "nan"):
        with pytest.raises(ValueError):
            screen._normalise_agent_budget_value(field, bad)


def test_large_values_are_accepted():
    """No ceiling: the whole point of the task is long expensive runs."""
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_total_tokens"]
    assert screen._normalise_agent_budget_value(field, "999999999") == 999999999


# -- the live hints ---------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (86400.0, "1d"),
        (3600.0, "1h"),
        (1800.0, "30m"),
        (90.0, "1m 30s"),
        (45.0, "45s"),
        (0.0, "unlimited"),
    ],
)
def test_seconds_are_rendered_as_a_readable_duration(seconds, expected):
    """86400 is not a number anyone reads as a day at a glance, and this
    field exists to be set to large values deliberately."""
    assert _Screen()._humanise_seconds(seconds) == expected


def test_the_unlimited_token_budget_says_what_it_costs_you():
    """0 is legal but removes the only runaway backstop -- the loop
    detector only catches identical repeated calls."""
    screen = _Screen({"agent_max_total_tokens": 0})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_total_tokens"]
    hint = screen._agent_budget_hint(field)
    assert "unlimited" in hint
    assert "backstop" in hint


def test_an_invalid_value_is_explained_in_place():
    screen = _Screen({})
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_model_turns"]
    screen._stage_agent_budget_value(field, "0")
    assert "at least" in screen._agent_budget_help_text(field)


# -- the derived step floor -------------------------------------------------


def test_no_warning_when_the_step_budget_clears_the_derived_floor():
    screen = _Screen({})  # shipped defaults: 25000 steps vs 5998 needed
    assert screen._agent_budget_step_floor_warning() == ""


def test_warning_when_the_step_budget_will_bind_first():
    """AC#6: 100 steps against 2000 turns means runs stop on 'step budget
    exhausted' at round ~34 -- a silent misconfiguration with a confusing
    symptom."""
    screen = _Screen({"agent_max_model_turns": 2000, "agent_max_steps": 100})
    warning = screen._agent_budget_step_floor_warning()
    assert warning
    assert "5998" in warning
    assert "2000" in warning


def test_the_warning_is_recomputed_from_staged_values():
    """It is a function of TWO fields, so editing either must update it --
    including before anything is saved."""
    screen = _Screen({})
    screen._stage_agent_budget_value(
        AGENT_BUDGET_FIELDS_BY_KEY["agent_max_steps"], "10"
    )
    assert screen._agent_budget_step_floor_warning()


def test_no_warning_while_a_field_is_mid_edit():
    """An empty or half-typed field must not flash a warning derived from
    a number the user has not finished entering."""
    screen = _Screen({})
    screen._stage_agent_budget_value(
        AGENT_BUDGET_FIELDS_BY_KEY["agent_max_steps"], ""
    )
    assert screen._agent_budget_step_floor_warning() == ""


# -- it actually mounts -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_budget_section_mounts_in_console_behavior():
    """compose() builds these five rows in a loop over the spec table, which
    is the one part of this feature the pure staging tests above cannot
    reach: a bad `with Horizontal(...)` nesting or a duplicate id would
    crash compose and leave the whole category unmounted."""
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from Tests.UI.test_settings_configuration_hub import _open_settings_category

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        for field in AGENT_BUDGET_FIELDS:
            await _wait_for_selector(
                screen, pilot, f"#{field.widget_id}", timeout=8.0
            )
        await _wait_for_selector(
            screen, pilot, "#settings-console-agent-budget-step-warning", timeout=8.0
        )


@pytest.mark.asyncio
async def test_opening_the_category_does_not_create_a_draft():
    """Mounting must not stage anything. Textual fires Input.Changed when a
    widget is created with an initial `value`, so five new inputs are five
    new chances to open the category already dirty -- which would enable
    Save with nothing changed and make 'revert' meaningless."""
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from Tests.UI.test_settings_configuration_hub import _open_settings_category

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        await _wait_for_selector(
            screen, pilot, f"#{AGENT_BUDGET_FIELDS[0].widget_id}", timeout=8.0
        )
        await pilot.pause()
        draft = screen._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        staged_budget_keys = (
            set(draft.values) & set(AGENT_BUDGET_KEYS) if draft else set()
        )
        assert not staged_budget_keys, (
            f"opening the category staged {sorted(staged_budget_keys)}"
        )


@pytest.mark.asyncio
async def test_an_edit_saves_and_reaches_the_run_budget_resolver(monkeypatch):
    """The full integration the staging tests cannot cover: a user edit in
    the mounted screen is staged, saved through the real save path, written
    into the app config the resolver reads, and `console_run_budget()`
    then returns the saved number. Any broken link in that chain -- a field
    that stages but does not save, a key saved under a different name, a
    resolver reading a different key -- fails here with the exact number
    that got lost.
    """
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from Tests.UI.test_settings_configuration_hub import (
        _open_settings_category,
        _wait_for_settings_text,
    )
    from tldw_chatbook.Chat import console_agent_bridge
    from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module

    app = _build_test_app()
    app.app_config["console"] = {}
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(
        settings_screen_module, "SettingsConfigAdapter", FakeAdapter
    )
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        tokens_field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_total_tokens"]
        await _wait_for_selector(
            screen, pilot, f"#{tokens_field.widget_id}", timeout=8.0
        )
        widget = screen.query_one(f"#{tokens_field.widget_id}", Input)
        widget.value = "1234567"
        screen.handle_console_agent_budget_changed(
            Input.Changed(widget, widget.value)
        )
        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(
            screen, pilot, "Console behavior settings saved."
        )

    # The save path persisted the staged value under the resolver's key...
    assert saved == [{"console": {"agent_max_total_tokens": 1234567}}]
    assert app.app_config["console"]["agent_max_total_tokens"] == 1234567
    # ...and the resolver returns it when the config it reads holds it.
    # `console_run_budget()` reads the on-disk config cache via
    # `get_cli_setting`, not this harness's in-memory `app.app_config`, so
    # pin the read to what the adapter just persisted -- the assertion is
    # that the resolver honours a saved value with no restart, which is
    # AC#2 of the task this closes.
    written = dict(app.app_config["console"])

    def _fake_get(section, key, default=None, *a, **k):
        if section != "console":
            return default
        return written.get(key, default)

    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting", _fake_get, raising=True
    )
    budget = console_agent_bridge.console_run_budget()
    assert budget.max_total_tokens == 1234567
