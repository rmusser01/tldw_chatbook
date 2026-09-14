"""TASK-32533: a provider Select never receives a value it does not offer.

Critique #3's P0 was one unguarded provider ``Select`` (the Console model
popover) whose value came from a draft rather than from its own options.
Settings has two more of the same class, both of which write a value read from
a catalog into a Select that was populated from a *different* read of that
catalog:

* ``#settings-provider-value`` (Providers & Models) -- options are built at
  compose time from ``_provider_select_options()``; ``_sync_provider_manual_widget``
  re-reads the catalog afterwards, and registering or removing a custom endpoint
  moves one without the other;
* ``#settings-speech-configure-provider`` (Speech) -- the deep-link target
  validates its provider against ``BUILT_IN_TTS_PROVIDER_IDS`` while the select
  is built from the separate, hand-maintained ``BUILT_IN_TTS_PROVIDER_ORDER``.

Both tests reproduce that disagreement the only way it can happen in one
process -- by dropping an option the caller still believes in -- and drive the
real route. Without ``assign_select_value`` (``Widgets/select_values.py``) both
raise ``InvalidSelectValueError``, which before this task exited the whole app.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, OptionList, Select

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Speech.speech_settings_contracts import SpeechTTSNavigationTarget


@pytest.mark.asyncio
async def test_provider_pane_survives_a_catalog_value_its_select_no_longer_offers():
    """The Providers pane's real sync route, with the catalog moved under it."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()

        provider_select = screen.query_one("#settings-provider-value", Select)
        offered = [
            (label, value)
            for label, value in provider_select._options
            if value is not Select.NULL
        ]
        # A provider the catalog still knows -- `_provider_select_value_for_provider`
        # returns it verbatim -- but which this mounted select no longer lists.
        dropped = next(
            value
            for _label, value in offered
            if value != provider_select.value and value != "__manual__"
        )
        provider_select.set_options(
            [(label, value) for label, value in offered if value != dropped]
        )
        await pilot.pause()

        screen._sync_provider_manual_widget(dropped)
        await pilot.pause()

        assert host.is_running, "the settings pane took the app down with it"
        assert host._exception is None, f"sync raised {host._exception!r}"
        # Not a dead pane, and not a third state either: a value the select can
        # no longer show falls back to the manual spelling, which names the real
        # provider in the row underneath instead of contradicting it.
        assert provider_select.value == "__manual__"
        manual_input = screen.query_one("#settings-provider-manual-value", Input)
        assert manual_input.value == dropped
        assert not manual_input.disabled


@pytest.mark.asyncio
async def test_speech_pane_survives_a_navigation_target_its_select_no_longer_offers():
    """The Speech deep-link route, with the two provider tuples disagreeing."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "speech-tts"})
        await _wait_for_selector(
            screen, pilot, "#settings-speech-configure-provider", timeout=8.0
        )
        await pilot.pause()

        provider_select = screen.query_one("#settings-speech-configure-provider", Select)
        offered = [
            (label, value)
            for label, value in provider_select._options
            if value is not Select.NULL
        ]
        dropped = next(
            value for _label, value in offered if value != provider_select.value
        )
        provider_select.set_options(
            [(label, value) for label, value in offered if value != dropped]
        )
        await pilot.pause()
        kept = provider_select.value

        # `SpeechTTSNavigationTarget` accepts it: it is a built-in provider id.
        screen._speech_tts_navigation_target = SpeechTTSNavigationTarget(dropped)
        screen._apply_speech_tts_navigation_context()
        await pilot.pause()

        assert host.is_running, "the speech pane took the app down with it"
        assert host._exception is None, f"navigation raised {host._exception!r}"
        assert provider_select.value == kept


@pytest.mark.asyncio
async def test_provider_picker_row_the_select_no_longer_offers_does_not_raise():
    """Clicking a picker row for a provider the mounted select dropped.

    The picker's rows are rebuilt from a live catalog read on every refresh
    (``build_provider_picker_groups``), while ``#settings-provider-value`` was
    populated once at compose time -- the same disagreement, one route further
    on. Without ``assign_select_value`` the assignment in
    ``handle_provider_picker_selected`` raises ``InvalidSelectValueError`` from
    a user click.
    """
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()

        provider_select = screen.query_one("#settings-provider-value", Select)
        offered = [
            (label, value)
            for label, value in provider_select._options
            if value is not Select.NULL
        ]
        dropped = next(
            value
            for _label, value in offered
            if value != provider_select.value and value != "__manual__"
        )
        provider_select.set_options(
            [(label, value) for label, value in offered if value != dropped]
        )
        await pilot.pause()

        picker = screen.query_one("#settings-provider-picker", OptionList)
        index = next(
            i
            for i in range(picker.option_count)
            if getattr(picker.get_option_at_index(i), "provider_id", None) == dropped
        )
        picker.focus()
        picker.highlighted = index
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert host.is_running, "the picker click took the app down with it"
        assert host._exception is None, f"picker selection raised {host._exception!r}"
        # The click still does what the user asked even though the select cannot
        # show it: the provider is applied directly.
        assert (
            str(screen._provider_setting_values_mapping().get("provider") or "")
            == dropped
        )


@pytest.mark.asyncio
async def test_provider_revert_with_a_saved_value_the_select_no_longer_offers():
    """Revert the Providers category while its select has lost the saved value.

    ``_revert_category`` re-reads the saved provider and writes it straight into
    ``#settings-provider-value``; the select's options are the compose-time set.
    """
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()

        provider_select = screen.query_one("#settings-provider-value", Select)
        saved = str(screen._provider_setting_values()["provider"])
        dropped = screen._provider_select_value_for_provider(saved)
        assert dropped != "__manual__", "fixture provider is not a catalog provider"
        offered = [
            (label, value)
            for label, value in provider_select._options
            if value is not Select.NULL
        ]
        provider_select.set_options(
            [(label, value) for label, value in offered if value != dropped]
        )
        await pilot.pause()

        screen._revert_category(SettingsCategoryId.PROVIDERS_MODELS)
        await pilot.pause()

        assert host.is_running, "revert took the app down with it"
        assert host._exception is None, f"revert raised {host._exception!r}"
        # `_sync_provider_manual_widget` repairs the refusal: manual spelling,
        # with the real provider named in the row beneath it.
        assert provider_select.value == "__manual__"
        manual_input = screen.query_one("#settings-provider-manual-value", Input)
        assert manual_input.value == saved
