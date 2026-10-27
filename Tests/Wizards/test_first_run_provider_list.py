"""TASK-34100.6 slice D: the Provider step's list is searchable and honest.

Review findings provider-11 (60 providers in an unsearchable five-row box with
no descriptions; typing on the list did nothing), provider-12 (legacy aliases
and the '#2' slot looked like duplicates of the real rows) and E2 / coverage-08
(a provider whose key is already exported was not marked anywhere in the list).
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Input

from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    key_input,
    pick_provider,
    provider_list,
    resumed_on_provider,
    static_text,
    wait_for_step,
    wait_until,
)
from tldw_chatbook.Chat.provider_setup_facts import LEGACY_PROVIDER_KEYS
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER
from tldw_chatbook.UI.Wizards.first_run_step_guard import first_run_provider_catalog

pytestmark = pytest.mark.bootstrap_profile

_FILTER = "#setup-provider-filter"
_COUNT = "#setup-provider-count"


def _option_ids(step) -> list[str]:
    choices = provider_list(step)
    return [
        str(choices.get_option_at_index(index).id)
        for index in range(choices.option_count)
    ]


def _provider_rows(step) -> list[str]:
    """Selectable provider rows of the normal groups, in order."""
    return [
        option_id.removeprefix("provider-")
        for option_id in _option_ids(step)
        if option_id.startswith("provider-")
    ]


def _prompt(step, option_id: str) -> str:
    option = provider_list(step).get_option(option_id)
    return str(getattr(option.prompt, "plain", option.prompt))


async def _on_provider(pilot, wizard):
    await wait_for_step(pilot, wizard, STEP_PROVIDER)
    _container, step = current_step(wizard)
    return step


@pytest.mark.asyncio
async def test_a_filter_field_and_a_count_line_sit_above_the_list():
    """provider-11: 'Filter providers — name or ID' and a count above the list."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        field = step.query_one(_FILTER, Input)
        assert field.placeholder == "Filter providers — name or ID"
        total = len(first_run_provider_catalog())
        assert static_text(step, _COUNT) == f"{total} providers · type to filter"
        choices = provider_list(step)
        assert field.region.y < choices.region.y
        # One row for both at 80x24's budget (live finding, TASK-34100.6).
        assert step.query_one(_COUNT).region.y == field.region.y


@pytest.mark.asyncio
async def test_letters_typed_on_the_list_filter_it_and_space_still_selects():
    """provider-11: typing 'openr' on the list finds OpenRouter; Space picks it."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        choices = provider_list(step)
        choices.focus()
        await pilot.pause()
        for key in "openr":
            await pilot.press(key)
        await pilot.pause()
        assert step.query_one(_FILTER, Input).value == "openr"
        assert _provider_rows(step) == ["openrouter"]
        total = len(first_run_provider_catalog())
        assert static_text(step, _COUNT) == f"1 of {total} match “openr”"
        assert app.focused is choices, "the list keeps focus while it filters"
        assert step.selected_provider_key == "", "filtering never picks a row"

        await pilot.press("space")
        await pilot.pause()
        assert step.selected_provider_key == "openrouter"
        assert app.focused is key_input(step), "an explicit pick focuses the key"

        choices.focus()
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.pause()
        assert step.query_one(_FILTER, Input).value == "open"


@pytest.mark.asyncio
async def test_a_burst_typed_on_the_list_lands_whole_in_the_filter():
    """The driver's burst (lessons-textual): no letter is lost or misrouted."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        choices = provider_list(step)
        choices.focus()
        await pilot.pause()
        for key in "groq":
            event = events.Key(key, key)
            event.set_sender(app)
            app._driver.send_message(event)  # noqa: SLF001 - the driver path
        field = step.query_one(_FILTER, Input)
        await wait_until(pilot, lambda: field.value == "groq")
        await pilot.pause()
        assert _provider_rows(step) == ["groq"]
        assert step.selected_provider_key == ""


@pytest.mark.asyncio
async def test_no_match_says_so_and_how_to_get_the_list_back():
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        step.query_one(_FILTER, Input).value = "zzzz-nothing"
        await pilot.pause()
        total = len(first_run_provider_catalog())
        assert static_text(step, _COUNT) == f"No match · clear to see all {total}"
        assert _provider_rows(step) == []


@pytest.mark.asyncio
async def test_each_row_says_whether_it_runs_here_or_needs_a_cloud_key():
    """provider-11: a dim second column on every provider row."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        assert _prompt(step, "provider-openai") == "OpenAI  · Cloud · API key"
        assert _prompt(step, "provider-ollama") == "Ollama  · This computer"
        assert _prompt(step, "provider-llama_cpp") == "llama.cpp  · This computer"


@pytest.mark.asyncio
async def test_legacy_rows_hide_behind_a_reveal_the_filter_still_finds_them():
    """provider-12: legacy aliases and '#2' are one reveal away, never gone."""
    catalog_keys = {entry.readiness_key for entry in first_run_provider_catalog()}
    legacy = sorted(LEGACY_PROVIDER_KEYS & catalog_keys)
    assert legacy, "the catalog lists legacy rows (sanity)"
    wizard = resumed_on_provider(
        lambda config: config.setdefault("chat_defaults", {}).update(
            {"provider": "openai"}
        )
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        rows = _provider_rows(step)
        assert not set(legacy) & set(rows)
        assert "show-legacy" in _option_ids(step)
        assert _prompt(step, "show-legacy") == f"Show legacy entries ({len(legacy)})"

        choices = provider_list(step)
        choices.focus()
        choices.highlighted = choices.get_option_index("show-legacy")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        rows = _provider_rows(step)
        assert set(legacy) <= set(rows)
        assert "show-legacy" not in _option_ids(step)
        assert step.selected_provider_key == ""

        step.query_one(_FILTER, Input).value = "legacy alias"
        await pilot.pause()
        assert {"local_ollama", "local_vllm", "local_llamacpp"} <= set(
            _provider_rows(step)
        )


@pytest.mark.asyncio
async def test_revealing_legacy_rows_highlights_the_first_one_where_the_reveal_was():
    """Live 2026-10-04: Enter on the reveal jumped the list back to its top
    row, so the rows just revealed (at the list's end) were off screen."""
    catalog_keys = {entry.readiness_key for entry in first_run_provider_catalog()}
    legacy = LEGACY_PROVIDER_KEYS & catalog_keys
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        choices = provider_list(step)
        choices.focus()
        choices.highlighted = choices.get_option_index("show-legacy")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        first_legacy = next(
            option_id
            for option_id in _option_ids(step)
            if option_id.removeprefix("provider-") in legacy
        )
        highlighted = choices.highlighted_option
        assert highlighted is not None and str(highlighted.id) == first_legacy
        top = int(choices.scroll_offset.y)
        assert top <= choices.highlighted < top + choices.scrollable_content_region.height
        assert step.selected_provider_key == ""  # revealing browses, never picks


@pytest.mark.asyncio
async def test_a_legacy_row_the_config_already_uses_stays_listed():
    wizard = resumed_on_provider(
        lambda config: config.setdefault("chat_defaults", {}).update(
            {"provider": "local_ollama"}
        )
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        rows = _provider_rows(step)
        assert "local_ollama" in rows
        assert "local_vllm" not in rows


@pytest.mark.asyncio
async def test_the_list_grows_into_free_height():
    """provider-11: more than the old five visible rows when there is room."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await pilot.pause()
        # The old cap was 7 rows including the border (5 visible).
        assert provider_list(step).outer_size.height >= 14


@pytest.mark.asyncio
async def test_at_80x24_the_key_field_is_on_screen_after_picking_openai(monkeypatch):
    """provider-11: growing the list never pushes the key field off screen."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(80, 24)) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "openai")
        await pilot.pause(0.1)
        field = key_input(step)
        assert app.focused is field
        region = field.region
        screen = app.screen.region
        assert region.height > 0
        assert screen.y <= region.y and region.bottom <= screen.bottom, (
            f"key field {region} is outside the screen {screen}"
        )


@pytest.mark.asyncio
async def test_a_provider_whose_key_is_exported_is_ready_on_this_machine(monkeypatch):
    """E2 / coverage-08: 'Ready on this machine' lists an exported key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 40)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "ready-anthropic" in _option_ids(step))
        ids = _option_ids(step)
        assert ids[0] == "group-ready-on-this-machine"
        assert _prompt(step, "ready-anthropic") == (
            "Anthropic  · key in ANTHROPIC_API_KEY"
        )
        assert ids.index("ready-anthropic") < ids.index("group-popular")



@pytest.mark.asyncio
async def test_a_provider_whose_key_is_saved_is_ready_on_this_machine(monkeypatch):
    """E2 / AC#29; review round 1, TE-04: the stored-key branch ('key saved')
    had no test -- only the exported-key row did."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    wizard = resumed_on_provider(
        lambda config: config["api_settings"].setdefault("groq", {}).update(
            api_key="gsk_" + "a" * 40
        )
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "ready-groq" in _option_ids(step))
        assert _prompt(step, "ready-groq") == "Groq  · key saved"
        ids = _option_ids(step)
        assert ids.index("ready-groq") < ids.index("group-popular")

@pytest.mark.asyncio
async def test_arrowing_between_keyed_and_local_rows_keeps_the_list_in_view(monkeypatch):
    """Live finding (TASK-34100.6): a provider switch toggles 'Authentication',
    and Textual's Collapsible scrolls itself into view on every toggle -- so
    arrowing Anthropic -> Ollama scrolled the list being browsed off screen."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "anthropic")
        choices = provider_list(step)
        choices.focus()
        await pilot.pause(0.3)
        await pilot.press("down")
        await pilot.pause(0.5)
        assert step.selected_provider_key == "ollama"
        assert app.focused is choices
        viewport = step.region
        row = choices.region
        assert row.y >= viewport.y and row.y < viewport.bottom, (
            f"the focused list {row} scrolled out of the step viewport {viewport}"
        )


@pytest.mark.asyncio
async def test_enter_in_the_filter_field_picks_the_first_match_and_stays(monkeypatch):
    """Enter in a step Input means 'continue' wizard-wide; in the filter it
    picks the highlighted match instead (it must not advance to Model)."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        field = step.query_one(_FILTER, Input)
        field.focus()
        await pilot.pause()
        for key in "groq":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert step.selected_provider_key == "groq"
        _container, current = current_step(wizard)
        assert current is step, "Enter in the filter advanced the wizard"


def _highlighted_row_y(step) -> int:
    """The screen row of the list's highlighted option."""
    choices = provider_list(step)
    line = choices._index_to_line[choices.highlighted]
    return choices.region.y + choices.gutter.top + line - int(choices.scroll_offset.y)


@pytest.mark.asyncio
async def test_browsing_after_the_step_scrolled_brings_the_row_back(monkeypatch):
    """Live F10, reproduced on this branch (g6-r1fix/02): with the list
    focused, the step scrolled down (here as a mouse wheel does), and the
    arrow keys went on browsing -- selecting rows, rewriting the endpoint --
    while every row of the list was above the step's viewport."""
    from tldw_chatbook.Chat import local_server_discovery

    async def nothing_answers(_app_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", nothing_answers)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, step = current_step(wizard)
        choices = provider_list(step)
        choices.focus()
        await pilot.press("down", "down", "down")  # browse onto a keyed row
        await pilot.pause(0.2)
        step.scroll_to(y=step.max_scroll_y, animate=False, immediate=True)
        await pilot.pause()
        assert step.scroll_y > 0 and app.focused is choices
        for press in range(3):
            await pilot.press("up")
            await pilot.pause(0.1)
            y = _highlighted_row_y(step)
            assert step.region.y <= y < step.region.bottom, (
                f"after {press + 1} Up: row y={y} outside the step {step.region}"
            )


def test_a_filter_that_only_finds_a_local_server_never_says_no_match():
    """Review round 1, live F9: filtering 'studio' read 'No match · clear to
    see all 60' while the list showed 'LM Studio on this computer ·
    127.0.0.1:1234'. The found row is a match, and the line says so."""
    from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
    from tldw_chatbook.UI.Wizards.first_run_provider_list import build_provider_options

    entries = first_run_provider_catalog()
    lm_studio = DiscoveredLocalServer(
        "custom", "http://127.0.0.1:1234", ("q",), engine="LM Studio"
    )
    build = build_provider_options(entries, query="studio", found=(lm_studio,))
    ids = [str(option.id) for option in build.options]
    assert "found-0" in ids
    assert build.count_line == (
        f"1 local server matches “studio” · clear to see all {len(entries)}"
    )
    two = build_provider_options(
        entries,
        query="this computer",
        found=(lm_studio, DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:9099", ("a",))),
    )
    assert not two.count_line.startswith("No match")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_arrowing_past_the_lists_viewport_scrolls_the_list(monkeypatch, size):
    """Review round 2, R2-01 / G6-VR2-01: the round-1 ``watch_highlighted``
    override replaced OptionList's own watcher (Textual looks it up with
    getattr), so the list never scrolled to the browsed row and never posted
    OptionHighlighted. Arrowing past the last visible row selected providers
    the user could not see."""
    from textual.widgets import OptionList

    from tldw_chatbook.Chat import local_server_discovery

    async def nothing_answers(_app_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", nothing_answers)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    highlighted: list[int] = []
    async with app.run_test(size=size) as pilot:
        step = await _on_provider(pilot, wizard)
        choices = provider_list(step)
        original = choices.post_message

        def spy(message):
            if isinstance(message, OptionList.OptionHighlighted):
                highlighted.append(message.option_index)
            return original(message)

        choices.post_message = spy
        choices.focus()
        await pilot.pause()
        window = choices.scrollable_content_region.height
        for _ in range(window + 6):
            await pilot.press("down")
        await pilot.pause(0.3)
        line = choices._index_to_line[choices.highlighted]
        top = int(choices.scroll_offset.y)
        assert line >= window, "the test never browsed past the first screenful"
        assert top <= line < top + choices.scrollable_content_region.height, (
            f"highlighted line {line} is outside the list's window "
            f"[{top}, {top + choices.scrollable_content_region.height})"
        )
        assert choices.highlighted in highlighted, "OptionHighlighted was never posted"
