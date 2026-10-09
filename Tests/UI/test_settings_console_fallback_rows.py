"""Settings ▸ Console Behavior ▸ Global fallback defaults: the shared rows (TASK-33007.7).

Console Behavior owns the global fallbacks (ADR-006, ADR-052) and now draws
them with Model defaults' row grammar: core rows first, Sampling in one closed
one-row disclosure, and streaming as an On/Off Select from the same family as
the model default's Inherit/On/Off Select (ADR-095). Mounted with the real
application stylesheet at 211x44.

The last test runs the whole app on a private profile: the real Settings save
writer, Providers & Models' inherited streaming row, and Ctrl+T in Console.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import toml
from rich.cells import cell_len
from textual.containers import Horizontal
from textual.widgets import Collapsible, Input, Select, Static
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen, _static_text
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_default_console_session_settings,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import (
    CONSOLE_PIN_BUILT_IN_HINT,
    MODEL_PROFILE_STREAMING_SELECT_OPTIONS,
)
from tldw_chatbook.UI.Settings_Modules.settings_field_rows import (
    UNSET_FALLBACK_HELP,
    sampling_state,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    BLANK_CHOICE_PROMPT,
    CORE_FIELDS,
    SAMPLING_FIELDS,
)

_SIZE = (211, 44)
CONSOLE_BEHAVIOR = SettingsCategoryId.CONSOLE_BEHAVIOR
_KEY = "sk-proj-abcdefghijklmnop1234"
_ENUM_FIELDS = ("reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort")


def _app(chat_defaults=None):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "openai",
        "model": "gpt-4.1",
        **(chat_defaults or {}),
    }
    return app


async def _open(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "console-behavior")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return _active_destination_screen(host)


def _cid(name: str) -> str:
    return "settings-console-default-" + name.replace("_", "-")


def _text(screen, selector: str) -> str:
    return _static_text(screen.query_one(selector, Static))


def _row_copy(screen, name: str) -> tuple[str, str]:
    return _text(screen, f"#{_cid(name)}-source"), _text(screen, f"#{_cid(name)}-help")


async def _wait_until(pilot, predicate, what: str, *, attempts: int = 250) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def _choose_streaming(pilot, select: Select, value: str) -> None:
    """Pick On ("true") or Off ("false") with the keyboard, as a user does."""
    select.focus()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.press("home" if value == "true" else "end", "enter")
    await _wait_until(pilot, lambda: select.value == value, f"streaming = {value}")
    await pilot.pause()


async def _revert(host, pilot, screen) -> None:
    """Press r with no field focused and confirm the discard dialog."""
    screen.set_focus(None)
    await pilot.pause()
    await pilot.press("r")
    await _wait_until(
        pilot, lambda: bool(host.screen.query("#confirm-button")), "the revert dialog"
    )
    await pilot.click("#confirm-button")
    await pilot.pause()


def _guide_row(screen, index: int) -> str:
    """One painted row of the Inspector's Focused field guide."""
    return _text(screen, f"#settings-console-behavior-field-guide-{index}")


def _sampling_title(screen) -> str:
    return str(screen.query_one("#settings-console-sampling", Collapsible).title)


def _dirty(screen) -> set[str]:
    draft = screen._settings_drafts.get(CONSOLE_BEHAVIOR)
    return set(draft.dirty_keys) if draft is not None else set()


@pytest.mark.asyncio
@private_profile_test
async def test_fallbacks_use_model_defaults_rows_core_first_then_closed_sampling(
    request,
):
    """AC#1/AC#2: Temperature, Max tokens, Streaming, then reasoning and
    thinking, each a label, a one-row control, a Source word and a help line;
    the six samplers sit in one closed one-row Sampling disclosure; streaming
    is an On/Off Select of the model default's family."""
    # The shipped template's own samplers (config.py [chat_defaults]).
    host = _SettingsCssHarness(
        _app({"top_p": 0.95, "min_p": 0.05, "top_k": 50}), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        fallbacks = screen.query_one("#settings-console-fallbacks")
        rows = [child for child in fallbacks.children if isinstance(child, Horizontal)]
        assert [row.id for row in rows] == [f"{_cid(name)}-row" for name in CORE_FIELDS]
        # The reasoning note heads the reasoning rows, not Temperature.
        children = list(fallbacks.children)
        note = screen.query_one("#settings-console-reasoning-help", Static)
        assert children.index(note) == children.index(rows[2]) + 1
        assert (
            children[children.index(note) + 1].id == f"{_cid('reasoning_effort')}-row"
        )
        # A blank enum shows the word Chat settings shows for the same state
        # ("default": nothing is sent), painted whole in the card's control
        # column (review note 9 widened it from Model defaults' 16 cells).
        for name in _ENUM_FIELDS:
            select = screen.query_one(f"#{_cid(name)}", Select)
            assert select.value is Select.NULL
            assert select.prompt == BLANK_CHOICE_PROMPT == "default"
            shown = select.query_one("SelectCurrent #label", Static)
            assert _static_text(shown) == BLANK_CHOICE_PROMPT
            assert shown.region.height == 1, (name, shown.region)
            assert shown.region.width >= cell_len(BLANK_CHOICE_PROMPT), (
                name,
                shown.region,
            )
        for name, row in zip(CORE_FIELDS, rows):
            label, control, source, help_line = row.children
            assert _static_text(label) == MODEL_FIELD_LABELS[name]
            assert label.has_class("settings-input-label")
            assert control.id == _cid(name)
            assert source.id == f"{_cid(name)}-source" and _static_text(source)
            assert help_line.id == f"{_cid(name)}-help" and _static_text(help_line)
            assert row.region.height == 1, (name, row.region)
            assert control.region.height == 1, (name, control.region)

        sampling = screen.query_one("#settings-console-sampling", Collapsible)
        assert list(fallbacks.children)[-1] is sampling
        assert sampling.collapsed is True
        assert sampling.region.height == 1
        assert sampling.query_one(CollapsibleTitle).region.height == 1
        # Every set sampler is named while the title fits its one row.
        assert str(sampling.title) == "Sampling · Top P 0.95 · Min P 0.05 · Top K 50"
        inner = [
            child
            for child in sampling.query_one("Contents").children
            if isinstance(child, Horizontal)
        ]
        assert [row.id for row in inner] == [
            f"{_cid(name)}-row" for name in SAMPLING_FIELDS
        ]

        # The model default's Select offers the same two options plus a
        # blank "Inherit"; the global fallback has nothing to inherit from.
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert tuple(map(tuple, streaming._options)) == (
            MODEL_PROFILE_STREAMING_SELECT_OPTIONS
        )
        assert streaming.value == "true"
        assert streaming.has_class("settings-compact-select")
        assert streaming.parent.has_class("settings-select-row")

        # '/' still reaches a sampler: landing opens the closed disclosure.
        screen._land_search_focus_on_field(_cid("seed"), "Seed")
        await pilot.pause()
        await pilot.pause()
        assert sampling.collapsed is False
        assert screen.app.focused is screen.query_one(f"#{_cid('seed')}", Input)


@pytest.mark.asyncio
@private_profile_test
async def test_each_fallback_says_its_source_and_a_blank_says_what_it_sends(request):
    """AC#1 (spec §6 grammar): a saved fallback reads "Console Behavior" and
    the field's help, an edited one "edited *", and a blank optional one says
    the provider decides; the Sampling title follows the fields."""
    host = _SettingsCssHarness(
        _app({"temperature": 0.4, "top_p": 0.9, "streaming": False}), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert _row_copy(screen, "temperature") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert _row_copy(screen, "streaming") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "false"
        assert _row_copy(screen, "seed") == ("provider", "blank = provider default")
        assert _row_copy(screen, "max_tokens") == (
            "provider",
            "blank = provider default",
        )

        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press("end", *["backspace"] * 4, *"0.8")
        sampling = screen.query_one("#settings-console-sampling", Collapsible)
        sampling.collapsed = False
        await pilot.pause()
        seed = screen.query_one(f"#{_cid('seed')}", Input)
        seed.focus()
        await pilot.press(*"7")
        await pilot.pause()

        assert _row_copy(screen, "temperature") == (
            "edited *",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert _row_copy(screen, "seed") == (
            "edited *",
            MODEL_CONFIG_FIELDS["seed"].help,
        )
        assert str(sampling.title) == "Sampling · Top P 0.9 · Seed 7"


@pytest.mark.asyncio
@private_profile_test
async def test_legacy_enable_streaming_is_still_read_and_shown(request):
    """AC#3: with only the legacy enable_streaming key the Select still shows
    its value (chat_defaults.streaming is read first when present)."""
    host = _SettingsCssHarness(_app({"enable_streaming": False}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "false"
        # The legacy key is [chat_defaults]' value, so the row may say so.
        assert _row_copy(screen, "streaming")[0] == "Console Behavior"


def test_sampling_state_names_what_fits_and_counts_the_rest():
    """The closed title names every set sampler while it fits ``cells`` and
    counts them otherwise; Console Behavior, where nothing inherits, has its
    own word for none."""
    values = {"top_p": "0.95", "min_p": "0.05", "top_k": "50", "seed": ""}
    named = "Top P 0.95 · Min P 0.05 · Top K 50"
    assert sampling_state(values, cells=cell_len(f"Sampling · {named}")) == named
    assert sampling_state(values, cells=cell_len(f"Sampling · {named}") - 1) == "3 set"
    # Model defaults keeps its two-name cap: its title also names hidden fields.
    assert sampling_state(values) == "3 set"
    assert sampling_state({"top_p": "0.9", "seed": "7"}) == "Top P 0.9 · Seed 7"
    assert sampling_state({"top_p": "", "seed": ""}) == "all inherit"
    assert sampling_state({"top_p": "", "seed": ""}, unset="none set") == "none set"
    assert sampling_state({}) == ""


@pytest.mark.asyncio
@private_profile_test
async def test_a_value_chat_defaults_does_not_hold_reads_built_in_like_model_defaults(
    request,
):
    """Review I1: with no streaming, temperature or top_p key the controls
    show tldw's own values, so the rows say "built-in" (never "Console
    Behavior") and that a provider's own setting comes first; Model defaults'
    inherit lines name the same layers for the same keys."""
    app = _app()
    # As the shipped template does for OpenAI: the provider table sets Off.
    app.app_config["api_settings"] = {"openai": {"api_key": _KEY, "streaming": False}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "true"
        assert screen.query_one(f"#{_cid('temperature')}", Input).value == "0.7"
        assert screen.query_one(f"#{_cid('top_p')}", Input).value == "0.95"
        for name in ("streaming", "temperature", "top_p"):
            assert _row_copy(screen, name) == ("built-in", UNSET_FALLBACK_HELP), name
        assert not _dirty(screen)
        # Review round 2 (3): the built-in Top P is shown, not set, so the
        # closed title does not count it.
        assert _sampling_title(screen) == "Sampling · none set"

        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#settings-model-profile-streaming-help")),
            "Model defaults",
        )
        await pilot.pause()
        await pilot.pause()
        inherits = {
            name: _text(
                screen, f"#settings-model-profile-{name.replace('_', '-')}-help"
            )
            for name in ("streaming", "temperature", "top_p")
        }
        # The provider's Off outranks the built-in On; where nothing is set
        # both surfaces show the same built-in value. The help names only the
        # value: the Source column already names the layer (checkpoint review).
        assert inherits == {
            "streaming": "inherits Off",
            "temperature": "inherits 0.7",
            "top_p": "inherits 0.95",
        }


@pytest.mark.asyncio
@private_profile_test
async def test_an_unset_streaming_fallback_can_be_pinned_on_and_reverted(request):
    """Review I1's decision: nothing is saved for streaming, so the control
    shows the built-in On; choosing Off and then On again is still an edit
    Save writes (it pins On over a provider's Off), and Revert clears it."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert _row_copy(screen, "streaming")[0] == "built-in"
        assert not _dirty(screen)
        # Review round 2 (5): while the row reads "built-in" its focused
        # guide says how to keep the value shown; an edit drops the hint.
        valid = MODEL_CONFIG_FIELDS["streaming"].valid_range
        streaming.focus()
        await pilot.pause()
        await pilot.pause()
        assert _guide_row(screen, 2) == (
            f"Validation: {valid}; {CONSOLE_PIN_BUILT_IN_HINT}"
        )

        await _choose_streaming(pilot, streaming, "false")
        await _choose_streaming(pilot, streaming, "true")

        assert _guide_row(screen, 2) == f"Validation: {valid}"
        assert _dirty(screen) == {"streaming"}
        assert screen._settings_drafts[CONSOLE_BEHAVIOR].values["streaming"] is True
        assert _row_copy(screen, "streaming") == (
            "edited *",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )

        await _revert(host, pilot, screen)
        await _wait_until(pilot, lambda: not _dirty(screen), "the revert")
        await pilot.pause()
        assert streaming.value == "true"
        assert _row_copy(screen, "streaming") == ("built-in", UNSET_FALLBACK_HELP)
        assert not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR)


@pytest.mark.asyncio
@private_profile_test
async def test_reverting_streaming_restores_the_saved_value_word_and_a_clean_draft(
    request,
):
    """Review M7(b): Off over a saved On, then r -- the Select shows On again,
    the row says "Console Behavior" and nothing is left staged."""
    host = _SettingsCssHarness(_app({"streaming": True}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert _row_copy(screen, "streaming")[0] == "Console Behavior"

        await _choose_streaming(pilot, streaming, "false")
        assert _dirty(screen) == {"streaming"}
        assert _row_copy(screen, "streaming")[0] == "edited *"

        await _revert(host, pilot, screen)
        await _wait_until(pilot, lambda: streaming.value == "true", "the revert")
        await pilot.pause()
        await pilot.pause()
        assert _row_copy(screen, "streaming") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )
        assert not _dirty(screen)
        assert not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR)


@pytest.mark.asyncio
@private_profile_test
async def test_a_cleared_required_fallback_says_it_needs_a_value_and_save_agrees(
    request,
):
    """Review I2: Temperature and Top P cannot be blank here, so a cleared row
    says so with its range (never "blank = provider default"), Save refuses
    with the same range, and the Sampling title does not say "inherit".

    Review round 2 (1): the words are Chat settings' own ("Required: 0.0 to
    2.0."), and the range is the field table's, so the row and the Focused
    field guide beside it spell it one way."""
    host = _SettingsCssHarness(_app({"temperature": 0.4, "top_p": 0.9}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.pause()
        await pilot.pause()
        # A value [chat_defaults] holds gets no pin hint in its guide.
        assert _guide_row(screen, 2) == "Validation: 0.0 to 2.0"
        await pilot.press("end", *["backspace"] * 4)
        await pilot.pause()
        assert temperature.value == ""
        assert _row_copy(screen, "temperature") == (
            "edited *",
            "Required: 0.0 to 2.0.",
        )
        assert _guide_row(screen, 2) == "Validation: 0.0 to 2.0"

        sampling = screen.query_one("#settings-console-sampling", Collapsible)
        sampling.collapsed = False
        await pilot.pause()
        top_p = screen.query_one(f"#{_cid('top_p')}", Input)
        top_p.focus()
        await pilot.press("end", *["backspace"] * 4)
        await pilot.pause()
        assert _row_copy(screen, "top_p") == (
            "edited *",
            f"Required: {MODEL_CONFIG_FIELDS['top_p'].valid_range}.",
        )
        assert str(sampling.title) == "Sampling · none set"
        # An optional field keeps the provider-decides line.
        assert _row_copy(screen, "seed") == ("provider", "blank = provider default")

        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: (
                "must be between" in _text(screen, "#settings-console-behavior-result")
            ),
            "Save to refuse the blank",
        )
        assert _text(screen, "#settings-console-behavior-result") == (
            "Temperature must be between 0.0 and 2.0."
        )
        assert _dirty(screen) == {"temperature", "top_p"}


@pytest.mark.asyncio
@private_profile_test
async def test_a_hand_edited_fallback_shows_the_value_and_word_a_new_chat_resolves(
    request,
):
    """Review round 2 (4): a row's value is read with the default chain's own
    coercion, as its Source word is, so a hand-edited value only one of the
    two used to accept cannot split them.

    ``streaming = 0`` is not a boolean the chain reads: the row shows the
    built-in On as "built-in" (it read "Off | built-in"). An out-of-range
    Temperature is what a new chat gets, so the row shows it (it read "0.7 |
    Console Behavior"). A fractional Top K is unusable, so its row is blank
    (it read "50 | built-in")."""
    app = _app({"streaming": 0, "temperature": 3.0, "top_p": 1.5, "top_k": 50.7})
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        # What [chat_defaults] and the built-in values alone resolve to.
        resolved = build_default_console_session_settings(
            {"chat_defaults": dict(app.app_config["chat_defaults"])}
        )
        assert (resolved.streaming, resolved.temperature) == (True, 3.0)
        assert (resolved.top_p, resolved.top_k) == (1.5, None)

        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "true"
        assert _row_copy(screen, "streaming") == ("built-in", UNSET_FALLBACK_HELP)
        assert screen.query_one(f"#{_cid('temperature')}", Input).value == "3.0"
        assert _row_copy(screen, "temperature") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert screen.query_one(f"#{_cid('top_p')}", Input).value == "1.5"
        assert _row_copy(screen, "top_p")[0] == "Console Behavior"
        assert screen.query_one(f"#{_cid('top_k')}", Input).value == ""
        assert _row_copy(screen, "top_k") == ("provider", "blank = provider default")
        assert _sampling_title(screen) == "Sampling · Top P 1.5"
        # Showing an out-of-range value stages nothing.
        assert not _dirty(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_a_hand_edited_choice_with_no_option_reads_saved_and_names_its_value(
    request,
):
    """Review round 3 (1): reasoning and thinking choices are read as a new
    chat reads them, without folding case. A saved "High" is no option, so
    the Select stays blank (it showed "high", which a new chat never gets),
    yet the row says "Console Behavior" and names what a new chat takes, as
    "bogus" does (it read "provider"). Choosing an option is an edit; Revert
    brings back the blank, held row with nothing staged."""
    app = _app({"reasoning_effort": "High", "verbosity": "bogus"})
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        resolved = build_default_console_session_settings(
            {"chat_defaults": dict(app.app_config["chat_defaults"])}
        )
        assert (resolved.reasoning_effort, resolved.verbosity) == ("High", "bogus")
        effort = screen.query_one(f"#{_cid('reasoning_effort')}", Select)
        assert effort.value is Select.NULL
        assert screen.query_one(f"#{_cid('verbosity')}", Select).value is Select.NULL
        assert _row_copy(screen, "reasoning_effort") == (
            "Console Behavior",
            "saved 'High' is not a choice",
        )
        assert _row_copy(screen, "verbosity") == (
            "Console Behavior",
            "saved 'bogus' is not a choice",
        )
        assert _row_copy(screen, "thinking_effort") == (
            "provider",
            "blank = provider default",
        )
        assert not _dirty(screen)

        effort.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("end", "up", "enter")
        await _wait_until(pilot, lambda: effort.value == "high", "reasoning = high")
        await pilot.pause()
        assert _dirty(screen) == {"reasoning_effort"}
        assert _row_copy(screen, "reasoning_effort")[0] == "edited *"

        await _revert(host, pilot, screen)
        await _wait_until(pilot, lambda: effort.value is Select.NULL, "the revert")
        await pilot.pause()
        await pilot.pause()
        assert _row_copy(screen, "reasoning_effort") == (
            "Console Behavior",
            "saved 'High' is not a choice",
        )
        assert not _dirty(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_default_after_an_option_stops_sending_a_choice_with_no_option(
    request,
):
    """Review round 5 (2), with the real writer on a private profile: a blank
    Select over a saved "bogus" is no edit at rest, but a blank chosen after
    an option is "default". It reads "edited *" with "blank = provider
    default", and Save writes it, so a new chat sends nothing."""
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    on_disk = toml.loads(config_path.read_text())
    on_disk.setdefault("chat_defaults", {})["verbosity"] = "bogus"
    config_path.write_text(toml.dumps(on_disk))
    host = _SettingsCssHarness(_app({"verbosity": "bogus"}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        verbosity = screen.query_one(f"#{_cid('verbosity')}", Select)
        assert verbosity.value is Select.NULL
        assert _row_copy(screen, "verbosity")[1] == "saved 'bogus' is not a choice"
        assert not _dirty(screen)

        verbosity.value = "low"
        await _wait_until(pilot, lambda: bool(_dirty(screen)), "the staged choice")
        verbosity.value = Select.NULL
        await _wait_until(
            pilot,
            lambda: (
                _row_copy(screen, "verbosity")[0] == "edited *"
                and verbosity.value is Select.NULL
            ),
            "the staged default",
        )
        await pilot.pause()
        assert _dirty(screen) == {"verbosity"}
        assert _row_copy(screen, "verbosity") == (
            "edited *",
            "blank = provider default",
        )
        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR),
            "the save",
        )
        await host.workers.wait_for_complete()

    saved = toml.loads(config_path.read_text())["chat_defaults"]
    assert saved["verbosity"] == ""
    resolved = build_default_console_session_settings({"chat_defaults": saved})
    assert resolved.verbosity is None


@pytest.mark.asyncio
@private_profile_test
async def test_a_shown_negative_integer_can_be_backspaced_away(request):
    """Review round 3 (2): a hand-edited ``seed = -1`` is shown as a new chat
    reads it, so its Input must take the edit that clears it. Backspace from
    the end leaves "-" (a digits-only restrict refused that, so nothing
    happened), then blank."""
    host = _SettingsCssHarness(_app({"seed": -1}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        screen.query_one("#settings-console-sampling", Collapsible).collapsed = False
        await pilot.pause()
        seed = screen.query_one(f"#{_cid('seed')}", Input)
        assert seed.value == "-1"
        assert _row_copy(screen, "seed")[0] == "Console Behavior"
        assert not _dirty(screen)

        seed.focus()
        await pilot.press("end", "backspace")
        await pilot.pause()
        assert seed.value == "-"
        await pilot.press("backspace")
        await pilot.pause()
        assert seed.value == ""
        assert _row_copy(screen, "seed") == ("edited *", "blank = provider default")


@pytest.mark.asyncio
@private_profile_test
async def test_the_streaming_key_fact_lives_in_the_inspector_config_key_disclosure(
    request,
):
    """AC#4: the raw 'chat_defaults.streaming is canonical...' line is gone
    from Settings' own copy; the Inspector's closed "config key" disclosure
    holds it."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        detail = screen.query_one("#settings-detail-pane-body")
        assert not [
            _static_text(widget)
            for widget in detail.query(Static)
            if "enable_streaming" in _static_text(widget)
        ]
        disclosure = screen.query_one(
            "#settings-console-behavior-config-key", Collapsible
        )
        assert disclosure.collapsed is True
        assert str(disclosure.title) == "config key"
        assert disclosure.region.height == 1
        body = screen.query_one("#settings-impact-pane-body")
        holders = [
            widget
            for widget in body.query(Static)
            if "enable_streaming" in _static_text(widget)
        ]
        assert holders
        assert all(disclosure in widget.ancestors for widget in holders)
        fact = " ".join(_static_text(widget) for widget in holders).replace("\n", "")
        assert "chat_defaults.streaming" in fact

        # Review M6: the focused field's "Saved as" row sits in that same
        # disclosure, not printed above it (Providers & Models does the same).
        def saved_as() -> list[Static]:
            return [
                widget
                for widget in body.query(Static)
                if _static_text(widget).startswith("Saved as")
            ]

        def guide() -> list[str]:
            return [
                _static_text(row)
                for row in body.query(Static)
                if str(row.id or "").startswith(
                    "settings-console-behavior-field-guide-"
                )
            ]

        assert [_static_text(widget) for widget in saved_as()] == [
            "Saved as: varies by field"
        ]
        assert all(disclosure in widget.ancestors for widget in saved_as())
        screen.query_one(f"#{_cid('temperature')}", Input).focus()
        await pilot.pause()
        await pilot.pause()
        field = MODEL_CONFIG_FIELDS["temperature"]
        # Nothing is saved for Temperature here, so the guide adds the hint.
        assert guide() == [
            f"Focused setting: {field.label}",
            f"Purpose: {field.help}",
            f"Validation: {field.valid_range}; {CONSOLE_PIN_BUILT_IN_HINT}",
        ]
        assert [_static_text(widget) for widget in saved_as()] == [
            "Saved as: chat_defaults.temperature"
        ]
        assert all(disclosure in widget.ancestors for widget in saved_as())


@pytest.mark.asyncio
@private_profile_test
async def test_saving_global_streaming_off_reaches_a_new_chat_and_model_defaults(
    request, monkeypatch
):
    """AC#3/AC#6: the whole app on a private profile -- the real Settings save
    writer sets chat_defaults.streaming = false, a model default left at
    Inherit then says "inherits Off" from Console Behavior, and Ctrl+T's new chat
    resolves streaming Off."""
    from Tests.UI.test_console_session_settings import (
        _build_live_config_test_app,
        _wait_for_screen,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    adapter = SettingsConfigAdapter()
    for section, values in (
        ("splash_screen", {"enabled": False}),
        ("first_run", {"setup_completed": True}),
        (
            "chat_defaults",
            {"provider": "llama_cpp", "model": "model-a", "streaming": True},
        ),
        ("api_settings.llama_cpp", {"api_url": "http://127.0.0.1:9099"}),
    ):
        assert adapter.save_values(section, values), section
    config_module.load_settings(force_reload=True)
    app = _build_live_config_test_app()

    async with app.run_test(size=_SIZE) as pilot:
        app.providers_models = {"llama_cpp": ["model-a"]}
        app.post_message(NavigateToScreen("settings", {"category": CONSOLE_BEHAVIOR}))
        screen = await _wait_for_screen(app, pilot, "SettingsScreen")
        await _wait_for_selector(screen, pilot, f"#{_cid('streaming')}")
        await pilot.pause()
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert streaming.value == "true"
        streaming.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await _wait_until(
            pilot,
            lambda: streaming.value == "false",
            "the Select to choose Off",
        )
        assert screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR)
        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR),
            "the save to finish",
        )
        saved = config_module.load_settings(force_reload=True)["chat_defaults"]
        assert saved["streaming"] is False
        assert _row_copy(screen, "streaming")[0] == "Console Behavior"

        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(
            screen, pilot, "#settings-model-profile-streaming-help"
        )
        await pilot.pause()
        await pilot.pause()
        inherit = screen.query_one("#settings-model-profile-streaming", Select)
        assert inherit.value is Select.NULL
        assert _text(screen, "#settings-model-profile-streaming-help") == (
            "inherits Off"
        )

        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        before = store.active_session_id
        await pilot.press("ctrl+t")
        await _wait_until(
            pilot,
            lambda: store.active_session_id not in (None, before),
            "Ctrl+T to open a new chat",
        )
        assert store.session_settings(store.active_session_id).streaming is False


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "seed",
    [{"streaming": False}, {"enable_streaming": False}, {}],
    ids=["saved-off", "legacy-off-only", "unset"],
)
async def test_choosing_streaming_on_saves_true_over_a_saved_legacy_or_unset_value(
    request, monkeypatch, seed
):
    """Review M7(a) and I1's decision, with the real writer on a private
    profile: On over a saved Off, over a legacy-only enable_streaming = false
    (the written streaming = true then wins) and over nothing saved (Off,
    then On) all write chat_defaults.streaming = true."""
    from Tests.UI.test_console_session_settings import (
        _build_live_config_test_app,
        _wait_for_screen,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    adapter = SettingsConfigAdapter()
    for section, values in (
        ("splash_screen", {"enabled": False}),
        ("first_run", {"setup_completed": True}),
        ("chat_defaults", {"provider": "llama_cpp", "model": "model-a", **seed}),
        ("api_settings.llama_cpp", {"api_url": "http://127.0.0.1:9099"}),
    ):
        assert adapter.save_values(section, values), section
    before = config_module.load_settings(force_reload=True)["chat_defaults"]
    assert "streaming" in before if "streaming" in seed else "streaming" not in before
    app = _build_live_config_test_app()

    async with app.run_test(size=_SIZE) as pilot:
        app.providers_models = {"llama_cpp": ["model-a"]}
        app.post_message(NavigateToScreen("settings", {"category": CONSOLE_BEHAVIOR}))
        screen = await _wait_for_screen(app, pilot, "SettingsScreen")
        await _wait_for_selector(screen, pilot, f"#{_cid('streaming')}")
        await pilot.pause()
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert streaming.value == ("false" if seed else "true")
        assert _row_copy(screen, "streaming")[0] == (
            "Console Behavior" if seed else "built-in"
        )
        if not seed:
            await _choose_streaming(pilot, streaming, "false")
        await _choose_streaming(pilot, streaming, "true")
        assert _row_copy(screen, "streaming")[0] == "edited *"
        assert screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR)
        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR),
            "the save to finish",
        )
        saved = config_module.load_settings(force_reload=True)
        assert saved["chat_defaults"]["streaming"] is True
        if "enable_streaming" in seed:
            assert saved["chat_defaults"]["enable_streaming"] is False
        assert build_default_console_session_settings(saved).streaming is True
        assert streaming.value == "true"
        assert _row_copy(screen, "streaming") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )
