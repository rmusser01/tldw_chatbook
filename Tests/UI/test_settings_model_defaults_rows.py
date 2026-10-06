"""Settings ▸ Providers & Models ▸ Model defaults: one-row field-truth rows (TASK-33007.5).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at; the painted screen, not only widget values, is read
where the AC is about what the user sees.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import toml
from textual.containers import Horizontal
from textual.widgets import Collapsible, Input, Select, Static
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_console_fallback_rows import _revert, _wait_until
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_default_console_session_settings,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import MODEL_PROFILE_INPUT_PLACEHOLDERS
from tldw_chatbook.UI.Settings_Modules.settings_field_rows import (
    inherited_values,
    row_copy,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CORE_FIELDS,
    SAMPLING_FIELDS,
)

_SIZE = (211, 44)
_KEY = "sk-ant-api03-abcdefghijklmnop1234"
_ANTHROPIC_HIDDEN = (
    "Min P",
    "Seed",
    "Presence penalty",
    "Frequency penalty",
    "Reasoning effort",
    "Reasoning summary",
    "Verbosity",
)


def _app(provider: str, model: str, *, profiles=None, chat_defaults=None):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": provider,
        "model": model,
        **(chat_defaults or {}),
    }
    app.app_config["api_settings"] = {
        "anthropic": {"api_key": _KEY},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    if profiles:
        app.app_config["api_settings"][provider]["model_defaults"] = profiles
    return app


async def _open(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "providers-models")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return _active_destination_screen(host)


def _cid(name: str) -> str:
    return "settings-model-profile-" + name.replace("_", "-")


def _text(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).renderable)


def _painted(screen, x: int, y: int):
    """The glyph and colours the compositor painted at ``(x, y)``."""
    position = 0
    for segment in screen._compositor.render_strips()[y]:
        if position + len(segment.text) > x:
            return (
                segment.text[x - position],
                segment.style.color,
                segment.style.bgcolor,
            )
        position += len(segment.text)
    raise AssertionError(f"({x}, {y}) is off screen")


@pytest.mark.asyncio
@private_profile_test
async def test_model_defaults_follow_default_model_open_and_name_the_pair(request):
    """AC#1: directly after Default model, open, titled with the pair; a model
    change re-titles it and shows that model's values."""
    profiles = {"claude-a": {"temperature": 0.3}, "claude-b": {"temperature": 0.9}}
    host = _SettingsCssHarness(
        _app("anthropic", "claude-a", profiles=profiles), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        card = screen.query_one("#settings-providers-models-card")
        defaults = screen.query_one("#settings-generation-defaults", Collapsible)
        assert defaults.collapsed is False
        assert str(defaults.title) == "Model defaults · Anthropic · claude-a"
        children = list(card.children)
        applies = screen.query_one("#settings-model-applies-row")
        assert children[children.index(applies) + 1] is defaults
        title = defaults.query_one(CollapsibleTitle)
        assert title.region.height == 1
        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        assert temperature.value == "0.3"

        screen._set_model_field_value("claude-b", quiet=False)
        await pilot.pause()
        await pilot.pause()

        assert str(defaults.title) == "Model defaults · Anthropic · claude-b"
        assert temperature.value == "0.9"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "model", "shown_core"),
    [
        (
            "anthropic",
            "claude-sonnet-4-5",
            (
                "temperature",
                "max_tokens",
                "streaming",
                "thinking_effort",
                "thinking_budget_tokens",
            ),
        ),
        (
            "llama_cpp",
            "qwen",
            (
                "temperature",
                "max_tokens",
                "streaming",
                "reasoning_effort",
                "thinking_budget_tokens",
            ),
        ),
    ],
)
async def test_core_rows_come_first_each_one_row_with_source_and_help(
    request, provider, model, shown_core
):
    """AC#2: Temperature, Max tokens, Streaming, then only the reasoning and
    thinking controls the provider accepts; each row is a label, a one-row
    control, a Source word and one help line; Sampling follows, closed."""
    host = _SettingsCssHarness(_app(provider, model), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        defaults = screen.query_one("#settings-generation-defaults", Collapsible)
        rows = [
            child
            for child in defaults.query_one("Contents").children
            if isinstance(child, Horizontal)
            and child.display
            and not child.has_class("settings-gated-profile-hidden")
        ]
        assert [row.id for row in rows] == [f"{_cid(name)}-row" for name in shown_core]
        for name, row in zip(shown_core, rows):
            label, control, source, help_line = row.children
            assert str(label.renderable) == MODEL_FIELD_LABELS[name]
            assert control.id == _cid(name)
            assert source.id == f"{_cid(name)}-source" and str(source.renderable)
            assert help_line.id == f"{_cid(name)}-help" and str(help_line.renderable)
            assert row.region.height == 1, (name, row.region)
            assert control.region.height == 1, (name, control.region)
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert {
            value for _label, value in streaming._options if value is not Select.NULL
        } == {
            "true",
            "false",
        }
        assert streaming.prompt == "Inherit"
        sampling = screen.query_one("#settings-model-sampling", Collapsible)
        children = list(defaults.query_one("Contents").children)
        assert children.index(sampling) > children.index(rows[-1])
        assert sampling.collapsed is True
        assert sampling.region.height == 1
        for name in set(CORE_FIELDS) - set(shown_core):
            row = screen.query_one(f"#{_cid(name)}-row")
            assert row.has_class("settings-gated-profile-hidden"), name
            assert screen.query_one(f"#{_cid(name)}").disabled, name


@pytest.mark.asyncio
@private_profile_test
async def test_every_select_row_is_one_row_on_both_cards(request):
    """AC#3: every Select row on Providers & Models and Console Behavior paints
    one row tall at 211x44; the three-row select-row height no longer applies."""
    host = _SettingsCssHarness(_app("anthropic", "claude-sonnet-4-5"), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        checked = 0
        for card_id, category in (
            ("#settings-providers-models-card", None),
            ("#settings-console-behavior-card", SettingsCategoryId.CONSOLE_BEHAVIOR),
        ):
            if category is not None:
                screen._select_category(category.value)
                await host.workers.wait_for_complete()
                await pilot.pause()
            card = screen.query_one(card_id)
            for row in card.query(".settings-select-row"):
                if not row.display or row.has_class("settings-gated-profile-hidden"):
                    continue
                for select in row.query(Select):
                    assert select.region.height == 1, (
                        card_id,
                        select.id,
                        select.region,
                    )
                assert row.region.height == 1, (card_id, row.id, row.region)
                checked += 1
        assert checked >= 15, checked


@pytest.mark.asyncio
@private_profile_test
async def test_a_blank_field_says_what_it_inherits_and_from_where(request):
    """AC#4: an empty field shows the inherited value and its layer; a set one
    says model default; placeholders state only a range or unit."""
    host = _SettingsCssHarness(
        _app(
            "anthropic",
            "claude-sonnet-4-5",
            profiles={"claude-sonnet-4-5": {"max_tokens": 8192}},
            chat_defaults={"temperature": 1.0, "streaming": False},
        ),
        "settings",
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)

        def row(name):
            return (
                _text(screen, f"#{_cid(name)}-source"),
                _text(screen, f"#{_cid(name)}-help"),
            )

        assert screen.query_one(f"#{_cid('temperature')}", Input).value == ""
        assert row("temperature") == (
            "Console Behavior",
            "inherits 1.0 · Console Behavior",
        )
        assert row("streaming") == (
            "Console Behavior",
            "inherits Off · Console Behavior",
        )
        assert row("max_tokens") == (
            "model default",
            MODEL_CONFIG_FIELDS["max_tokens"].help,
        )
        assert row("top_p") == ("built-in", "inherits 0.95 · built-in")
        assert row("thinking_budget_tokens") == ("provider", "blank = provider default")
        for name in (
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "thinking_budget_tokens",
        ):
            placeholder = screen.query_one(f"#{_cid(name)}", Input).placeholder
            assert (
                placeholder == MODEL_PROFILE_INPUT_PLACEHOLDERS[f"model_profile_{name}"]
            )
            assert "optional" not in placeholder and "inherit" not in placeholder

        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press(*"0.4")
        max_tokens = screen.query_one(f"#{_cid('max_tokens')}", Input)
        max_tokens.focus()
        await pilot.press("end", *["backspace"] * 8)
        await pilot.pause()

        assert row("temperature") == (
            "edited *",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert row("max_tokens") == ("edited *", "blank = provider default")


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("theme", ["agentic_terminal", "textual-light"])
async def test_help_and_source_text_clear_aa_against_the_detail_pane(request, theme):
    """AC#5: Source words and help lines paint at >= 4.5:1 on the pane."""
    from textual.color import Color

    from tldw_chatbook.css.Themes.themes import _contrast_ratio, agentic_terminal_theme

    app = _app("anthropic", "claude-sonnet-4-5", chat_defaults={"temperature": 1.0})
    host = _SettingsCssHarness(app, "settings")
    host.register_theme(agentic_terminal_theme)

    async with host.run_test(size=_SIZE) as pilot:
        host.theme = theme
        screen = await _open(host, pilot)
        await pilot.pause()
        checked = 0
        for name in ("temperature", "max_tokens", "streaming"):
            for part in ("source", "help"):
                widget = screen.query_one(f"#{_cid(name)}-{part}", Static)
                text = str(widget.renderable)
                glyph, fg, bg = _painted(
                    screen, widget.content_region.x, widget.content_region.y
                )
                assert glyph == text[0], (name, part, glyph, text)
                ratio = _contrast_ratio(
                    Color.from_rich_color(fg), Color.from_rich_color(bg)
                )
                assert ratio >= 4.5, (theme, name, part, ratio)
                checked += 1
        assert checked == 6


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("size", [(211, 44), (235, 52)], ids=["211x44", "235x52"])
async def test_sampling_is_one_closed_row_naming_what_anthropic_does_not_accept(
    request, size
):
    """AC#6: Top P, Top K, Min P, Seed, Presence and Frequency sit in one closed
    one-row Sampling disclosure whose title says their state; the fields the
    provider does not accept are hidden and named -- in the title when they
    fit, else counted there and listed inside (owner ruling 2026-10-02, the
    same line Chat settings prints)."""
    host = _SettingsCssHarness(
        _app(
            "anthropic",
            "claude-sonnet-4-5",
            profiles={"claude-sonnet-4-5": {"top_p": 0.9}},
        ),
        "settings",
    )

    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        sampling = screen.query_one("#settings-model-sampling", Collapsible)
        title = str(sampling.title)
        assert sampling.collapsed is True
        assert sampling.region.height == 1
        assert sampling.query_one(CollapsibleTitle).region.height == 1
        assert title == (
            "Sampling · Top P 0.9 · Anthropic does not accept 7 fields "
            "(open to list them)"
        )
        assert {
            name
            for name in SAMPLING_FIELDS
            if screen.query_one(f"#{_cid(name)}-row").has_class(
                "settings-gated-profile-hidden"
            )
        } == {"min_p", "seed", "presence_penalty", "frequency_penalty"}

        sampling.collapsed = False
        await pilot.pause()
        hidden_list = screen.query_one("#settings-provider-generation-support", Static)
        assert str(hidden_list.renderable) == (
            f"Anthropic does not accept: {', '.join(_ANTHROPIC_HIDDEN)}."
        )
        assert hidden_list.region.height >= 1
        for name in ("top_p", "top_k"):
            assert screen.query_one(f"#{_cid(name)}-row").region.height == 1, name


@pytest.mark.asyncio
@private_profile_test
async def test_saving_an_edit_and_a_blank_changes_exactly_those_config_keys(request):
    """AC#7: the real save path on a private profile: one edited field and one
    blanked field; on disk exactly those two keys change, and the blank one is
    deleted so the lower layers apply."""
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    on_disk = toml.loads(config_path.read_text())
    on_disk.setdefault("chat_defaults", {}).update(
        {"provider": "llama_cpp", "model": "qwen"}
    )
    llama = on_disk.setdefault("api_settings", {}).setdefault("llama_cpp", {})
    llama["api_url"] = "http://127.0.0.1:9099"
    llama["model_defaults"] = {
        "qwen": {"temperature": 0.5, "max_tokens": 2048, "seed": 7},
        "other": {"temperature": 0.2},
    }
    config_path.write_text(toml.dumps(on_disk))
    before = toml.loads(config_path.read_text())
    app = _build_test_app()
    app.app_config["chat_defaults"] = dict(before["chat_defaults"])
    app.app_config["api_settings"] = {
        "llama_cpp": dict(before["api_settings"]["llama_cpp"])
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press("end", *["backspace"] * 8, *"0.8")
        max_tokens = screen.query_one(f"#{_cid('max_tokens')}", Input)
        max_tokens.focus()
        await pilot.press("end", *["backspace"] * 8, "escape", "s")
        for _ in range(100):
            await pilot.pause(0.05)
            if not screen._category_has_unsaved_changes(
                SettingsCategoryId.PROVIDERS_MODELS
            ):
                break
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.PROVIDERS_MODELS
        )

    after = toml.loads(config_path.read_text())
    expected = toml.loads(toml.dumps(before))
    expected["api_settings"]["llama_cpp"]["model_defaults"]["qwen"] = {
        "temperature": 0.8,
        "seed": 7,
    }
    assert after == expected


def _dirty(screen) -> set[str]:
    draft = screen._provider_draft()
    return set(draft.dirty_keys) if draft is not None else set()


def _row(screen, name: str) -> tuple[str, str]:
    return _text(screen, f"#{_cid(name)}-source"), _text(screen, f"#{_cid(name)}-help")


@pytest.mark.asyncio
@private_profile_test
async def test_a_hand_edited_choice_with_no_option_reads_saved_like_the_fallbacks(
    request,
):
    """Task 7 review round 4 (1): Model defaults reads a choice as a new chat
    does, without folding case, as Console Behavior's fallbacks do. A saved
    "High" is no option, so the Select stays blank (it showed "high", which a
    new chat never gets) and the row says "model default" and names the
    value, as "extreme" does (it read "provider"). Choosing an option is an
    edit; Revert brings back the blank row with nothing staged. Round 5 (4):
    a ``reasoning_summary = 1`` a new chat ignores is not called a saved
    choice; the row says what it inherits."""
    app = _app("openai", "gpt-4.1")
    app.app_config["api_settings"]["openai"] = {
        "model_defaults": {
            "gpt-4.1": {
                "reasoning_effort": "High",
                "verbosity": "extreme",
                "reasoning_summary": 1,
            }
        }
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        resolved = build_default_console_session_settings(
            app.app_config, "openai", "gpt-4.1"
        )
        assert (resolved.reasoning_effort, resolved.verbosity) == ("High", "extreme")
        assert resolved.reasoning_summary is None
        effort = screen.query_one(f"#{_cid('reasoning_effort')}", Select)
        assert effort.value is Select.NULL
        assert screen.query_one(f"#{_cid('verbosity')}", Select).value is Select.NULL
        assert _row(screen, "reasoning_effort") == (
            "model default",
            "saved 'High' is not a choice",
        )
        assert _row(screen, "verbosity") == (
            "model default",
            "saved 'extreme' is not a choice",
        )
        assert _row(screen, "reasoning_summary") == (
            "provider",
            "blank = provider default",
        )
        assert not _dirty(screen)

        effort.value = "high"
        await _wait_until(pilot, lambda: bool(_dirty(screen)), "the staged choice")
        await pilot.pause()
        assert _dirty(screen) == {"model_profile_reasoning_effort"}
        assert _row(screen, "reasoning_effort")[0] == "edited *"

        await _revert(host, pilot, screen)
        await _wait_until(pilot, lambda: effort.value is Select.NULL, "the revert")
        await pilot.pause()
        await pilot.pause()
        assert _row(screen, "reasoning_effort") == (
            "model default",
            "saved 'High' is not a choice",
        )
        assert not _dirty(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_saving_another_field_keeps_a_choice_the_select_cannot_show(request):
    """Task 7 review round 4 (1): Save reads Model defaults from its widgets,
    so a blank Select over a saved "High" would have deleted it on any other
    edit's save. The real save path on a private profile keeps it exactly, as
    Console Behavior's fallbacks keep theirs."""
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    on_disk = toml.loads(config_path.read_text())
    on_disk.setdefault("chat_defaults", {}).update(
        {"provider": "llama_cpp", "model": "qwen"}
    )
    llama = on_disk.setdefault("api_settings", {}).setdefault("llama_cpp", {})
    llama["api_url"] = "http://127.0.0.1:9099"
    llama["model_defaults"] = {
        "qwen": {"temperature": 0.5, "reasoning_effort": "High"},
    }
    config_path.write_text(toml.dumps(on_disk))
    before = toml.loads(config_path.read_text())
    app = _build_test_app()
    app.app_config["chat_defaults"] = dict(before["chat_defaults"])
    app.app_config["api_settings"] = {
        "llama_cpp": dict(before["api_settings"]["llama_cpp"])
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert (
            screen.query_one(f"#{_cid('reasoning_effort')}", Select).value
            is Select.NULL
        )
        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press("end", *["backspace"] * 8, *"0.8", "escape", "s")
        for _ in range(100):
            await pilot.pause(0.05)
            if not screen._category_has_unsaved_changes(
                SettingsCategoryId.PROVIDERS_MODELS
            ):
                break
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.PROVIDERS_MODELS
        )

    after = toml.loads(config_path.read_text())
    expected = toml.loads(toml.dumps(before))
    expected["api_settings"]["llama_cpp"]["model_defaults"]["qwen"] = {
        "temperature": 0.8,
        "reasoning_effort": "High",
    }
    assert after == expected


@pytest.mark.asyncio
@private_profile_test
async def test_a_shown_negative_integer_can_be_backspaced_away(request):
    """Task 7 review round 4 (2): a hand-edited ``seed = -1`` is shown as a
    new chat reads it, so its Input must take the edit that clears it, as the
    global fallbacks' Input does. Backspace from the end leaves "-" (a
    digits-only restrict refused that, so nothing happened), then blank.
    At rest it, and a refused ``temperature = 3.0``, read "model default"
    with nothing staged (the mount's own Changed staged both as edits)."""
    app = _app("llama_cpp", "qwen", profiles={"qwen": {"seed": -1, "temperature": 3.0}})
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        resolved = build_default_console_session_settings(
            app.app_config, "llama_cpp", "qwen"
        )
        assert (resolved.seed, resolved.temperature) == (-1, 3.0)
        screen.query_one("#settings-model-sampling", Collapsible).collapsed = False
        await pilot.pause()
        seed = screen.query_one(f"#{_cid('seed')}", Input)
        assert seed.value == "-1"
        assert screen.query_one(f"#{_cid('temperature')}", Input).value == "3.0"
        assert _row(screen, "seed")[0] == "model default"
        assert _row(screen, "temperature")[0] == "model default"
        assert not _dirty(screen)

        seed.focus()
        await pilot.press("end", "backspace")
        await pilot.pause()
        assert seed.value == "-"
        await pilot.press("backspace")
        await pilot.pause()
        assert seed.value == ""
        assert _row(screen, "seed") == ("edited *", "blank = provider default")


def _private_qwen_profile(qwen: dict) -> tuple[Path, dict, object]:
    """Save llama_cpp/qwen model defaults to the private profile's config.

    Returns the config path, the file as written, and an app reading it.
    """
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    on_disk = toml.loads(config_path.read_text())
    on_disk.setdefault("chat_defaults", {}).update(
        {"provider": "llama_cpp", "model": "qwen"}
    )
    llama = on_disk.setdefault("api_settings", {}).setdefault("llama_cpp", {})
    llama["api_url"] = "http://127.0.0.1:9099"
    llama["model_defaults"] = {"qwen": qwen}
    config_path.write_text(toml.dumps(on_disk))
    before = toml.loads(config_path.read_text())
    app = _build_test_app()
    app.app_config["chat_defaults"] = dict(before["chat_defaults"])
    app.app_config["api_settings"] = {
        "llama_cpp": dict(before["api_settings"]["llama_cpp"])
    }
    return config_path, before, app


async def _save_providers_models(host, pilot, screen) -> None:
    """Press s with no field focused and wait for the save to land."""
    screen.set_focus(None)
    await pilot.pause()
    await pilot.press("s")
    await _wait_until(
        pilot,
        lambda: (
            not screen._category_has_unsaved_changes(
                SettingsCategoryId.PROVIDERS_MODELS
            )
        ),
        "the save",
    )
    await host.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(("streaming", "shown"), [("0", "false"), ("1", "true")])
async def test_saving_another_field_keeps_every_untouched_hand_edit_exactly(
    request, streaming, shown
):
    """Task 7 review round 5 (1, 3, 4), with the real writer on a private
    profile. Model defaults reads each value as a new chat does: "0"/"1"
    streaming shows Off/On as a model default (it read Inherit and an
    unrelated save dropped it), and a ``top_k = 2.5`` or ``min_p = "abc"``
    a new chat ignores shows blank with what it inherits (it read "model
    default"). A refused ``seed = -1`` no longer blocks the save: editing
    Temperature saves 0.8 and leaves every untouched row exactly as saved."""
    qwen = {
        "temperature": 0.5,
        "seed": -1,
        "top_k": 2.5,
        "min_p": "abc",
        "streaming": streaming,
    }
    config_path, before, app = _private_qwen_profile(qwen)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        resolved = build_default_console_session_settings(
            app.app_config, "llama_cpp", "qwen"
        )
        inherited = inherited_values(app.app_config, "llama_cpp", "qwen")
        # A new chat skips the profile's top_k and min_p for the next layer.
        assert (resolved.seed, resolved.top_k, resolved.min_p) == (
            -1,
            inherited["top_k"][0],
            inherited["min_p"][0],
        )
        assert resolved.streaming is (streaming == "1")
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == shown
        assert _row(screen, "streaming") == (
            "model default",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )
        assert screen.query_one(f"#{_cid('seed')}", Input).value == "-1"
        assert _row(screen, "seed")[0] == "model default"
        for name in ("top_k", "min_p"):
            assert screen.query_one(f"#{_cid(name)}", Input).value == "", name
            assert _row(screen, name) == row_copy(name, "", False, inherited[name])
            assert _row(screen, name)[0] != "model default", name
        assert not _dirty(screen)

        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press("end", *["backspace"] * 8, *"0.8", "escape")
        await _save_providers_models(host, pilot, screen)

    after = toml.loads(config_path.read_text())
    expected = toml.loads(toml.dumps(before))
    expected["api_settings"]["llama_cpp"]["model_defaults"]["qwen"] = {
        **qwen,
        "temperature": 0.8,
    }
    assert after == expected


@pytest.mark.asyncio
@private_profile_test
async def test_inherit_after_an_option_clears_a_choice_the_select_cannot_show(
    request,
):
    """Task 7 review round 5 (2), with the real writer on a private profile:
    a blank Select over a saved "High" is no edit at rest, but a blank chosen
    after an option is Inherit. It reads "edited *" with what it inherits,
    and Save deletes the override."""
    config_path, before, app = _private_qwen_profile(
        {"temperature": 0.5, "reasoning_effort": "High"}
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        effort = screen.query_one(f"#{_cid('reasoning_effort')}", Select)
        assert effort.value is Select.NULL
        assert _row(screen, "reasoning_effort")[1] == "saved 'High' is not a choice"
        assert not _dirty(screen)

        effort.value = "high"
        await _wait_until(pilot, lambda: bool(_dirty(screen)), "the staged choice")
        effort.value = Select.NULL
        await _wait_until(
            pilot,
            lambda: (
                _row(screen, "reasoning_effort")[0] == "edited *"
                and effort.value is Select.NULL
            ),
            "the staged Inherit",
        )
        await pilot.pause()
        assert _dirty(screen) == {"model_profile_reasoning_effort"}
        assert _row(screen, "reasoning_effort") == (
            "edited *",
            "blank = provider default",
        )
        await _save_providers_models(host, pilot, screen)

    after = toml.loads(config_path.read_text())
    expected = toml.loads(toml.dumps(before))
    expected["api_settings"]["llama_cpp"]["model_defaults"]["qwen"] = {
        "temperature": 0.5
    }
    assert after == expected
