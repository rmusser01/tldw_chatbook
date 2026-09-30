"""Switch model: the Alt+M provider·model pair list (TASK-33004.4).

Pilot tests against the real popover on a harness app that loads the app's
own stylesheets. Rows are read the way a user sees them: the painted screen.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pytest
from textual.containers import Grid
from textual.widgets import Button, Input, OptionList, Select

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    CURRENT_MARK,
    HIGHLIGHT_GLYPH,
    ConsoleModelPopover,
    switcher_readiness_words,
)

pytestmark = pytest.mark.asyncio

READY = ConsoleSettingsReadiness("Ready", "Ready.", True)
NO_KEY = ConsoleSettingsReadiness(
    "Not ready",
    "API key missing",
    False,
    operability="not_ready",
    blocker="credential_missing",
    recovery_action="configure_credential",
    configuration="incomplete",
    configuration_issue="credential_missing",
    credential="missing",
)
READY_PROVIDERS = frozenset({"llama_cpp", "anthropic", "ollama", "openrouter"})
PROVIDERS_MODELS = {
    "Llama_cpp": ["model-a", "model-b"],
    "Anthropic": [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-1",
        "claude-3-7-sonnet",
        "claude-3-5-haiku",
    ],
    "OpenAI": ["gpt-5.1"],
    "OpenRouter": ["openai/gpt-y"],
    "local_llamacpp": ["legacy-model"],
}


@dataclass(frozen=True)
class _Use:
    provider: str
    model: str
    age: timedelta
    in_this_chat: bool = False

    def used_label(self, now: datetime) -> str:
        hours = int(self.age.total_seconds() // 3600)
        text = f"used {hours}h ago"
        return f"{text} in this chat" if self.in_this_chat else text


class SwitcherHarness(ConsolidatedCSSApp):
    """Empty app with the production stylesheets."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self) -> None:
        super().__init__()
        self.result: object = "unset"


class Recorder:
    """Everything the switcher asked its injected seams for."""

    def __init__(self, *, ready: frozenset[str] = READY_PROVIDERS) -> None:
        self.ready = ready
        self.readiness_calls: list[tuple[str, str | None, int]] = []
        self.catalog_calls: list[str] = []
        self.submissions: list[ConsoleSettingsSubmission] = []
        self.setup: list[str] = []

    def readiness(self, provider: str, model: str | None) -> ConsoleSettingsReadiness:
        self.readiness_calls.append((provider, model, threading.get_ident()))
        return READY if provider in self.ready or "*" in self.ready else NO_KEY

    def commit(
        self, submission: ConsoleSettingsSubmission
    ) -> ConsoleSettingsLiveCommit:
        self.submissions.append(submission)
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id="session-a",
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )


def _draft(
    provider: str = "llama_cpp",
    model: str | None = "model-a",
    *,
    compaction: ContextCompactionMode | None = None,
) -> ConsoleSettingsDraftState:
    settings = ConsoleSessionSettings(provider=provider, model=model, temperature=0.7)
    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(
            compaction_mode=compaction
        ),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "max_tokens", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )


def _rebase(
    state: ConsoleSettingsDraftState, **kwargs: object
) -> ConsoleSettingsDraftState:
    return replace(
        state,
        settings=replace(
            state.settings, provider=kwargs["provider"], model=kwargs["model"]
        ),
    )


def build_switcher(
    recorder: Recorder,
    *,
    draft: ConsoleSettingsDraftState | None = None,
    providers_models: dict[str, list[str]] | None = None,
    app_config: dict[str, object] | None = None,
    recent: tuple[_Use, ...] | None = None,
    remembered_previous: _Use | None = None,
    catalog_loader=None,
) -> ConsoleModelPopover:
    """One switcher with recording seams. RECENT loads after open."""

    async def load_recent():
        return recent or ()

    def previous(uses):
        if remembered_previous is not None:
            return remembered_previous
        return next((use for use in uses if use.model != "model-a"), None)

    return ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session-a", None, 0),
        app_config=app_config or {"api_settings": {}},
        initial_draft=draft or _draft(),
        providers_models=PROVIDERS_MODELS
        if providers_models is None
        else providers_models,
        scope_copy="Applies to: this chat only",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=_rebase,
        live_committer=recorder.commit,
        default_readiness_resolver=recorder.readiness,
        recent_pairs_loader=load_recent if recent is not None else None,
        previous_pair=previous,
        catalog_loader=catalog_loader,
        setup_opener=lambda provider, model: recorder.setup.append((provider, model)),
    )


async def _settle(app, pilot) -> None:
    for _ in range(3):
        await pilot.pause()
        await app.workers.wait_for_complete()
    await pilot.pause()


async def open_switcher(
    app, pilot, switcher: ConsoleModelPopover
) -> ConsoleModelPopover:
    def capture(result: object) -> None:
        app.result = result

    await app.push_screen(switcher, callback=capture)
    await _settle(app, pilot)
    return switcher


def painted_lines(app) -> list[str]:
    return [strip.text for strip in app.screen._compositor.render_strips()]


def list_lines(app, switcher: ConsoleModelPopover) -> list[str]:
    region = switcher.query_one("#console-popover-pairs", OptionList).region
    lines = painted_lines(app)[region.y : region.bottom]
    return [line[region.x : region.right].rstrip() for line in lines]


def line_with(lines: list[str], *needles: str) -> str:
    for line in lines:
        if all(needle in line for needle in needles):
            return line
    raise AssertionError(f"no line has {needles!r}:\n" + "\n".join(lines))


async def test_switcher_lists_four_groups_of_one_line_pairs() -> None:
    """AC#1, AC#3: PREVIOUS, RECENT, READY PROVIDERS (3 + more) and NEEDS SETUP.

    Every row names its model, the provider's display name (never the raw
    config key), a context size, readiness words and its last use.
    """
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                recent=(
                    _Use("anthropic", "claude-sonnet-4-5", timedelta(hours=2), True),
                    _Use("llama_cpp", "model-a", timedelta(minutes=1)),
                    _Use("openai", "gpt-5.1", timedelta(hours=72)),
                ),
            ),
        )
        lines = list_lines(app, switcher)
        headers = [
            line.strip()
            for line in lines
            if line.strip().isupper()
            or "·" in line
            and line.strip().split(" ")[0].isupper()
        ]
        order = [
            next(i for i, line in enumerate(lines) if line.startswith(title))
            for title in ("PREVIOUS", "RECENT", "READY PROVIDERS", "NEEDS SETUP")
        ]
        assert order == sorted(order), headers

        previous = line_with(lines, "claude-sonnet-4-5")
        assert "Anthropic" in previous
        assert "~200k" in previous or "200k" in previous
        assert "Ready · not tested" in previous
        assert "used 2h ago in this chat" in previous

        current = line_with(lines, "model-a")
        assert "llama.cpp" in current and "llama_cpp" not in current
        assert CURRENT_MARK in current

        # Anthropic lists three models (sonnet is PREVIOUS) plus a more row.
        anthropic_rows = [
            line
            for line in lines[order[2] : order[3]]
            if "Anthropic" in line and "Ready" in line
        ]
        assert len(anthropic_rows) == 3, anthropic_rows
        assert any("… 1 more Anthropic models" in line for line in lines)

        setup = line_with(lines, "OpenAI", "Not ready · no key")
        assert "(any model)" in setup or "gpt-5.1" in setup
        assert "Enter: add key in Settings" in line_with(
            lines[order[3] :], "Not ready · no key"
        )
        assert all("llama_cpp" not in line for line in lines)


async def test_previous_row_is_highlighted_on_open_and_enter_swaps_to_it() -> None:
    """AC#5 and AC#2: PREVIOUS is highlighted (glyph), so Enter applies that pair."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                remembered_previous=_Use(
                    "anthropic", "claude-sonnet-4-5", timedelta(hours=2), True
                ),
            ),
        )
        assert app.focused is switcher.query_one("#console-popover-find", Input)
        row = switcher.highlighted_row()
        assert row is not None and (row.provider, row.model) == (
            "anthropic",
            "claude-sonnet-4-5",
        )
        assert line_with(list_lines(app, switcher), "claude-sonnet-4-5").startswith(
            HIGHLIGHT_GLYPH
        )

        await pilot.press("enter")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    submission = recorder.submissions[0]
    assert submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submission.surface is ConsoleSettingsSurface.QUICK_POPOVER
    assert submission.default_field_mask == frozenset()
    assert (submission.draft.settings.provider, submission.draft.settings.model) == (
        "anthropic",
        "claude-sonnet-4-5",
    )


async def test_recent_fills_in_after_open_and_previous_falls_back_to_it() -> None:
    """RECENT arrives from the loader after open; with no remembered PREVIOUS
    the newest other recent pair becomes PREVIOUS and is highlighted."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                recent=(
                    _Use("llama_cpp", "model-a", timedelta(minutes=1)),
                    _Use("openai", "gpt-5.1", timedelta(hours=3)),
                ),
            ),
        )
        lines = list_lines(app, switcher)
        assert lines[0].startswith("PREVIOUS")
        assert "gpt-5.1" in lines[1] and lines[1].startswith(HIGHLIGHT_GLYPH)
        # A not-ready provider's recent pair stays honest about it.
        assert "Not ready · no key" in lines[1]


async def test_current_pair_is_marked_even_when_its_catalog_omits_it() -> None:
    """AC#4 (ADR-020): the active model is always listed and marked in text."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, draft=_draft(model="unlisted-model"))
        )
        current = line_with(list_lines(app, switcher), "unlisted-model")
        assert CURRENT_MARK in current and "llama.cpp" in current


async def test_find_filters_every_catalog_in_memory_and_highlights_the_best_match() -> (
    None
):
    """AC#6: typing narrows every provider's list with no load or readiness call."""
    loads: list[str] = []

    async def catalog(provider: str) -> list[str]:
        loads.append(provider)
        return list(PROVIDERS_MODELS.get(provider.capitalize(), ())) or [
            "catalog-only-sonnet"
        ]

    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, catalog_loader=catalog)
        )
        loads_before, readiness_before = list(loads), len(recorder.readiness_calls)
        await pilot.press(*"son")
        await _settle(app, pilot)

        lines = list_lines(app, switcher)
        assert loads == loads_before
        assert len(recorder.readiness_calls) == readiness_before
        pair_lines = [line for line in lines if "Ready" in line]
        assert pair_lines and all("son" in line for line in pair_lines), lines
        row = switcher.highlighted_row()
        assert row is not None and row.model is not None and "son" in row.model
        assert "catalog-only-sonnet" in "\n".join(lines)


async def test_filtering_a_2000_model_catalog_takes_under_50ms_per_keystroke() -> None:
    """AC#7: one keystroke against an OpenRouter-sized catalog, on the UI thread."""
    models = [f"vendor-{index % 40}/model-{index:04d}" for index in range(2_000)]
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder, providers_models={**PROVIDERS_MODELS, "OpenRouter": models}
            ),
        )
        find = switcher.query_one("#console-popover-find", Input)
        timings = []
        for query in ("m", "mo", "model-1", "model-19"):
            started = time.perf_counter()
            find.value = query  # Input.Changed runs the filter synchronously below
            switcher._find_changed(Input.Changed(find, query))
            timings.append(time.perf_counter() - started)
            await pilot.pause()
        assert max(timings) < 0.050, timings
        row = switcher.highlighted_row()
        assert row is not None and row.model is not None and "model-19" in row.model


async def test_a_typed_model_id_applies_as_a_pair_and_bad_ids_never_do() -> None:
    """AC#8: an id no catalog lists applies with the chat's provider, and only
    as bounded single-line text (TASK-14812 AC#5, AC#7)."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        find = switcher.query_one("#console-popover-find", Input)
        find.value = "x" * 300
        await pilot.pause()
        assert all(row.kind != "typed" for row in switcher._rows)

        find.value = "private/model-id"
        await pilot.pause()
        typed = [row for row in switcher._rows if row.kind == "typed"]
        assert [(row.provider, row.model) for row in typed] == [
            ("llama_cpp", "private/model-id")
        ]
        assert "TYPED MODEL ID" in "\n".join(list_lines(app, switcher))
        await pilot.press("enter")
        await pilot.pause()

    assert recorder.submissions[0].draft.settings.model == "private/model-id"
    assert recorder.submissions[0].draft.settings.provider == "llama_cpp"


async def test_loading_empty_and_unavailable_catalogs_each_show_a_row() -> None:
    """AC#9: a provider never silently shows no rows."""
    release = asyncio.Event()

    async def catalog(provider: str) -> list[str]:
        if provider == "ollama":
            await release.wait()
            return []
        if provider == "openrouter":
            raise OSError("catalog store unreadable")
        return list(PROVIDERS_MODELS.get(provider.capitalize(), ()))

    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build_switcher(
            recorder,
            providers_models={**PROVIDERS_MODELS, "OpenRouter": []},
            catalog_loader=catalog,
        )
        app.push_screen(switcher)
        for _ in range(20):
            await pilot.pause()
            if "loading models…" in "\n".join(list_lines(app, switcher)):
                break
        assert "Ollama · loading models…" in "\n".join(list_lines(app, switcher))
        release.set()
        await _settle(app, pilot)
        text = "\n".join(list_lines(app, switcher))
        assert "Ollama · no models reported · type its name and an id" in text
        assert "OpenRouter · catalog unavailable · type its name and an id" in text

        # ...and typing the provider's name then an id pairs the id with it.
        switcher.query_one("#console-popover-find", Input).value = "Ollama qwen3:32b"
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("typed", "ollama", "qwen3:32b")


async def test_readiness_resolves_once_per_provider_off_the_ui_thread() -> None:
    """AC#10: one resolver call per provider per open, in a worker thread,
    however many keystrokes, recents and catalog loads follow."""

    async def catalog(provider: str) -> list[str]:
        return list(PROVIDERS_MODELS.get(provider.capitalize(), ()))

    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        ui_thread = threading.get_ident()
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                recent=(_Use("openai", "gpt-5.1", timedelta(hours=1)),),
                catalog_loader=catalog,
            ),
        )
        await pilot.press(*"claude")
        await pilot.press(*["backspace"] * 6)
        await _settle(app, pilot)
        assert switcher.is_mounted

    providers = [provider for provider, _model, _thread in recorder.readiness_calls]
    assert providers and len(providers) == len(set(providers)), providers
    assert all(thread != ui_thread for _p, _m, thread in recorder.readiness_calls)


def test_readiness_words_never_claim_verified_or_reachable() -> None:
    """AC#11: config-only readiness reads 'Ready · not tested' or 'Not ready · <reason>'."""
    reachable = replace(READY, endpoint="reachable")
    assert switcher_readiness_words(reachable) == "Ready · not tested"
    assert switcher_readiness_words(READY) == "Ready · not tested"
    assert switcher_readiness_words(NO_KEY) == "Not ready · no key"
    words = {switcher_readiness_words(r) for r in (READY, NO_KEY, reachable)}
    assert not any("verified" in word or "reachable" in word for word in words)


async def test_legacy_aliases_are_hidden_unless_current_then_labelled() -> None:
    """AC#12 (ADR-066): hidden, not deleted; shown when current, as a legacy alias."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        assert all(row.provider != "local_llamacpp" for row in switcher._rows)
        assert "local_llamacpp" not in {p for p, _m, _t in recorder.readiness_calls}
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(recorder, draft=_draft("local_llamacpp", "legacy-model")),
        )
        row = line_with(list_lines(app, switcher), "legacy-model")
        assert "legacy alias" in row and CURRENT_MARK in row


async def test_custom_endpoints_show_display_names_and_builtin_slots_stay() -> None:
    """AC#13 (ADR-146): registry entries by display name; custom/custom_2 listable."""
    recorder = Recorder()
    app_config = {
        "api_settings": {},
        "custom_endpoints": {
            "vale": {
                "display_name": "Vale endpoint",
                "base_url": "http://127.0.0.1:9999/v1",
                "family": "openai_compatible",
                "models": ["vale-model"],
            }
        },
    }
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, app_config=app_config)
        )
        find = switcher.query_one("#console-popover-find", Input)
        find.value = "vale"
        await pilot.pause()
        vale = line_with(list_lines(app, switcher), "vale-model")
        assert "Vale endpoint" in vale and "custom-ep" not in vale
        assert ("custom-ep:vale", "vale-model") in {
            (row.provider, row.model) for row in switcher._rows
        }
        find.value = "custom"
        await pilot.pause()
        custom = {row.provider for row in switcher._rows if row.provider}
        text = "\n".join(list_lines(app, switcher))
    assert {"custom", "custom_2"} <= custom
    assert "Custom OpenAI-co…ble" in text and "Custom OpenAI-co… #2" in text


async def test_enter_on_needs_setup_opens_that_providers_settings_fix() -> None:
    """AC#14, AC#15: Enter never applies a NEEDS SETUP row and there is no key input."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        assert all(
            not widget.password
            and widget.id in {"console-popover-find", "console-popover-temperature"}
            for widget in switcher.query(Input)
        )
        await pilot.press(*"gpt-5")
        await pilot.pause()
        row = switcher.highlighted_row()
        assert row is not None and row.kind == "setup" and row.provider == "openai"
        assert "Enter: add key in Settings" in line_with(
            list_lines(app, switcher), "gpt-5.1"
        )
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not switcher

    assert recorder.setup == [("openai", "gpt-5.1")]
    assert recorder.submissions == []
    assert app.result is None


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
async def test_switcher_is_120_wide_auto_height_to_80_percent_keys_visible(
    size,
) -> None:
    """AC#16, AC#17: width token, auto height capped at 80%, key rows on screen."""
    recorder = Recorder()
    many = {**PROVIDERS_MODELS, "Ollama": [f"m{i}" for i in range(80)]}
    app = SwitcherHarness()
    async with app.run_test(size=size) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, providers_models=many)
        )
        await pilot.press(*"m")
        await _settle(app, pilot)
        box = switcher.query_one("#console-model-popover")
        assert box.region.width == 120
        assert box.region.height <= int(size[1] * 0.8)
        for key_id in ("#console-popover-apply", "#console-popover-save-model-default"):
            key = switcher.query_one(key_id, Button)
            assert box.region.contains_region(key.region), key_id
            assert key.region.height == 1
        assert app.focused is switcher.query_one("#console-popover-find", Input)

        await pilot.press(*"-not-a-model-anywhere")
        await _settle(app, pilot)
        assert box.region.height < int(size[1] * 0.8)
        assert box.region.contains_region(
            switcher.query_one("#console-popover-apply", Button).region
        )


def _painted_cell(screen, x: int, y: int):
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


def _ratio(first, second) -> float:
    from textual.color import Color

    from tldw_chatbook.css.Themes.themes import _contrast_ratio

    return _contrast_ratio(Color.from_rich_color(first), Color.from_rich_color(second))


@pytest.mark.parametrize("theme", ["agentic_terminal", "textual-light"])
async def test_headers_read_at_4_5_and_the_highlight_is_a_3_to_1_bar(theme) -> None:
    """AC#18, AC#19: group headers 4.5:1 on the switcher; the highlighted row
    carries the glyph and a background 3:1 away from the other rows."""
    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme

    recorder = Recorder()
    app = SwitcherHarness()
    app.register_theme(agentic_terminal_theme)
    async with app.run_test(size=(211, 44)) as pilot:
        app.theme = theme
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        pairs = switcher.query_one("#console-popover-pairs", OptionList)
        top = pairs.region.y
        header_index = next(
            i for i, row in enumerate(switcher._rows) if row.kind == "header"
        )
        highlighted = pairs.highlighted
        other = next(
            i
            for i, row in enumerate(switcher._rows)
            if row.kind == "pair" and i != highlighted
        )
        x = pairs.region.x + 2
        _, header_ink, header_bg = _painted_cell(app.screen, x, top + header_index)
        glyph, _, bar = _painted_cell(app.screen, pairs.region.x, top + highlighted)
        _, _, rest = _painted_cell(app.screen, pairs.region.x, top + other)
        assert _ratio(header_ink, header_bg) >= 4.5, (theme, header_ink, header_bg)
        assert glyph == HIGHLIGHT_GLYPH
        assert _ratio(bar, rest) >= 3.0, (theme, bar, rest)


async def test_the_old_form_is_gone_and_the_kept_ids_remain() -> None:
    """AC#20, AC#21."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        assert not switcher.query(Select)
        assert not switcher.query(Grid)
        assert not switcher.query("ModelSearchPicker")
        text = "\n".join(painted_lines(app))
        for gone in (
            "Conversation settings",
            "Compaction",
            "Defaults…",
            "Model window",
        ):
            assert gone not in text
        for kept in (
            "apply",
            "temperature",
            "streaming",
            "save-model-default",
            "make-new-chat-default",
        ):
            assert switcher.query_one(f"#console-popover-{kept}")


async def test_apply_keeps_the_compaction_override_it_was_given() -> None:
    """ADR-095: quick Apply still submits the unchanged compaction override."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder, draft=_draft(compaction=ContextCompactionMode.OFF)
            ),
        )
        await pilot.press("enter")
        await pilot.pause()
    overrides = recorder.submissions[0].draft.context_policy_overrides
    assert overrides.compaction_mode is ContextCompactionMode.OFF


async def test_no_model_never_applies_a_bare_provider() -> None:
    """AC#2: with no pair to apply, Enter explains instead of committing."""
    recorder = Recorder(ready=frozenset({"*"}))
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(recorder, draft=_draft(model=None), providers_models={}),
        )
        switcher.query_one("#console-popover-apply", Button).press()
        await pilot.pause()
        error = str(switcher.query_one("#console-popover-error").render())
        assert "Choose a model" in error
    assert recorder.submissions == []
    for row in switcher._rows:
        if row.kind in {"pair", "typed"}:
            assert row.model


def test_needs_setup_route_names_the_provider_its_model_and_the_field() -> None:
    """AC#14 (R11): the route reuses the recovery keys and carries the pair."""
    from types import SimpleNamespace

    from tldw_chatbook.Constants import TAB_SETTINGS
    from tldw_chatbook.UI.Console_Modules.model_switcher import open_provider_setup

    posted: list[object] = []
    resolved: list[tuple[str, str | None]] = []
    screen = SimpleNamespace(
        _console_default_readiness=lambda provider, model: (
            resolved.append((provider, model)) or NO_KEY
        ),
        post_message=posted.append,
    )

    open_provider_setup(screen, "openrouter", "openai/gpt-4o-mini")

    (message,) = posted
    assert message.screen_name == TAB_SETTINGS
    assert message.screen_context == {
        "category": "providers-models",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "field": "api_key",
    }
    assert resolved == [("openrouter", "openai/gpt-4o-mini")]
