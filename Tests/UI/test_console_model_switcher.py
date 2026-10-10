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
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_provider_support import MODEL_FIELD_LABELS
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from tldw_chatbook.Chat.console_settings_apply import (
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
    ConsoleSettingsTransfer,
)
from tldw_chatbook.Utils.token_counter import resolve_context_window
from tldw_chatbook.Widgets.Console.console_model_popover import (
    CURRENT_MARK,
    HIGHLIGHT_GLYPH,
    VALUE_FIELDS,
    ConsoleModelPopover,
    UnsavedEditsGuard,
    context_copy,
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


#: Per-model effective values for ``_rebase_with_values``:
#: (temperature, max_tokens, streaming).
PAIR_VALUES = {
    "model-a": (0.7, None, False),
    "claude-sonnet-4-5": (0.2, 1024, True),
    "model-b": (1.3, 2048, True),
}


def _rebase_with_values(
    state: ConsoleSettingsDraftState, **kwargs: object
) -> ConsoleSettingsDraftState:
    """A rebaser whose pairs differ in value, like the controller's: each target
    starts from its own inherited Temperature and Max tokens."""
    temperature, max_tokens, streaming = PAIR_VALUES.get(
        str(kwargs["model"]), (1.0, 4096, True)
    )
    settings = replace(
        state.settings,
        provider=kwargs["provider"],
        model=kwargs["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    return replace(
        state,
        settings=settings,
        field_drafts=tuple(
            replace(
                field,
                effective_value=getattr(settings, field.name),
                profile_override=getattr(settings, field.name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for field in state.field_drafts
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
    draft_rebaser=_rebase,
    pick_only: bool = False,
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
        draft_rebaser=draft_rebaser,
        live_committer=recorder.commit,
        default_readiness_resolver=recorder.readiness,
        recent_pairs_loader=load_recent if recent is not None else None,
        previous_pair=previous,
        catalog_loader=catalog_loader,
        setup_opener=lambda provider, model: recorder.setup.append((provider, model)),
        pick_only=pick_only,
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
        # The size the one resolver gives: "200k" from the catalog, or "?"
        # if the catalog is unavailable (a fallback is unknown, TASK-33007 #12).
        window = resolve_context_window("anthropic", "claude-sonnet-4-5")
        assert f" {context_copy(window.tokens, window.verified)}  " in previous
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


async def test_typed_rows_are_headed_as_matches_not_top_3_each() -> None:
    """cubic #2947: typing lists every match, up to 30, so the READY PROVIDERS
    header stops promising three models per provider."""
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(Recorder()))
        assert any("top 3 each" in line for line in list_lines(app, switcher))
        await pilot.press(*"claude")
        await _settle(app, pilot)

        lines = list_lines(app, switcher)
        anthropic = [line for line in lines if "Anthropic" in line and "Ready" in line]
        assert len(anthropic) > 3, lines
        assert not any("top 3 each" in line for line in lines), lines
        assert any("READY PROVIDERS · matches from every" in line for line in lines)


async def test_page_keys_from_find_always_leave_a_pair_highlighted() -> None:
    """cubic #2947: PageDown and PageUp move the list while Find keeps focus.
    Textual's page move onto a trailing info row or a leading header
    highlights nothing, which left Enter with no pair to apply."""
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(Recorder()))
        find = switcher.query_one("#console-popover-find", Input)
        assert switcher._rows[-1].kind in {"info", "header"}  # the trap's premise
        for key in ("pagedown", "pagedown", "pageup", "pageup"):
            await pilot.press(key)
            await pilot.pause()
            assert switcher.highlighted_row() is not None, key
            assert app.focused is find, key


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
            # Time the handler itself (the filter runs synchronously in it),
            # best of three: a GC pause or a busy runner inflates one sample,
            # while a slow filter is slow every time. Measured 3-7 ms locally.
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                find.value = query
                switcher._find_changed(Input.Changed(find, query))
                samples.append(time.perf_counter() - started)
                await pilot.pause()
            timings.append(min(samples))
        assert max(timings) < 0.050, timings
        row = switcher.highlighted_row()
        assert row is not None and row.model is not None and "model-19" in row.model


async def test_a_typed_model_id_applies_as_a_pair_and_bad_ids_never_do() -> None:
    """AC#8: an id no catalog lists applies with the chat's provider, and only
    as bounded single-line text (TASK-14812 AC#5, AC#7) with no markup
    bracket, which other Console surfaces would still parse and crash on."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        find = switcher.query_one("#console-popover-find", Input)
        for bad in ("x" * 300, "foo[/]", "a[b]"):
            find.value = bad
            await pilot.pause()
            assert all(row.kind != "typed" for row in switcher._rows), bad

        # The highlight (and the draft with it) moves to Anthropic first; the
        # typed id still pairs with the chat's provider, not the highlight's.
        find.value = "claude"
        await pilot.pause()
        assert switcher.highlighted_row().provider == "anthropic"
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


def test_switcher_rows_read_the_one_readiness_vocabulary() -> None:
    """TASK-33005.3 (rewritten on purpose from P4's
    ``test_readiness_words_never_claim_verified_or_reachable``): rows read
    the shared spec §5 words. Config-only readiness, or a reachable facet
    with no observed connection, still never claims more than 'not tested';
    an observed local listing reads 'reachable HH:MM' and a refusal names
    the port, as every other surface does."""
    from datetime import datetime

    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

    llama = ProviderDraftIdentity(
        provider_key="llama_cpp",
        connection_identity=canonical_connection_identity(
            "llama_cpp", "http://127.0.0.1:9099"
        ),
        credential_source="none",
        credential_revision=0,
        draft_generation=0,
    )
    seen = datetime(2026, 10, 1, 14, 1).astimezone()
    reachable = replace(READY, endpoint="reachable")
    refused = ConsoleSettingsReadiness(
        "Not ready",
        "",
        False,
        operability="not_ready",
        blocker="endpoint_unreachable",
        recovery_action="retry_connection",
        configuration="configured",
        credential="not_required",
        endpoint="unreachable",
        endpoint_category="connection_refused",
        model="unconfirmed",
        connection=llama,
        observed_at=seen,
    )

    assert switcher_readiness_words(None) == "checking…"
    assert switcher_readiness_words(READY) == "Ready · not tested"
    assert switcher_readiness_words(reachable) == "Ready · not tested"
    assert switcher_readiness_words(NO_KEY) == "Not ready · no key"
    assert switcher_readiness_words(
        replace(reachable, connection=llama, observed_at=seen)
    ) == ("Ready · reachable 14:01")
    assert switcher_readiness_words(refused) == "Not ready · refused :9099"


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
        # TASK-33004.5: Max tokens joined the value row; still no key input.
        assert all(
            not widget.password
            and widget.id
            in {
                "console-popover-find",
                "console-popover-temperature",
                "console-popover-max-tokens",
            }
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
async def test_switcher_is_140_wide_auto_height_to_80_percent_keys_visible(
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
        # Rewritten on purpose (TASK-33004 final review): 140 columns so the
        # model column shows ids whole (spec: never truncated).
        assert box.region.width == 140
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


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.parametrize(
    ("long_id", "query"),
    [
        ("anthropic/claude-3.7-sonnet:thinking", "sonnet:thinking"),
        # Qodo #2947: past the old 44-column cap.
        ("accounts/fireworks/models/deepseek-r1-distill-llama-70b-fast", "distill"),
    ],
    ids=["36-columns", "60-columns"],
)
async def test_switcher_shows_long_model_ids_whole(size, long_id, query) -> None:
    """Spec rule: full ids, never truncated, in the row and on screen."""
    recorder = Recorder()
    models = {**PROVIDERS_MODELS, "Ollama": [long_id]}
    app = SwitcherHarness()
    async with app.run_test(size=size) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, providers_models=models)
        )
        await pilot.press(*query)
        await _settle(app, pilot)
        rendered = [
            str(switcher._prompt(row, False)) for row in switcher._rows
        ]
        painted = list_lines(app, switcher)
    assert any(long_id in line for line in rendered), rendered
    assert not any("…" in line and long_id[:12] in line for line in rendered)
    assert any(long_id in line for line in painted), painted


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
        # TASK-33004.5: the one Select left is the value row's Streaming On/Off.
        assert [select.id for select in switcher.query(Select)] == [
            "console-popover-streaming"
        ]
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


def _strip(app) -> str:
    return line_with(painted_lines(app), "Values for")


def _controls(switcher: ConsoleModelPopover) -> tuple[str, str, object]:
    """Temperature text, Max tokens text and the Streaming Select's value."""
    return (
        switcher.query_one("#console-popover-temperature", Input).value,
        switcher.query_one("#console-popover-max-tokens", Input).value,
        switcher.query_one("#console-popover-streaming", Select).value,
    )


async def test_value_strip_follows_every_highlight_move_and_invents_no_edit() -> None:
    """The strip names and shows the pair that Enter, Ctrl+N and Save act on:
    PREVIOUS on open, then every Up/Down, from Find and from Temperature. Moving
    edits nothing, and Save right after open saves exactly what the strip shows.

    Rewritten for TASK-33004.5: the values are the row's controls (Max tokens
    an Input, Streaming an On/Off Select), not copy in the label line."""
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
                draft_rebaser=_rebase_with_values,
            ),
        )
        temperature = switcher.query_one("#console-popover-temperature", Input)

        def shown() -> tuple[str | None, str, tuple[str, str, object]]:
            row = switcher.highlighted_row()
            return (row.model if row else None, _strip(app), _controls(switcher))

        model, strip, values = shown()
        assert model == "claude-sonnet-4-5"
        assert "Values for claude-sonnet-4-5" in strip
        assert values == ("0.2", "1024", True)

        await pilot.press("down")  # from Find
        await pilot.pause()
        model, strip, values = shown()
        assert model == "model-a"
        assert "Values for model-a" in strip
        assert values == ("0.7", "", False)

        temperature.focus()
        await pilot.press("up")  # from Temperature
        await pilot.pause()
        model, strip, values = shown()
        assert model == "claude-sonnet-4-5"
        assert "Values for claude-sonnet-4-5" in strip and values[0] == "0.2"
        for name in ("temperature", "max-tokens", "streaming"):
            word = switcher.query_one(f"#console-popover-{name}-source", Static)
            assert str(word.render()) != "edited *", name

        switcher.query_one("#console-popover-save-model-default", Button).press()
        await pilot.pause()

    (submission,) = recorder.submissions
    settings = submission.draft.settings
    assert submission.action is ConsoleSettingsAction.SAVE_MODEL_DEFAULT
    assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-4-5")
    assert (settings.temperature, settings.max_tokens) == (0.2, 1024)
    assert not any(field.dirty for field in submission.draft.field_drafts)


async def test_a_row_without_a_pair_shows_and_applies_this_chats_pair() -> None:
    """With no pair highlighted (no match, or a NEEDS SETUP row with no model)
    the strip falls back to this chat's pair, never a pair the highlight
    passed while typing; with no row at all, Enter applies nothing."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, draft_rebaser=_rebase_with_values)
        )
        find = switcher.query_one("#console-popover-find", Input)
        find.value = "claude-sonnet"
        await pilot.pause()
        assert "Values for claude-sonnet-4-5" in _strip(app)

        find.value = ""
        await pilot.pause()
        setup = next(
            index
            for index, row in enumerate(switcher._rows)
            if row.kind == "setup" and row.model is None
        )
        switcher._set_highlight(setup)
        await pilot.pause()
        assert "Values for model-a" in _strip(app)

        find.value = "claude-sonnet"
        await pilot.pause()
        find.value = "x" * 300  # matches nothing, too long to be a typed id
        await pilot.pause()
        assert switcher.highlighted_row() is None
        assert "Values for model-a" in _strip(app)
        await pilot.press("enter")
        await pilot.pause()
        assert "Choose a model" in str(
            switcher.query_one("#console-popover-error").render()
        )
        assert app.screen is switcher

    assert recorder.submissions == []


async def test_chat_settings_carries_the_highlighted_pair() -> None:
    """'Chat settings…' transfers the pair the strip names, unapplied."""
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
                draft_rebaser=_rebase_with_values,
            ),
        )
        switcher.query_one("#console-popover-full-settings", Button).press()
        await pilot.pause()

    assert recorder.submissions == []
    assert isinstance(app.result, ConsoleSettingsTransfer)
    settings = app.result.draft.settings
    assert (settings.provider, settings.model, settings.temperature) == (
        "anthropic",
        "claude-sonnet-4-5",
        0.2,
    )


@pytest.mark.parametrize(
    ("draft", "app_config", "shown"),
    [
        (_draft(model="foo[/]"), None, "llama.cpp · foo[/]"),
        (
            _draft("custom-ep:lab", "lab-model"),
            {
                "api_settings": {},
                "custom_endpoints": {
                    "lab": {
                        "display_name": "[x] lab",
                        "base_url": "http://127.0.0.1:9999/v1",
                        "family": "openai_compatible",
                        "models": ["lab-model"],
                    }
                },
            },
            "[x] lab · lab-model",
        ),
    ],
)
async def test_title_shows_bracketed_model_ids_and_names_literally(
    draft, app_config, shown
) -> None:
    """A model id or endpoint name is never parsed as markup: 'foo[/]' raised
    MarkupError on every open, and '[x] lab' lost its '[x]'."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, draft=draft, app_config=app_config)
        )
        box = switcher.query_one("#console-model-popover")
        assert f"Switch model · now: {shown}" in painted_lines(app)[box.region.y]


async def test_a_value_edit_pins_its_pair_against_a_late_recent_fill() -> None:
    """RECENT arriving after an edit must not move the draft, and the edit,
    to the PREVIOUS pair it brings."""
    release = asyncio.Event()

    async def late_recent():
        await release.wait()
        return (_Use("anthropic", "claude-sonnet-4-5", timedelta(hours=1)),)

    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build_switcher(recorder, draft_rebaser=_rebase_with_values)
        switcher._recent_pairs_loader = late_recent
        app.push_screen(switcher)
        for _ in range(3):
            await pilot.pause()
        assert switcher.highlighted_row().model == "model-a"
        temperature = switcher.query_one("#console-popover-temperature", Input)
        temperature.focus()
        temperature.value = "0.3"
        await pilot.pause()
        release.set()
        await _settle(app, pilot)
        assert any(row.model == "claude-sonnet-4-5" for row in switcher._rows)
        assert switcher.highlighted_row().model == "model-a"
        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model, settings.temperature) == (
        "llama_cpp",
        "model-a",
        0.3,
    )


# -- TASK-33004.5: the value row, Source words and commit keys ---------------

#: A real config for the controller's own rebase and the Source-word resolver:
#: sonnet's Temperature and Max tokens are a model default, Streaming is
#: chat_defaults (Console Behavior), and the chat's model-a values are built-in.
REAL_CONFIG = {
    "chat_defaults": {"provider": "llama_cpp", "model": "model-a", "streaming": True},
    "api_settings": {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "anthropic": {
            "model": "claude-sonnet-4-5",
            "model_defaults": {
                "claude-sonnet-4-5": {"temperature": 0.2, "max_tokens": 1024},
                "claude-haiku-4-5": {"temperature": 0.5},
            },
        },
    },
}
SPEC_SOURCE_WORDS = {
    "edited *",
    "this chat",
    "model default",
    "Console Behavior",
    "provider",
    "built-in",
}


def _real_rebase(
    state: ConsoleSettingsDraftState, **kwargs: object
) -> ConsoleSettingsDraftState:
    """The controller's rebase: remembered drafts, carried edits, defaults."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    return ConsoleChatController.rebase_console_settings_draft(
        object(), state, **kwargs
    )


def _real_switcher(recorder: Recorder, **kwargs) -> ConsoleModelPopover:
    return build_switcher(
        recorder, app_config=REAL_CONFIG, draft_rebaser=_real_rebase, **kwargs
    )


def _words(switcher: ConsoleModelPopover) -> dict[str, str]:
    return {
        name: str(
            switcher.query_one(
                f"#console-popover-{name.replace('_', '-')}-source", Static
            ).render()
        )
        for name in VALUE_FIELDS
    }


async def _find(pilot, text: str) -> None:
    """From Temperature, Shift+Tab back to Find (its text selected) and type."""
    await pilot.press("shift+tab", *text)
    await pilot.pause()


def test_the_value_row_is_exactly_the_quick_default_mask() -> None:
    """AC#1, AC#10 (R3): one field list, so Save keeps every value it shows."""
    assert set(VALUE_FIELDS) == QUICK_MODEL_DEFAULT_FIELDS
    assert "thinking_effort" not in VALUE_FIELDS


async def test_value_row_shows_the_highlighted_pairs_values_one_row_each() -> None:
    """AC#1, AC#2, AC#6: Temperature, Max tokens and Streaming for the
    highlighted pair, each one row tall, each with its spec §6 Source word;
    Streaming is one Select offering On and Off; Thinking is not here."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            _real_switcher(
                recorder,
                remembered_previous=_Use(
                    "anthropic", "claude-sonnet-4-5", timedelta(hours=2), True
                ),
            ),
        )
        assert _controls(switcher) == ("0.2", "1024", True)
        assert _words(switcher) == {
            "temperature": "model default",
            "max_tokens": "model default",
            "streaming": "Console Behavior",
        }
        row = switcher.query_one("#console-popover-values")
        labels = [
            str(child.render())
            for child in row.children
            if isinstance(child, Static)
            and "console-popover-source" not in child.classes
        ]
        assert labels == [MODEL_FIELD_LABELS[name] for name in VALUE_FIELDS]
        box = switcher.query_one("#console-model-popover")
        for child in row.children:
            assert child.region.height == 1, child
            assert box.region.contains_region(child.region), child
        painted = line_with(
            painted_lines(app), "Temperature", "Max tokens", "Streaming"
        )
        assert "Thinking" not in painted
        assert "model default" in painted and "Console Behavior" in painted

        streaming = switcher.query_one("#console-popover-streaming", Select)
        assert [(str(prompt), value) for prompt, value in streaming._options] == [
            ("On", True),
            ("Off", False),
        ]

        await pilot.press("down")  # the chat's own pair: built-in values
        await pilot.pause()
        assert switcher.highlighted_row().model == "model-a"
        assert _words(switcher) == {
            "temperature": "built-in",
            "max_tokens": "built-in",
            "streaming": "Console Behavior",
        }
        assert set(_words(switcher).values()) <= SPEC_SOURCE_WORDS


async def test_a_chat_value_that_differs_from_its_defaults_says_this_chat() -> None:
    """AC#2: the chat's own pair shows what the chat holds, sourced 'this chat'."""
    recorder = Recorder()
    app = SwitcherHarness()
    draft = _draft()
    draft = replace(draft, settings=replace(draft.settings, temperature=1.3))
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, _real_switcher(recorder, draft=draft)
        )
        assert switcher.highlighted_row().model == "model-a"
        assert _controls(switcher)[0] == "1.3"
        assert _words(switcher)["temperature"] == "this chat"


async def test_alex_path_tab_from_find_edits_the_highlighted_pairs_values() -> None:
    """AC#4 (R13): Tab from Find rebases to the highlighted pair and focuses
    Temperature with its value selected, so typing replaces it; Tab again
    selects Max tokens; Enter applies both as 'edited *' values."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, _real_switcher(recorder))
        await pilot.press(*"son")
        await pilot.pause()
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"

        await pilot.press("tab")
        await pilot.pause()
        temperature = switcher.query_one("#console-popover-temperature", Input)
        assert app.focused is temperature
        assert "Values for claude-sonnet-4-5" in _strip(app)
        assert temperature.selected_text == temperature.value == "0.2"
        await pilot.press(*"0.9", "tab")
        await pilot.pause()
        max_tokens = switcher.query_one("#console-popover-max-tokens", Input)
        assert app.focused is max_tokens
        assert max_tokens.selected_text == max_tokens.value == "1024"
        await pilot.press(*"8192")
        await pilot.pause()
        assert _controls(switcher)[:2] == ("0.9", "8192")
        assert _words(switcher)["temperature"] == "edited *"
        assert _words(switcher)["max_tokens"] == "edited *"
        await pilot.press("enter")
        await pilot.pause()

    (submission,) = recorder.submissions
    settings = submission.draft.settings
    assert submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-4-5")
    assert (settings.temperature, settings.max_tokens) == (0.9, 8192)


async def test_edits_for_a_pair_come_back_after_switching_away_and_back() -> None:
    """AC#5: A→B→A in one open restores A's edits, not B's."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, _real_switcher(recorder))
        await pilot.press(*"claude-sonnet-4-5", "tab", *"0.9")
        await pilot.pause()
        await _find(pilot, "claude-haiku-4-5")
        assert switcher.highlighted_row().model == "claude-haiku-4-5"
        await pilot.press("tab", *"0.4")
        await pilot.pause()

        await _find(pilot, "claude-sonnet-4-5")
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"
        assert _controls(switcher)[0] == "0.9"
        assert _words(switcher)["temperature"] == "edited *"
        await pilot.press("tab")  # into Temperature, so Shift+Tab lands in Find
        await _find(pilot, "claude-haiku-4-5")
        assert _controls(switcher)[0] == "0.4"


@pytest.mark.parametrize(
    "previous, max_tokens", [(None, None), ("claude-sonnet-4-5", 2048)]
)
async def test_opening_and_closing_leaves_no_edit(previous, max_tokens) -> None:
    """AC#7, AC#13: the Inputs' and the Select's mount-time Changed echoes are
    not edits, even after the highlight has rebased the row to PREVIOUS's
    different values; Esc then closes at once with no change. Every echo the
    switcher waits for arrives (a blank Input posts none, so none is awaited)."""
    recorder = Recorder()
    app = SwitcherHarness()
    draft = _draft()
    draft = replace(draft, settings=replace(draft.settings, max_tokens=max_tokens))
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            _real_switcher(
                recorder,
                draft=draft,
                remembered_previous=(
                    _Use("anthropic", previous, timedelta(hours=1))
                    if previous
                    else None
                ),
            ),
        )
        assert switcher._mount_echo == {}
        drafts = (
            switcher._draft.field_drafts,
            *(remembered.field_drafts for remembered in switcher._draft.model_drafts),
        )
        assert not any(field.dirty for fields in drafts for field in fields)
        assert "edited *" not in _words(switcher).values()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not switcher
    assert app.result is None
    assert recorder.submissions == []


def _key_row(app, switcher: ConsoleModelPopover) -> str:
    """The painted text of the switcher's key row."""
    region = switcher.query_one("#console-popover-keys").region
    lines = painted_lines(app)[region.y : region.bottom]
    return "\n".join(line[region.x : region.right] for line in lines)


async def _edited_switcher(app, pilot, recorder: Recorder) -> ConsoleModelPopover:
    switcher = await open_switcher(app, pilot, _real_switcher(recorder))
    await pilot.press(*"son", "tab", *"0.9")
    await pilot.pause()
    await pilot.press(
        "shift+tab"
    )  # back to Find: Enter and d must still reach the guard
    await pilot.pause()
    return switcher


async def test_esc_with_edits_asks_and_keeps_editing_then_d_discards() -> None:
    """AC#14 (R14): Esc shows the prompt and discards nothing; Esc again keeps
    editing; in the prompt ``d`` discards (focus left Find, so it is not typed)."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await _edited_switcher(app, pilot, recorder)
        find = switcher.query_one("#console-popover-find", Input)
        guard = switcher.query_one("#console-popover-guard", UnsavedEditsGuard)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is switcher and guard.display
        assert app.focused is guard
        prompt = str(guard.render())
        assert "Temperature" in prompt
        assert "Enter apply · d discard · Esc keep editing" in prompt
        assert "Enter apply · d discard · Esc keep editing" in "\n".join(
            painted_lines(app)
        )
        # Its focus outline closes at the bottom (Textual 8.2.8 repeats the
        # top edge on a bottom padding row).
        region = guard.region
        edges = [
            line[region.x : region.right]
            for line in painted_lines(app)[region.y : region.bottom]
        ]
        assert edges[0][0] + edges[-1][0] == "┌└", edges
        # cubic #2947: the key row under the prompt agrees with it on Esc.
        assert "Esc keep editing" in _key_row(app, switcher)
        assert "Esc cancel" not in _key_row(app, switcher)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is switcher and not guard.display
        assert app.focused is find
        assert _controls(switcher)[0] == "0.9"
        assert "Esc cancel" in _key_row(app, switcher)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert app.screen is not switcher
        assert find.value == "son"
    assert app.result is None
    assert recorder.submissions == []


async def test_esc_with_edits_then_enter_applies_them() -> None:
    """AC#14: the prompt's Enter applies the edited pair to this chat."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await _edited_switcher(app, pilot, recorder)
        await pilot.press("escape")
        await pilot.pause()
        assert switcher.query_one("#console-popover-guard").display
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not switcher
    (submission,) = recorder.submissions
    assert submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submission.draft.settings.temperature == 0.9


async def test_the_key_rows_print_every_key_the_switcher_binds() -> None:
    """AC#15: Enter, Tab, Ctrl+N, Ctrl+O, Esc and the scope line are printed,
    and Save names the three fields it saves (AC#10)."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        text = "\n".join(painted_lines(app))
        for copy in (
            "Enter apply to this chat",
            "Tab edit values",
            "Ctrl+N default for new chats",
            "Ctrl+O chat settings",
            "Esc cancel",
            "Applies to: this chat only",
            "Save as model default",
            "saves Temperature, Max tokens, Streaming",
        ):
            assert copy in text, copy
        box = switcher.query_one("#console-model-popover")
        for key_id in ("apply", "make-new-chat-default", "full-settings"):
            key = switcher.query_one(f"#console-popover-{key_id}", Button)
            assert key.region.height == 1 and box.region.contains_region(key.region)


@pytest.mark.parametrize("key", ["enter", "tab", "ctrl+n", "ctrl+o", "escape"])
async def test_every_printed_key_works_while_find_has_focus(key) -> None:
    """AC#16: each printed accelerator acts from Find, on the highlighted pair."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, _real_switcher(recorder))
        await pilot.press(*"son")
        await pilot.pause()
        assert app.focused is switcher.query_one("#console-popover-find", Input)
        await pilot.press(key)
        await pilot.pause()
        if key == "tab":
            assert app.focused is switcher.query_one(
                "#console-popover-temperature", Input
            )
            return
        assert app.screen is not switcher
    if key == "escape":
        assert app.result is None and recorder.submissions == []
        return
    if key == "ctrl+o":
        assert recorder.submissions == []
        assert isinstance(app.result, ConsoleSettingsTransfer)
        assert app.result.draft.settings.model == "claude-sonnet-4-5"
        return
    (submission,) = recorder.submissions
    assert submission.draft.settings.model == "claude-sonnet-4-5"
    assert submission.action is (
        ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        if key == "ctrl+n"
        else ConsoleSettingsAction.APPLY_TO_CHAT
    )


@pytest.mark.parametrize(
    ("press", "action"),
    [
        ("ctrl+n", ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT),
        ("save", ConsoleSettingsAction.SAVE_MODEL_DEFAULT),
    ],
)
async def test_default_actions_save_exactly_the_three_values(press, action) -> None:
    """AC#10: Ctrl+N and Save carry the quick mask with the edited Max tokens."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, _real_switcher(recorder))
        await pilot.press(*"son", "tab", "tab", *"8192")
        await pilot.pause()
        if press == "save":
            switcher.query_one("#console-popover-save-model-default", Button).press()
        else:
            await pilot.press(press)
        await pilot.pause()
    (submission,) = recorder.submissions
    assert submission.action is action
    assert submission.default_field_mask == QUICK_MODEL_DEFAULT_FIELDS
    assert submission.draft.settings.max_tokens == 8192


async def test_ctrl_o_carries_the_highlighted_pair_and_its_edits_unapplied() -> None:
    """AC#12: Chat settings gets the draft; nothing is applied or discarded."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        await open_switcher(app, pilot, _real_switcher(recorder))
        await pilot.press(*"son", "tab", *"0.9", "ctrl+o")
        await pilot.pause()
    assert recorder.submissions == []
    assert isinstance(app.result, ConsoleSettingsTransfer)
    draft = app.result.draft
    assert (draft.settings.provider, draft.settings.model) == (
        "anthropic",
        "claude-sonnet-4-5",
    )
    assert draft.settings.temperature == 0.9
    temperature = next(f for f in draft.field_drafts if f.name == "temperature")
    assert temperature.dirty


def test_the_switcher_binds_no_terminal_convention_key() -> None:
    """AC#17 (ADR-031 rule 2): no Ctrl+C/V/X/S/D/Z/A/R/W on the switcher."""
    banned = {f"ctrl+{letter}" for letter in "cvxsdzarw"}
    bound = {
        key.strip()
        for owner in (ConsoleModelPopover, UnsavedEditsGuard)
        for binding in owner.BINDINGS
        for key in binding.key.split(",")
    }
    assert {"ctrl+n", "ctrl+o", "escape"} <= bound
    assert not bound & banned


async def test_tab_into_the_values_pins_the_pair_and_typing_re_ranks() -> None:
    """R13: once Tab moves into a pair's values, a late fill that brings a
    better match for the query cannot retarget the strip under the cursor;
    typing in Find again picks the best match."""
    release = asyncio.Event()

    async def late_recent():
        await release.wait()
        return (_Use("anthropic", "sonic-1", timedelta(hours=1)),)

    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = _real_switcher(recorder)
        switcher._recent_pairs_loader = late_recent
        app.push_screen(switcher)
        for _ in range(3):
            await pilot.pause()
        await pilot.press(*"son", "tab")
        await pilot.pause()
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"
        release.set()
        await _settle(app, pilot)
        assert any(row.model == "sonic-1" for row in switcher._rows)
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"
        assert "Values for claude-sonnet-4-5" in _strip(app)

        # The same query typed again: both rows still match, and the new
        # query ranks (prefix beats substring) instead of keeping the pin.
        await pilot.press("shift+tab", *"son")
        await pilot.pause()
        assert switcher.highlighted_row().model == "sonic-1"


@pytest.mark.parametrize("before, after", [("0.9", ""), ("0", ".9")])
async def test_a_typed_id_tabbed_into_before_readiness_yields_to_the_catalog_match(
    before, after
) -> None:
    """T5 review fix: before readiness lists any catalog, 'son' highlights only
    the TYPED MODEL ID row. Tab and an edit typed ahead of readiness must not
    pin that row: when readiness lands, the catalog match takes the highlight,
    the edit carries to it, and the text being typed is not rewritten."""
    gate = threading.Event()
    recorder = Recorder()

    def gated_readiness(provider: str, model: str | None) -> ConsoleSettingsReadiness:
        gate.wait(5)
        return recorder.readiness(provider, model)

    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = _real_switcher(recorder)
        switcher._default_readiness_resolver = gated_readiness
        app.push_screen(switcher)
        for _ in range(3):
            await pilot.pause()
        await pilot.press(*"son", "tab", *before)
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("typed", "llama_cpp", "son")

        gate.set()
        await _settle(app, pilot)
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"
        assert "Values for claude-sonnet-4-5" in _strip(app)
        assert _controls(switcher)[0] == before
        await pilot.press(*after, "tab", *"8192", "enter")
        await pilot.pause()

    (submission,) = recorder.submissions
    settings = submission.draft.settings
    assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-4-5")
    assert (settings.temperature, settings.max_tokens) == (0.9, 8192)


async def test_clearing_a_value_the_chat_opened_blank_is_an_edit() -> None:
    """T5 review fix: a blank Input posts no mount-time Changed, so clearing
    PREVIOUS's Max tokens (blank = no cap) on a chat that opened with no cap
    is the user's edit, not a swallowed echo, and Enter applies no cap."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            _real_switcher(
                recorder,
                remembered_previous=_Use(
                    "anthropic", "claude-sonnet-4-5", timedelta(hours=1)
                ),
            ),
        )
        assert _controls(switcher)[1] == "1024"
        await pilot.press("tab", "tab", "backspace")
        await pilot.pause()
        assert _controls(switcher)[1] == ""
        assert _words(switcher)["max_tokens"] == "edited *"
        await pilot.press("enter")
        await pilot.pause()

    (submission,) = recorder.submissions
    settings = submission.draft.settings
    assert (settings.model, settings.max_tokens) == ("claude-sonnet-4-5", None)


def _row_keys(switcher: ConsoleModelPopover) -> list[tuple[str, str, str | None]]:
    """Every listed row but the headers, whose copy differs by mode."""
    return [row.key for row in switcher._rows if row.kind != "header"]


async def test_pick_only_lists_the_same_pairs_and_returns_the_pick_unapplied() -> None:
    """TASK-33004.6 AC#1, AC#4: the pick-only result contract.

    Pick-only Switch model lists the pairs the switcher lists, has no value
    row and no default action, prints only the keys it binds, and Enter hands
    the highlighted pair back to the opener as ``(provider, model)``, with the
    provider's canonical key (a stored chat may hold "Anthropic"): nothing is
    rebased, applied, saved or routed to Settings. Ctrl+N and Ctrl+O, the
    switcher's default-action keys, do nothing here.
    """
    previous = _Use("Anthropic", "claude-sonnet-4-5", timedelta(hours=2), True)
    recent = (_Use("openai", "gpt-5.1", timedelta(hours=3)),)
    listed = {}
    rebased: list[object] = []

    def recording_rebaser(state, **kwargs):
        rebased.append(kwargs["model"])
        return _rebase(state, **kwargs)

    for pick_only in (False, True):
        recorder = Recorder()
        app = SwitcherHarness()
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = await open_switcher(
                app,
                pilot,
                build_switcher(
                    recorder,
                    recent=recent,
                    remembered_previous=previous,
                    draft_rebaser=recording_rebaser if pick_only else _rebase,
                    pick_only=pick_only,
                ),
            )
            listed[pick_only] = _row_keys(switcher)
            if not pick_only:
                continue
            assert app.focused is switcher.query_one("#console-popover-find", Input)
            assert [widget.id for widget in switcher.query(Input)] == [
                "console-popover-find"
            ]
            assert not switcher.query(Select) and not switcher.query(Button)
            text = "\n".join(painted_lines(app))
            assert "Enter picks · Esc cancel" in text
            for gone in (
                "Values for",
                "Temperature",
                "Max tokens",
                "Save as model default",
                "Ctrl+N",
                "Ctrl+O",
                "Applies to",
                "Enter apply",
                "Enter applies",
                "swaps back",
            ):
                assert gone not in text, gone

            await pilot.press("ctrl+n", "ctrl+o")
            await pilot.pause()
            assert app.screen is switcher and app.result == "unset"
            row = switcher.highlighted_row()
            assert row is not None and row.model == "claude-sonnet-4-5"
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is not switcher

    assert listed[True] == listed[False]
    assert app.result == ("anthropic", "claude-sonnet-4-5")
    assert recorder.submissions == [] and recorder.setup == [] and rebased == []


@pytest.mark.parametrize(
    "theme", ["agentic_terminal", "textual-dark", "pastel_dreams", "solarized_light"]
)
async def test_pick_only_needs_setup_rows_are_muted_and_readable(theme) -> None:
    """TASK-33004.6 AC#2, review fix round 1: a disabled NEEDS SETUP row is
    painted in $text-muted, unlike a header or a pickable row, and clears
    4.5:1 on the light themes where the old dimmed ink fell to 1.64:1."""
    from tldw_chatbook.css.Themes.themes import ALL_THEMES

    recorder = Recorder(ready=frozenset({"llama_cpp"}))
    app = SwitcherHarness()
    for shipped in ALL_THEMES:
        app.register_theme(shipped)
    async with app.run_test(size=(211, 44)) as pilot:
        app.theme = theme
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={
                    "Llama_cpp": ["model-a", "model-b"],
                    "OpenAI": ["gpt-5.1"],
                },
                pick_only=True,
            ),
        )
        pairs = switcher.query_one("#console-popover-pairs", OptionList)
        kinds = [row.kind for row in switcher._rows]
        pickable = next(
            index
            for index, kind in enumerate(kinds)
            if kind not in {"header", "info", "setup"} and index != pairs.highlighted
        )
        x, top = pairs.region.x + 2, pairs.region.y
        _, header_ink, _ = _painted_cell(app.screen, x, top + kinds.index("header"))
        _, pick_ink, _ = _painted_cell(app.screen, x, top + pickable)
        _, setup_ink, setup_bg = _painted_cell(
            app.screen, x, top + kinds.index("setup")
        )
        assert setup_ink not in {header_ink, pick_ink}, theme
        assert _ratio(setup_ink, setup_bg) >= 4.5, (theme, setup_ink, setup_bg)


async def test_pick_only_never_picks_needs_setup_and_esc_returns_nothing() -> None:
    """TASK-33004.6 AC#2: NEEDS SETUP rows are listed but cannot be
    highlighted or picked, their copy promises no Enter, and Esc returns
    None at once, even when the opener's draft carries unsaved edits."""
    recorder = Recorder(ready=frozenset({"llama_cpp"}))
    edited = _draft()
    edited = replace(
        edited,
        field_drafts=tuple(
            replace(field, dirty=field.name == "temperature")
            for field in edited.field_drafts
        ),
    )
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                draft=edited,
                providers_models={
                    "Llama_cpp": ["model-a", "model-b"],
                    "OpenAI": ["gpt-5.1"],
                },
                pick_only=True,
            ),
        )
        pairs = switcher.query_one("#console-popover-pairs", OptionList)
        setup = [i for i, row in enumerate(switcher._rows) if row.kind == "setup"]
        assert setup and all(pairs.get_option_at_index(i).disabled for i in setup)
        lines = list_lines(app, switcher)
        assert line_with(lines, "NEEDS SETUP")
        for line in lines[min(setup) - 1 : max(setup) + 1]:
            assert "Enter" not in line, line

        seen = set()
        for _ in switcher._rows:
            await pilot.press("down")
            seen.add(switcher.highlighted_row().kind)
        assert "setup" not in seen

        await pilot.press(*"openai")
        await pilot.pause()
        assert {row.kind for row in switcher._rows} >= {"setup"}
        assert switcher.highlighted_row() is None
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is switcher and app.result == "unset"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not switcher

    assert app.result is None
    assert recorder.submissions == [] and recorder.setup == []
