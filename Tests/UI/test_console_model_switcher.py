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
from rich.cells import cell_len
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
#: A key is saved (or none is needed) and no model is chosen or listed.
NO_MODEL = ConsoleSettingsReadiness(
    "Not ready",
    "No model selected",
    False,
    operability="not_ready",
    blocker="model_missing",
    recovery_action="select_model",
    configuration="configured",
    credential="not_required",
    model="missing",
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
    **extra: object,
) -> ConsoleModelPopover:
    """One switcher with recording seams. RECENT loads after open.

    ``extra`` reaches the popover as is (``query``, ``served_models``, ...).
    """

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
        **extra,
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
        # Capture fix 3 finding 2: one more is "model", not "models".
        assert any("… 1 more Anthropic model" in line for line in lines)
        assert not any("… 1 more Anthropic models" in line for line in lines)

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


async def test_a_served_legacy_alias_is_listed_with_the_models_it_served() -> None:
    """TASK-33008.3: a server detected on a legacy alias's endpoint (llama.cpp
    on :8001) lists its models, so the card's switcher is not empty: serving
    them is use, which lists a legacy alias (and lets Find type an id on it)."""
    recorder = Recorder(ready=READY_PROVIDERS | {"local_llamacpp"})
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"Llama_cpp": ["model-a"]},
                served_models={"local_llamacpp": ("alias-a", "alias-b")},
                query="llama.cpp (legacy alias) ",
            ),
        )
        assert "local_llamacpp" in {p for p, _m, _t in recorder.readiness_calls}
        for model in ("alias-a", "alias-b"):
            assert "legacy alias" in line_with(list_lines(app, switcher), model)
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.kind, row.provider, row.model) == (
            "pair",
            "local_llamacpp",
            "alias-a",
        )


async def test_a_highlight_pair_outranks_previous_and_current_until_find_changes() -> (
    None
):
    """TASK-33008.3 AC#1: the card's detected switcher highlights the server's
    first listed model, although PREVIOUS and the chat's own model on that
    provider rank above it; once Find changes, typing ranks as ever."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                remembered_previous=_Use(
                    "llama_cpp", "model-b", timedelta(hours=2), True
                ),
                served_models={"llama_cpp": ("srv-1", "srv-2")},
                query="llama.cpp ",
                highlight_pair=("llama_cpp", "srv-1"),
            ),
        )
        rows = [(row.provider, row.model) for row in switcher._rows if row.model]
        # PREVIOUS and the chat's current model are listed ahead of it.
        assert rows.index(("llama_cpp", "model-b")) < rows.index(("llama_cpp", "srv-1"))
        assert rows.index(("llama_cpp", "model-a")) < rows.index(("llama_cpp", "srv-1"))
        row = switcher.highlighted_row()
        assert row is not None and (row.provider, row.model) == ("llama_cpp", "srv-1")

        find = switcher.query_one("#console-popover-find", Input)
        find.value = "srv-2"
        await pilot.pause()
        row = switcher.highlighted_row()
        assert row is not None and (row.provider, row.model) == ("llama_cpp", "srv-2")

        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("llama_cpp", "srv-2")


def _held_recorder() -> tuple[Recorder, threading.Event]:
    """A recorder whose readiness waits for the event (10 s at most)."""
    release = threading.Event()

    class SlowRecorder(Recorder):
        def readiness(self, provider, model):  # type: ignore[no-untyped-def]
            release.wait(10)
            return super().readiness(provider, model)

    return SlowRecorder(), release


async def test_a_highlight_pair_applies_while_providers_are_still_checked() -> None:
    """TASK-33008.3 (parent AC#1, Enter, Enter at speed): before the first
    readiness lands, the list holds only the chat's pair and "checking
    providers…". The detected server's first model is listed and highlighted
    there too, so a fast second Enter applies it instead of an error."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                draft=_draft("openai", None),
                served_models={"llama_cpp": ("srv-1", "srv-2")},
                query="llama.cpp ",
                highlight_pair=("llama_cpp", "srv-1"),
            )
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            assert any("checking providers" in row.text for row in switcher._rows)
            row = switcher.highlighted_row()
            assert row is not None and (row.kind, row.provider, row.model) == (
                "pair",
                "llama_cpp",
                "srv-1",
            )

            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is not switcher  # applied and closed, no error
            release.set()
    finally:
        release.set()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("llama_cpp", "srv-1")


_KEYLESS_USE = _Use("openai", "gpt-5.1", timedelta(hours=2))


@pytest.mark.parametrize(
    "held",
    [
        {},
        {"remembered_previous": _KEYLESS_USE},
        {
            "recent": (
                _Use("anthropic", "claude-sonnet-4-5", timedelta(hours=1)),
                _KEYLESS_USE,
            )
        },
    ],
    ids=["nothing-recent", "previous", "recent"],
)
async def test_a_highlight_pair_no_server_listed_waits_for_readiness(held) -> None:
    """TASK-33008.2: Switch model reopens on the pair of the NEEDS SETUP row
    the user left to fix. Unless a server listed that pair, nothing says it
    can send yet. While providers are checked it is not offered as a pair,
    and nothing is highlighted: PREVIOUS and RECENT list a pair whatever its
    readiness, and a fast Enter on their copy would apply a pair whose key
    was never saved. That Enter says providers are still checked, not
    "Choose a model: type to search": there is nothing to type or choose.
    Once readiness lands the line goes and the pair is highlighted as what
    it is, here its NEEDS SETUP row and not the copy above it, so Enter
    opens its fix again."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                query="OpenAI ",
                highlight_pair=("openai", "gpt-5.1"),
                **held,
            )
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            assert any("checking providers" in row.text for row in switcher._rows)
            # PREVIOUS's or RECENT's copy if there is one, and the chat's pair.
            listed = [(row.provider, row.model) for row in switcher._rows if row.model]
            assert listed == [("openai", "gpt-5.1")] * bool(held) + [
                ("llama_cpp", "model-a")
            ]
            assert switcher.highlighted_row() is None
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is switcher  # nothing applied, nothing opened
            assert recorder.submissions == [] and recorder.setup == []
            error = switcher.query_one("#console-popover-error")
            assert error.display and str(error.render()) == (
                "Still checking providers: press Enter again when the list fills."
            )

            release.set()
            await _settle(app, pilot)
            assert not error.display and str(error.render()) == ""
            kinds = [
                row.kind
                for row in switcher._rows
                if (row.provider, row.model) == ("openai", "gpt-5.1")
            ]
            assert kinds == ["pair"] * bool(held) + ["setup"]
            row = switcher.highlighted_row()
            assert row is not None and (row.kind, row.provider, row.model) == (
                "setup",
                "openai",
                "gpt-5.1",
            )
            await pilot.press("enter")
            await pilot.pause()
    finally:
        release.set()

    assert recorder.setup == [("openai", "gpt-5.1")]
    assert recorder.submissions == []


async def test_first_readiness_clears_no_error_but_the_one_about_the_wait() -> None:
    """TASK-33008.2: when the first readiness lands it clears Enter's "Still
    checking providers" line and no other. A value error raised while
    providers were checked is as true once the list fills."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            # RECENT's load lists the chat's own pair before readiness.
            switcher = build_switcher(recorder, recent=())
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            row = switcher.highlighted_row()
            assert row is not None and row.model == "model-a"
            await pilot.press("tab", *"hot", "enter")
            await pilot.pause()
            error = switcher.query_one("#console-popover-error")
            invalid = "Temperature must be a finite number from 0 to 2."
            assert str(error.render()) == invalid

            release.set()
            await _settle(app, pilot)
            assert switcher._first_readiness_done
            assert error.display and str(error.render()) == invalid
    finally:
        release.set()

    assert recorder.submissions == []


async def test_ctrl_o_with_no_model_before_first_readiness_names_the_wait() -> None:
    """TASK-33008.2: Ctrl+O and Chat settings… validate the draft without
    passing through Enter's path. On a chat with no model, nothing is
    highlighted while providers are checked when no PREVIOUS or RECENT row
    is listed yet (here) or the list reopened on a NEEDS SETUP pair, and
    with no row the draft has no pair to carry. Ctrl+O then gets the line
    Enter gets, not "Choose a model: type to search", and the first
    readiness clears it. A highlighted row gives the draft its pair, so
    Ctrl+O is not refused (the next test)."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(recorder, draft=_draft("openai", None))
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            assert switcher.highlighted_row() is None
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert app.screen is switcher  # nothing to carry, nothing opened
            error = switcher.query_one("#console-popover-error")
            assert error.display and str(error.render()) == (
                "Still checking providers: press Enter again when the list fills."
            )

            release.set()
            await _settle(app, pilot)
            assert switcher._first_readiness_done
            assert not error.display and str(error.render()) == ""
    finally:
        release.set()

    assert recorder.submissions == []


async def test_ctrl_o_with_a_row_highlighted_before_first_readiness_carries_it() -> (
    None
):
    """TASK-33008.2: the guide's "with no row highlighted" decides Ctrl+O
    too. On a chat with no model, RECENT's load lists PREVIOUS and highlights
    it while providers are still checked. The draft has that row's pair, so
    Ctrl+O opens Chat settings on it and names no wait."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    previous = ("anthropic", "claude-sonnet-4-5")
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                draft=_draft("openai", None),
                recent=(),
                remembered_previous=_Use(*previous, timedelta(hours=2)),
            )
            await app.push_screen(
                switcher, callback=lambda result: setattr(app, "result", result)
            )
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            assert any("checking providers" in row.text for row in switcher._rows)
            row = switcher.highlighted_row()
            assert row is not None and (row.provider, row.model) == previous
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert app.screen is not switcher
    finally:
        release.set()

    assert recorder.submissions == []
    assert isinstance(app.result, ConsoleSettingsTransfer)
    settings = app.result.draft.settings
    assert (settings.provider, settings.model) == previous


@pytest.mark.parametrize(
    "own_models, pair, reopened, kept",
    [
        (["gpt-5.1"], "gpt-5.1", None, "gpt-5.1"),
        # A return from Settings with no row for its pair: the row named no
        # model, or the catalog no longer lists the one it named.
        (["gpt-5.1", "gpt-b"], None, "openai", "gpt-5.1"),
        (["gpt-a", "gpt-b"], "gpt-5.1", "openai", "gpt-a"),
    ],
    ids=["the-pair", "returned-no-pair", "returned-pair-no-longer-listed"],
)
async def test_a_highlight_pair_past_the_match_cap_stays_listed(
    own_models, pair, reopened, kept
) -> None:
    """TASK-33008.2: "OpenAI " in Find also matches every aggregator id that
    starts with it, and those rank above OpenAI's own models, which match by
    the provider's name alone. With 30 of them the cut at 30 rows left the
    row's pair out, and Enter applied Find's best match: another provider's
    model. While Find is as it opened, the pair takes the last listed place.
    The cap is unchanged, and so is the ranking once Find changes.

    Controller ruling of 2026-10-11: a list reopened from Settings with no
    row for its pair keeps the returned provider's first row the same way.
    Cut, none of that provider's rows was listed, nothing was highlighted and
    no line named the provider (``kept`` is the model of the row kept)."""
    recorder = Recorder(ready=READY_PROVIDERS | {"openai"})
    aggregator = [f"openai/gpt-{number}" for number in range(30)]
    app = SwitcherHarness()

    def listed(switcher) -> list[tuple[str, str | None]]:
        return [(row.provider, row.model) for row in switcher._rows if row.model]

    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"OpenAI": own_models, "OpenRouter": aggregator},
                query="OpenAI ",
                highlight_pair=("openai", pair) if pair else None,
                reopened_provider=reopened,
            ),
        )
        assert listed(switcher) == [
            *(("openrouter", model) for model in aggregator[:29]),
            ("openai", kept),
        ]
        more = len(aggregator) + len(own_models) - 30
        # Capture fix 3 finding 2: "… 1 more matches" counts one in the singular.
        matches = "match" if more == 1 else "matches"
        assert any(
            row.text == f"    … {more} more {matches} · keep typing"
            for row in switcher._rows
        )
        row = switcher.highlighted_row()
        assert row is not None and (row.kind, row.provider, row.model) == (
            "pair",
            "openai",
            kept,
        )

        find = switcher.query_one("#console-popover-find", Input)
        find.value = "OpenAI"  # the user's own query: ranked as ever
        await pilot.pause()
        assert listed(switcher) == [("openrouter", model) for model in aggregator]
        find.value = "OpenAI "
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("openai", kept)


#: Three keyless providers listed ahead of OpenAI, three rows each for
#: "OpenAI" in Find: they fill NEEDS SETUP's eight rows before its turn.
_EIGHT_ROWS_AHEAD = {
    name: [f"openai-{name}-{number}" for number in range(3)]
    for name in ("Anthropic", "DeepSeek", "Groq")
}


@pytest.mark.parametrize(
    "more, line",
    [
        ({"OpenAI": ["zzq-x"]}, "… 1 more provider needs setup"),
        (
            {"OpenAI": ["zzq-x"], "OpenRouter": ["zzq-y"]},
            "… 2 more providers need setup",
        ),
    ],
    ids=["one", "two"],
)
async def test_the_needs_setup_overflow_line_counts_one_in_the_singular(
    more, line
) -> None:
    """Capture fix 3 finding 2 (05a, 05c): NEEDS SETUP's last line read
    "… 1 more providers need setup". Only these providers list a "zzq" id:
    three fill the group's eight rows, the rest are counted."""
    recorder = Recorder(ready=frozenset())
    ahead = {name: [f"zzq-{name}-{number}" for number in range(3)] for name in _EIGHT_ROWS_AHEAD}
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={**ahead, **more},
                query="zzq",
            ),
        )
        texts = [row.text for row in switcher._rows if row.kind == "info"]
    assert f"    {line} · type a name to find one" in texts, texts


@pytest.mark.parametrize(
    "providers_models, chat, model, own_rows, typed_rows",
    [
        # A provider lists three NEEDS SETUP rows; the pair is its fourth
        # model, and a ready aggregator's id outranks them all.
        (
            {
                "OpenAI": ["gpt-a", "gpt-b", "gpt-c", "gpt-5.1"],
                "OpenRouter": ["openai/gpt-y"],
            },
            ("llama_cpp", "model-a"),
            "gpt-5.1",
            ["gpt-a", "gpt-b", "gpt-5.1"],
            ["gpt-a", "gpt-b", "gpt-c"],
        ),
        # Eight NEEDS SETUP rows end the group; three providers listed ahead
        # of OpenAI fill it.
        (
            {**_EIGHT_ROWS_AHEAD, "OpenAI": ["gpt-5.1"]},
            ("llama_cpp", "model-a"),
            "gpt-5.1",
            ["gpt-5.1"],
            [],
        ),
        # The same eight rows, and a return that names no model: OpenAI
        # lists none, so its one NEEDS SETUP row names none and is no pair's.
        (
            {**_EIGHT_ROWS_AHEAD, "OpenAI": []},
            ("llama_cpp", "model-a"),
            None,
            [None],
            [],
        ),
        # The pair is the chat's own, on a model no saved list holds: with
        # nothing in Find its NEEDS SETUP row names that model, and a query
        # lists saved models only.
        (
            {"OpenAI": ["gpt-a"], "OpenRouter": ["openai/gpt-y"]},
            ("openai", "gpt-5.1"),
            "gpt-5.1",
            ["gpt-a", "gpt-5.1"],
            ["gpt-a"],
        ),
    ],
    ids=[
        "three-rows-a-provider",
        "eight-rows-a-group",
        "eight-rows-a-group-no-pair",
        "a-model-no-list-holds",
    ],
)
async def test_a_highlight_pair_keeps_its_needs_setup_row(
    providers_models, chat, model, own_rows, typed_rows
) -> None:
    """TASK-33008.2: the NEEDS SETUP rows left the row's pair out too, and
    Enter then acted on Find's best match: it applied a ready aggregator's
    model, or opened another provider's fix. While Find is as it opened, the
    pair's own row is listed (in the third row's place, when the provider
    has more) and highlighted, and Enter opens its fix. A return from
    Settings that names no model (``model`` is None: its provider lists none)
    keeps that provider's one row the same way; the row went into "… N more
    providers need setup" and nothing was highlighted. Once Find changes the
    caps cut as ever (``typed_rows``)."""
    recorder = Recorder(ready=frozenset({"openrouter"}))
    app = SwitcherHarness()

    def own(switcher) -> list[str | None]:
        return [row.model for row in switcher._rows if row.provider == "openai"]

    def highlight(switcher) -> tuple[str, str, str | None] | None:
        row = switcher.highlighted_row()
        return None if row is None else (row.kind, row.provider, row.model)

    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                draft=_draft(*chat),
                providers_models=providers_models,
                query="OpenAI ",
                highlight_pair=("openai", model) if model else None,
                reopened_provider=None if model else "openai",
            ),
        )
        assert own(switcher) == own_rows
        assert highlight(switcher) == ("setup", "openai", model)

        find = switcher.query_one("#console-popover-find", Input)
        find.value = "OpenAI"  # the user's own query: the caps cut as ever
        await pilot.pause()
        assert own(switcher) == typed_rows
        find.value = "OpenAI "
        await pilot.pause()
        assert own(switcher) == own_rows
        assert highlight(switcher) == ("setup", "openai", model)

        await pilot.press("enter")
        await pilot.pause()

    assert recorder.setup == [("openai", model)]
    assert recorder.submissions == []


_AGGREGATOR_ROW = ("openrouter", "openai/gpt-y")


@pytest.mark.parametrize(
    "own_models, saved, model, highlighted",
    [
        # The row named no model (its provider lists none). With the key
        # saved the provider has no row to offer; unsaved, its own NEEDS
        # SETUP row is still there.
        ([], True, None, None),
        ([], False, None, ("setup", "openai", None)),
        # The key is saved, and the catalog no longer lists the row's model.
        (["gpt-a", "gpt-b"], True, "gpt-5.1", ("pair", "openai", "gpt-a")),
    ],
    ids=["no-models-saved", "no-models-unsaved", "model-no-longer-listed"],
)
async def test_a_list_reopened_from_settings_never_highlights_another_providers_row(
    own_models, saved, model, highlighted
) -> None:
    """TASK-33008.2 (controller ruling of 2026-10-11): a NEEDS SETUP row's
    return reopens Switch model with the provider's name in Find. A ready
    aggregator's id that starts with that name is Find's best match, and with
    no row for the returned pair the highlight fell back to it, so the Enter
    the guide describes applied another provider's model. Reopened this way,
    the highlight is the returned pair's row, else that provider's first row,
    else nothing, until Find changes.

    With nothing to highlight (the key is saved and the provider lists no
    models) the list said nothing about that provider: text in Find lists
    matches only. Until Find changes, READY PROVIDERS opens with that
    provider's own line, and the id typed after its name applies."""
    ready = {"openrouter", "openai"} if saved else {"openrouter"}
    recorder = Recorder(ready=frozenset(ready))
    app = SwitcherHarness()

    def highlight(switcher) -> tuple[str, str, str | None] | None:
        row = switcher.highlighted_row()
        return None if row is None else (row.kind, row.provider, row.model)

    def says_no_models(switcher) -> bool:
        lines = list_lines(app, switcher)
        line = "OpenAI · no models reported · type its name and an id"
        said = [index for index, text in enumerate(lines) if line in text]
        # First under the heading, where 30 other matches cannot hide it.
        assert all("READY PROVIDERS" in lines[index - 1] for index in said)
        return bool(said)

    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"OpenAI": own_models, "OpenRouter": ["openai/gpt-y"]},
                query="OpenAI ",
                highlight_pair=("openai", model) if model else None,
                reopened_provider="openai",
            ),
        )
        listed = [(row.provider, row.model) for row in switcher._rows if row.model]
        assert _AGGREGATOR_ROW in listed and ("openai", model) not in listed
        assert highlight(switcher) == highlighted
        assert says_no_models(switcher) == (highlighted is None)

        find = switcher.query_one("#console-popover-find", Input)
        find.value = "OpenAI"  # the user's own query: Find's best match again
        await pilot.pause()
        assert highlight(switcher) == ("pair", *_AGGREGATOR_ROW)
        assert not says_no_models(switcher)
        find.value = "OpenAI "
        await pilot.pause()
        assert highlight(switcher) == highlighted
        assert says_no_models(switcher) == (highlighted is None)

        await pilot.press("enter")
        await pilot.pause()
        expected = [highlighted[1:]] if highlighted and saved else []
        if highlighted is None:
            assert app.screen is switcher  # nothing applied by accident
            error = switcher.query_one("#console-popover-error")
            assert str(error.render()) == "Choose a model: type to search, then Enter."
            assert recorder.submissions == []

            find.value = "OpenAI gpt-new"  # what the line asks for
            await pilot.pause()
            assert highlight(switcher) == ("typed", "openai", "gpt-new")
            await pilot.press("enter")
            await pilot.pause()
            expected = [("openai", "gpt-new")]

    applied = [
        (submission.draft.settings.provider, submission.draft.settings.model)
        for submission in recorder.submissions
    ]
    assert applied == expected
    assert recorder.setup == ([] if saved else [("openai", None)])


async def test_a_list_reopened_with_no_pair_waits_for_readiness_too() -> None:
    """TASK-33008.2: before readiness RECENT can already list the
    aggregator's id, and it is Find's best match there too. A list reopened
    from Settings highlights nothing until providers are checked, with or
    without a pair, so a fast Enter applies no other provider's model."""
    recorder, release = _held_recorder()
    recorder.ready = frozenset({"openrouter"})
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                providers_models={"OpenAI": [], "OpenRouter": ["openai/gpt-y"]},
                recent=(_Use(*_AGGREGATOR_ROW, timedelta(hours=1)),),
                query="OpenAI ",
                reopened_provider="openai",
            )
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            listed = [(row.provider, row.model) for row in switcher._rows if row.model]
            assert _AGGREGATOR_ROW in listed
            assert switcher.highlighted_row() is None
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is switcher and recorder.submissions == []

            release.set()
            await _settle(app, pilot)
            row = switcher.highlighted_row()
            assert row is not None and (row.kind, row.provider, row.model) == (
                "setup",
                "openai",
                None,
            )
    finally:
        release.set()

    assert recorder.submissions == [] and recorder.setup == []


async def test_a_returned_provider_says_once_that_its_models_are_loading() -> None:
    """TASK-33008.2: the returned provider's line leads READY PROVIDERS, and
    every loading catalog has a line after the matches. While its own catalog
    loads it has the one line, in the leading place, which then becomes "no
    models reported"."""
    release = asyncio.Event()

    async def catalog(provider: str) -> list[str]:
        if provider == "openai":
            await release.wait()
            return []
        return ["openai/gpt-y"]

    def about_openai(switcher) -> list[str]:
        lines = [line.strip() for line in list_lines(app, switcher)]
        ready = next(i for i, line in enumerate(lines) if "READY PROVIDERS" in line)
        assert lines[ready + 1].startswith("OpenAI · "), lines
        return [line for line in lines if line.startswith("OpenAI · ")]

    recorder = Recorder(ready=frozenset({"openrouter", "openai"}))
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build_switcher(
            recorder,
            providers_models={"OpenAI": [], "OpenRouter": ["openai/gpt-y"]},
            catalog_loader=catalog,
            query="OpenAI ",
            reopened_provider="openai",
        )
        app.push_screen(switcher)
        for _ in range(20):
            await pilot.pause()
            if "loading models…" in "\n".join(list_lines(app, switcher)):
                break
        assert about_openai(switcher) == ["OpenAI · loading models…"]
        release.set()
        await _settle(app, pilot)
        assert about_openai(switcher) == [
            "OpenAI · no models reported · type its name and an id"
        ]
        assert switcher.highlighted_row() is None


async def test_a_returned_provider_with_a_key_and_no_model_says_what_to_type() -> None:
    """Controller ruling (c) of 2026-10-11, on the reopened list. A NEEDS
    SETUP row of a provider that lists no models routes no model, and
    Settings no longer keeps another provider's model beside it. With its key
    saved the provider is therefore still under NEEDS SETUP, now for its
    model, and Enter on its row opens Settings again. Until Find changes,
    READY PROVIDERS opens with the provider's own line, as it does for a
    ready provider that lists nothing, and the id typed after its name
    applies."""
    recorder = Recorder(ready=frozenset({"openrouter"}))
    others = recorder.readiness
    recorder.readiness = lambda provider, model: (
        NO_MODEL if provider == "openai" else others(provider, model)
    )
    app = SwitcherHarness()
    line = "OpenAI · no models reported · type its name and an id"

    def said(switcher) -> bool:
        lines = list_lines(app, switcher)
        at = [index for index, text in enumerate(lines) if line in text]
        assert all("READY PROVIDERS" in lines[index - 1] for index in at)
        return bool(at)

    def highlight(switcher) -> tuple[str, str, str | None] | None:
        row = switcher.highlighted_row()
        return None if row is None else (row.kind, row.provider, row.model)

    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"OpenAI": [], "OpenRouter": ["openai/gpt-y"]},
                query="OpenAI ",
                reopened_provider="openai",
            ),
        )
        assert highlight(switcher) == ("setup", "openai", None)
        assert said(switcher)

        find = switcher.query_one("#console-popover-find", Input)
        find.value = "OpenAI"  # the user's own query: matches only
        await pilot.pause()
        assert not said(switcher)

        find.value = "OpenAI gpt-new"  # what the line asks for
        await pilot.pause()
        assert highlight(switcher) == ("typed", "openai", "gpt-new")
        await pilot.press("enter")
        await pilot.pause()

    applied = [
        (submission.draft.settings.provider, submission.draft.settings.model)
        for submission in recorder.submissions
    ]
    assert applied == [("openai", "gpt-new")]
    assert recorder.setup == []


@pytest.mark.parametrize("reopened", [False, True], ids=["any-open", "reopened"])
async def test_a_provider_whose_only_blocker_is_its_model_lists_its_catalog(
    reopened: bool,
) -> None:
    """Controller ruling of 2026-10-12 (final fix wave 1, item 1, Option A).

    A provider with its key saved and no model (24 of the 60 shipped lists
    are empty) read model_missing, and catalogs loaded for ready providers
    only, so its models were never listed. Its catalog loads too, on every
    open, and its readiness is asked again with the first model it lists:
    its models are READY rows, and a list reopened on it highlights the first
    and applies it on Enter. A provider with another blocker lists nothing."""
    recorder = Recorder(ready=frozenset({"openrouter"}))
    others = recorder.readiness

    def readiness(provider: str, model: str | None) -> ConsoleSettingsReadiness:
        verdict = others(provider, model)  # records the call
        return (READY if model else NO_MODEL) if provider == "openai" else verdict

    async def catalog(provider: str) -> list[str]:
        recorder.catalog_calls.append(provider)
        return {"openai": ["gpt-new", "gpt-old"], "groq": ["llama-g"]}.get(
            provider, ["openai/gpt-y"]
        )

    recorder.readiness = readiness
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                draft=_draft("openrouter", "openai/gpt-y"),
                providers_models={"OpenAI": [], "Groq": [], "OpenRouter": []},
                catalog_loader=catalog,
                **(
                    {"query": "OpenAI ", "reopened_provider": "openai"}
                    if reopened
                    else {}
                ),
            ),
        )
        keys = _row_keys(switcher)
        assert ("pair", "openai", "gpt-new") in keys
        assert not any(key[:2] == ("setup", "openai") for key in keys)
        assert ("openai", "gpt-new") in [call[:2] for call in recorder.readiness_calls]
        # Another blocker: no catalog, no model rows.
        assert "groq" not in recorder.catalog_calls
        assert not any(key[1] == "groq" and key[0] != "setup" for key in keys)
        if reopened:
            row = switcher.highlighted_row()
            assert (row.kind, row.provider, row.model) == ("pair", "openai", "gpt-new")
            await pilot.press("enter")
            await pilot.pause()
    if reopened:
        applied = [
            (submission.draft.settings.provider, submission.draft.settings.model)
            for submission in recorder.submissions
        ]
        assert applied == [("openai", "gpt-new")]


async def test_a_reopened_list_highlights_nothing_while_its_providers_models_arrive() -> (
    None
):
    """Final fix wave 3, item 1. A list reopened on a provider whose only
    blocker is its model settles on its catalog's first model, READY, and
    Enter applies it. Until then the list held the NEEDS SETUP "(any model)"
    row while the catalog loaded, then up to three NEEDS SETUP catalog rows
    while readiness was asked again, and a fast Enter on either opened
    Settings again. Nothing is highlighted until both have answered.

    Final fix wave 4: readiness is asked again as soon as the provider's own
    catalog answers, not after every later provider's catalog, so a rebuild
    in between (a local server's probe settling) highlights no NEEDS SETUP
    catalog row. That early Enter says providers are still checked, and the
    line goes when the list settles, as it does for the first readiness."""
    catalog_release = asyncio.Event()
    later_release = asyncio.Event()  # the provider after it in the order
    readiness_release = threading.Event()
    checking = "Still checking providers: press Enter again when the list fills."
    recorder = Recorder(ready=frozenset({"openrouter"}))
    others = recorder.readiness

    def readiness(provider: str, model: str | None) -> ConsoleSettingsReadiness:
        verdict = others(provider, model)
        if provider != "openai":
            return verdict
        if model:
            readiness_release.wait(10)
            return READY
        return NO_MODEL

    async def catalog(provider: str) -> list[str]:
        if provider == "openai":
            await catalog_release.wait()
            return ["gpt-new", "gpt-old"]
        await later_release.wait()
        return ["openai/gpt-y"]

    def error_line() -> str:
        error = switcher.query_one("#console-popover-error")
        return str(error.render()) if error.display else ""

    async def enter_does_nothing(pilot) -> None:
        assert switcher.highlighted_row() is None
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is switcher
        assert recorder.setup == [] and recorder.submissions == []
        assert error_line() == checking

    recorder.readiness = readiness
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                draft=_draft("openrouter", "openai/gpt-y"),
                providers_models={"OpenAI": [], "OpenRouter": []},
                catalog_loader=catalog,
                query="OpenAI ",
                reopened_provider="openai",
            )
            app.push_screen(switcher, callback=lambda result: None)
            for _ in range(40):
                await pilot.pause()
                if switcher._catalog_state.get("openai") == "loading":
                    break
            assert switcher._catalog_state.get("openai") == "loading"
            await enter_does_nothing(pilot)

            catalog_release.set()
            for _ in range(40):
                await pilot.pause()
                if "openai" in switcher._catalogs:
                    break
            # OpenRouter's catalog is still loading; a rebuild lands now.
            assert switcher._catalog_state.get("openrouter") == "loading"
            switcher._rebuild_rows()
            assert ("setup", "openai", "gpt-new") in _row_keys(switcher)
            await enter_does_nothing(pilot)
            assert "openai" in switcher._readiness_pending

            later_release.set()
            for _ in range(40):
                await pilot.pause()
                if switcher._catalog_state.get("openrouter") == "ready":
                    break
            assert switcher._catalog_state.get("openrouter") == "ready"
            assert switcher.highlighted_row() is None
            assert error_line() == checking

            readiness_release.set()
            await _settle(app, pilot)
            row = switcher.highlighted_row()
            assert row is not None
            assert (row.kind, row.provider, row.model) == ("pair", "openai", "gpt-new")
            assert error_line() == ""
            await pilot.press("enter")
            await pilot.pause()
    finally:
        catalog_release.set()
        later_release.set()
        readiness_release.set()

    applied = [
        (submission.draft.settings.provider, submission.draft.settings.model)
        for submission in recorder.submissions
    ]
    assert applied == [("openai", "gpt-new")]
    assert recorder.setup == []


async def test_the_wait_line_goes_when_the_returned_catalog_lists_nothing() -> None:
    """Final fix wave 4, item 2. An early Enter while the returned provider's
    catalog loads says providers are still checked. A catalog that lists
    nothing starts no second readiness, and the line still goes when it
    settles."""
    release = asyncio.Event()
    recorder = Recorder(ready=frozenset({"openrouter"}))
    others = recorder.readiness

    def readiness(provider: str, model: str | None) -> ConsoleSettingsReadiness:
        verdict = others(provider, model)
        return NO_MODEL if provider == "openai" else verdict

    async def catalog(provider: str) -> list[str]:
        if provider == "openai":
            await release.wait()
            return []
        return ["openai/gpt-y"]

    recorder.readiness = readiness
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                draft=_draft("openrouter", "openai/gpt-y"),
                providers_models={"OpenAI": [], "OpenRouter": []},
                catalog_loader=catalog,
                query="OpenAI ",
                reopened_provider="openai",
            )
            app.push_screen(switcher, callback=lambda result: None)
            for _ in range(40):
                await pilot.pause()
                if switcher._catalog_state.get("openai") == "loading":
                    break
            assert switcher._catalog_state.get("openai") == "loading"
            await pilot.press("enter")
            await pilot.pause()
            error = switcher.query_one("#console-popover-error")
            assert error.display and str(error.render()) == (
                "Still checking providers: press Enter again when the list fills."
            )

            release.set()
            await _settle(app, pilot)
            assert switcher._catalog_state.get("openai") == "empty"
            assert not error.display and str(error.render()) == ""
    finally:
        release.set()

    assert recorder.setup == [] and recorder.submissions == []


async def test_the_wait_line_holds_when_enter_lands_during_the_first_readiness() -> (
    None
):
    """Final fix wave 5, item 1. The double Enter's second press usually lands
    before the first readiness. That readiness rebuilt the list before the
    returned provider's catalog was marked loading, so the rebuild cleared the
    checking line (and briefly highlighted its NEEDS SETUP row) while its
    models were still on the way. The line holds until the list settles."""
    readiness_release = threading.Event()
    catalog_release = asyncio.Event()
    checking = "Still checking providers: press Enter again when the list fills."
    recorder = Recorder(ready=frozenset({"openrouter"}))
    others = recorder.readiness

    def readiness(provider: str, model: str | None) -> ConsoleSettingsReadiness:
        verdict = others(provider, model)
        if provider != "openai":
            return verdict
        if model:
            return READY
        readiness_release.wait(10)
        return NO_MODEL

    async def catalog(provider: str) -> list[str]:
        if provider == "openai":
            await catalog_release.wait()
            return ["gpt-new", "gpt-old"]
        return ["openai/gpt-y"]

    def error_line() -> str:
        error = switcher.query_one("#console-popover-error")
        return str(error.render()) if error.display else ""

    recorder.readiness = readiness
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(
                recorder,
                draft=_draft("openrouter", "openai/gpt-y"),
                providers_models={"OpenAI": [], "OpenRouter": []},
                catalog_loader=catalog,
                query="OpenAI ",
                reopened_provider="openai",
            )
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            assert not switcher._first_readiness_done
            await pilot.press("enter")
            await pilot.pause()
            assert error_line() == checking

            highlighted: list[tuple[str, str] | None] = []
            rebuild = switcher._rebuild_rows

            def spied_rebuild() -> None:
                rebuild()
                row = switcher.highlighted_row()
                highlighted.append(None if row is None else (row.kind, row.provider))

            switcher._rebuild_rows = spied_rebuild
            readiness_release.set()
            for _ in range(40):
                await pilot.pause()
                if switcher._catalog_state.get("openai") == "loading":
                    break
            assert switcher._catalog_state.get("openai") == "loading"
            assert switcher._first_readiness_done
            assert switcher.highlighted_row() is None
            assert error_line() == checking
            assert highlighted and ("setup", "openai") not in highlighted

            catalog_release.set()
            await _settle(app, pilot)
            row = switcher.highlighted_row()
            assert row is not None
            assert (row.kind, row.provider, row.model) == ("pair", "openai", "gpt-new")
            assert error_line() == ""
    finally:
        readiness_release.set()
        catalog_release.set()

    assert recorder.setup == [] and recorder.submissions == []


@pytest.mark.parametrize(
    "catalog", [None, ["gpt-new", "gpt-old"]], ids=["lists-nothing", "lists-others"]
)
async def test_a_ready_returned_provider_lists_the_model_just_saved(
    catalog: list[str] | None,
) -> None:
    """Final fix wave 3, item 3. A model saved in Settings' Model field for a
    provider is the pair the list reopens on, and the provider is ready on it.
    READY PROVIDERS listed only its [providers] list or catalog, never its
    saved model: with neither the list said "no models reported" and
    highlighted nothing, and with a catalog Enter applied the catalog's first
    model. A pair that is not the saved model and that no list holds is still
    not listed (test_a_list_reopened_from_settings_never_highlights_another_
    providers_row[model-no-longer-listed])."""

    async def load(provider: str) -> list[str]:
        return list(catalog or ()) if provider == "openai" else ["openai/gpt-y"]

    recorder = Recorder(ready=frozenset({"openrouter", "openai"}))
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                draft=_draft("openrouter", "openai/gpt-y"),
                providers_models={"OpenAI": [], "OpenRouter": []},
                app_config={"api_settings": {"openai": {"model": "llama-x"}}},
                catalog_loader=load,
                query="OpenAI ",
                highlight_pair=("openai", "llama-x"),
                reopened_provider="openai",
            ),
        )
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.kind, row.provider, row.model) == ("pair", "openai", "llama-x")
        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("openai", "llama-x")


async def test_a_query_the_list_opened_with_is_typed_after_not_over() -> None:
    """TASK-33008.2: the list reopens with the provider's name and a space in
    Find, and its "no models reported" line asks for an id after the name.
    Focus selected the whole query, so the first key typed replaced the name
    and the id paired with the chat's provider (seen live: Enter moved the
    chat to OpenAI · llama-live-x where SambaNova was meant). A query the
    list opens with starts with the cursor after it and nothing selected, as
    the "… N more" row leaves it."""
    recorder = Recorder(ready=frozenset({"openrouter", "openai"}))
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"OpenAI": [], "OpenRouter": ["openai/gpt-y"]},
                query="OpenAI ",
                reopened_provider="openai",
            ),
        )
        await pilot.press(*"gpt-new")  # real keys: a set value selects nothing
        await pilot.pause()
        find = switcher.query_one("#console-popover-find", Input)
        assert find.value == "OpenAI gpt-new"
        row = switcher.highlighted_row()
        assert row is not None and (row.kind, row.provider, row.model) == (
            "typed",
            "openai",
            "gpt-new",
        )
        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("openai", "gpt-new")


async def test_a_typed_id_another_provider_lists_under_the_name_is_one_down_away() -> (
    None
):
    """TASK-33008.2: what console.md says of the id typed after the returned
    provider's name. Once Find changes the highlight is Find's best match, and
    a listed id outranks the TYPED MODEL ID row: an aggregator's
    "openai/gpt-y" holds both words of "OpenAI gpt-y", so it is highlighted
    first. The typed row is the list's last, so Down reaches it and Enter
    there applies the returned provider's pair.

    It pins the guide's sentence, not a fix: whether the reopened list should
    rank only the returned provider's rows here is with the controller."""
    recorder = Recorder(ready=frozenset({"openrouter", "openai"}))
    app = SwitcherHarness()

    def highlight(switcher) -> tuple[str, str, str | None]:
        row = switcher.highlighted_row()
        return (row.kind, row.provider, row.model)

    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                providers_models={"OpenAI": [], "OpenRouter": ["openai/gpt-y"]},
                query="OpenAI ",
                reopened_provider="openai",
            ),
        )
        assert switcher.highlighted_row() is None
        await pilot.press(*"gpt-y")
        await pilot.pause()
        assert highlight(switcher) == ("pair", *_AGGREGATOR_ROW)
        assert switcher._rows[-1].kind == "typed"
        await pilot.press("down")
        await pilot.pause()
        assert highlight(switcher) == ("typed", "openai", "gpt-y")
        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("openai", "gpt-y")


async def test_a_more_rows_provider_name_is_typed_after_from_the_values_too() -> None:
    """TASK-33008.2: Enter on an "… N more" row puts the provider's name in
    Find and focuses it. With focus in Temperature (Down moves the highlight
    from the values) that focus selected the whole name, because a list
    opened with an empty Find selects on focus: the first key replaced the
    name and the search was no longer that provider's."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        find = switcher.query_one("#console-popover-find", Input)
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is switcher.query_one("#console-popover-temperature", Input)
        for _ in switcher._rows:
            if switcher.highlighted_row().kind == "more":
                break
            await pilot.press("down")
            await pilot.pause()
        assert "more Anthropic models" in switcher.highlighted_row().text
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is find and find.value == "Anthropic "
        await pilot.press(*"son")
        await pilot.pause()
        assert find.value == "Anthropic son"
        assert switcher.highlighted_row().model == "claude-sonnet-4-5"


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


def _setup_route_screen(readiness=lambda _provider, _model: NO_KEY):
    """A Console stand-in with the real handoff store the route stages into."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore

    posted: list[object] = []
    screen = SimpleNamespace(
        _console_default_readiness=readiness,
        _ensure_console_chat_store=lambda: SimpleNamespace(
            active_session_id="session-1"
        ),
        app_instance=SimpleNamespace(pending_handoffs=PendingHandoffStore()),
        post_message=posted.append,
    )
    return screen, posted


def test_needs_setup_route_names_the_provider_its_model_and_the_field() -> None:
    """AC#14 (R11): the route reuses the recovery keys and carries the pair.

    TASK-33008.1 (R32): it is also the Console's way back. It stages a
    return naming only the chat, and the context carries that revision.
    """
    from tldw_chatbook.Constants import TAB_SETTINGS
    from tldw_chatbook.UI.Console_Modules.model_switcher import open_provider_setup
    from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
        ConsoleProviderReturnIntent,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    resolved: list[tuple[str, str | None]] = []
    screen, posted = _setup_route_screen(
        lambda provider, model: resolved.append((provider, model)) or NO_KEY
    )

    open_provider_setup(screen, "openrouter", "openai/gpt-4o-mini")

    (message,) = posted
    assert message.screen_name == TAB_SETTINGS
    assert message.screen_context == {
        "category": "providers-models",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "field": "api_key",
        "return_revision": 1,
        "session_id": "session-1",
    }
    assert resolved == [("openrouter", "openai/gpt-4o-mini")]
    handoffs = screen.app_instance.pending_handoffs
    claim = handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert claim is not None and claim.revision == 1
    assert claim.value == ConsoleProviderReturnIntent("session-1")
    handoffs.release(claim)
    # A navigation that never leaves the Console takes its return back.
    message.report_completion(False)
    assert handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


@pytest.mark.parametrize(
    "recovery_action, field",
    [
        ("configure_credential", "api_key"),
        ("configure_endpoint", "endpoint"),
        ("save_endpoint", "endpoint"),
        # A key and no model: the Default model field, not no field at all.
        ("select_model", "model"),
    ],
)
def test_a_needs_setup_row_route_carries_its_field_and_the_pair_to_reopen_on(
    recovery_action, field
) -> None:
    """TASK-33008.2 AC#1 (R42, R43): a NEEDS SETUP row leaves through the
    return route, at the control its blocker names. Its return also names the
    provider and model Switch model reopens on. Settings is not told them:
    its context is the recovery's own."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.model_switcher import open_provider_setup
    from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
        ConsoleProviderReturnIntent,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    # The route reads the readiness for its recovery action and nothing else.
    screen, posted = _setup_route_screen(
        lambda _provider, _model: SimpleNamespace(recovery_action=recovery_action)
    )

    open_provider_setup(
        screen, "openrouter", "openai/gpt-4o-mini", reopen_switcher=True
    )

    (message,) = posted
    assert message.screen_context == {
        "category": "providers-models",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "field": field,
        "return_revision": 1,
        "session_id": "session-1",
    }
    claim = screen.app_instance.pending_handoffs.claim(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN
    )
    assert claim is not None and claim.value == ConsoleProviderReturnIntent(
        "session-1", "openrouter", "openai/gpt-4o-mini"
    )


def test_setup_route_a_target_settings_cannot_name_goes_one_way() -> None:
    """R35: a provider the return target rejects stages nothing; the route
    still opens Settings as before (a custom endpoint's ``custom-ep:`` key)."""
    from tldw_chatbook.UI.Console_Modules.model_switcher import open_provider_setup
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    screen, posted = _setup_route_screen()

    open_provider_setup(screen, "custom-ep:3f2a", "local-model")

    (message,) = posted
    assert message.screen_context == {
        "category": "providers-models",
        "provider": "custom-ep:3f2a",
        "model": "local-model",
        "field": "api_key",
    }
    assert not screen.app_instance.pending_handoffs.has_pending(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN
    )


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


@pytest.mark.parametrize(
    ("provider", "model", "refused"),
    (("anthropic", "claude-sonnet-4-5", True), ("llama_cpp", "model-a", False)),
)
async def test_temperature_is_held_to_the_drafts_provider_range(
    provider: str, model: str, refused: bool
) -> None:
    """TASK-33007 capture finding 25 (ruling cr1-N1 (a)): Alt+M refuses what
    Ctrl+O refuses. Anthropic takes 0 to 1, so 1.5 is refused with that
    range; llama.cpp takes 0 to 2 and applies it."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, build_switcher(recorder, draft=_draft(provider, model))
        )
        assert switcher.highlighted_row().model == model
        switcher.query_one("#console-popover-temperature", Input).value = "1.5"
        await pilot.pause()
        switcher.query_one("#console-popover-apply", Button).press()
        await pilot.pause()
        error = (
            str(switcher.query_one("#console-popover-error", Static).render())
            if app.screen is switcher
            else None  # applied and closed
        )

    if refused:
        assert recorder.submissions == []
        assert error == "Temperature must be a finite number from 0 to 1."
    else:
        assert error is None
        assert recorder.submissions[0].draft.settings.temperature == 1.5


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


def _assert_enter_drawn_whole(app, switcher: ConsoleModelPopover) -> None:
    """Capture fix 3 finding 1: Enter's button is as wide as its label, so a
    longer label is never cut and a shorter one leaves no gap after it."""
    apply = switcher.query_one("#console-popover-apply", Button)
    label = str(apply.label)
    assert apply.region.width == cell_len(label) + 2, (label, apply.region.width)
    assert f" {label}  · Tab edit values" in _key_row(app, switcher)


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


async def test_the_enter_label_follows_the_highlighted_rows_kind() -> None:
    """Round 2 finding 4 (05a): with a NEEDS SETUP row highlighted the key row
    said "Enter apply to this chat", but Enter there opens Settings; on a
    "… N more" row it lists that provider's models. The label is the row's."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(app, pilot, build_switcher(recorder))
        apply = switcher.query_one("#console-popover-apply", Button)
        labels: dict[str, set[str]] = {}
        for _ in switcher._rows:
            row = switcher.highlighted_row()
            labels.setdefault(row.kind, set()).add(str(apply.label))
            _assert_enter_drawn_whole(app, switcher)
            await pilot.press("down")
            await pilot.pause()

    assert labels == {
        "pair": {"Enter apply to this chat"},
        "more": {"Enter show them"},
        "setup": {"Enter set up in Settings"},
    }, labels


@pytest.mark.parametrize(
    "held",
    [
        {"draft": _draft("openai", None)},
        {
            "query": "OpenAI ",
            "highlight_pair": ("openai", "gpt-5.1"),
            "recent": (_KEYLESS_USE,),
        },
        {"draft": _draft("openai", None), "enter_sets_default": True},
    ],
    ids=["no-rows-yet", "reopened-unhighlighted", "label-grows"],
)
async def test_the_enter_label_names_the_refusal_while_nothing_is_highlighted(
    held,
) -> None:
    """Capture fix 2 round 1 finding 1: while providers are checked nothing
    is highlighted (no rows yet, or a reopened list kept unhighlighted), and
    Enter applies nothing there. The key row said "Enter apply to this chat";
    it names the refusal until a row is highlighted.

    Capture fix 3 finding 1 (05c): Button.label is no layout reactive, so the
    button kept the refusal's width after the label changed: a gap after a
    shorter label, a cut in a longer one ("label-grows")."""
    recorder, release = _held_recorder()
    app = SwitcherHarness()
    try:
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = build_switcher(recorder, **held)
            app.push_screen(switcher, callback=lambda result: None)
            await pilot.pause()
            await pilot.pause()
            assert not switcher._first_readiness_done
            assert switcher.highlighted_row() is None
            apply = switcher.query_one("#console-popover-apply", Button)
            assert str(apply.label) == "Enter choose a model first"
            _assert_enter_drawn_whole(app, switcher)

            release.set()
            await _settle(app, pilot)
            assert switcher.highlighted_row() is not None
            assert str(apply.label) != "Enter choose a model first"
            _assert_enter_drawn_whole(app, switcher)
    finally:
        release.set()


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
