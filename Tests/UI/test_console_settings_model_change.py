"""Chat settings changes the model only in Switch model's pick mode (TASK-33006.4).

Spec rule 1, pairs only: the MODEL row's Change (or Alt+M) opens the real
pick-only switcher over the modal, and the picked provider·model pair rebases
the draft through the controller's rebaser. Nothing is applied until Apply.
Every test mounts the real modal under the production stylesheets and drives
it with real keypresses; ``pick_mode_opener`` builds the switcher exactly as
``model_switcher.open_model_picker`` does, from the harness's configuration.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_console_settings_core_first import (
    CoreFirstHarness,
    _open,
    _painted,
    _settings,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleSettingsContextEstimate,
    ConsoleValueLayer,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    readiness_words,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CONNECTION_DISCLOSURE_ID,
    MODEL_CHANGE_ID,
    MODEL_CHANGE_LABEL,
    NAME_DISCLOSURE_ID,
    REQUEST_ESTIMATE_DISCLOSURE_ID,
    SAMPLING_DISCLOSURE_ID,
    field_control_id,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    MODEL_DISCOVER_STATUS_ID,
    ConsoleSettingsModal,
)

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

#: The harness's saved model lists, which pick mode lists from.
PROVIDERS_MODELS = {
    "llama_cpp": ["model-a", "model-b"],
    "anthropic": ["claude-sonnet-4-5"],
    "openai": ["gpt-5"],
}


def _never(*_args, **_kwargs):
    raise AssertionError("pick mode never rebases, applies or opens Settings")


def pick_mode_opener(app, app_config, providers_models=PROVIDERS_MODELS):
    """Return the Change opener the Console passes: the real pick-only switcher.

    Args:
        app: The harness app the switcher is pushed on.
        app_config: The configuration its rows and readiness read.
        providers_models: The saved model lists it lists.

    Returns:
        A ``model_picker`` for ``ConsoleSettingsModal``.
    """

    def open_picker(origin, draft, query, on_pick, served):
        app.push_screen(
            ConsoleModelPopover(
                origin=origin,
                app_config=app_config,
                initial_draft=draft,
                providers_models=providers_models,
                scope_copy="",
                durability_copy="",
                draft_rebaser=_never,
                live_committer=_never,
                default_readiness_resolver=readiness_resolver(app_config),
                pick_only=True,
                query=query,
                served_models=served,
            ),
            callback=on_pick,
        )

    return open_picker


def readiness_resolver(app_config):
    """Return the configuration-only readiness pick mode reads, as the Console's.

    Args:
        app_config: The configuration the readiness is built from.

    Returns:
        A ``default_readiness_resolver`` for ``ConsoleModelPopover``.
    """

    def readiness(provider, model):
        settings = build_target_default_console_session_settings(
            app_config, provider, model
        )
        return build_console_settings_readiness(settings, app_config=app_config)

    return readiness


def real_rebase(state, **kwargs):
    """The controller's rebaser: the seam a pick lands on (AC#2)."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    return ConsoleChatController.rebase_console_settings_draft(object(), state, **kwargs)


def pick_modal(app, settings, **kwargs):
    """A Chat settings modal wired like the Console's: picker and rebaser."""
    kwargs.setdefault("draft_rebaser", real_rebase)
    kwargs.setdefault("model_picker", pick_mode_opener(app, app.app_config))
    kwargs.setdefault(
        "context_estimate", ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
    )
    return ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models=PROVIDERS_MODELS,
        can_save=True,
        **kwargs,
    )


async def settle(pilot, app, rounds: int = 4) -> None:
    for _ in range(rounds):
        await pilot.pause()
        await app.workers.wait_for_complete()
    await pilot.pause()


async def press_change(pilot, app, modal) -> ConsoleModelPopover:
    """Press the MODEL row's Change with real keys; return the pick-mode switcher."""
    modal.query_one(f"#{MODEL_CHANGE_ID}", Button).focus()
    await pilot.press("enter")
    await settle(pilot, app)
    switcher = app.screen
    assert isinstance(switcher, ConsoleModelPopover), switcher
    return switcher


async def pick(pilot, app, modal, text: str) -> None:
    """Change, type ``text`` in Find, Enter: the pick lands back in the modal."""
    await press_change(pilot, app, modal)
    await pilot.press(*text)
    await settle(pilot, app)
    await pilot.press("enter")
    await settle(pilot, app)
    assert app.screen is modal


def _model_row_line(app, modal) -> str:
    return _painted(app.screen)[modal.query_one("#console-settings-model-row").region.y]


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.asyncio
async def test_model_row_paints_pair_source_readiness_context_and_change(size) -> None:
    """AC#1: model · provider name | Source | readiness · context | Change."""
    app = CoreFirstHarness()
    settings = _settings("anthropic", "claude-sonnet-4-5", temperature=0.7)
    estimate = ConsoleSettingsContextEstimate(
        10, 200_000, "10 / 200k", token_limit_verified=True
    )
    modal = pick_modal(app, settings, context_estimate=estimate)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        readiness = build_console_settings_readiness(
            settings, app_config=app.app_config
        )
        line = _model_row_line(app, modal)
        parts = (
            "Model",
            "claude-sonnet-4-5 · Anthropic",
            CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.THIS_CHAT],
            f"{readiness_words(readiness)} · 200k context",
            MODEL_CHANGE_LABEL,
        )
        positions = [line.find(part) for part in parts]
        assert -1 not in positions and positions == sorted(positions), (line, parts)
        change = modal.query_one(f"#{MODEL_CHANGE_ID}", Button)
        assert change.region.height == 1
        container = modal.query_one("#console-settings-modal")
        assert container.region.contains_region(change.region)


#: TASK-33007 #12: the MODEL row's context words for the two models whose
#: Settings ▸ Advanced titles test_settings_advanced_disclosures pins
#: (CONTEXT_WINDOW_TITLES): a known size, or "unknown" -- the size assumed
#: is said in Request estimate and the Context view, as Settings says it.
CONTEXT_WINDOW_WORDS = {
    "gpt-4o": " · 128k context",
    "gpt-5.6-terra": " · context unknown ",
}


@pytest.mark.parametrize("model", sorted(CONTEXT_WINDOW_WORDS))
@pytest.mark.asyncio
async def test_model_row_says_what_settings_says_about_the_context_window(model):
    """TASK-33007 #12: "~32k context" read as a known size while Settings ▸
    Advanced said "unknown". The modal publishes the shared resolver's answer
    (as the Console wires it) and a fallback says "unknown" in every place
    the modal names the window: the MODEL row, Request estimate and the
    Context view."""
    from textual.widgets import Collapsible

    from tldw_chatbook.Utils.token_counter import resolve_context_window

    async def resolver(draft):
        return resolve_context_window(draft.provider, draft.model or "")

    app = CoreFirstHarness()
    modal = pick_modal(
        app, _settings("openai", model), context_window_resolver=resolver
    )
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        await settle(pilot, app)
        line = _model_row_line(app, modal)
        assert CONTEXT_WINDOW_WORDS[model] in line, line
        assert "~" not in line, line
        title = str(
            modal.query_one(f"#{REQUEST_ESTIMATE_DISCLOSURE_ID}", Collapsible).title
        )
        window = str(modal.query_one("#console-context-model-window", Static).render())
        if model == "gpt-4o":
            assert "/ 128,000 tokens" in title and "unknown" not in title, title
            assert window.endswith("128,000 tokens (model catalog)"), window
        else:
            assert title.endswith("/ 32,000 tokens (assumed; window unknown)"), title
            assert window.endswith("unknown, 32,000 assumed"), window


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("fireworks", "accounts/fireworks/models/deepseek-v3-0324"),
        ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free-Long"),
    ],
)
@pytest.mark.asyncio
async def test_model_row_shows_a_long_id_and_the_provider_name_whole(
    size, provider, model
) -> None:
    """Review round 1 (AC#1): the pair takes the row's free width, so ids
    that passed the old 48-cell cap render whole with the provider name."""
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(provider, model))
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        name = provider_display_name(provider, app.app_config)
        line = _model_row_line(app, modal)
        assert f"{model} · {name} " in line, line
        assert MODEL_CHANGE_LABEL in line


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.asyncio
async def test_model_row_shortens_only_an_id_wider_than_the_row(size) -> None:
    """Review round 1: an id wider than the free width gives way in the
    middle, so its prefix and its quant suffix stay; the provider name,
    the readiness words and Change are never cut."""
    model = (
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/"
        "Meta-Llama-3.1-70B-Instruct-abliterated-Q4_K_M.gguf"
    )
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings("llama_cpp", model))
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        line = _model_row_line(app, modal)
        assert model not in line
        assert "bartowski/Meta-Llama" in line and "…" in line, line
        assert "Q4_K_M.gguf · llama.cpp " in line, line
        status = modal.query_one("#console-settings-model-status")
        assert str(status.render()) in line
        change = modal.query_one(f"#{MODEL_CHANGE_ID}", Button)
        container = modal.query_one("#console-settings-modal")
        assert container.region.contains_region(change.region)
        assert MODEL_CHANGE_LABEL in line


@pytest.mark.asyncio
async def test_compact_model_row_keeps_the_pair_and_change_and_names_the_window() -> None:
    """Review round 1 (finding 7): below 100 columns, not redesigned, the
    MODEL row shows the pair and Change only; the context window still reads
    in Request estimate's one-row title."""
    from textual.widgets import Collapsible

    app = CoreFirstHarness()
    estimate = ConsoleSettingsContextEstimate(10, 200_000, "10 / 200k")
    modal = pick_modal(app, _settings(), context_estimate=estimate)
    async with app.run_test(size=(80, 24)) as pilot:
        await _open(pilot, app, modal)
        assert modal.has_class("-conversation-settings-compact")
        line = _model_row_line(app, modal)
        assert "model-a · llama.cpp" in line and MODEL_CHANGE_LABEL in line, line
        for hidden in ("#console-settings-model-source", "#console-settings-model-status"):
            assert modal.query_one(hidden).display is False
        change = modal.query_one(f"#{MODEL_CHANGE_ID}", Button)
        assert modal.query_one("#console-settings-modal").region.contains_region(
            change.region
        )
        title = modal.query_one(f"#{REQUEST_ESTIMATE_DISCLOSURE_ID}", Collapsible).title
        assert "200k" in str(title)


@pytest.mark.asyncio
async def test_change_opens_pick_mode_and_the_pick_rebases_without_applying() -> None:
    """AC#2: Change opens pick-only Switch model over the modal; the pick
    rebases the draft through the controller's rebaser to that exact pair,
    the row says "edited *", and nothing is applied until Apply."""
    app = CoreFirstHarness()
    rebased: list[tuple[str, str]] = []
    committed: list[object] = []

    def recording_rebase(state, **kwargs):
        rebased.append((kwargs["provider"], kwargs["model"]))
        return real_rebase(state, **kwargs)

    def commit(submission):
        committed.append(submission)
        return modal._transitional_live_commit(submission)

    modal = pick_modal(
        app, _settings(), draft_rebaser=recording_rebase, live_committer=commit
    )
    results: list[object] = []
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await settle(pilot, app)
        switcher = await press_change(pilot, app, modal)
        assert switcher._pick_only is True
        assert modal in app.screen_stack  # over the modal, not instead of it
        await pilot.press(*"claude-sonnet")
        await settle(pilot, app)
        await pilot.press("enter")
        await settle(pilot, app)
        assert app.screen is modal and results == [] and committed == []
        assert rebased == [("anthropic", "claude-sonnet-4-5")]
        assert (modal._draft.settings.provider, modal._draft.settings.model) == (
            "anthropic",
            "claude-sonnet-4-5",
        )
        line = _model_row_line(app, modal)
        assert "claude-sonnet-4-5 · Anthropic" in line
        assert CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.EDITED_DRAFT] in line
        assert app.focused is modal.query_one(f"#{MODEL_CHANGE_ID}", Button)

        modal.query_one("#console-settings-save", Button).focus()
        await pilot.press("enter")
        await settle(pilot, app)
    assert len(committed) == 1
    applied = committed[0].draft.settings
    assert (applied.provider, applied.model) == ("anthropic", "claude-sonnet-4-5")


@pytest.mark.asyncio
async def test_the_consoles_pick_opener_never_rebases_and_its_committer_refuses(
    monkeypatch,
) -> None:
    """Review round 1 (finding 6): the Console's own opener
    (``model_switcher.open_model_picker``) hands the pair back. Pick mode
    never calls the rebaser it is given, even as highlights move, and its
    committer refuses, so no pick can commit to the chat behind the modal."""
    from functools import partial
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import model_switcher

    app = CoreFirstHarness()
    calls: list[str] = []

    def recording_rebase(state, **kwargs):
        calls.append("rebase")
        return real_rebase(state, **kwargs)

    def sources(_screen, _session_id, _before):
        return {
            "app_config": app.app_config,
            "providers_models": PROVIDERS_MODELS,
            "draft_rebaser": recording_rebase,
            "default_readiness_resolver": readiness_resolver(app.app_config),
        }

    monkeypatch.setattr(model_switcher, "_switcher_sources", sources)
    console = SimpleNamespace(
        app=app, _commit_console_settings_submission_live=lambda _s: calls.append("commit")
    )
    opener = partial(model_switcher.open_model_picker, console)
    modal = pick_modal(app, _settings(), model_picker=opener)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        switcher = await press_change(pilot, app, modal)
        with pytest.raises(ValueError):
            switcher._live_committer(object())
        await pilot.press("down", "down", "up", *"claude")
        await settle(pilot, app)
        assert calls == []  # an apply-mode highlight rebases; pick mode never
        await pilot.press("enter")
        await settle(pilot, app)
        assert app.screen is modal
        assert modal._draft.settings.provider == "anthropic"
    assert calls == []


@pytest.mark.asyncio
async def test_alt_m_opens_pick_mode_from_a_focused_field() -> None:
    """AC#2: Alt+M inside Chat settings is Change, even from a typing field.

    A terminal delivers Alt+M as key "alt+m" carrying the character "m"
    (Textual's xterm parser), which a focused Input would type; the event is
    posted as the driver posts it, since ``pilot.press`` sends no character.
    """
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        temperature = modal.query_one("#console-settings-temperature", Input)
        assert app.focused is temperature
        app.post_message(events.Key("alt+m", "m"))
        await settle(pilot, app)
        assert isinstance(app.screen, ConsoleModelPopover)
        assert app.screen._pick_only is True
        assert temperature.value == "0.4"  # Alt+M typed nothing into the field


@pytest.mark.asyncio
async def test_no_provider_picker_or_model_search_and_a_typed_id_still_picks() -> None:
    """AC#3: the modal hosts no provider picker or model search, so no
    provider is chosen without a model; an unlisted model id is still
    chosen through pick mode's typed row (TASK-30012 AC#4)."""
    from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert not modal.query(ModelSearchPicker)
        for gone in (
            "#console-settings-provider",
            "#console-settings-provider-picker",
            "#console-settings-model-picker",
            "#console-settings-model-select",
            "#console-settings-model-input",
            "#console-settings-model-custom",
            "#console-settings-keep-unverified-model",
        ):
            assert not modal.query(gone), gone
        choices = {
            field_control_id(name)
            for name in (
                "streaming",
                "reasoning_effort",
                "reasoning_summary",
                "verbosity",
                "thinking_effort",
            )
        }
        for select in modal.query(Select):  # value choices only, no provider
            assert select.id in choices or select.id.startswith("console-context-")

        await pick(pilot, app, modal, "my-own-gguf")
        assert (modal._draft.settings.provider, modal._draft.settings.model) == (
            "llama_cpp",
            "my-own-gguf",
        )
        assert "my-own-gguf · llama.cpp" in _model_row_line(app, modal)


@pytest.mark.asyncio
async def test_tab_order_runs_change_then_core_then_disclosures_to_apply() -> None:
    """R7: the final Model view order, from the view tabs: Change, the shown
    CORE fields, the four disclosure titles, then Apply."""
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one("#console-settings-view-model", Button).focus()
        await pilot.pause()
        seen: list[str] = []
        for _ in range(20):
            await pilot.press("tab")
            focused = app.focused
            seen.append(focused.id or focused.parent.id)
            if focused.id == "console-settings-save":
                break
        shown_core = [
            field_control_id(name)
            for name in (
                "temperature",
                "max_tokens",
                "streaming",
                "reasoning_effort",
                "reasoning_summary",
                "verbosity",
                "thinking_effort",
                "thinking_budget_tokens",
            )
            if modal.query_one(f"#{field_control_id(name)}-row").display
        ]
        body = [
            "console-settings-view-context",
            MODEL_CHANGE_ID,
            *shown_core,
            SAMPLING_DISCLOSURE_ID,
            CONNECTION_DISCLOSURE_ID,
            REQUEST_ESTIMATE_DISCLOSURE_ID,
            NAME_DISCLOSURE_ID,
        ]
        assert seen[: len(body)] == body
        footer = seen[len(body) :]  # the shown footer actions, Apply last
        assert footer[-1] == "console-settings-save"
        assert set(footer[:-1]) <= {
            "console-settings-save-default",
            "console-settings-make-default",
        }


@pytest.mark.asyncio
async def test_new_endpoint_lands_on_a_pair_never_on_a_provider_alone() -> None:
    """R9: a created entry is listed first, then pick mode opens on it with
    what it serves beside its configured models; the pick lands on the entry
    and one of its models, and Esc leaves the old pair (review round 1:
    pick mode lists a new entry's served models before the pick)."""
    from tldw_chatbook.Chat.provider_test_evidence import ProviderProbeResult

    listed: list[object] = []

    async def connection(identity):
        listed.append((identity.custom_endpoint_id, identity.connection_identity))
        return ProviderProbeResult("reachable", ("served-x",))

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), connection_tester=connection)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.pick_created_endpoint("custom-ep:gpu-box")
        await settle(pilot, app)
        assert listed == [("custom-ep:gpu-box", ("llama_cpp", "http://192.168.1.9:8080"))]
        switcher = app.screen
        assert isinstance(switcher, ConsoleModelPopover) and switcher._pick_only
        assert switcher.query_one("#console-popover-find", Input).value == "GPU box "
        pairs = {(row.provider, row.model) for row in switcher._rows}
        assert {("custom-ep:gpu-box", "model-a"), ("custom-ep:gpu-box", "served-x")} <= pairs
        await pilot.press("escape")
        await settle(pilot, app)
        assert app.screen is modal
        assert (modal._draft.settings.provider, modal._draft.settings.model) == (
            "llama_cpp",
            "model-a",
        )

        modal.pick_created_endpoint("custom-ep:gpu-box")
        await settle(pilot, app)
        await pilot.press(*"served-x", "enter")
        await settle(pilot, app)
        assert app.screen is modal
        assert (modal._draft.settings.provider, modal._draft.settings.model) == (
            "custom-ep:gpu-box",
            "served-x",
        )


@pytest.mark.asyncio
async def test_a_created_entry_says_it_is_listing_and_opens_pick_mode_even_if_that_fails(
    monkeypatch,
) -> None:
    """Final review I7: while a created entry is listed, the status line says
    so; when resolving its connection raises, pick mode still opens on it,
    so the entry is never left behind without a word."""
    import asyncio

    import tldw_chatbook.Widgets.Console.console_settings_field_row as field_row
    from tldw_chatbook.Chat.provider_test_evidence import ProviderProbeResult

    release = asyncio.Event()

    async def slow_listing(_identity):
        await release.wait()
        return ProviderProbeResult("reachable", ("served-x",))

    def status(modal) -> str:
        line = modal.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
        return str(line.content) if line.display else ""

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), connection_tester=slow_listing)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.pick_created_endpoint("custom-ep:gpu-box")
        for _ in range(4):
            await pilot.pause()
        assert status(modal) == "Listing the models GPU box serves…"
        release.set()
        await settle(pilot, app)
        assert isinstance(app.screen, ConsoleModelPopover)
        await pilot.press("escape")
        await settle(pilot, app)
        assert status(modal) == ""

        def broken(*_args, **_kwargs):
            raise RuntimeError("identity resolve failed")

        monkeypatch.setattr(field_row, "console_send_connection", broken)
        modal.pick_created_endpoint("custom-ep:gpu-box")
        await settle(pilot, app)
        switcher = app.screen
        assert isinstance(switcher, ConsoleModelPopover) and switcher._pick_only
        assert switcher.query_one("#console-popover-find", Input).value == "GPU box "


@pytest.mark.asyncio
async def test_a_pick_during_the_paid_test_cancels_it_and_keeps_the_billing_warning() -> None:
    """TASK-33006.8 AC#1: a model change made the user's way (Change, type in
    pick mode's Find, Enter) while the paid generation test runs cancels the
    test, keeps the billing warning and fences the late result. The tester
    ignores cancellation, so this waits on pauses, never on the workers."""
    import asyncio

    from tldw_chatbook.Chat.provider_test_evidence import ProviderGenerationProbeResult

    entered = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_tester(_request):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return ProviderGenerationProbeResult("succeeded")

    async def pause(pilot) -> None:
        for _ in range(6):
            await pilot.pause()

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), generation_tester=cancellation_resistant_tester)
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(modal)
        await pause(pilot)
        modal.query_one("#console-settings-test-generation", Button).press()
        modal.query_one("#console-settings-confirm-generation", Button).press()
        await asyncio.wait_for(entered.wait(), 1)

        modal.query_one(f"#{MODEL_CHANGE_ID}", Button).focus()
        await pilot.press("enter")
        await pause(pilot)
        assert isinstance(app.screen, ConsoleModelPopover)
        await pilot.press(*"model-b")
        await pause(pilot)
        await pilot.press("enter")
        await pause(pilot)
        assert app.screen is modal
        assert modal._current_model_value() == "model-b"
        button = modal.query_one("#console-settings-test-generation", Button)
        assert str(button.label) == "Test generation" and not button.disabled
        status = str(
            modal.query_one("#console-settings-generation-test-status", Static).content
        )
        assert "Stopped waiting" in status and "may still be billed" in status

        release.set()
        await pause(pilot)
        readiness = str(modal.query_one("#console-settings-readiness", Static).content)
        assert "Generation · Succeeded" not in readiness


@pytest.mark.asyncio
async def test_a_listing_feeds_pick_mode_but_never_picks() -> None:
    """Review round 1: what Test connection lists is offered in pick mode
    (Change), and still nothing changes until a pick lands."""
    from tldw_chatbook.Chat.provider_test_evidence import ProviderProbeResult

    async def connection(_identity):
        return ProviderProbeResult("reachable", ("model-a", "served-y"))

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), connection_tester=connection)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one("#console-settings-model-discover", Button).press()
        await settle(pilot, app)
        assert modal._current_model_value() == "model-a"
        switcher = await press_change(pilot, app, modal)
        pairs = {(row.provider, row.model) for row in switcher._rows}
        assert ("llama_cpp", "served-y") in pairs
        await pilot.press(*"served-y", "enter")
        await settle(pilot, app)
        assert (modal._draft.settings.provider, modal._draft.settings.model) == (
            "llama_cpp",
            "served-y",
        )
