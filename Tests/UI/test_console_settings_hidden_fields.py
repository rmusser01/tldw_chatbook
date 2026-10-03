"""Chat settings hides the fields the provider does not accept (TASK-33006.2).

Spec rule 2: supported fields come from the one shared support decision
(``supported_generation_fields`` for the samplers, and
``console_generation_control_support`` for the reasoning and thinking
controls, whose "unknown" stays visible, TASK-30012 AC#3). Hidden fields are
named on the Sampling disclosure's one summary line. Every test mounts the
real modal under the production stylesheets and reads what was painted.
"""

from __future__ import annotations

import time

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Button, Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_settings_core_first import (
    CoreFirstHarness,
    _modal,
    _open,
    _painted,
    _settings,
)
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_FIELD_LABELS,
    console_generation_control_support,
    supported_generation_fields,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CORE_FIELDS,
    FIELD_ROW_FIELDS,
    GENERATION_CONTROL_UNKNOWN_COPY,
    HIDDEN_FIELDS_REASON,
    SAMPLING_DISCLOSURE_ID,
    SAMPLING_FIELDS,
    field_control_id,
)

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

#: The reasoning and thinking controls; every other row is a sampler.
_CONTROLS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
    }
)
_ANTHROPIC_HIDDEN = (
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
)


def _expected_line(provider: str, hidden: tuple[str, ...]) -> str:
    """The Sampling title, built from the field table's labels (R3)."""
    if not hidden:
        return "Sampling"
    names = ", ".join(MODEL_FIELD_LABELS[name] for name in hidden)
    return (
        f"Sampling · hidden for {provider_display_name(provider)}: {names} "
        f"{HIDDEN_FIELDS_REASON}"
    )


def _shared_hidden(app_config, provider: str, model: str) -> tuple[str, ...]:
    """The fields the shared support functions reject, in line order."""
    supported = supported_generation_fields(provider, model, app_config)
    return tuple(
        name
        for name in (*SAMPLING_FIELDS, *CORE_FIELDS)
        if (
            console_generation_control_support(provider, model, name) == "unsupported"
            if name in _CONTROLS
            else name not in supported
        )
    )


def _hidden_rows(modal) -> tuple[str, ...]:
    return tuple(
        name
        for name in (*SAMPLING_FIELDS, *CORE_FIELDS)
        if not modal.query_one(f"#{field_control_id(name)}-row").display
    )


def _sampling_title(modal) -> str:
    return str(modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).title)


def _painted_title(app, modal) -> str:
    """The Sampling title as painted, its wrapped rows joined by one space."""
    title = modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}").query_one(CollapsibleTitle)
    region = title.content_region
    rows = _painted(app.screen)[region.y : region.y + region.height]
    return " ".join(
        " ".join(row[region.x : region.x + region.width].split()) for row in rows
    )


def _real_rebase(state, **kwargs):
    """The controller's rebaser, the seam the switcher's pick mode lands on."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    return ConsoleChatController.rebase_console_settings_draft(
        object(), state, **kwargs
    )


@pytest.mark.asyncio
async def test_anthropic_hides_unaccepted_fields_behind_the_sampling_line() -> None:
    """AC#1/#2: Min P, Seed, the penalties and the reasoning controls
    Anthropic does not accept are not rendered and never take focus; the
    Sampling title names them all, by their field-table labels."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings("anthropic", "claude-sonnet-4-5", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert _hidden_rows(modal) == _ANTHROPIC_HIDDEN
        expected = _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        assert expected.startswith(
            "Sampling · hidden for Anthropic: Min P, Seed, Presence penalty, "
            "Frequency penalty, Reasoning effort"
        )
        assert expected.endswith("(this provider does not accept them)")
        assert _sampling_title(modal) == expected
        assert _painted_title(app, modal).endswith(expected)
        # The line does not cost the footer its place (T1 AC#9).
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        assert body.max_scroll_y == 0

        # Open Sampling from its title with real keys, then Tab to Apply.
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}").query_one(
            CollapsibleTitle
        ).focus()
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        modal.query_one("#console-settings-temperature").focus()
        await pilot.pause()
        visited: list[str | None] = []
        for _ in range(40):
            await pilot.press("tab")
            visited.append(app.focused.id if app.focused else None)
            if app.focused is modal.query_one("#console-settings-save", Button):
                break
        else:
            raise AssertionError(f"Tab never reached Apply: {visited}")
        assert {"console-settings-top-p", "console-settings-top-k"} <= set(visited)
        assert not {field_control_id(name) for name in _ANTHROPIC_HIDDEN} & set(
            visited
        )
        painted = "\n".join(_painted(app.screen))
        for name in ("min_p", "seed", "presence_penalty", "frequency_penalty"):
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert not row.region.area, name
        assert "Top K" in painted


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-4-5"),
        ("anthropic", "claude-opus-4-7"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.5-pro"),
        ("llama_cpp", "model-a"),
        ("ollama", "qwen3"),
        ("custom-ep:gpu-box", "model-a"),
    ],
)
@pytest.mark.asyncio
async def test_hidden_set_and_line_come_from_the_shared_support_functions(
    provider, model
) -> None:
    """AC#3: the modal hides exactly what the shared functions reject and
    holds no support table of its own."""
    import tldw_chatbook.Widgets.Console.console_settings_modal as modal_module

    for name in (
        "_GENERATION_CONTROL_INPUTS",
        "PROVIDER_CHOICE_NO_EFFECT_SUFFIX",
        "GENERATION_CONTROL_UNKNOWN_COPY",
    ):
        assert not hasattr(modal_module, name), name
    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, model, temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        hidden = _shared_hidden(app.app_config, provider, model)
        assert _hidden_rows(modal) == hidden
        expected = _expected_line(provider, hidden)
        if provider.startswith("custom-ep:"):
            expected = expected.replace(
                provider_display_name(provider),
                provider_display_name(provider, app.app_config),
            )
        assert _sampling_title(modal) == expected
        assert _painted_title(app, modal).endswith(expected)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "future-custom-model"),
        # No PROVIDER_PARAM_MAP entry: every field stays (TASK-30012 AC#3).
        ("unmapped-provider", "model-x"),
    ],
)
@pytest.mark.asyncio
async def test_unknown_support_stays_visible_with_neutral_copy_in_its_help_line(
    provider, model
) -> None:
    """AC#4: an unknown control stays visible and says so in its own help
    line; the per-row "Support not verified" statics are gone."""
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, model, temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert not modal.query(".console-settings-control-support")
        for name in FIELD_ROW_FIELDS:
            assert not modal.query(f"#{field_control_id(name)}-support"), name
        unknown = [
            name
            for name in _CONTROLS
            if console_generation_control_support(provider, model, name) == "unknown"
        ]
        assert unknown, "a new or unmapped model's controls are unknown"
        if provider not in PROVIDER_PARAM_MAP:
            assert _hidden_rows(modal) == ()
            assert _sampling_title(modal) == "Sampling"
        painted = _painted(app.screen)
        for name in unknown:
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert row.display, name
            help_line = modal.query_one(f"#{field_control_id(name)}-help", Static)
            assert str(help_line.content).startswith(GENERATION_CONTROL_UNKNOWN_COPY)
            assert GENERATION_CONTROL_UNKNOWN_COPY in painted[row.region.y], name
        # Known support keeps the field table's plain help.
        temperature_help = modal.query_one("#console-settings-temperature-help", Static)
        assert GENERATION_CONTROL_UNKNOWN_COPY not in str(temperature_help.content)


@pytest.mark.asyncio
async def test_changing_the_model_updates_the_hidden_set_and_line_at_once() -> None:
    """AC#6: the rebase a pick lands on re-decides the hidden rows and the
    Sampling line in the same call, both ways."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(), draft_rebaser=_real_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        llama_hidden = _shared_hidden(app.app_config, "llama_cpp", "model-a")
        assert _hidden_rows(modal) == llama_hidden
        assert _sampling_title(modal) == _expected_line("llama_cpp", llama_hidden)

        assert modal._rebase_to("anthropic", "claude-sonnet-4-5") is True
        # No pause: the same call already changed the rows and the line.
        assert _hidden_rows(modal) == _ANTHROPIC_HIDDEN
        assert _sampling_title(modal) == _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        for _ in range(3):
            await pilot.pause()
        assert _painted_title(app, modal).endswith(
            _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        )

        assert modal._rebase_to("llama_cpp", "model-a") is True
        assert _hidden_rows(modal) == llama_hidden
        assert _sampling_title(modal) == _expected_line("llama_cpp", llama_hidden)


@pytest.mark.asyncio
async def test_a_focused_field_that_becomes_hidden_hands_focus_on() -> None:
    """AC#1: a field hidden under focus leaves the Tab order at once, and
    focus goes to where the view opens (R13), not nowhere."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(), draft_rebaser=_real_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        seed = modal.query_one("#console-settings-seed")
        seed.focus()
        await pilot.pause()
        assert app.focused is seed
        assert modal._rebase_to("anthropic", "claude-sonnet-4-5") is True
        for _ in range(4):
            await pilot.pause()
        assert app.focused is modal.query_one("#console-settings-temperature")


#: Every generic request key a hidden Anthropic field could ride on.
_HIDDEN_REQUEST_KEYS = frozenset(
    {
        "minp",
        "min_p",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
    }
)


@pytest.mark.asyncio
@private_profile_test
async def test_applying_on_anthropic_submits_none_of_the_hidden_fields(
    request, tmp_path, monkeypatch
) -> None:
    """AC#7: the real Console opens Chat settings on an Anthropic chat whose
    defaults carry Min P, Seed, the penalties and reasoning values. Apply
    commits through the controller with none of them, and the next send hands
    the Anthropic handler none of them, while a shown field (Temperature)
    still goes."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _drain_settings_tasks,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    assert SettingsConfigAdapter().save_sections(
        {
            "first_run": {"setup_completed": True},
            "chat_defaults": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "temperature": 0.6,
                "min_p": 0.05,
                "seed": 7,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "reasoning_effort": "high",
                "verbosity": "high",
            },
            "api_settings.anthropic": {
                "api_key": "sk-ant-test-hidden-fields-0000",
                "model": "claude-sonnet-4-5",
            },
        }
    )
    app = _build_test_app()
    # File-backed: a send commits its trace from a worker thread, and each
    # thread's ":memory:" connection is an empty database of its own.
    app.chachanotes_db = CharactersRAGDB(
        tmp_path / "chachanotes.sqlite", client_id="hidden-fields"
    )
    app.chat_api_provider_value = "anthropic"
    app.chat_api_model_value = "claude-sonnet-4-5"
    app.providers_models = {"anthropic": ["claude-sonnet-4-5"]}
    captured: list[dict] = []

    def anthropic_handler(**kwargs):
        captured.append(kwargs)
        return "hidden fields reply"

    monkeypatch.setitem(API_CALL_HANDLERS, "anthropic", anthropic_handler)
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        before = store.session_settings(session_id)
        assert (before.provider, before.min_p, before.seed) == ("anthropic", 0.05, 7)

        assert await console._open_console_settings() is True
        modal = None
        for _ in range(80):
            if isinstance(harness.screen, ConsoleSettingsModal):
                modal = harness.screen
                break
            await pilot.pause(0.05)
        assert modal is not None
        for _ in range(4):
            await pilot.pause()
        assert _hidden_rows(modal) == _ANTHROPIC_HIDDEN
        assert _sampling_title(modal) == _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        modal.query_one("#console-settings-save", Button).press()
        for _ in range(80):
            if harness.screen is console:
                break
            await pilot.pause(0.05)
        assert harness.screen is console
        await _drain_settings_tasks(app)
        # What Apply committed for the chat: none of the hidden fields.
        after = store.session_settings(session_id)
        assert after.provider == "anthropic"
        for name in _ANTHROPIC_HIDDEN:
            assert getattr(after, name) is None, name

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        await pilot.pause(0.1)
        console.query_one("#console-send-message", Button).press()
        deadline = time.monotonic() + 30.0
        while not captured and time.monotonic() < deadline:
            await pilot.pause(0.05)

    assert captured, "the Anthropic handler was never called"
    sent = captured[-1]
    assert not _HIDDEN_REQUEST_KEYS & set(sent), sorted(sent)
    assert sent["temp"] == pytest.approx(0.6)
    assert sent["model"] == "claude-sonnet-4-5"
