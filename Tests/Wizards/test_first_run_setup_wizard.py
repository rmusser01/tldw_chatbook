"""Pilot tests for the first-run setup wizard skeleton."""

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Input, RadioButton, RadioSet, Static, Switch

from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    AppearanceStep,
    CLOUD_PROBE_TIMEOUT_SECONDS,
    FirstRunSetupWizard,
    ModelStep,
    NotesSyncStep,
    ProtectKeysStep,
    ProviderStep,
    RagStep,
    SetupWizardContainer,
    SummaryStep,
    ToolsStep,
)
from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardNavigation,
    WizardProgress,
    WizardStepConfig,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SUMMARY,
    TRACK_FULL,
    TRACK_QUICK,
)


class _HostApp(App):
    def __init__(self, wizard: FirstRunSetupWizard):
        super().__init__()
        self._wizard = wizard
        self.wizard_result = "UNSET"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self._capture)

    def _capture(self, result) -> None:
        self.wizard_result = result


def _make_wizard(**kwargs) -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance, **kwargs)
    return wizard


@pytest.mark.asyncio
async def test_welcome_track_choice_activates_quick_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        assert STEP_PROVIDER in container.active_ids
        assert STEP_RAG not in container.active_ids
        assert container.active_ids[-1] == STEP_SUMMARY


@pytest.mark.asyncio
async def test_select_track_rebuilds_progress_in_original_slot():
    """F-C regression (live-verified via tmux screenshot): _rebuild_progress
    replaces the WizardProgress widget wholesale on every track change, but
    ``parent.mount(fresh)`` with no ``before=``/``after=`` appends at the
    container's END -- after WizardNavigation -- so the whole progress bar
    rendered BELOW the Back/Next buttons instead of staying in its original
    slot right after the title."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.2)
        children = list(container.children)
        progress = container.query_one(".wizard-progress", WizardProgress)
        nav = container.query_one(".wizard-navigation", WizardNavigation)
        steps_container = container.query_one(".wizard-steps-container")
        assert children.index(progress) < children.index(steps_container)
        assert children.index(progress) < children.index(nav)


@pytest.mark.asyncio
async def test_welcome_full_track_activates_all_non_conditional_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_FULL)
        assert STEP_RAG in container.active_ids


@pytest.mark.asyncio
async def test_escape_asks_for_confirmation_instead_of_dismissing():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        # The wizard must still be open (confirm dialog on top), not dismissed.
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_next_button_click_drives_quick_track_to_completion():
    """Regression test for a real Textual double-dispatch trap.

    Textual's @on-decorated handlers are collected across the WHOLE MRO
    (textual.message_pump.MessagePump._get_dispatch_methods), so both
    WizardContainer.handle_next (base) and SetupWizardContainer.handle_next
    (override) fire on a single Button.Pressed("#wizard-next"). Without
    event.prevent_default() in the override, the base handler flat-advances
    current_step by one BEFORE the override's own worker runs — silently
    breaking track selection (select_track() on the Welcome step never
    actually applies) and skipping/duplicating steps. This test drives the
    real click path (not container.select_track() directly) so a regression
    of that suppression would fail it.
    """
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)

        seen_step_ids = []
        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            await pilot.click("#wizard-next")
            await pilot.pause(0.2)
            step = container.steps[container.current_step]
            seen_step_ids.append(step.config.id if step.config else None)

        assert app.wizard_result == {"completed": True, "exit_route": None}
        # Exactly the quick-track subset, each step visited once, in order.
        assert seen_step_ids == ["provider", "model", "summary", "summary"]
        assert set(container.wizard_data.keys()) == {
            "welcome",
            "provider",
            "model",
            "summary",
        }


def _provider_step(wizard=None, environ=None, discover=None, probe=None):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = wizard or SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    return ProviderStep(
        wizard=wizard,
        config=WizardStepConfig(id="provider", title="Provider", step_number=2),
        discover=discover or AsyncMock(return_value=()),
        probe=probe or AsyncMock(),
        environ=environ or {},
    )


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


@pytest.mark.asyncio
async def test_provider_step_env_key_shows_found_in_environment():
    step = _provider_step(environ={"OPENAI_API_KEY": "sk-x"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        assert "environment" in str(status.render()).lower()


@pytest.mark.asyncio
async def test_provider_step_stale_probe_result_is_discarded():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        generation_before = step.probe_generation
        step.select_provider("anthropic")
        # A result stamped with the old generation must not render.
        step.apply_probe_result(generation_before, reachable=True, summary="stale ok")
        status = step.query_one("#setup-provider-probe-status", Static)
        assert "stale ok" not in str(status.render())


@pytest.mark.asyncio
async def test_provider_step_commit_writes_key_and_notes_key_entered():
    from unittest.mock import AsyncMock

    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-key-input", Input).value = "sk-new"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["api_settings.openai"]["api_key"] == "sk-new"
        wizard.note_key_entered.assert_called_once()


@pytest.mark.asyncio
async def test_provider_step_commit_reads_pressed_radio_without_changed_event():
    """F-A regression (UAT): ProviderStep relied solely on RadioSet.Changed
    to set selected_provider_key. Textual's RadioSet distinguishes the
    merely-*highlighted* button (arrow-key navigation, see
    RadioSet.action_next_button) from the *pressed* one (pressed_button,
    only set by an explicit toggle or an initial value=True at mount --
    RadioSet._on_mount's "switched_on" handling never fires Changed). This
    test simulates exactly that: a button IS pressed per the RadioSet's own
    bookkeeping, but Changed genuinely never fired, so ProviderStep's own
    handler never ran. commit() must still recover the real choice instead
    of silently skipping (which is what left chat_defaults untouched during
    live UAT)."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-provider-choice", RadioSet)
        target = step.query_one("#setup-provider-anthropic", RadioButton)
        # Simulate the mount-time "switched_on" bookkeeping (or any other
        # path) that leaves a button pressed without ever posting
        # RadioButton.Changed / RadioSet.Changed.
        radio_set._pressed_button = target
        assert step.selected_provider_key == ""  # sanity: Changed truly never fired

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_provider_key == "anthropic"
        committed = wizard.commit_config.call_args.args[0]
        assert committed["chat_defaults"]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_provider_step_nothing_pressed_still_legitimately_skips():
    """The other half of the F-A fix: when the RadioSet genuinely reports no
    pressed_button (nothing was ever toggled -- just the default focus
    highlight), commit() must still skip, not fabricate a selection."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-provider-choice", RadioSet)
        assert radio_set.pressed_button is None  # sanity: nothing pressed

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_provider_key == ""
        wizard.commit_config.assert_not_called()


def test_provider_grouping_orders_cloud_then_local_then_custom():
    """Mirror settings_screen.py:6423's grouping rule (task-6 brief interface)."""
    from tldw_chatbook.Chat.console_provider_support import ConsoleProviderCatalogEntry

    entries = (
        ConsoleProviderCatalogEntry(
            readiness_key="ollama", execution_key="ollama",
            display_name="Ollama", requires_api_key=False,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="local_llamacpp", execution_key="custom-openai-api",
            display_name="local llama.cpp", requires_api_key=False,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="openai", execution_key="openai",
            display_name="OpenAI", requires_api_key=True,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="anthropic", execution_key="anthropic",
            display_name="Anthropic", requires_api_key=True,
        ),
    )
    grouped = ProviderStep._grouped(entries)
    # Cloud (alpha) -> Local (alpha) -> Custom/legacy alias keys, last.
    assert [entry.readiness_key for entry in grouped] == [
        "anthropic", "openai", "ollama", "local_llamacpp",
    ]


@pytest.mark.asyncio
async def test_provider_step_one_click_connect_adopts_discovered_server():
    """Discovered local server: one click selects it; commit persists the
    endpoint but never calls note_key_entered (no secret was involved)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    server = DiscoveredLocalServer(
        provider_key="llama_cpp", base_url="http://127.0.0.1:8080", model_ids=("m1",)
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=AsyncMock(return_value=(server,)))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        detected = step.query_one("#setup-provider-detected", Static)
        assert "127.0.0.1:8080" in str(detected.render())
        use_button = step.query_one("#setup-provider-use-detected", Button)
        assert "hidden" not in use_button.classes

        await pilot.click("#setup-provider-use-detected")
        await pilot.pause()
        assert step.selected_provider_key == "llama_cpp"
        assert step.detected_base_url == "http://127.0.0.1:8080"

        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        # Bug-3 fix: this app_config has no persisted chat_defaults, so the
        # persisted-fallback previous provider is "" -- selecting
        # "llama_cpp" differs from that, so chat_defaults now syncs
        # alongside the endpoint, exactly like a first-ever cloud-provider
        # selection would (see the dedicated Bug-3 tests below).
        assert committed == {
            "api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"},
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
        }
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_masked_key_never_round_trips_configured_secret():
    """A configured (non-env) secret renders as presence only -- never a value."""
    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={"api_settings": {"openai": {"api_key": "sk-existing-secret"}}}
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        key_input = step.query_one("#setup-provider-key-input", Input)
        assert key_input.password is True
        assert key_input.value == ""
        status = step.query_one("#setup-provider-key-status", Static)
        assert "sk-existing-secret" not in str(status.render())
        actions = step.query_one("#setup-provider-key-actions")
        assert "hidden" not in actions.classes


@pytest.mark.asyncio
async def test_provider_step_keep_preserves_existing_key_without_note():
    """Keep must not touch the stored secret nor trigger the protect-keys gate."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"api_settings": {"openai": {"api_key": "sk-existing"}}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        # Direct handler call: the full provider catalog can push these
        # buttons below the visible test-terminal region, and this test is
        # about the handler's effect, not the click hit-region.
        step._on_keep()
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        # Bug-3 fix: this app_config has no persisted chat_defaults, so
        # selecting "openai" (differing from the persisted-fallback "")
        # now syncs chat_defaults alongside the untouched (Keep) credential
        # -- the secret itself is still not written, and note_key_entered is
        # still not called, which is the whole point of this test.
        assert committed == {"chat_defaults": {"provider": "openai", "model": ""}}
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_clear_persists_empty_key_without_note():
    """Clear must explicitly erase the stored secret (not just skip writing)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"api_settings": {"openai": {"api_key": "sk-existing"}}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step._on_clear()  # see comment in the Keep test above
        await pilot.pause()
        key_input = step.query_one("#setup-provider-key-input", Input)
        assert key_input.value == ""
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        # Bug-3 fix: as with Keep above, selecting "openai" here also
        # differs from the persisted-fallback "" (no chat_defaults in this
        # app_config), so the commit picks up the provider sync alongside
        # the explicit empty-key erasure.
        assert committed == {
            "api_settings.openai": {"api_key": ""},
            "chat_defaults": {"provider": "openai", "model": ""},
        }
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_switching_provider_clears_key_input():
    """Bug-1: a key typed for provider A must not commit under provider B.
    select_provider() previously left the shared key Input's value alone
    when the provider selection changed, so a key typed under one provider
    silently carried over and would commit under whichever provider was
    selected next."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-key-input", Input).value = "sk-under-openai"
        step.select_provider("anthropic")
        key_input = step.query_one("#setup-provider-key-input", Input)
        assert key_input.value == ""

        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert "api_settings.anthropic" not in committed
        assert "api_settings.openai" not in committed


@pytest.mark.asyncio
async def test_provider_step_reselecting_same_provider_keeps_typed_key():
    """Guards the boundary of the Bug-1 fix above: the key Input must only
    be cleared on an actual provider CHANGE, not on every select_provider()
    call (e.g. a redundant re-selection of the currently-active provider)."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-key-input", Input).value = "sk-under-openai"
        step.select_provider("openai")
        key_input = step.query_one("#setup-provider-key-input", Input)
        assert key_input.value == "sk-under-openai"


@pytest.mark.asyncio
async def test_provider_step_first_selection_persists_chat_defaults_provider():
    """Bug-3: a first-ever provider selection on an empty config previously
    left chat_defaults.provider untouched (invalidate_model_for_provider_change
    only fired for a non-empty in-session previous value), so Model-step-
    skipped left the template default provider active even though
    credentials landed under api_settings. ProviderStep must fall back to
    the PERSISTED chat_defaults.provider (empty here) and still sync it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-key-input", Input).value = "sk-new"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["chat_defaults"] == {"provider": "openai", "model": ""}


@pytest.mark.asyncio
async def test_provider_step_rerun_same_provider_leaves_chat_defaults_untouched():
    """Bug-3: a rerun that re-selects the SAME persisted provider must not
    blank chat_defaults.model -- only an actual provider CHANGE should."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
                "api_settings": {"openai": {"api_key": "sk-existing"}},
            }
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step._on_keep()
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert "chat_defaults" not in committed


@pytest.mark.asyncio
async def test_provider_step_rerun_different_provider_blanks_model():
    """Bug-3: a rerun that picks a DIFFERENT provider than the persisted one
    must sync chat_defaults.provider and blank the stale model."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"chat_defaults": {"provider": "openai", "model": "gpt-4o"}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("anthropic")
        step.query_one("#setup-provider-key-input", Input).value = "sk-new-anthropic"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["chat_defaults"] == {"provider": "anthropic", "model": ""}


@pytest.mark.asyncio
async def test_provider_step_probe_budgets_cloud_vs_local():
    """8.0s for a cloud key probe; 2.5s for a bare local-endpoint probe."""
    from unittest.mock import AsyncMock

    probe = AsyncMock()
    step = _provider_step(probe=probe)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")

        step._launch_probe(api_key="sk-cloud-key")
        await pilot.pause()
        assert probe.call_args.kwargs["timeout"] == CLOUD_PROBE_TIMEOUT_SECONDS
        assert probe.call_args.kwargs["http_client"] is not None

        step.detected_base_url = "http://127.0.0.1:8080"
        step._launch_probe(api_key=None)
        await pilot.pause()
        assert probe.call_args.kwargs["timeout"] == 2.5
        assert probe.call_args.kwargs["http_client"] is None


def _model_step(wizard, discover_models=None):
    from unittest.mock import AsyncMock

    return ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover_models or AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_model_step_provider_change_resets_selection():
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        step.set_selected_model("gpt-5.6-terra")
        assert step.selected_model_id == "gpt-5.6-terra"
        wizard.wizard_data["provider"] = {
            "provider_key": "anthropic", "provider_value": "Anthropic",
        }
        step.on_show()
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_model_step_commit_writes_chat_defaults():
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        step.set_selected_model("gpt-5.6-terra")
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}
        }


@pytest.mark.asyncio
async def test_model_step_empty_selection_commits_nothing():
    """Skip-safe: leaving the model step untouched must not touch config."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_model_step_clearing_custom_input_clears_stale_selection():
    """Bug-5: typing then clearing the custom-model Input previously left
    selected_model_id stuck at the last typed value (Input.Changed only
    assigned when the value was non-empty) -- clearing it must reset the
    selection so a skip-safe commit doesn't silently keep committing it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        custom_input = step.query_one("#setup-model-custom", Input)
        custom_input.value = "my-custom-model"
        await pilot.pause()
        assert step.selected_model_id == "my-custom-model"

        custom_input.value = ""
        await pilot.pause()
        assert step.selected_model_id == ""

        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_model_step_clearing_custom_input_falls_back_to_radio_selection():
    """Guards the "fall back to a radio selection if one is active" half of
    the Bug-5 fix: clearing the custom Input after a radio pick was also
    made must restore the radio's model, not blank it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=["radio-model-a"]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        radio_set.query_one(RadioButton).value = True
        await pilot.pause()
        assert step.selected_model_id == "radio-model-a"

        custom_input = step.query_one("#setup-model-custom", Input)
        custom_input.value = "my-custom-model"
        await pilot.pause()
        assert step.selected_model_id == "my-custom-model"

        custom_input.value = ""
        await pilot.pause()
        assert step.selected_model_id == "radio-model-a"


@pytest.mark.asyncio
async def test_model_step_commit_reads_pressed_radio_without_changed_event():
    """F-A regression, same pattern as ProviderStep: a RadioButton pressed
    without ever firing Changed must still be recovered at commit time."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=["radio-model-a"]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        target = radio_set.query_one(RadioButton)
        radio_set._pressed_button = target
        assert step.selected_model_id == ""  # sanity: Changed truly never fired

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_model_id == "radio-model-a"
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "chat_defaults": {"provider": "OpenAI", "model": "radio-model-a"}
        }


@pytest.mark.asyncio
async def test_model_step_provider_switch_does_not_resurrect_stale_pressed_radio():
    """F1 regression: Textual's ``RadioSet._pressed_button`` is a plain
    instance attribute that ``remove_children()`` never touches (confirmed
    by reading ``textual/widgets/_radio_set.py`` -- pruning children is
    purely a DOM operation with no watcher on ``_pressed_button``).
    ``_render_models`` calls ``remove_children()``/``mount_all()`` on every
    provider switch, but the OLD, now-detached RadioButton object stays
    referenced by ``_pressed_button`` until a NEW button is pressed in the
    fresh set. Sequence: press a real radio for provider A (via ``.value =
    True``, a genuine toggle -- not manipulating ``_pressed_button``
    directly), switch to provider B via wizard_data + on_show, let the
    re-render happen, then commit with nothing pressed in B's list yet --
    the commit must NOT resurrect provider A's model."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    async def discover(provider_key):
        return {"openai": ["model-a"], "anthropic": ["model-b"]}[provider_key]

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=discover)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        radio_set.query_one(RadioButton).value = True  # real press, fires Changed
        await pilot.pause()
        assert step.selected_model_id == "model-a"  # sanity: the press registered

        wizard.wizard_data["provider"] = {
            "provider_key": "anthropic", "provider_value": "Anthropic",
        }
        step.on_show()
        await pilot.pause(0.1)
        labels = [str(b.label) for b in radio_set.query(RadioButton)]
        assert labels == ["model-b"]  # the re-render itself landed correctly

        ok, error = await step.commit()
        assert ok, error
        assert step._effective_model_id() != "model-a"
        wizard.commit_config.assert_not_called()  # skip-safe: nothing pressed in B's list


@pytest.mark.asyncio
async def test_model_step_no_provider_shows_pick_a_provider_copy():
    """F-F regression: with no provider chosen yet, on_show must not leave
    the initial "(loading models...)" placeholder forever -- there is
    nothing to discover against, so the old code's ``if provider_key:``
    guard just skipped the load entirely and the placeholder never got
    replaced."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={},  # no "provider" entry at all -- provider_key is ""
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=[]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        labels = [str(b.label) for b in radio_set.query(RadioButton)]
        assert "(loading models…)" not in labels
        assert any("pick a provider" in label.lower() for label in labels)


@pytest.mark.asyncio
async def test_model_step_curated_fallback_bridges_raw_provider_key(monkeypatch):
    """Task-6/7 finding: ProviderStep persists chat_defaults.provider as the
    RAW provider_key (e.g. "openai"), but config.toml's curated [providers]
    table is keyed by display name (e.g. "OpenAI"). A naive
    ``catalog.get(provider_value)`` would silently return [] for the raw-key
    form even though a matching curated entry exists -- the fallback must
    bridge key forms regardless of which form ProviderStep handed it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module, "get_cli_providers_and_models",
        lambda: {"OpenAI": ["gpt-curated-1", "gpt-curated-2"]},
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        # provider_value in the RAW form ProviderStep actually persists.
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "openai"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=[]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        labels = [str(button.label) for button in radio_set.query(RadioButton)]
        assert labels == ["gpt-curated-1", "gpt-curated-2"]


@pytest.mark.asyncio
async def test_model_step_uses_scope_service_when_available():
    """The scope-service path (no injected discover_models) renders whatever
    the service reports on a "success" result -- mirrors
    settings_screen.py:7079's call shape."""
    from unittest.mock import AsyncMock, MagicMock as Mock
    from types import SimpleNamespace

    scope_result = SimpleNamespace(status="success", models=("svc-model-a", "svc-model-b"))
    scope_service = Mock()
    scope_service.discover_models = AsyncMock(return_value=scope_result)
    app_instance = MagicMock(app_config={})
    app_instance.llm_provider_catalog_scope_service = scope_service
    wizard = SimpleNamespace(
        app_instance=app_instance,
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=None,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        # _StepHost mounts the step directly (no hidden/visible toggling like
        # the real wizard), so Textual's own Show event fires on top of this
        # test's explicit on_show() call -- exclusive=True on the worker
        # group (like ProviderStep._start_discovery) means only the shape of
        # the *last* call matters here, not the exact invocation count.
        assert scope_service.discover_models.await_args.kwargs == {
            "mode": "local", "provider": "openai", "staged_settings": None
        }
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        labels = [str(button.label) for button in radio_set.query(RadioButton)]
        assert labels == ["svc-model-a", "svc-model-b"]


@pytest.mark.asyncio
async def test_model_step_discovery_timeout_falls_back_to_curated(monkeypatch):
    """Behavior spec: an 8s guard on model discovery -- a slow/hanging
    discover() must not block the step forever; it degrades to the curated
    fallback instead of hanging Next indefinitely."""
    import asyncio as asyncio_module
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    import tldw_chatbook.config as config_module
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(wizard_module, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        config_module, "get_cli_providers_and_models",
        lambda: {"OpenAI": ["fallback-model"]},
    )

    async def _hangs(_provider_key):
        await asyncio_module.sleep(1.0)
        return ["too-slow-model"]

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=_hangs)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.3)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        labels = [str(button.label) for button in radio_set.query(RadioButton)]
        assert labels == ["fallback-model"]


def test_model_step_worker_group_is_not_wizard_advance():
    """Parked Task-5 finding: "setup-wizard-advance" is reserved for the
    container's own commit-on-Next worker; a step reusing it would race or
    duplicate with that worker."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    calls = []

    def _fake_run_worker(coro, **kwargs):
        coro.close()  # never actually scheduled; avoid a "never awaited" warning
        calls.append(kwargs)

    step.run_worker = _fake_run_worker
    step.query_one = MagicMock(side_effect=Exception("not mounted"))
    step.on_show()
    assert calls, "expected on_show to schedule a model-load worker"
    assert calls[0]["group"] == "setup-model-load"


@pytest.mark.asyncio
async def test_rag_step_missing_deps_shows_install_copy_and_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        body = str(step.query_one("#setup-rag-status", Static).render())
        assert "tldw_chatbook[embeddings_rag]" in body
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_step_commit_reads_pressed_radio_without_changed_event():
    """F-A regression, same pattern as ProviderStep/ModelStep, applied to
    RagStep's embedding-model RadioSet."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"embedding_config": {"models": {"embed-a": {}, "embed-b": {}}}}
        ),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-rag-model-choice", RadioSet)
        target = step.query_one(RadioButton)
        radio_set._pressed_button = target
        assert step.selected_embedding_model == ""  # sanity: Changed never fired

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_embedding_model == str(target.label)
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "embedding_config": {"default_model_id": str(target.label)}
        }


@pytest.mark.asyncio
async def test_tools_step_commits_only_changed_gates():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
        assert switches, "tools step must render one switch per gateable tool"
        assert all(sw.value is False for sw in switches)  # default OFF
        switches[0].value = True
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["tools"][step.gate_key_for(switches[0])] is True


@pytest.mark.asyncio
async def test_tools_step_fresh_config_no_changes_commits_nothing():
    """Pin the no-op: on a fresh config every switch starts and stays False,
    so the delta-aware commit added by the final-review fix wave must not regress the
    original "commits nothing when nothing changed" behavior."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_tools_step_on_to_off_transition_writes_false():
    """Re-run prefills a previously-enabled gate ON; turning it back off in
    the UI must persist False, not silently no-op (final-review finding 3)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"tools": {"read_file_enabled": True}}),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch = step.query_one("#setup-tool-read_file", Switch)
        assert switch.value is True  # prefilled ON from config
        switch.value = False  # user turns it back off
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"read_file_enabled": False}}


@pytest.mark.asyncio
async def test_notes_step_commit_writes_directory_and_toggle():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-notes-enable", Switch).value = True
        step.query_one("#setup-notes-directory", Input).value = "~/MyNotes"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "notes": {"sync_directory": "~/MyNotes", "auto_sync_enabled": True}
        }


@pytest.mark.asyncio
async def test_notes_step_disabled_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_notes_step_enabled_to_disabled_writes_auto_sync_false():
    """Re-run prefills the toggle ON from a previously-enabled sync;
    turning it off must persist auto_sync_enabled=False while leaving
    sync_directory untouched (final-review finding 3)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"notes": {"sync_directory": "~/Notes", "auto_sync_enabled": True}}
        ),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch = step.query_one("#setup-notes-enable", Switch)
        assert switch.value is True  # prefilled ON from config
        switch.value = False  # user disables sync
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"notes": {"auto_sync_enabled": False}}


@pytest.mark.asyncio
async def test_protect_keys_enables_encryption_via_injected_callable():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    calls = []
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: calls.append(pw) or True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok = await step.apply_password("hunter2-long-password")
        assert ok is True
        assert calls == ["hunter2-long-password"]


@pytest.mark.asyncio
async def test_protect_keys_failure_leaves_step_skippable_with_inline_error():
    """Failure must not raise nor block Next -- keys stay plaintext, skippable."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok = await step.apply_password("hunter2-long-password")
        assert ok is False
        ok2, error = await step.commit()
        assert ok2, error  # the step itself never blocks Next


def test_protect_keys_password_worker_uses_dedicated_group_not_wizard_advance():
    """Parked Task-5 finding (deviation from the task-10 brief's pseudocode):
    "setup-wizard-advance" is the CONTAINER's own advance/finalize worker
    group. Reusing it here for the password-apply worker would let a slow
    password-hash operation race the container's own commit-on-Next worker
    (both exclusive=True on the same group cancels/blocks the other). Use a
    dedicated group instead; the config RLock inside enable_config_encryption
    is what actually serializes writes, not this group name."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: True,
    )
    calls = []

    def _fake_run_worker(coro, **kwargs):
        coro.close()
        calls.append(kwargs)

    step.run_worker = _fake_run_worker
    step._on_password_result("hunter2-long-password")
    assert calls, "expected a worker to be scheduled for the password result"
    assert calls[0]["group"] == "setup-protect-encrypt"
    assert calls[0]["group"] != "setup-wizard-advance"


@pytest.mark.asyncio
async def test_summary_step_renders_rows_from_read_back():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
        },
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        rendered = str(step.query_one("#setup-summary-rows", Static).render())
        assert "Provider" in rendered
        assert "✓" in rendered and "✗" in rendered


@pytest.mark.asyncio
async def test_summary_footer_shows_the_effective_config_path(monkeypatch, tmp_path):
    """F-D regression (UAT): the footer's "Config file:" line must show the
    REAL effective path -- resolved fresh via get_cli_config_path(), which
    honors a TLDW_CONFIG_PATH override -- not an empty value."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    scratch_config = tmp_path / "scratch-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch_config))

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        footer = str(step.query_one("#setup-summary-footer", Static).render())
        assert str(scratch_config) in footer
        assert "Config file:" in footer


@pytest.mark.asyncio
async def test_summary_quick_track_shows_defaults_note():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        note = str(step.query_one("#setup-summary-defaults-note", Static).render())
        assert "recommended defaults" in note.lower()


@pytest.mark.asyncio
async def test_summary_first_run_exit_buttons_set_expected_routes():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
        advance_programmatically=MagicMock(),
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert {b.id for b in step.query(Button)} == {
            "setup-exit-chat", "setup-exit-home",
        }
        # Direct handler call, not pilot.click(): the actions row sits below
        # what fits in this fixed 120x40 test viewport (same clipping the
        # provider-catalog tests above hit -- see _on_keep's comment), so a
        # click here actually lands on the docked WizardNavigation bar
        # instead of this button. The test is about the handler's effect.
        step._exit_home()
        await pilot.pause()
        assert step.get_step_data() == {"exit_route": TAB_HOME}
        wizard.advance_programmatically.assert_called_once()


@pytest.mark.asyncio
async def test_summary_rerun_exit_buttons_are_done_and_go_to_chat():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.Constants import TAB_CHAT

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
        wizard_data={"welcome": {"track": "quick"}},
        advance_programmatically=MagicMock(),
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert {b.id for b in step.query(Button)} == {
            "setup-exit-done", "setup-exit-chat",
        }
        # See the comment in the first-run-exit test above: direct handler
        # call, not pilot.click() -- the actions row is clipped below this
        # fixed test viewport.
        step._exit_chat()
        await pilot.pause()
        assert step.get_step_data() == {"exit_route": TAB_CHAT}
        wizard.advance_programmatically.assert_called_once()


@pytest.mark.asyncio
async def test_summary_exit_button_advances_the_wizard_without_an_event():
    """SummaryStep's own exit buttons must drive the SAME advance/finalize
    path as the wizard-level Next button (Summary is the last active step),
    but they have no Button.Pressed event targeting "#wizard-next" to hand
    to SetupWizardContainer.handle_next(event) -- which requires one to call
    event.prevent_default(). Exercises the real container end to end (not a
    stub wizard) so a regression back to calling handle_next() with no/None
    event would fail loudly instead of being masked by a mock.

    Reaches Summary via real "#wizard-next" clicks (that button is clear of
    the viewport), then calls the exit handler directly rather than
    pilot.click("#setup-exit-chat") -- the actions row sits below what fits
    in this fixed 120x40 viewport, same as the provider-catalog tests above
    (see _on_keep's comment): a click there actually lands on the docked
    WizardNavigation bar. This test is about advance_programmatically()'s
    wiring, not click hit-regions.
    """
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)
        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            step = container.steps[container.current_step]
            if isinstance(step, SummaryStep):
                step._exit_chat()
            else:
                await pilot.click("#wizard-next")
            await pilot.pause(0.2)
        from tldw_chatbook.Constants import TAB_CHAT

        assert app.wizard_result == {"completed": True, "exit_route": TAB_CHAT}


@pytest.mark.asyncio
async def test_ctrl_n_on_summary_dismisses_and_completes():
    """F-B regression (UAT): pressing ctrl+n while ON the Summary step (the
    last active step) must finish the wizard exactly like clicking its own
    exit buttons or the WizardNavigation "Finish" button does -- dismiss the
    screen and persist first_run.setup_completed.

    Reaches Summary directly via select_track + show_step (not by clicking
    through every prior step) so this test isolates ctrl+n's own dispatch
    and _advance()/complete_wizard()/_handle_complete()'s worker wiring from
    anything upstream."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        summary_index = container._step_index_for_id(STEP_SUMMARY)
        container.show_step(summary_index)
        await pilot.pause(0.2)
        assert isinstance(container.steps[container.current_step], SummaryStep)

        await pilot.press("ctrl+n")
        await pilot.pause(0.3)

        assert app.wizard_result == {"completed": True, "exit_route": None}


@pytest.mark.asyncio
async def test_ctrl_n_still_works_after_focus_was_on_a_now_hidden_widget():
    """F-B ROOT CAUSE (found via live tmux repro + diagnostic instrumentation,
    not the worker-group theory below): Textual's own focus-recovery when
    the currently-focused widget becomes hidden (Screen._reset_focus, run
    when a step's container gets `display: none` on every step change --
    BaseWizard.show_step()'s `current.add_class("hidden")`) is unreliable:
    depending on what else happens to sit in the global focus chain at that
    moment, it can land back on None, OR on some OTHER incidentally-hidden
    widget from the very step that just got hidden (observed live and
    reproduced here: with nothing else to fall back to it goes fully None;
    with an unrelated hidden sibling button present as a candidate, Textual
    quietly refocuses THAT non-interactive widget instead -- neither is a
    real focus target). Either way, a user whose last interaction was with a
    control INSIDE a step's own content (a RadioButton, an Input -- as
    opposed to the persistent WizardNavigation bar, which is never hidden)
    ends up with no RELIABLE focus target; ctrl+n/ctrl+b (bound several
    ancestors up from wherever the user last interacted) then have no
    guaranteed focus chain to resolve bindings through and can go silently
    inert -- confirmed live: a diagnostic log line inside
    advance_programmatically() fired for three consecutive successful
    ctrl+n presses and produced NOTHING on the fourth (Summary -> Finish),
    while clicking the same "Finish" button worked immediately after and
    also proved app.focused had indeed become None by then.

    Round-2 regression + fix: the FIRST cut of this fix always re-focused
    the persistent nav bar's own Next/Cancel button after every step change.
    That broke direct keyboard interaction with the new step's own content
    -- landing on Provider with focus already parked on "Next" meant
    Down/Space (which only act on a FOCUSED RadioSet) silently did nothing,
    reproducing the exact "selection doesn't commit" symptom one level up
    in the UI. The corrected fix prefers the incoming step's own first
    focusable descendant (DOM order) and falls back to the nav bar only
    when the step truly has none. Pin that exact invariant -- "not None" is
    too weak a check, since Textual's own incidental fallback can
    accidentally satisfy it without the wizard being reliably
    keyboard-navigable, and "always the nav bar" is now the wrong
    behavior."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.1)

        def _first_focusable(step):
            return next((w for w in step.walk_children(Widget) if w.focusable), None)

        def _assert_focus_on_current_step_content() -> None:
            current = container.steps[container.current_step]
            expected = _first_focusable(current)
            assert expected is not None, f"{current!r} has no focusable widget"
            assert app.focused is expected, (
                f"expected focus on {current!r}'s first focusable widget "
                f"{expected!r}, got {app.focused!r}"
            )

        await pilot.press("ctrl+n")  # Welcome -> Provider
        await pilot.pause(0.2)
        _assert_focus_on_current_step_content()
        provider_step = container.steps[container.current_step]
        assert isinstance(provider_step, ProviderStep)
        radio_set = provider_step.query_one("#setup-provider-choice", RadioSet)
        assert app.focused is radio_set  # the auto-focus landed here, no Tab needed

        await pilot.press("ctrl+n")  # Provider -> Model
        await pilot.pause(0.2)
        _assert_focus_on_current_step_content()

        model_step = container.steps[container.current_step]
        assert isinstance(model_step, ModelStep)
        # Simulate the live UAT sequence: the user clicks into the custom-
        # model Input specifically (overriding the RadioSet the auto-focus
        # landed on) rather than accepting a curated radio option.
        custom_input = model_step.query_one("#setup-model-custom", Input)
        custom_input.focus()
        await pilot.pause(0.1)
        assert app.focused is custom_input  # sanity: focus is inside Model's own Input

        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            # Once the wizard has actually completed, the screen is
            # dismissed and app.focused legitimately going None reflects
            # that there is no more wizard to hold it -- only check the
            # focus invariant while the wizard is still open.
            if app.wizard_result == "UNSET":
                _assert_focus_on_current_step_content()

        assert app.wizard_result == {"completed": True, "exit_route": None}


@pytest.mark.asyncio
async def test_down_space_selects_provider_with_no_tab_presses():
    """Round-2 regression pin (live-confirmed by the controller): Down then
    Space on the Provider step, immediately after ctrl+n from Welcome, with
    NO Tab press in between, must select a provider. The first cut of the
    F-B focus fix parked focus on the nav bar's Next button after every step
    change, so Down/Space (RadioSet-only bindings) landed on the wrong
    widget and silently selected nothing -- reproducing F-A's "no provider
    commit" symptom purely through keyboard navigation, no click/RadioSet
    stub involved."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.1)

        await pilot.press("ctrl+n")  # Welcome -> Provider
        await pilot.pause(0.2)
        provider_step = container.steps[container.current_step]
        assert isinstance(provider_step, ProviderStep)
        radio_set = provider_step.query_one("#setup-provider-choice", RadioSet)
        assert app.focused is radio_set  # sanity: no Tab needed to reach it

        await pilot.press("down")
        await pilot.press("space")
        await pilot.pause(0.2)

        assert radio_set.pressed_button is not None
        assert provider_step.selected_provider_key != ""


def test_finalize_worker_uses_a_dedicated_group_not_wizard_advance():
    """F-B fix pin: _handle_complete() runs synchronously from inside
    complete_wizard(), itself called synchronously from _advance() -- the
    body of the CURRENTLY-RUNNING "setup-wizard-advance" worker whenever the
    step being advanced past has no real await in its own commit() (true for
    SummaryStep, which never overrides SetupStep's trivial default commit).
    Scheduling _finalize into that same exclusive group asks Textual to
    cancel_group() the group it is currently executing from inside itself --
    confirmed harmless only by scheduling luck (a separately-created task
    survives regardless), not by design. Pin the dedicated group so this
    does not regress back to relying on that accident."""
    app_instance = MagicMock()
    app_instance.app_config = {}
    real_container = SetupWizardContainer(app_instance)
    calls = []

    def _fake_run_worker(coro, **kwargs):
        coro.close()  # never actually scheduled; avoid a "never awaited" warning
        calls.append(kwargs)

    real_container.run_worker = _fake_run_worker
    real_container._handle_complete({"summary": {"exit_route": None}})
    assert calls, "expected _handle_complete to schedule the finalize worker"
    assert calls[0]["group"] == "setup-wizard-finalize"
    assert calls[0]["group"] != "setup-wizard-advance"


@pytest.mark.asyncio
async def test_finalize_and_dismiss_screen_never_double_dismiss():
    """F3 hardening: a duplicate entry into _finalize/_dismiss_screen (e.g.
    a stray extra Finish click/ctrl+n racing the "setup-wizard-finalize"
    worker, or Skip-entirely arriving after Finish already completed) must
    be a clean no-op, not a second Screen.dismiss() call."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        summary_index = container._step_index_for_id(STEP_SUMMARY)
        container.show_step(summary_index)
        await pilot.pause(0.2)

        dismiss_calls = []
        wizard.dismiss = lambda result=None: dismiss_calls.append(result)

        await pilot.press("ctrl+n")
        await pilot.pause(0.3)
        assert len(dismiss_calls) == 1
        assert container._finalized is True

        # Duplicate entries via BOTH public entry points must be no-ops.
        await container._finalize(None)
        container._dismiss_screen({"completed": True, "exit_route": "duplicate"})
        assert len(dismiss_calls) == 1


@pytest.mark.asyncio
async def test_appearance_step_commits_theme_and_card():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.selected_theme = "textual-light"
        step.selected_splash_card = "matrix"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["general"] == {"default_theme": "textual-light"}
        assert committed["splash_screen"] == {"card_selection": "matrix"}


@pytest.mark.asyncio
async def test_appearance_step_rerun_preselects_configured_theme():
    """Added scope (Task-11 controller decision): re-run must prefill every
    step from current config. AppearanceStep previously always rendered its
    theme RadioSet with nothing pressed, even when general.default_theme was
    already set -- pre-select the RadioButton matching it, when the theme is
    in the rendered list."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"general": {"default_theme": "nord"}}),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-theme-choice", RadioSet)
        pressed = radio_set.pressed_button
        assert pressed is not None
        assert str(pressed.label) == "nord"


@pytest.mark.asyncio
async def test_appearance_step_no_config_theme_preselects_nothing():
    """First-run behavior must stay unchanged: with no general.default_theme,
    no RadioButton is pre-pressed."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-theme-choice", RadioSet)
        assert radio_set.pressed_button is None


@pytest.mark.asyncio
async def test_appearance_step_rerun_change_only_splash_card_leaves_theme_untouched():
    """Bug-2a/b: AppearanceStep.commit() used to fall back to a hardcoded
    "textual-dark" default whenever selected_theme was empty, clobbering a
    persisted theme on a rerun that only touches the splash card. compose()
    must initialize selected_theme from the persisted default (a), and the
    delta-aware commit must omit general.default_theme when the chosen
    theme matches what's already persisted (b)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"general": {"default_theme": "nord"}}),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # compose() must have initialized selected_theme from the persisted
        # default -- pin that directly, since it's the crux of fix (a).
        assert step.selected_theme == "nord"

        # Only the splash card changes this run; the theme RadioSet is left
        # untouched at its pre-selected ("nord") position.
        step.selected_splash_card = "matrix"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert "general" not in committed
        assert committed["splash_screen"] == {"card_selection": "matrix"}


@pytest.mark.asyncio
async def test_appearance_step_surprise_me_over_persisted_card_writes_random():
    """Bug-2c: "Surprise me (random)" maps to splash_card=None, which the
    old commit() unconditionally treated as "nothing to write" -- so a
    previously persisted specific card could never be reset back to random.
    Explicitly re-picking "Surprise me" over a persisted specific card must
    write card_selection="random"."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"splash_screen": {"card_selection": "matrix"}}
        ),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-splash-choice", RadioSet)
        buttons = list(radio_set.query(RadioButton))
        surprise_button = next(
            b for b in buttons if str(b.label).startswith("Surprise me")
        )
        other_button = next(
            b for b in buttons if not str(b.label).startswith("Surprise me")
        )
        # "Surprise me" is already the default mount-time pre-selection, and
        # RadioSet does not fire Changed for its own initial state -- press
        # a different card first, then explicitly re-press "Surprise me",
        # to mirror a real user re-picking it.
        other_button.value = True
        await pilot.pause()
        surprise_button.value = True
        await pilot.pause()

        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["splash_screen"] == {"card_selection": "random"}


@pytest.mark.asyncio
async def test_appearance_step_fresh_run_untouched_commits_nothing():
    """Bug-2 regression guard: a truly fresh run where the user never
    touches either RadioSet must still commit nothing at all (unchanged
    skip-safe behavior), even now that selected_theme is initialized from
    prefill in compose()."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_tools_step_rerun_prefills_switches_from_config():
    """Added scope: ToolsStep previously always initialized every Switch to
    False, even on re-run with gates already enabled in config -- initialize
    each Switch from prefill.tool_gates instead. First-run behavior (no
    "tools" section, or a section with everything off) is unchanged since
    tool_gates comes back empty/False in that case."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"tools": {"read_file_enabled": True}}
        ),
        commit_config=AsyncMock(return_value=True), rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        enabled_switch = step.query_one("#setup-tool-read_file", Switch)
        assert enabled_switch.value is True
        other_switches = [
            sw for sw in step.query(Switch) if sw.id != "setup-tool-read_file"
        ]
        assert other_switches, "expect more than one gateable tool"
        assert all(sw.value is False for sw in other_switches)


@pytest.mark.asyncio
async def test_model_step_rerun_prefills_from_config_when_no_provider_entry_yet():
    """Added scope: a re-run user who reaches Model before wizard_data has a
    "provider" entry this session (e.g. jumping forward) must see the
    persisted chat_defaults.model resurface as the initial selection and in
    the custom-model Input, rather than a blank slate."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"chat_defaults": {"model": "gpt-4o"}}),
        wizard_data={},  # no "provider" key yet
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        assert step.selected_model_id == "gpt-4o"
        assert step.query_one("#setup-model-custom", Input).value == "gpt-4o"


@pytest.mark.asyncio
async def test_model_step_with_provider_entry_present_does_not_prefill_stale_model():
    """Guards the boundary of the added scope above: once a "provider" entry
    exists in wizard_data (the normal sequential path, and a real
    Back-and-switch), the existing reset-to-blank behavior must still apply
    -- the prefill path is only for the "no entry yet" case."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"chat_defaults": {"model": "gpt-4o"}}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        assert step.selected_model_id == ""
        assert step.query_one("#setup-model-custom", Input).value == ""


@pytest.mark.asyncio
async def test_rerun_with_stored_plaintext_key_activates_protect_step_without_typing():
    """Bug-4: active_step_ids previously dropped STEP_PROTECT unless a
    secret was typed THIS run, so a rerun over a config that already has a
    plaintext key on disk could never reach Protect Keys without retyping a
    credential. The gate must also fire from config alone."""
    app_instance = MagicMock()
    app_instance.app_config = {"api_settings": {"openai": {"api_key": "sk-existing"}}}
    wizard = FirstRunSetupWizard(app_instance, rerun=True)
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert STEP_PROTECT in container.active_ids


@pytest.mark.asyncio
async def test_fresh_config_without_stored_key_omits_protect_step():
    """Regression guard for the Bug-4 fix above: a fresh config with no
    stored key and nothing typed this run must still omit STEP_PROTECT."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert STEP_PROTECT not in container.active_ids


class TestAppOfferGating:
    """The app hook is thin; assert the state functions drive it correctly."""

    def test_fresh_config_offers_and_upgrader_does_not(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import should_offer_wizard

        assert should_offer_wizard({}, {}) is True
        upgrader = {"api_settings": {"openai": {"api_key": "sk-x"}}}
        assert should_offer_wizard(upgrader, {}) is False

    def test_rerun_flag_reaches_container(self):
        wizard = _make_wizard(rerun=True)
        assert wizard.rerun is True


class TestCommandPaletteReentry:
    """AC #4 (task-1264): "re-runnable from Settings and the command
    palette". The Settings re-entry button is covered app-level in
    Tests/UI/test_first_run_wizard_live_contract.py; a Task 12 audit found
    NOTHING anywhere exercised SetupWizardProvider (app.py), the command
    palette's entire bridge to the wizard -- this closes that gap.
    """

    def test_run_setup_wizard_action_pushes_rerun_wizard(self):
        from tldw_chatbook.app import SetupWizardProvider
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        screen = MagicMock()
        provider = SetupWizardProvider(screen)

        provider.handle_setup_wizard_action("run_setup_wizard")

        screen.app.push_screen.assert_called_once()
        (pushed_wizard, callback), _kwargs = screen.app.push_screen.call_args
        assert isinstance(pushed_wizard, FirstRunSetupWizard)
        assert pushed_wizard.rerun is True
        # Final-review finding 2: this push must wire the app-level result
        # callback, exactly like the Settings button and the auto-offer
        # path (app.py's _push_first_run_wizard) already do -- without it,
        # a truthy exit_route off the Summary step's "Go to Chat" button is
        # silently dropped instead of navigating anywhere.
        assert callback == screen.app.handle_first_run_wizard_result

    def test_unknown_action_id_is_a_no_op(self):
        from tldw_chatbook.app import SetupWizardProvider

        screen = MagicMock()
        provider = SetupWizardProvider(screen)

        provider.handle_setup_wizard_action("something_else")

        screen.app.push_screen.assert_not_called()
