"""Pilot tests for the first-run setup wizard skeleton."""

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
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
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
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
        assert committed == {
            "api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"}
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
        assert committed == {}
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
        assert committed == {"api_settings.openai": {"api_key": ""}}
        wizard.note_key_entered.assert_not_called()


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
        (pushed_wizard,), _kwargs = screen.app.push_screen.call_args
        assert isinstance(pushed_wizard, FirstRunSetupWizard)
        assert pushed_wizard.rerun is True

    def test_unknown_action_id_is_a_no_op(self):
        from tldw_chatbook.app import SetupWizardProvider

        screen = MagicMock()
        provider = SetupWizardProvider(screen)

        provider.handle_setup_wizard_action("something_else")

        screen.app.push_screen.assert_not_called()
