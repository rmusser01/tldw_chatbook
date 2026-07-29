"""Pilot tests for the first-run setup wizard skeleton."""

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, RadioButton, RadioSet, Static, Switch

from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    CLOUD_PROBE_TIMEOUT_SECONDS,
    FirstRunSetupWizard,
    ModelStep,
    ProviderStep,
    RagStep,
    SetupWizardContainer,
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
