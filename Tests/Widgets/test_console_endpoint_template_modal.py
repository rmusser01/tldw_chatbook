"""Component tests for the Console "New endpoint from template" modal (ADR-146)."""

import tomllib
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints
from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
    ConsoleEndpointTemplateModal,
)


class _TemplateModalHarness(ConsolidatedCSSApp):
    """StyledModalHarness-equivalent minimal host for the template modal."""

    CSS = """
    Screen {
        layout: vertical;
    }
    """
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, app_config: dict) -> None:
        super().__init__()
        self.app_config = app_config
        self.created_provider_id: str | None = None

    def on_console_endpoint_template_modal_endpoint_created(
        self, event: ConsoleEndpointTemplateModal.EndpointCreated
    ) -> None:
        self.created_provider_id = event.provider_id


@pytest.mark.asyncio
async def test_template_modal_creates_entry_and_dismisses_with_id(
    tmp_path, monkeypatch
):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _TemplateModalHarness(app_config={})
        async with app.run_test(size=(100, 40)) as pilot:
            modal = ConsoleEndpointTemplateModal(
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                template_provider="llama_cpp",
            )
            await app.push_screen(modal)
            await pilot.click("#endpoint-template-name")
            await pilot.press(*"GPU box")
            await pilot.click("#endpoint-template-url")
            # pilot.press types nothing for multi-char strings (char-less Key
            # events), so unpack the URL into single-character keys.
            await pilot.press("ctrl+a", *"http://192.168.1.9:8080")
            await pilot.click("#endpoint-template-create")
            await pilot.pause()
        assert app.created_provider_id == "custom-ep:gpu-box"
        entry = load_custom_endpoints(app.app_config)["gpu-box"]
        assert entry.family == "llama_cpp"
        assert entry.base_url == "http://192.168.1.9:8080"
        # The [custom_endpoints.<slug>] section round-trips through the real
        # (unmonkeypatched) atomic writer into the temp config file itself.
        raw = tomllib.loads(config_path.read_text())
        assert raw["custom_endpoints"]["gpu-box"] == {
            "display_name": "GPU box",
            "family": "llama_cpp",
            "base_url": "http://192.168.1.9:8080",
            "models": ["model-a"],
            "created_from": "llama_cpp",
        }
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


@pytest.mark.asyncio
async def test_template_modal_shows_validation_inline(tmp_path):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp",
        )
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-url")
        # Unpacked for the same multi-char press limitation as above.
        await pilot.press("ctrl+a", *"ftp://192.168.1.9:8080")
        error = app.screen.query_one("#endpoint-template-error", Static)
        assert "http(s)" in error.renderable
        create = app.screen.query_one("#endpoint-template-create", Button)
        assert create.disabled is True
        assert app.created_provider_id is None
        assert load_custom_endpoints(app.app_config) == {}


@pytest.mark.asyncio
async def test_template_modal_surfaces_slug_exhaustion_inline(tmp_path, monkeypatch):
    # Every derivable slug for the name is taken: Create must surface the
    # collision inline (error-banner pattern) instead of persisting an
    # entry that overwrites an existing slug's config section.
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        # gpu-box plus gpu-box-2 .. gpu-box-9999: exhausts bare base and
        # every suffixed candidate through the 4-digit suffix search.
        squatted = {"gpu-box"} | {f"gpu-box-{n}" for n in range(2, 10000)}
        app_config = {
            "custom_endpoints": {
                slug: {
                    "display_name": "Squatter",
                    "family": "llama_cpp",
                    "base_url": "http://127.0.0.1:8080",
                }
                for slug in squatted
            }
        }
        app = _TemplateModalHarness(app_config=app_config)
        async with app.run_test(size=(100, 40)) as pilot:
            modal = ConsoleEndpointTemplateModal(
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                template_provider="llama_cpp",
            )
            await app.push_screen(modal)
            await pilot.click("#endpoint-template-name")
            await pilot.press(*"GPU box")
            await pilot.click("#endpoint-template-create")
            await pilot.pause()
            error = app.screen.query_one("#endpoint-template-error", Static)
            assert "already in use" in str(error.renderable)
            assert error.display is True
            # Create stays usable (retry with another name) and no entry
            # was persisted -- the modal is still up and the file untouched.
            create = app.screen.query_one("#endpoint-template-create", Button)
            assert create.disabled is False
        assert app.created_provider_id is None
        assert "custom_endpoints" not in tomllib.loads(config_path.read_text())
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


def _registry_section() -> dict:
    """A valid raw ``[custom_endpoints.<slug>]`` table for fake loads."""
    return {
        "display_name": "Competitor",
        "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
    }


@pytest.mark.asyncio
async def test_template_modal_rederives_slug_when_concurrent_create_takes_it(
    monkeypatch,
):
    """A competing same-process create that lands between derivation and the
    atomic write must not be overwritten: the slug is re-derived against the
    fresh registry before writing."""
    import tldw_chatbook.Widgets.Console.console_endpoint_template_modal as modal_module

    loads = []

    def fake_load(app_config):
        # Load 1 is the picker build (modal init); load 2 is Create's
        # initial derivation -- both see an empty registry, deriving
        # "gpu-box". Every later load (the pre-write re-check and any
        # re-derivation) sees a competitor that took "gpu-box" in between.
        loads.append(1)
        if len(loads) <= 2:
            return {}
        return {"gpu-box": _registry_section()}

    saved_mutations = []

    def fake_save(mutation):
        saved_mutations.append(mutation)
        return True

    monkeypatch.setattr(modal_module, "load_custom_endpoints", fake_load)
    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", fake_save)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp",
        )
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-create")
        await pilot.pause()

    # The write went to a re-derived slug, not the collided one.
    assert saved_mutations, "entry was never persisted"
    assert list(saved_mutations[0]) == ["custom_endpoints.gpu-box-2"]
    assert app.created_provider_id == "custom-ep:gpu-box-2"
    assert "gpu-box-2" in load_custom_endpoints(app.app_config)


@pytest.mark.asyncio
async def test_template_modal_surfaces_in_use_error_when_rederive_attempts_exhaust(
    monkeypatch,
):
    """When every bounded re-derivation attempt collides with a concurrent
    create, Create surfaces the inline name-in-use error and writes nothing."""
    import tldw_chatbook.Widgets.Console.console_endpoint_template_modal as modal_module

    loads = []

    def fake_load(app_config):
        # Loads 1-2 (picker build, initial derivation) see {} (->
        # "gpu-box"); each pre-write re-check then reveals a competitor
        # that sniped the candidate just derived.
        loads.append(1)
        n = len(loads)
        if n <= 2:
            return {}
        if n <= 4:
            return {"gpu-box": _registry_section()}
        if n <= 6:
            return {
                "gpu-box": _registry_section(),
                "gpu-box-2": _registry_section(),
            }
        return {
            "gpu-box": _registry_section(),
            "gpu-box-2": _registry_section(),
            "gpu-box-3": _registry_section(),
        }

    saved_mutations = []

    def fake_save(mutation):
        saved_mutations.append(mutation)
        return True

    monkeypatch.setattr(modal_module, "load_custom_endpoints", fake_load)
    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", fake_save)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp",
        )
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-create")
        await pilot.pause()
        # Still mounted, error banner carries the in-use copy, Create usable.
        error = app.screen.query_one("#endpoint-template-error", Static)
        assert "already in use" in str(error.renderable)
        create = app.screen.query_one("#endpoint-template-create", Button)
        assert create.disabled is False

    assert saved_mutations == []
    assert app.created_provider_id is None


@pytest.mark.asyncio
async def test_template_modal_duplicate_carries_env_ref_and_copy_name(
    tmp_path, monkeypatch
):
    """Duplicating an existing entry copies the credential *reference*
    (api_key_env, never the stored api_key) and prefills the display name
    suffixed ' (copy)' (spec: creation-from-template decision)."""
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app_config = {
            "custom_endpoints": {
                "paid": {
                    "display_name": "Paid",
                    "family": "openai_compatible",
                    "base_url": "https://api.example.com/v1",
                    "api_key_env": "PAID_KEY",
                    "api_key": "stored-secret",
                    "models": ["m1"],
                }
            }
        }
        app = _TemplateModalHarness(app_config=app_config)
        async with app.run_test(size=(100, 40)) as pilot:
            modal = ConsoleEndpointTemplateModal(
                app_config=app.app_config,
                providers_models={},
                template_provider="custom-ep:paid",
            )
            await app.push_screen(modal)
            # The duplicate template prefills the display name with (copy).
            name = app.screen.query_one("#endpoint-template-name", Input)
            assert name.value == "Paid (copy)"
            await pilot.click("#endpoint-template-create")
            await pilot.pause()

        assert app.created_provider_id == "custom-ep:paid-copy"
        # The stored secret never crosses over; the env reference does.
        raw = tomllib.loads(config_path.read_text())
        duplicated = raw["custom_endpoints"]["paid-copy"]
        assert duplicated["api_key_env"] == "PAID_KEY"
        assert "api_key" not in duplicated
        assert duplicated["display_name"] == "Paid (copy)"
        assert duplicated["base_url"] == "https://api.example.com/v1"
        assert duplicated["created_from"] == "custom-ep:paid"
        # The in-memory mirror keeps the credential reference too, so the
        # opener's readiness/credential resolution sees it without a reload.
        assert (
            load_custom_endpoints(app.app_config)["paid-copy"].api_key_env
            == "PAID_KEY"
        )
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
