"""Component tests for the Console "New endpoint from template" modal (ADR-146)."""

import threading
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


def _conflict_result() -> "config_module.ConfigMutationResult":
    """A locked-snapshot precondition abort (slug taken under the lock)."""
    return config_module.ConfigMutationResult(
        False, False, None, conflict=True, conflict_reason="identity_changed"
    )


def _saved_result() -> "config_module.ConfigMutationResult":
    return config_module.ConfigMutationResult(True, True, None)


@pytest.mark.asyncio
async def test_template_modal_rederives_slug_when_concurrent_create_takes_it(
    monkeypatch,
):
    """A competing same-process create that commits the derived slug under
    the config writer lock must not be overwritten: the create-only write
    aborts with a collision and the slug is re-derived for a retry."""
    import tldw_chatbook.Widgets.Console.console_endpoint_template_modal as modal_module

    saved_mutations = []
    write_results = [_conflict_result(), _saved_result()]

    def fake_apply(section_values, **_kwargs):
        # First create-only write collides (a competitor committed
        # "gpu-box" under the lock); the retry for the re-derived slug
        # succeeds.
        saved_mutations.append(section_values)
        return write_results.pop(0)

    monkeypatch.setattr(modal_module, "load_custom_endpoints", lambda _cfg: {})
    monkeypatch.setattr(modal_module, "apply_settings_mutation_to_cli_config", fake_apply)
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

    # The successful write went to a re-derived slug, not the collided one.
    assert [list(mutation) for mutation in saved_mutations] == [
        ["custom_endpoints.gpu-box"],
        ["custom_endpoints.gpu-box-2"],
    ]
    assert app.created_provider_id == "custom-ep:gpu-box-2"
    assert "gpu-box-2" in load_custom_endpoints(app.app_config)


@pytest.mark.asyncio
async def test_template_modal_surfaces_in_use_error_when_rederive_attempts_exhaust(
    monkeypatch,
):
    """When every bounded re-derivation attempt collides with a concurrent
    create, Create surfaces the inline name-in-use error and writes nothing
    (not the generic save-failure copy)."""
    import tldw_chatbook.Widgets.Console.console_endpoint_template_modal as modal_module

    saved_mutations = []

    def fake_apply(section_values, **_kwargs):
        # Every create-only write collides: a competitor keeps committing
        # each derived slug under the lock.
        saved_mutations.append(section_values)
        return _conflict_result()

    monkeypatch.setattr(modal_module, "load_custom_endpoints", lambda _cfg: {})
    monkeypatch.setattr(modal_module, "apply_settings_mutation_to_cli_config", fake_apply)
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

    # All three bounded attempts collided and nothing was persisted.
    assert [list(mutation) for mutation in saved_mutations] == [
        ["custom_endpoints.gpu-box"],
        ["custom_endpoints.gpu-box-2"],
        ["custom_endpoints.gpu-box-3"],
    ]
    assert app.created_provider_id is None


@pytest.mark.asyncio
async def test_template_modal_does_not_overwrite_disk_entry_missing_from_stale_view(
    tmp_path, monkeypatch
):
    """A competitor entry committed to disk after this modal's in-memory view
    was built must survive Create: the write shares the config writer lock,
    so the occupied section aborts the mutation and the slug re-derives
    instead of replacing the competitor's URL/models/credentials."""
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        # The competitor commits "gpu-box" to the real config file; the
        # modal's in-memory app_config below stays stale (empty), exactly
        # like a create racing a writer whose commit is not mirrored into
        # this mapping.
        assert config_module.save_settings_to_cli_config(
            {
                "custom_endpoints": {
                    "gpu-box": {
                        "display_name": "Competitor",
                        "family": "llama_cpp",
                        "base_url": "http://127.0.0.1:8080",
                    }
                }
            }
        )
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

        raw = tomllib.loads(config_path.read_text())
        # The competitor's section is intact and ours went to a fresh slug.
        assert raw["custom_endpoints"]["gpu-box"]["display_name"] == "Competitor"
        assert raw["custom_endpoints"]["gpu-box-2"]["display_name"] == "GPU box"
        assert app.created_provider_id == "custom-ep:gpu-box-2"
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


@pytest.mark.asyncio
async def test_template_modal_create_runs_registry_derivation_off_the_event_loop(
    monkeypatch,
):
    """Registry loading and slug derivation during Create run in a worker
    thread (asyncio.to_thread), never on the UI event loop -- a registry
    with thousands of occupied slugs must not freeze the modal's controls."""
    import tldw_chatbook.Widgets.Console.console_endpoint_template_modal as modal_module

    loads_on_main_thread = []

    def fake_load(_app_config):
        loads_on_main_thread.append(
            threading.current_thread() is threading.main_thread()
        )
        return {}

    monkeypatch.setattr(modal_module, "load_custom_endpoints", fake_load)
    monkeypatch.setattr(
        modal_module,
        "apply_settings_mutation_to_cli_config",
        lambda _section_values, **_kwargs: _saved_result(),
    )
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

    assert app.created_provider_id == "custom-ep:gpu-box"
    # Modal init loads on the UI thread; Create's derivation load must not.
    assert loads_on_main_thread[0] is True
    assert any(entry is False for entry in loads_on_main_thread[1:])


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


@pytest.mark.asyncio
async def test_template_modal_duplicate_prefill_respects_display_name_limit(
    tmp_path, monkeypatch
):
    """Duplicating an entry named at the 80-character maximum must prefill a
    still-valid duplicate name: the source is truncated to reserve the
    seven-character ' (copy)' suffix instead of leaving Create disabled."""
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app_config = {
            "custom_endpoints": {
                "long": {
                    # Valid at exactly the registry's 80-char maximum.
                    "display_name": "L" * 80,
                    "family": "openai_compatible",
                    "base_url": "https://api.example.com/v1",
                }
            }
        }
        app = _TemplateModalHarness(app_config=app_config)
        async with app.run_test(size=(100, 40)) as pilot:
            modal = ConsoleEndpointTemplateModal(
                app_config=app.app_config,
                providers_models={},
                template_provider="custom-ep:long",
            )
            await app.push_screen(modal)
            name = app.screen.query_one("#endpoint-template-name", Input)
            # 73 truncated source chars + " (copy)" = exactly 80.
            assert name.value == "L" * 73 + " (copy)"
            create = app.screen.query_one("#endpoint-template-create", Button)
            assert create.disabled is False
            await pilot.click("#endpoint-template-create")
            await pilot.pause()

        assert app.created_provider_id == "custom-ep:" + "l" * 64
        raw = tomllib.loads(config_path.read_text())
        duplicated = raw["custom_endpoints"]["l" * 64]
        assert duplicated["display_name"] == "L" * 73 + " (copy)"
        assert duplicated["base_url"] == "https://api.example.com/v1"
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
