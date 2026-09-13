"""F9 Settings custom endpoint management seams (ADR-146, task-7).

The four brief-verbatim tests exercise the panel-action seams in
``settings_provider_view_model``: overview rows render safe endpoint
displays, the delete guard lists referencing sessions, detach-then-delete
re-points sessions and removes the entry, and slot conversion creates a
named endpoint while leaving the slot untouched. Config round-trips run
against a ``TLDW_CONFIG_PATH`` temp config (the Task 6 pattern in
Tests/Widgets/test_console_endpoint_template_modal.py).
"""


import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.custom_endpoint_registry import (
    CustomEndpointEntry,
    load_custom_endpoints,
)
from tldw_chatbook.config import load_settings, save_settings_to_cli_config
from tldw_chatbook.UI.Screens.settings_provider_view_model import (
    build_entry_edit_mutation,
    conversations_referencing_endpoint,
    convert_slot_to_named_endpoint,
    custom_endpoint_rows,
    detach_and_delete_entry,
)

_REGISTRY_ENTRY_TOML = """\
[custom_endpoints.gpu]
display_name = "GPU llama"
family = "llama_cpp"
base_url = "http://192.168.1.5:8080"
models = ["model-a", "model-b"]
api_key = "sk-test-never-rendered"
"""

_SLOT_TOML = """\
[api_settings.custom]
api_url = "http://127.0.0.1:5000/v1"
model = "my-model"
"""


def _registry_config() -> dict:
    """Mirror of ``_REGISTRY_ENTRY_TOML`` as the in-memory app_config view."""
    return {
        "custom_endpoints": {
            "gpu": {
                "display_name": "GPU llama",
                "family": "llama_cpp",
                "base_url": "http://192.168.1.5:8080",
                "models": ["model-a", "model-b"],
                "api_key": "sk-test-never-rendered",
            }
        }
    }


#: Llama-family entry whose raw URL carries the ``/v1`` suffix that load
#: normalization strips — proves detach writes the normalized origin.
_REGISTRY_LLAMA_V1_TOML = """\
[custom_endpoints.gpu]
display_name = "GPU llama"
family = "llama_cpp"
base_url = "http://192.168.1.5:8080/v1"
models = ["model-a"]
"""

#: Entry carrying a credential reference the Edit flow must be able to clear.
_REGISTRY_ENV_TOML = """\
[custom_endpoints.gpu]
display_name = "GPU llama"
family = "llama_cpp"
base_url = "http://192.168.1.5:8080"
models = ["model-a"]
api_key_env = "GPU_KEY"
"""


def _store_with_session(
    provider: str, base_url: str | None = "http://192.168.1.5:8080"
) -> ConsoleChatStore:
    """Minimal store with one session pinned to ``provider``.

    A real ``ConsoleChatStore`` (not a stub): detach must flow through the
    store's own settings mutation path. ``base_url`` defaults to the GPU
    endpoint; pass ``None`` for a session that never carried its own URL
    (the registry entry was its only endpoint source).
    """
    store = ConsoleChatStore()
    store.create_session(
        title="GPU chat",
        settings=ConsoleSessionSettings(
            provider=provider,
            base_url=base_url,
        ),
    )
    return store


def _activate_temp_config(tmp_path, monkeypatch, raw_toml: str) -> None:
    """Point the config cache at a temp file pre-seeded with ``raw_toml``."""
    config_path = tmp_path / "settings-custom-endpoints.toml"
    config_path.write_text(raw_toml)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def _reload_config() -> None:
    """Restore the process config caches after a temp-config test."""
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def test_custom_endpoint_rows_render_safe_display():
    rows = custom_endpoint_rows(_registry_config())
    row = next(r for r in rows if "GPU llama" in r.label)
    assert "192.168.1.5:8080" in row.value
    assert "api_key" not in row.value


def test_delete_guard_lists_referencing_sessions():
    store = _store_with_session(provider="custom-ep:gpu")  # minimal store stub
    assert conversations_referencing_endpoint(store, "custom-ep:gpu") != []


def test_delete_after_detach_removes_entry(tmp_path, monkeypatch):
    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENTRY_TOML)
    app_config = _registry_config()
    store = _store_with_session(provider="custom-ep:gpu")  # minimal store stub
    try:
        detach_and_delete_entry(app_config, store, "custom-ep:gpu")  # panel action seam
        assert load_custom_endpoints(load_settings()) == {}
        assert store.session_settings(store.sessions()[0].id).provider == "llama_cpp"
        # Detach keeps the session's current URL as a session-only base_url.
        assert (
            store.session_settings(store.sessions()[0].id).base_url
            == "http://192.168.1.5:8080"
        )
    finally:
        _reload_config()


def test_detach_preserves_entry_url_for_blank_session_base_url(tmp_path, monkeypatch):
    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_LLAMA_V1_TOML)
    app_config = {
        "custom_endpoints": {
            "gpu": {
                "display_name": "GPU llama",
                "family": "llama_cpp",
                "base_url": "http://192.168.1.5:8080/v1",
                "models": ["model-a"],
            }
        }
    }
    store = _store_with_session(provider="custom-ep:gpu", base_url=None)
    try:
        detach_and_delete_entry(app_config, store, "custom-ep:gpu")
        settings = store.session_settings(store.sessions()[0].id)
        assert settings.provider == "llama_cpp"
        # The blank session URL takes the entry's family-normalized URL
        # (llama normalization strips the /v1 suffix), so the conversation
        # keeps its current endpoint as conversation-only instead of
        # falling back to the family default.
        assert settings.base_url == "http://192.168.1.5:8080"
        assert load_custom_endpoints(load_settings()) == {}
    finally:
        _reload_config()


def test_convert_custom_slot_creates_entry_and_keeps_slot(tmp_path, monkeypatch):
    _activate_temp_config(tmp_path, monkeypatch, _SLOT_TOML)
    app_config = {"api_settings": {"custom": {
        "api_url": "http://127.0.0.1:5000/v1", "model": "my-model"}}}
    try:
        provider_id = convert_slot_to_named_endpoint(app_config, "custom")  # panel seam
        assert provider_id == "custom-ep:custom"
        entry = load_custom_endpoints(load_settings())["custom"]
        assert entry.family == "openai_compatible"
        assert entry.base_url == "http://127.0.0.1:5000/v1"
        assert load_settings()["api_settings"]["custom"]["model"] == "my-model"
    finally:
        _reload_config()


def _entry_with_env() -> CustomEndpointEntry:
    """Loaded-shape entry carrying a credential reference."""
    return CustomEndpointEntry(
        slug="gpu",
        display_name="GPU llama",
        family="llama_cpp",
        base_url="http://192.168.1.5:8080",
        api_key_env="GPU_KEY",
        models=("model-a",),
    )


def test_edit_mutation_rejects_malformed_env_var_reference():
    entry = _entry_with_env()
    with pytest.raises(ValueError, match="letters, digits, and underscores"):
        build_entry_edit_mutation(entry, entry.base_url, "GPU KEY!", "model-a")


def test_edit_mutation_keeps_valid_env_var_without_delete():
    entry = _entry_with_env()
    mutation, delete_keys = build_entry_edit_mutation(
        entry, entry.base_url, "GPU_KEY_2", "model-a, model-b ,model-a"
    )
    assert delete_keys is None
    values = mutation["custom_endpoints.gpu"]
    assert values["api_key_env"] == "GPU_KEY_2"
    # Comma-separated models parse with blanks and duplicates dropped.
    assert values["models"] == ["model-a", "model-b"]


def test_edit_mutation_clearing_env_var_deletes_reference_on_disk(
    tmp_path, monkeypatch
):
    """A cleared credential reference is removed from disk, not just omitted
    (config saves merge, so omission alone would keep the old key active)."""
    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENV_TOML)
    config_path = tmp_path / "settings-custom-endpoints.toml"
    try:
        entry = load_custom_endpoints(load_settings())["gpu"]
        assert entry.api_key_env == "GPU_KEY"
        mutation, delete_keys = build_entry_edit_mutation(
            entry, entry.base_url, "", "model-a"
        )
        assert delete_keys == {"custom_endpoints.gpu": ("api_key_env",)}
        assert save_settings_to_cli_config(mutation, delete_keys=delete_keys)
        reloaded = load_custom_endpoints(load_settings())
        assert reloaded["gpu"].api_key_env is None
        assert "api_key_env" not in config_path.read_text()
    finally:
        _reload_config()


@pytest.mark.asyncio
async def test_settings_screen_duplicate_button_opens_template_modal(
    tmp_path, monkeypatch
):
    """H5: each Custom endpoints row has a Duplicate button that opens the
    Console template-creation modal seeded from that entry (same-family
    starter per H5/P2-6), and a completed create confirms in the panel's
    shared status line."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
    )
    from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
        ConsoleEndpointTemplateModal,
    )

    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENTRY_TOML)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(120, 35)) as pilot:
            screen = _active_destination_screen(host)
            for _ in range(8):
                await pilot.pause(0.05)
            await pilot.app.workers.wait_for_complete()
            screen.query_one("#settings-category-providers-models", Button).press()
            for _ in range(8):
                await pilot.pause(0.05)
            screen.query_one("#settings-cep-duplicate-gpu", Button).press()
            for _ in range(4):
                await pilot.pause(0.05)

            modal = pilot.app.screen
            assert isinstance(modal, ConsoleEndpointTemplateModal)
            from textual.widgets import Input, Select

            # Seeded from the entry: llama family starter, blank name/URL.
            assert (
                modal.query_one("#endpoint-template-family", Select).value
                == "llama_cpp"
            )
            assert modal.query_one("#endpoint-template-url", Input).value == ""
            assert modal.query_one("#endpoint-template-name", Input).value == ""

            # Create a sibling endpoint from the starter prefill.
            modal.query_one("#endpoint-template-name", Input).value = "GPU two"
            modal.query_one("#endpoint-template-url", Input).value = (
                "http://192.168.1.7:8080"
            )
            await pilot.pause(0.1)
            modal.query_one("#endpoint-template-create", Button).press()
            for _ in range(15):
                await pilot.pause(0.1)

            status = screen.query_one("#settings-custom-endpoints-status", Static)
            status_text = str(
                getattr(status.renderable, "plain", status.renderable)
            )
            assert "Created endpoint 'GPU two'" in status_text
            assert "gpu-two" in load_custom_endpoints(load_settings())
    finally:
        _reload_config()


@pytest.mark.asyncio
async def test_settings_screen_duplicate_seeds_from_disk_when_app_snapshot_stale(
    tmp_path, monkeypatch
):
    """Qodo PR-2646: the Duplicate handler must seed from the same source the
    visible rows were built from (``_custom_endpoints_view_config``). When the
    shared in-memory app snapshot has not seen the on-disk registry yet, the
    modal still opens on the entry's same-family starter instead of a blank
    template."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
    )
    from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
        ConsoleEndpointTemplateModal,
    )

    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENTRY_TOML)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(120, 35)) as pilot:
            screen = _active_destination_screen(host)
            for _ in range(8):
                await pilot.pause(0.05)
            await pilot.app.workers.wait_for_complete()
            screen.query_one("#settings-category-providers-models", Button).press()
            for _ in range(8):
                await pilot.pause(0.05)

            # Stale shared snapshot: the rows above were rendered from the
            # freshest registry, but the shared app snapshot object lost
            # the section in both shapes it can live in. The view-config
            # seam stays pinned to the fresh registry (the real staleness
            # shape: the shared mapping lags a registry the view config
            # still resolves).
            from tldw_chatbook.Chat.custom_endpoint_registry import (
                load_custom_endpoints,
            )

            app_config = getattr(app, "app_config", None)
            assert isinstance(app_config, dict)
            app_config.pop("custom_endpoints", None)
            raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
            if isinstance(raw, dict):
                raw.pop("custom_endpoints", None)
            assert "gpu" not in load_custom_endpoints(app_config)
            fresh_view = _registry_config()
            assert "gpu" in load_custom_endpoints(fresh_view)
            screen._custom_endpoints_view_config = lambda: fresh_view

            screen.query_one("#settings-cep-duplicate-gpu", Button).press()
            for _ in range(4):
                await pilot.pause(0.05)

            modal = pilot.app.screen
            assert isinstance(modal, ConsoleEndpointTemplateModal)
            from textual.widgets import Select

            # Seeded from the entry: same-family starter active (blank
            # fallback would carry no starter at all), llama family, blank
            # name/URL.
            assert modal._same_family_starter is not None
            assert (
                modal.query_one("#endpoint-template-family", Select).value
                == "llama_cpp"
            )
            from textual.widgets import Input

            assert modal.query_one("#endpoint-template-name", Input).value == ""
            assert modal.query_one("#endpoint-template-url", Input).value == ""
    finally:
        _reload_config()


@pytest.mark.asyncio
async def test_settings_screen_duplicate_reports_missing_slug_instead_of_blank(
    tmp_path, monkeypatch
):
    """Qodo PR-2646: a slug that no longer resolves in the freshest registry
    reports through the shared status line; no blank template modal opens."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
    )
    from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
        ConsoleEndpointTemplateModal,
    )

    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENTRY_TOML)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(120, 35)) as pilot:
            screen = _active_destination_screen(host)
            for _ in range(8):
                await pilot.pause(0.05)
            await pilot.app.workers.wait_for_complete()
            screen.query_one("#settings-category-providers-models", Button).press()
            for _ in range(8):
                await pilot.pause(0.05)

            screen._custom_endpoint_duplicate_requested("ghost")
            for _ in range(4):
                await pilot.pause(0.05)

            assert not isinstance(pilot.app.screen, ConsoleEndpointTemplateModal)
            status = screen.query_one("#settings-custom-endpoints-status", Static)
            status_text = str(
                getattr(status.renderable, "plain", status.renderable)
            )
            assert "no longer available" in status_text
    finally:
        _reload_config()


@pytest.mark.asyncio
async def test_settings_screen_duplicate_falls_back_to_view_config_when_snapshot_read_only(
    tmp_path, monkeypatch
):
    """Qodo PR-2646: a read-only (non-mutable) app snapshot cannot be poked;
    the handler then hands the modal the fresh on-disk registry itself."""
    from types import MappingProxyType

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
    )
    from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
        ConsoleEndpointTemplateModal,
    )

    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENTRY_TOML)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(120, 35)) as pilot:
            screen = _active_destination_screen(host)
            for _ in range(8):
                await pilot.pause(0.05)
            await pilot.app.workers.wait_for_complete()
            screen.query_one("#settings-category-providers-models", Button).press()
            for _ in range(8):
                await pilot.pause(0.05)

            screen._app_config_mapping = lambda: MappingProxyType({"stale": True})
            screen._custom_endpoint_duplicate_requested("gpu")
            for _ in range(4):
                await pilot.pause(0.05)

            modal = pilot.app.screen
            assert isinstance(modal, ConsoleEndpointTemplateModal)
            from textual.widgets import Select

            assert modal._same_family_starter is not None
            assert (
                modal.query_one("#endpoint-template-family", Select).value
                == "llama_cpp"
            )
    finally:
        _reload_config()


@pytest.mark.asyncio
async def test_settings_screen_edit_flow_clears_env_var_reference(
    tmp_path, monkeypatch
):
    """Integration: the F9 panel's Edit form saves through its real button
    dispatch and threaded worker, and a cleared Env var disappears from disk."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
    )

    _activate_temp_config(tmp_path, monkeypatch, _REGISTRY_ENV_TOML)
    config_path = tmp_path / "settings-custom-endpoints.toml"
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(120, 35)) as pilot:
            screen = _active_destination_screen(host)
            for _ in range(8):
                await pilot.pause(0.05)
            await pilot.app.workers.wait_for_complete()
            screen.query_one("#settings-category-providers-models", Button).press()
            for _ in range(8):
                await pilot.pause(0.05)
            screen.query_one("#settings-cep-edit-gpu", Button).press()
            for _ in range(4):
                await pilot.pause(0.05)
            env_input = screen.query_one("#settings-cep-edit-key-env", Input)
            assert env_input.value == "GPU_KEY"
            env_input.value = ""
            screen.query_one("#settings-cep-edit-save", Button).press()
            await pilot.app.workers.wait_for_complete()
            for _ in range(8):
                await pilot.pause(0.05)
            status = screen.query_one("#settings-custom-endpoints-status", Static)
            status_text = str(getattr(status.renderable, "plain", status.renderable))
            assert "Saved endpoint 'gpu'" in status_text
            # The form closed after the successful report.
            from textual.css.query import NoMatches

            with pytest.raises(NoMatches):
                screen.query_one("#settings-cep-edit-key-env", Input)
        assert "api_key_env" not in config_path.read_text()
        assert load_custom_endpoints(load_settings())["gpu"].api_key_env is None
    finally:
        _reload_config()
