"""F9 Settings custom endpoint management seams (ADR-146, task-7).

The four brief-verbatim tests exercise the panel-action seams in
``settings_provider_view_model``: overview rows render safe endpoint
displays, the delete guard lists referencing sessions, detach-then-delete
re-points sessions and removes the entry, and slot conversion creates a
named entry while leaving the slot untouched. Config round-trips run
against a ``TLDW_CONFIG_PATH`` temp config (the Task 6 pattern in
Tests/Widgets/test_console_endpoint_template_modal.py).
"""


from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints
from tldw_chatbook.config import load_settings
from tldw_chatbook.UI.Screens.settings_provider_view_model import (
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


def _store_with_session(provider: str) -> ConsoleChatStore:
    """Minimal store with one session pinned to ``provider``.

    A real ``ConsoleChatStore`` (not a stub): detach must flow through the
    store's own settings mutation path.
    """
    store = ConsoleChatStore()
    store.create_session(
        title="GPU chat",
        settings=ConsoleSessionSettings(
            provider=provider,
            base_url="http://192.168.1.5:8080",
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
