import inspect
import re
import time
import builtins
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Button, Input, Select, SelectionList, Static, TextArea

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.Utils import input_validation as input_validation_module
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_effective_provider_model,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import (
    SettingsConfigAdapter,
    redact_secret_text,
)
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
    SettingsValidationResult,
)
from tldw_chatbook.ACP_Interop.runtime_session import ACPRuntimeSessionState
from tldw_chatbook.Chat.console_chat_models import ConsoleWorkspaceContext
from tldw_chatbook.Home.dashboard_state import HomeDashboardInput, summarize_home_dashboard
from tldw_chatbook.Sync_Interop.sync_promotion_state import SyncPromotionState
from tldw_chatbook.Sync_Interop.manual_sync_control import (
    ManualSyncPreview,
    ManualSyncRunResult,
)
from tldw_chatbook.Sync_Interop.conflict_review import SyncV2ConflictReviewItem
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Workspaces.display_state import LIBRARY_WORKSPACE_VISIBILITY_COPY
from tldw_chatbook.Workspaces.models import (
    WorkspaceAuthority,
    WorkspaceRecord,
    WorkspaceSyncStatus,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryError,
    ModelDiscoveryResult,
    PersistenceResult,
)

DUMMY_REDACTION_ENV_VALUE = "redaction-fixture-env-value"
DUMMY_REDACTION_CONFIG_VALUE = "redaction-fixture-config-value"
DUMMY_REDACTION_SERVER_VALUE = "redaction-fixture-server-value"


class StyledSettingsDestinationHarness(DestinationHarness):
    CSS_PATH = str(Path(__file__).parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss")


class FakeSettingsModelDiscoveryScope:
    def __init__(
        self,
        *,
        result: ModelDiscoveryResult,
        persistence_result: PersistenceResult | None = None,
    ) -> None:
        self.result = result
        self.persistence_result = persistence_result or PersistenceResult(
            provider=result.provider,
            provider_list_key=result.provider_list_key,
            status="saved",
            saved_model_ids=(),
            message="No new discovered models to save.",
        )
        self.discover_calls = []
        self.persist_calls = []
        self.clear_calls = []

    async def discover_models(self, **kwargs):
        self.discover_calls.append(kwargs)
        return self.result

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        return self.persistence_result

    async def clear_discovered_models(self, **kwargs):
        self.clear_calls.append(kwargs)


def _discovered_model(model_id: str, *, provider: str = "openai") -> DiscoveredModel:
    return DiscoveredModel(
        provider=provider,
        provider_list_key=provider,
        model_id=model_id,
        display_name=model_id,
        source="runtime_discovered",
        endpoint_fingerprint="fp-test",
        discovered_at="2026-06-04T00:00:00Z",
        metadata_raw_safe={"owned_by": "test-provider"},
    )


def _app(
    *,
    provider=None,
    api_model=None,
    model=None,
    defaults=None,
):
    return SimpleNamespace(
        app_config={"chat_defaults": defaults or {}},
        chat_api_provider_value=provider,
        chat_api_model_value=api_model,
        chat_model_value=model,
    )


async def _wait_for_settings_text(screen, pilot, expected_text: str, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected_text in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {expected_text!r}. Visible text: {_visible_text(screen)}")


async def _select_settings_category(
    screen,
    pilot,
    category: SettingsCategoryId | str,
    *,
    expected_text: str | None = None,
    selector: str | None = None,
    timeout: float = 4.0,
) -> None:
    category_value = category.value if isinstance(category, SettingsCategoryId) else str(category)
    button_selector = f"#settings-category-{category_value}"
    await _wait_for_selector(screen, pilot, button_selector, timeout=timeout)
    await pilot.click(button_selector)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        active = getattr(screen, "active_category", None) == category_value
        has_text = expected_text is None or expected_text in _visible_text(screen)
        has_selector = selector is None or bool(screen.query(selector))
        if active and has_text and has_selector:
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for Settings category {category_value!r}. "
        f"Visible text: {_visible_text(screen)}"
    )


async def _click_scrolled_settings_button(screen, pilot, selector: str) -> Button:
    button = screen.query_one(selector, Button)
    detail_pane = screen.query_one("#settings-detail-pane", VerticalScroll)
    detail_pane.scroll_to_widget(
        button,
        animate=False,
        immediate=True,
        top=True,
        force=True,
    )
    await pilot.pause()
    await pilot.click(selector)
    return button


async def _wait_for_settings_search_focus(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        search = screen.query_one("#settings-category-search", Input)
        if search.has_focus:
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for Settings category search focus")


def test_effective_provider_model_prefers_console_overrides():
    app = _app(
        provider="OpenAI",
        api_model="gpt-4.1",
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        console_provider="Anthropic",
        console_model="claude",
    )

    assert result.provider == "Anthropic"
    assert result.model == "claude"
    assert result.provider_source == "console_control"
    assert result.model_source == "console_control"


def test_effective_provider_model_preserves_configured_provider_when_reactive_is_default_openai():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(app)

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"
    assert result.model == "qwen"


def test_effective_provider_model_prefers_settings_draft_values():
    app = _app(
        provider="OpenAI",
        api_model="gpt-4.1",
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        settings_provider="Ollama",
        settings_model="llama3.1",
    )

    assert result.provider == "Ollama"
    assert result.model == "llama3.1"
    assert result.provider_source == "settings_draft"
    assert result.model_source == "settings_draft"


def test_effective_provider_model_ignores_blank_provider_overrides_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        settings_provider=" ",
        console_provider="None",
    )

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"


def test_effective_provider_model_ignores_blank_reactive_provider_for_default_fallback():
    for reactive_provider in ("", " ", "None"):
        app = _app(
            provider=reactive_provider,
            api_model=None,
            model=None,
            defaults={"provider": "llama_cpp", "model": "qwen"},
        )

        result = resolve_effective_provider_model(app)

        assert result.provider == "llama_cpp"
        assert result.provider_source == "chat_defaults"


def test_effective_provider_model_ignores_textual_blank_select_provider_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(app, settings_provider=Select.BLANK)

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"


def test_effective_provider_model_ignores_blank_model_overrides_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    for blank_model in ("", " ", "None", Select.BLANK):
        result = resolve_effective_provider_model(
            app,
            settings_model=blank_model,
            console_model=" ",
        )

        assert result.model == "qwen"
        assert result.model_source == "chat_defaults"


def test_effective_provider_model_handles_non_mapping_app_config():
    app = SimpleNamespace(
        app_config=[],
        chat_api_provider_value=None,
        chat_api_model_value=None,
        chat_model_value=None,
    )

    result = resolve_effective_provider_model(app)

    assert result.provider is None
    assert result.model is None


def test_settings_draft_tracks_dirty_values():
    draft = SettingsDraft(category=SettingsCategoryId.CONSOLE_BEHAVIOR)
    draft.set_value("collapse_large_pastes", True, False)

    assert draft.is_dirty
    assert draft.dirty_keys == {"collapse_large_pastes"}


def test_redact_secret_text_removes_api_key_like_values():
    text = "failed with OPENAI_API_KEY=sk-secret-token and token abc"

    redacted = redact_secret_text(text)

    assert "sk-secret-token" not in redacted
    assert "OPENAI_API_KEY=<redacted>" in redacted


def test_adapter_rejects_non_mapping_toml():
    adapter = SettingsConfigAdapter()

    result = adapter.validate_raw_toml('"not a mapping"')

    assert not result.valid
    assert "top-level TOML value must be a table" in result.message


def test_adapter_rejects_scalar_like_toml_with_table_message():
    adapter = SettingsConfigAdapter()

    for value in (
        "42",
        "true",
        "[1, 2]",
        "nan",
        "inf",
        "0xDEADBEEF",
        "1979-05-27",
        "1979-05-27T07:32:00Z",
    ):
        result = adapter.validate_raw_toml(value)

        assert not result.valid
        assert "top-level TOML value must be a table" in result.message


def test_adapter_accepts_table_headers_before_scalar_fallback():
    adapter = SettingsConfigAdapter()

    for value in ("[section]", "[[items]]"):
        result = adapter.validate_raw_toml(value)

        assert result.valid


def test_adapter_validate_config_file_rejects_corrupt_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")

    result = SettingsConfigAdapter().validate_config_file(config_path)

    assert not result.valid
    assert "Expected" in result.message or "Invalid" in result.message


def test_adapter_save_values_attempts_all_keys_when_one_save_fails(monkeypatch):
    calls = []

    def fake_save(section, key, value):
        calls.append((section, key, value))
        return key != "provider"

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        fake_save,
    )

    result = SettingsConfigAdapter().save_values(
        "chat_defaults",
        {"provider": "bad", "model": "still-attempted"},
    )

    assert not result
    assert calls == [
        ("chat_defaults", "provider", "bad"),
        ("chat_defaults", "model", "still-attempted"),
    ]


def test_adapter_save_sections_batches_sections(monkeypatch):
    calls = []

    def fake_save(section_values):
        calls.append(section_values)
        return True

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_settings_to_cli_config",
        fake_save,
    )

    assert SettingsConfigAdapter().save_sections(
        {
            "console": {"collapse_large_pastes": False},
            "chat_defaults": {"streaming": False},
        }
    )

    assert calls == [
        {
            "console": {"collapse_large_pastes": False},
            "chat_defaults": {"streaming": False},
        }
    ]


def test_settings_console_default_max_tokens_rejects_raw_zero():
    screen = SettingsScreen(_build_test_app())

    with pytest.raises(ValueError, match="Max tokens"):
        screen._normalise_console_default_max_tokens(0)


def test_settings_optional_int_defaults_load_invalid_values_as_blank():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"seed": "not-an-int", "top_k": "not-an-int"}
    screen = SettingsScreen(app)

    assert screen._loaded_console_default_seed() == ""
    assert screen._loaded_console_default_top_k() == ""


@pytest.mark.asyncio
async def test_settings_defaults_to_overview_category():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-overview")
        text = _visible_text(screen)

        assert "Overview" in text
        assert "Provider readiness" in text
        assert "Storage" in text
        assert "Privacy" in text
        assert "Console paste collapse" in text


def test_settings_ownership_records_cover_categories_and_runtime_boundaries():
    app = _build_test_app()
    screen = SettingsScreen(app)

    records = screen._category_ownership_records()
    records_by_category = {record.category: record for record in records}

    assert set(records_by_category) == {
        summary.category for summary in screen._category_summaries()
    }
    assert all(record.__class__.__name__ == "SettingsOwnershipRecord" for record in records)
    assert records_by_category[SettingsCategoryId.PROVIDERS_MODELS].owns_config_sections == (
        "chat_defaults.provider",
        "chat_defaults.model",
        "api_settings.<provider>.endpoint",
        "api_settings.<provider>.api_key",
        "api_settings.<provider>.api_key_env_var",
        "api_settings.<provider>.model_defaults.<model>",
    )
    assert records_by_category[SettingsCategoryId.CONSOLE_BEHAVIOR].owns_config_sections == (
        "console.collapse_large_pastes",
        "console.paste_collapse_threshold",
        "console.background_effects.*",
        "chat_defaults.streaming",
        "chat_defaults.temperature",
        "chat_defaults.top_p",
        "chat_defaults.max_tokens",
    )
    storage_record = records_by_category[SettingsCategoryId.STORAGE]
    assert storage_record.writes_allowed is True
    assert "database.media_db_path" in storage_record.owns_config_sections
    assert "restart" in storage_record.recovery_copy.lower()
    assert not storage_record.read_only_reason

    overview = records_by_category[SettingsCategoryId.OVERVIEW]
    boundary_text = " ".join(
        (
            overview.boundary_copy,
            overview.recovery_copy,
            *overview.reads_runtime_state_from,
        )
    )
    for owner in ("Console", "MCP", "ACP", "sync", "workspace"):
        assert owner in boundary_text


def test_settings_overview_ownership_rows_are_sourced_from_record():
    app = _build_test_app()
    screen = SettingsScreen(app)

    ownership = screen._ownership_record(SettingsCategoryId.OVERVIEW)
    rows = dict(screen._overview_ownership_rows())
    rendered_copy = " ".join(rows.values())

    for boundary in ownership.boundary_copy.split("; "):
        assert boundary in rendered_copy
    assert rows["Recovery"] == ownership.recovery_copy


def test_settings_server_sync_workspace_source_contracts_are_explicit():
    contracts = dict(settings_screen_module.SETTINGS_SERVER_SYNC_WORKSPACE_SOURCE_CONTRACTS)

    assert "runtime_policy.types.RuntimeSourceState" in contracts["Server profile"]
    assert "runtime_policy.server_context.RuntimeServerContextProvider" in contracts["Server profile"]
    assert "Sync_Interop.sync_scope_service.SyncScopeService" in contracts["Sync safety"]
    assert "Chat.console_chat_store.ConsoleChatStore.workspace_context" in contracts["Workspace context"]
    assert "Workspaces.display_state.LIBRARY_WORKSPACE_VISIBILITY_COPY" in contracts["Workspace context"]
    assert "Workspaces.models.WorkspaceTransferPolicy" in contracts["Handoff policy"]
    assert "ACP_Interop.runtime_session.ACPRuntimeSessionState" in contracts["ACP handoff readiness"]


def test_settings_domain_category_contracts_are_explicit_about_mutation_scope():
    app = _build_test_app()
    screen = SettingsScreen(app)
    contracts = {contract.category: contract for contract in screen._domain_category_contracts()}
    expected_categories = {
        SettingsCategoryId.LIBRARY_RAG,
        SettingsCategoryId.ARTIFACTS,
        SettingsCategoryId.PERSONAS,
        SettingsCategoryId.SKILLS,
        SettingsCategoryId.SCHEDULES,
        SettingsCategoryId.WATCHLISTS,
        SettingsCategoryId.WORKFLOWS,
        SettingsCategoryId.MCP_DEFAULTS,
        SettingsCategoryId.ACP_DEFAULTS,
    }

    assert set(contracts) == expected_categories
    for category in expected_categories:
        contract = contracts[category]
        assert contract.owner_destination
        assert contract.source_of_truth
        assert contract.follow_up
        if category is SettingsCategoryId.LIBRARY_RAG:
            assert contract.settings_can_mutate is True
        else:
            assert contract.settings_can_mutate is False

    library_contract = contracts[SettingsCategoryId.LIBRARY_RAG]
    library_copy = " ".join(
        (
            *(value for _label, value in library_contract.rows),
            library_contract.follow_up,
        )
    )
    assert "citations" in library_copy
    assert "snippets" in library_copy
    assert "AppRAGSearchConfig.rag.search" in library_copy


def test_settings_domain_category_ids_are_derived_from_contract_mapping():
    assert settings_screen_module.DOMAIN_SETTINGS_CATEGORY_IDS == frozenset(
        settings_screen_module.DOMAIN_CONTRACT_BY_CATEGORY
    )


def test_settings_domain_contract_mapping_rejects_duplicate_categories():
    contract = settings_screen_module.SETTINGS_DOMAIN_CATEGORY_CONTRACTS[0]

    with pytest.raises(ValueError, match="Duplicate Settings domain category"):
        settings_screen_module._build_domain_contract_by_category((contract, contract))


def test_settings_domain_categories_are_grouped_and_have_ownership_records():
    app = _build_test_app()
    screen = SettingsScreen(app)
    domain_categories = {
        SettingsCategoryId.LIBRARY_RAG,
        SettingsCategoryId.ARTIFACTS,
        SettingsCategoryId.PERSONAS,
        SettingsCategoryId.SKILLS,
        SettingsCategoryId.SCHEDULES,
        SettingsCategoryId.WATCHLISTS,
        SettingsCategoryId.WORKFLOWS,
        SettingsCategoryId.MCP_DEFAULTS,
        SettingsCategoryId.ACP_DEFAULTS,
    }

    grouped_categories = {
        category
        for _group_name, category_ids in screen._category_groups()
        for category in category_ids
    }
    records = {record.category: record for record in screen._category_ownership_records()}

    assert domain_categories <= grouped_categories
    assert domain_categories <= set(records)
    for category in domain_categories:
        record = records[category]
        if category is SettingsCategoryId.LIBRARY_RAG:
            assert record.writes_allowed
            assert "AppRAGSearchConfig.rag.search" in " ".join(record.owns_config_sections)
        else:
            assert not record.writes_allowed
            assert record.read_only_reason
        assert record.recovery_copy
        assert record.runtime_owner


@pytest.mark.asyncio
async def test_settings_library_rag_renders_guided_defaults_and_validates(monkeypatch):
    app = _build_test_app()
    app.app_config["AppRAGSearchConfig"] = {
        "rag": {
            "search": {
                "default_search_mode": "semantic",
                "default_top_k": 10,
                "score_threshold": 0.0,
                "include_citations": True,
                "citation_style": "inline",
                "snippet_max_chars": 240,
                "max_context_size": 16000,
            },
            "retriever": {
                "fts_top_k": 10,
                "vector_top_k": 10,
                "hybrid_alpha": 0.5,
            },
        }
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await pilot.click("#settings-category-library-rag")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Library & RAG" in text
        assert "Search defaults" in text
        assert "Citation and snippets" in text
        assert "Preview defaults" in text
        assert "Semantic search | 10 results | Inline citations" in text
        assert "Context budget: 16000 chars" in text
        assert "Config keys" in text
        assert "10 editable defaults under AppRAGSearchConfig" in text
        assert "AppRAGSearchConfig.rag.search.default_search_mode" not in text
        assert screen.query_one("#settings-library-rag-search-mode", Select).value == "semantic"
        assert screen.query_one("#settings-library-rag-default-top-k", Input).value == "10"
        assert screen.query_one("#settings-library-rag-snippet-max-chars", Input).value == "240"
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        top_k.value = "12"
        screen.handle_library_rag_default_top_k_changed(Input.Changed(top_k, top_k.value))
        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert "Unsaved" in _visible_text(screen)

        snippet = screen.query_one("#settings-library-rag-snippet-max-chars", Input)
        snippet.value = "20"
        screen.handle_library_rag_snippet_max_chars_changed(Input.Changed(snippet, snippet.value))
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "Snippet characters" in _visible_text(screen)
        assert snippet.has_class("settings-invalid-input")

        snippet.value = "360"
        screen.handle_library_rag_snippet_max_chars_changed(Input.Changed(snippet, snippet.value))
        assert not snippet.has_class("settings-invalid-input")
        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Library/RAG defaults saved.")

    assert saved
    rag = saved[-1]["AppRAGSearchConfig"]["rag"]
    assert rag["search"]["default_top_k"] == 12
    assert rag["search"]["snippet_max_chars"] == 360


@pytest.mark.asyncio
async def test_settings_library_rag_sync_clamps_invalid_select_values():
    app = _build_test_app()
    app.app_config["AppRAGSearchConfig"] = {
        "rag": {
            "search": {
                "default_search_mode": "not-a-mode",
                "citation_style": "not-a-style",
            },
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await pilot.click("#settings-category-library-rag")
        screen = _active_destination_screen(host)

        screen._sync_library_rag_widgets()

        assert screen.query_one("#settings-library-rag-search-mode", Select).value == "semantic"
        assert screen.query_one("#settings-library-rag-citation-style", Select).value == "inline"


def test_settings_library_rag_save_uses_exclusive_thread_worker():
    worker = SettingsScreen.__dict__["_settings_save_library_rag_worker"]
    source = inspect.getsource(SettingsScreen)

    assert getattr(worker, "__wrapped__", None) is not None
    assert (
        "@work(exclusive=True, thread=True)\n"
        "    def _settings_save_library_rag_worker"
    ) in source


@pytest.mark.asyncio
async def test_settings_appearance_renders_guided_defaults_and_validates(monkeypatch):
    app = _build_test_app()
    app.app_config["general"] = {
        "default_theme": "textual-dark",
        "palette_theme_limit": 1,
    }
    app.app_config["web_server"] = {"font_size": 12}
    app.app_config["appearance"] = {
        "density": "normal",
        "animations_enabled": True,
        "smooth_scrolling": True,
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await pilot.click("#settings-category-appearance")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Appearance" in text
        assert "Global visual defaults" in text
        assert "Customize owns full theme editing" in text
        assert "Save targets: general, web_server, and appearance" in text
        assert "Open Customize" in text
        assert screen.query_one("#settings-appearance-theme", Select).value == "textual-dark"
        assert screen.query_one("#settings-appearance-palette-theme-limit", Input).value == "1"
        assert screen.query_one("#settings-appearance-font-size", Input).value == "12"
        assert screen.query_one("#settings-appearance-density", Select).value == "normal"
        assert str(screen.query_one("#settings-appearance-animations-enabled", Button).label) == "Enabled"
        assert str(screen.query_one("#settings-appearance-smooth-scrolling", Button).label) == "Enabled"
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        palette_limit = screen.query_one("#settings-appearance-palette-theme-limit", Input)
        palette_limit.value = "5"
        screen.handle_appearance_palette_theme_limit_changed(
            Input.Changed(palette_limit, palette_limit.value)
        )

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert "Unsaved" in _visible_text(screen)

        font_size = screen.query_one("#settings-appearance-font-size", Input)
        font_size.value = "99"
        screen.handle_appearance_font_size_changed(Input.Changed(font_size, font_size.value))

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "Font size" in _visible_text(screen)
        assert font_size.has_class("settings-invalid-input")

        font_size.value = "14"
        screen.handle_appearance_font_size_changed(Input.Changed(font_size, font_size.value))
        assert not font_size.has_class("settings-invalid-input")

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Appearance defaults saved.")

    assert saved
    assert saved[-1]["general"]["default_theme"] == "textual-dark"
    assert saved[-1]["general"]["palette_theme_limit"] == 5
    assert saved[-1]["web_server"]["font_size"] == 14
    assert saved[-1]["appearance"]["density"] == "normal"


@pytest.mark.asyncio
async def test_settings_appearance_revert_restores_loaded_values():
    app = _build_test_app()
    app.app_config["general"] = {"palette_theme_limit": 1}
    app.app_config["web_server"] = {"font_size": 12}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-appearance")
        screen = _active_destination_screen(host)
        font_size = screen.query_one("#settings-appearance-font-size", Input)
        font_size.value = "16"
        screen.handle_appearance_font_size_changed(Input.Changed(font_size, font_size.value))

        assert "Unsaved" in _visible_text(screen)

        screen.action_settings_revert_category()
        text = _visible_text(screen)

        assert font_size.value == "12"
        assert "Appearance defaults reverted to last loaded values." in text
        assert "No unsaved changes" in text


@pytest.mark.asyncio
async def test_settings_appearance_preview_updates_runtime_without_saving(monkeypatch):
    app = _build_test_app()
    app.app_config["general"] = {"default_theme": "textual-dark"}
    app.theme = "textual-dark"
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-appearance")
        screen = _active_destination_screen(host)
        theme = screen.query_one("#settings-appearance-theme", Select)
        theme.value = "textual-light"
        screen.handle_appearance_theme_changed(Select.Changed(theme, theme.value))

        await pilot.click("#settings-preview-appearance")
        text = _visible_text(screen)

        assert app.theme == "textual-light"
        assert saved == []
        assert "Appearance preview applied for this session only." in text
        assert "Unsaved" in text


@pytest.mark.asyncio
async def test_settings_appearance_focused_input_keeps_typed_text_visible():
    app = _build_test_app()
    app.app_config["general"] = {"palette_theme_limit": 1}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-appearance")
        screen = _active_destination_screen(host)
        palette_limit = screen.query_one("#settings-appearance-palette-theme-limit", Input)
        palette_limit.focus()

        await pilot.press("2")

        assert palette_limit.value.endswith("2")
        assert palette_limit.region.width > 10
        assert screen.app.focused is palette_limit


def test_settings_appearance_save_uses_exclusive_thread_worker():
    worker = SettingsScreen.__dict__["_settings_save_appearance_worker"]
    source = inspect.getsource(SettingsScreen)

    assert getattr(worker, "__wrapped__", None) is not None
    assert (
        "@work(exclusive=True, thread=True)\n"
        "    def _settings_save_appearance_worker"
    ) in source


def test_settings_appearance_theme_options_use_specific_import_fallback():
    source = inspect.getsource(SettingsScreen._appearance_theme_options)

    assert "from tldw_chatbook.css.Themes.themes import ALL_THEMES" in source
    assert "except (ImportError, ModuleNotFoundError):" in source
    assert "except Exception:" not in source


def test_settings_storage_defaults_load_validate_and_build_save_payload(tmp_path):
    try:
        from tldw_chatbook.UI.Screens.settings_storage_defaults import (
            SettingsStorageDefaults,
            build_storage_check_rows,
            build_storage_save_sections,
            load_storage_defaults,
            validate_storage_defaults,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Storage defaults helper is missing: {exc}")

    base_dir = tmp_path / "tldw"
    base_dir.mkdir()
    db_dir = base_dir / "db"
    db_dir.mkdir()
    app_config = {
        "database": {
            "USER_DB_BASE_DIR": str(base_dir),
            "chachanotes_db_path": str(db_dir / "chatbook.db"),
            "prompts_db_path": str(db_dir / "prompts.db"),
            "media_db_path": str(db_dir / "media.db"),
            "research_db_path": str(db_dir / "research.db"),
            "writing_db_path": str(db_dir / "writing.db"),
            "library_collections_db_path": str(db_dir / "collections.db"),
            "workspaces_db_path": str(db_dir / "workspaces.db"),
            "check_integrity_on_startup": True,
        }
    }

    defaults = load_storage_defaults(app_config)

    assert defaults.user_db_base_dir == str(base_dir)
    assert defaults.chachanotes_db_path.endswith("chatbook.db")
    assert validate_storage_defaults(defaults).valid is True

    invalid = SettingsStorageDefaults(
        user_db_base_dir="",
        chachanotes_db_path="../outside.db",
        prompts_db_path=str(db_dir / "prompts.db"),
        media_db_path=str(db_dir / "media.db"),
        research_db_path=str(db_dir / "research.db"),
        writing_db_path=str(db_dir / "writing.db"),
        library_collections_db_path=str(db_dir / "collections.db"),
        workspaces_db_path=str(db_dir / "workspaces.db"),
    )
    validation = validate_storage_defaults(invalid)

    assert validation.valid is False
    assert "Base data directory" in validation.message

    edited = SettingsStorageDefaults(
        **{
            **defaults.__dict__,
            "media_db_path": "~/custom/tldw-media.db",
        }
    )
    section_values = build_storage_save_sections(app_config, edited)

    assert section_values["database"]["media_db_path"] == "~/custom/tldw-media.db"
    assert section_values["database"]["check_integrity_on_startup"] is True

    null_byte = SettingsStorageDefaults(
        **{
            **defaults.__dict__,
            "media_db_path": f"{db_dir / 'media'}\x00.db",
        }
    )
    rows = build_storage_check_rows(null_byte)

    assert rows[0] == "Storage check: complete"
    assert "Media DB must be a single filesystem path." in rows
    assert "Storage check blocked: fix invalid paths first." in rows

    windows_traversal = SettingsStorageDefaults(
        **{
            **defaults.__dict__,
            "media_db_path": r"C:\tmp\..\outside.db",
        }
    )

    validation = validate_storage_defaults(windows_traversal)

    assert validation.valid is False
    assert "Media DB cannot contain parent-directory traversal." in validation.message

    existing_directory = db_dir / "media-directory.db"
    existing_directory.mkdir()
    directory_as_database = SettingsStorageDefaults(
        **{
            **defaults.__dict__,
            "media_db_path": str(existing_directory),
        }
    )

    validation = validate_storage_defaults(directory_as_database)

    assert validation.valid is False
    assert "Media DB must be a database file path, not a directory." in validation.message


@pytest.mark.asyncio
async def test_settings_storage_renders_guided_defaults_and_validates(tmp_path):
    app = _build_test_app()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    app.app_config["database"] = {
        "USER_DB_BASE_DIR": str(tmp_path),
        "chachanotes_db_path": str(db_dir / "chatbook.db"),
        "prompts_db_path": str(db_dir / "prompts.db"),
        "media_db_path": str(db_dir / "media.db"),
        "research_db_path": str(db_dir / "research.db"),
        "writing_db_path": str(db_dir / "writing.db"),
        "library_collections_db_path": str(db_dir / "collections.db"),
        "workspaces_db_path": str(db_dir / "workspaces.db"),
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(195, 55)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Storage defaults" in text
        assert "Changes apply on next launch" in text
        assert "no files are moved or reconnected" in text
        assert screen.query_one("#settings-storage-user-db-base-dir", Input).value == str(tmp_path)
        assert screen.query_one("#settings-storage-media-db-path", Input).value.endswith("media.db")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        media = screen.query_one("#settings-storage-media-db-path", Input)
        media.value = "../outside.db"
        screen.handle_storage_media_db_path_changed(Input.Changed(media, media.value))

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "Media DB cannot contain parent-directory traversal." in _visible_text(screen)
        assert media.has_class("settings-invalid-input")

        media.value = str(db_dir / "media-next.db")
        screen.handle_storage_media_db_path_changed(Input.Changed(media, media.value))

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert not media.has_class("settings-invalid-input")
        assert "Unsaved" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_storage_surfaces_check_action_before_long_path_editor(tmp_path):
    app = _build_test_app()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    app.app_config["database"] = {
        "USER_DB_BASE_DIR": str(tmp_path),
        "chachanotes_db_path": str(db_dir / "chatbook.db"),
        "prompts_db_path": str(db_dir / "prompts.db"),
        "media_db_path": str(db_dir / "media.db"),
        "research_db_path": str(db_dir / "research.db"),
        "writing_db_path": str(db_dir / "writing.db"),
        "library_collections_db_path": str(db_dir / "collections.db"),
        "workspaces_db_path": str(db_dir / "workspaces.db"),
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert text.index("Draft path check") < text.index("Database paths")


@pytest.mark.asyncio
async def test_settings_storage_save_and_revert_defaults(monkeypatch, tmp_path):
    app = _build_test_app()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    app.app_config["database"] = {
        "USER_DB_BASE_DIR": str(tmp_path),
        "chachanotes_db_path": str(db_dir / "chatbook.db"),
        "prompts_db_path": str(db_dir / "prompts.db"),
        "media_db_path": str(db_dir / "media.db"),
        "research_db_path": str(db_dir / "research.db"),
        "writing_db_path": str(db_dir / "writing.db"),
        "library_collections_db_path": str(db_dir / "collections.db"),
        "workspaces_db_path": str(db_dir / "workspaces.db"),
        "check_integrity_on_startup": True,
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(195, 55)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        workspaces = screen.query_one("#settings-storage-workspaces-db-path", Input)
        workspaces.value = str(db_dir / "workspaces-next.db")
        screen.handle_storage_workspaces_db_path_changed(
            Input.Changed(workspaces, workspaces.value)
        )

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Storage defaults saved.")

        assert saved
        assert saved[-1]["database"]["workspaces_db_path"].endswith("workspaces-next.db")
        assert app.app_config["database"]["workspaces_db_path"].endswith("workspaces-next.db")
        assert "Restart Chatbook" in _visible_text(screen)

        prompts = screen.query_one("#settings-storage-prompts-db-path", Input)
        prompts.value = str(db_dir / "prompts-next.db")
        screen.handle_storage_prompts_db_path_changed(Input.Changed(prompts, prompts.value))

        assert "Unsaved" in _visible_text(screen)

        screen.action_settings_revert_category()

        assert prompts.value.endswith("prompts.db")
        assert "Storage defaults reverted to last loaded values." in _visible_text(screen)


def test_settings_storage_save_uses_exclusive_thread_worker():
    worker = SettingsScreen.__dict__["_settings_save_storage_worker"]
    source = inspect.getsource(SettingsScreen)

    assert getattr(worker, "__wrapped__", None) is not None
    assert (
        "@work(exclusive=True, thread=True)\n"
        "    def _settings_save_storage_worker"
    ) in source


@pytest.mark.asyncio
async def test_settings_library_rag_save_preserves_mapping_like_app_config(monkeypatch):
    app = _build_test_app()
    app.app_config = UserDict(
        {
            "AppRAGSearchConfig": {
                "rag": {
                    "search": {"default_top_k": 10},
                    "retriever": {},
                }
            }
        }
    )
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _select_settings_category(
            screen,
            pilot,
            SettingsCategoryId.LIBRARY_RAG,
            selector="#settings-library-rag-default-top-k",
        )
        top_k = screen.query_one("#settings-library-rag-default-top-k", Input)
        top_k.value = "12"
        screen.handle_library_rag_default_top_k_changed(Input.Changed(top_k, top_k.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Library/RAG defaults saved.")

    assert saved
    assert isinstance(app.app_config, UserDict)
    assert app.app_config["AppRAGSearchConfig"]["rag"]["search"]["default_top_k"] == 12


@pytest.mark.asyncio
async def test_settings_domain_category_renders_read_only_owner_contract():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-search")
        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)
        await pilot.press(*"mcp")
        await _wait_for_settings_text(screen, pilot, "Filter: mcp")
        await pilot.press("enter")
        await _wait_for_settings_text(screen, pilot, "MCP Defaults")
        text = _visible_text(screen)

        assert "MCP Defaults" in text
        assert "Owner destination: MCP" in text
        assert "MCP owns server/tool runtime" in text


def test_settings_server_sync_workspace_rows_use_source_contracts():
    captured_sync_kwargs = {}

    class FakeSyncScopeService:
        def list_write_sync_promotion_states(self, **kwargs):
            captured_sync_kwargs.update(kwargs)
            return (
                SyncPromotionState(
                    domain="library_collections",
                    surface_label="Collections",
                    status="dry-run",
                    authority_label="Authority: local-first",
                    sync_label="Sync: dry-run only",
                    review_label="Review: required before writes",
                    conflict_label="Conflicts: none reported",
                    rollback_label="Rollback: not required",
                    mirror_label="Mirror: no dry-run report yet",
                    primary_recovery="Review dry-run results before enabling writes.",
                ),
            )

    class FakeWorkspaceRegistry:
        def get_active_workspace(self):
            return WorkspaceRecord(
                workspace_id="research",
                name="Research",
                authority=WorkspaceAuthority.SERVER_BACKED,
                sync_status=WorkspaceSyncStatus.READY,
                active=True,
            )

    app = SimpleNamespace(
        app_config={},
        runtime_policy=SimpleNamespace(
            state=RuntimeSourceState(
                active_source="server",
                active_server_id="server-main",
                server_configured=True,
                last_known_server_label="Main Server",
            )
        ),
        server_context_provider=SimpleNamespace(
            get_active_context=lambda: SimpleNamespace(
                auth_token="settings-scope-token",
                credential_source="test",
            )
        ),
        sync_scope_service=FakeSyncScopeService(),
        workspace_registry_service=FakeWorkspaceRegistry(),
        console_chat_store=SimpleNamespace(
            workspace_context=SimpleNamespace(active_workspace_id="research")
        ),
        get_acp_runtime_session_state=lambda: ACPRuntimeSessionState(
            runtime_id="local-acp",
            runtime_label="Local ACP",
        ),
    )
    screen = SettingsScreen(app)

    rows = dict(screen._server_sync_workspace_handoff_rows())

    assert captured_sync_kwargs["server_profile_id"] == "server-main"
    assert captured_sync_kwargs["authenticated_principal_id"].startswith(
        "credential-fingerprint:test:"
    )
    assert captured_sync_kwargs["workspace_scope"] == "research"
    assert rows["Active server profile"] == "Main Server (server-main)"
    assert rows["Local/server authority"] == "server; Settings is read-only"
    assert rows["Sync safety"] == "Collections: Sync: dry-run only"
    assert rows["Sync recovery"] == "Review dry-run results before enabling writes."
    assert rows["Workspace default"] == (
        "Workspace: Research (research); Authority: server-backed; Sync: ready"
    )
    assert rows["Library visibility"] == LIBRARY_WORKSPACE_VISIBILITY_COPY
    assert rows["Handoff policy"] == (
        "copy/reference/metadata-only by source policy; Console staging is limited to the active workspace"
    )
    assert rows["ACP handoff readiness"] == (
        "ACP runtime configured: Local ACP; no session payload"
    )


@pytest.mark.asyncio
async def test_settings_overview_detail_uses_cached_server_sync_rows(monkeypatch):
    def fail_if_render_blocks_on_source_contracts(*_args, **_kwargs):
        raise AssertionError("Overview render must use cached source-contract rows")

    monkeypatch.setattr(
        SettingsScreen,
        "_refresh_server_sync_workspace_handoff_rows",
        lambda _self: None,
        raising=False,
    )
    monkeypatch.setattr(
        SettingsScreen,
        "_server_sync_workspace_handoff_rows",
        fail_if_render_blocks_on_source_contracts,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Active server profile: Loading Settings source contracts" in text


@pytest.mark.asyncio
async def test_settings_overview_reselect_refreshes_cached_source_rows(monkeypatch):
    refresh_calls = 0

    def fake_refresh(self):
        nonlocal refresh_calls
        refresh_calls += 1

    monkeypatch.setattr(SettingsScreen, "_refresh_server_sync_workspace_handoff_rows", fake_refresh)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-category-overview")

        assert refresh_calls >= 2


@pytest.mark.asyncio
async def test_settings_screen_resume_refreshes_cached_source_rows(monkeypatch):
    refresh_calls = 0

    def fake_refresh(self):
        nonlocal refresh_calls
        refresh_calls += 1

    monkeypatch.setattr(SettingsScreen, "_refresh_server_sync_workspace_handoff_rows", fake_refresh)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        screen.on_screen_resume()

        assert refresh_calls >= 2


def test_settings_server_sync_workspace_rows_fallback_to_read_only_wip_copy():
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    rows = dict(screen._server_sync_workspace_handoff_rows())

    assert rows["Active server profile"] == "local-only; no active server profile"
    assert rows["Local/server authority"] == "local; Settings is read-only"
    assert "Collections: Sync: dry-run only" in rows["Sync safety"]
    assert "Workspaces: Sync: dry-run only" in rows["Sync safety"]
    assert rows["Workspace default"] == (
        "Workspace: Local Default; Console/Home/Library own workspace switching"
    )
    assert rows["Library visibility"] == LIBRARY_WORKSPACE_VISIBILITY_COPY
    assert rows["ACP handoff readiness"] == (
        "ACP runtime not configured; configure runtime and sessions in ACP"
    )


def test_settings_manual_sync_rows_use_preview_service():
    captured_kwargs = {}

    class FakeManualSyncControl:
        def preview(self, **kwargs):
            captured_kwargs.update(kwargs)
            return ManualSyncPreview(
                status="ready",
                can_run=True,
                pending_total=3,
                pending_by_domain={"notes": 2, "chat": 1},
                user_message="Manual Sync preview: 3 pending outgoing changes.",
            )

    app = SimpleNamespace(
        app_config={},
        runtime_policy=SimpleNamespace(
            state=RuntimeSourceState(
                active_source="server",
                active_server_id="server-main",
                server_configured=True,
            )
        ),
        server_context_provider=SimpleNamespace(
            get_active_context=lambda: SimpleNamespace(
                auth_token="settings-sync-token",
                credential_source="test",
            )
        ),
        manual_sync_control_service=FakeManualSyncControl(),
    )
    screen = SettingsScreen(app)

    rows = dict(screen._manual_sync_rows())

    assert captured_kwargs["server_profile_id"] == "server-main"
    assert captured_kwargs["authenticated_principal_id"].startswith(
        "credential-fingerprint:test:"
    )
    assert rows["Manual sync status"] == "ready"
    assert rows["Manual sync preview"] == "Manual Sync preview: 3 pending outgoing changes."
    assert rows["Pending outgoing"] == "notes: 2; chat: 1"


def test_settings_manual_sync_rows_block_without_control_service():
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    rows = dict(screen._manual_sync_rows())

    assert rows["Manual sync status"] == "blocked"
    assert rows["Manual sync preview"] == "Manual Sync control is not available."


def test_settings_apply_manual_sync_result_updates_rows():
    preview = ManualSyncPreview(
        status="ready",
        can_run=True,
        pending_total=1,
        pending_by_domain={"chat": 1},
        user_message="Manual Sync preview: 1 pending outgoing change.",
    )
    result = ManualSyncRunResult(
        status="partial-failure",
        user_message="Manual Sync partially completed.",
        summary={"outbox_retained": 1},
        preview=preview,
    )
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    screen._apply_manual_sync_result(result)

    rows = dict(screen.manual_sync_rows)
    assert rows["Manual sync status"] == "partial-failure"
    assert rows["Manual sync result"] == "Manual Sync partially completed."


def test_settings_apply_manual_sync_result_includes_conflict_review_summary():
    preview = ManualSyncPreview(
        status="ready",
        can_run=True,
        pending_total=1,
        pending_by_domain={"notes": 1},
        user_message="Manual Sync preview: 1 pending outgoing change.",
    )
    result = ManualSyncRunResult(
        status="conflict",
        user_message="Manual Sync found 1 conflict.",
        summary={"outbox_retained": 1},
        preview=preview,
        conflict_reviews=(
            SyncV2ConflictReviewItem(
                domain="notes",
                item_label="Research note",
                cause="Remote edit conflicts with local edit.",
                local_summary="Local title changed.",
                remote_summary="Remote body changed.",
                recovery_options={
                    "retry": "available",
                    "keep-local": "available",
                    "accept-remote": "available",
                    "duplicate-fork": "available",
                    "defer-later": "available",
                },
            ),
        ),
    )
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    screen._apply_manual_sync_result(result)

    rows = dict(screen.manual_sync_rows)
    assert rows["Conflict review"] == (
        "notes | Research note | Remote edit conflicts with local edit. | "
        "local: Local title changed. | remote: Remote body changed."
    )
    assert rows["Recovery options"] == (
        "retry: available; keep-local: available; accept-remote: available; "
        "duplicate-fork: available; defer-later: available"
    )


def test_settings_manual_sync_run_worker_uses_main_event_loop_async_worker():
    worker = SettingsScreen.__dict__["_manual_sync_run_worker"]
    wrapped = getattr(worker, "__wrapped__", worker)

    assert inspect.iscoroutinefunction(wrapped)


def test_settings_model_discovery_button_handlers_dispatch_workers():
    assert not inspect.iscoroutinefunction(SettingsScreen.__dict__["handle_discover_provider_models"])
    assert not inspect.iscoroutinefunction(SettingsScreen.__dict__["handle_save_discovered_provider_models"])
    assert not inspect.iscoroutinefunction(SettingsScreen.__dict__["handle_clear_discovered_provider_models"])

    for worker_name in (
        "_discover_provider_models_worker",
        "_save_selected_discovered_provider_models_worker",
        "_clear_discovered_provider_models_worker",
    ):
        worker = SettingsScreen.__dict__[worker_name]
        wrapped = getattr(worker, "__wrapped__", worker)
        assert inspect.iscoroutinefunction(wrapped)


def test_settings_status_language_agrees_with_home_console_and_library_contracts():
    app = SimpleNamespace(
        app_config={},
        runtime_policy=SimpleNamespace(
            state=RuntimeSourceState(
                active_source="server",
                active_server_id="server-main",
                server_configured=True,
                last_known_server_label="Main Server",
            )
        ),
        console_chat_store=SimpleNamespace(
            workspace_context=ConsoleWorkspaceContext(active_workspace_id="workspace-a")
        ),
    )
    screen = SettingsScreen(app)

    rows = dict(screen._server_sync_workspace_handoff_rows())
    library_scope = library_screen_module._active_library_sync_scope(app)
    home = summarize_home_dashboard(
        HomeDashboardInput(
            runtime_source="server",
            active_server_id="server-main",
            server_configured=True,
            server_reachability="reachable",
            server_auth_state="authenticated",
        )
    )

    assert rows["Local/server authority"].startswith("server")
    assert library_scope["source_authority"] == "server"
    assert "Mode: Server" in home.sections[0].lines[0]
    assert "Server: Ready" in home.sections[0].lines[0]
    assert rows["Workspace default"] == (
        "Workspace: workspace-a; Console context active; Library browse/search remains global"
    )
    assert rows["Library visibility"] == LIBRARY_WORKSPACE_VISIBILITY_COPY


@pytest.mark.asyncio
async def test_settings_overview_renders_server_sync_workspace_handoff_contracts():
    class FakeWorkspaceRegistry:
        def get_active_workspace(self):
            return WorkspaceRecord(
                workspace_id="research",
                name="Research",
                authority=WorkspaceAuthority.LOCAL_ONLY,
                sync_status=WorkspaceSyncStatus.NOT_CONFIGURED,
                active=True,
            )

    app = _build_test_app()
    app.runtime_policy.state = RuntimeSourceState(
        active_source="server",
        active_server_id="server-main",
        server_configured=True,
        last_known_server_label="Main Server",
    )
    app.workspace_registry_service = FakeWorkspaceRegistry()
    app.acp_runtime_session_state = ACPRuntimeSessionState(
        runtime_id="local-acp",
        runtime_label="Local ACP",
        session_id="session-1",
        session_title="Ticket triage",
        session_status="running",
        session_payload={"id": "session-1"},
    )

    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_settings_text(screen, pilot, "Active server profile: Main Server")
        text = _visible_text(screen)

        assert "Server, sync, workspace, and handoff" in text
        assert "Active server profile: Main Server (server-main)" in text
        assert "Local/server authority: server; Settings is read-only" in text
        assert "Collections: Sync: dry-run only" in text
        assert "Workspaces: Sync: dry-run only" in text
        assert "Workspace: Research (research); Authority: local-only; Sync: not-configured" in text
        assert LIBRARY_WORKSPACE_VISIBILITY_COPY in text
        assert "ACP handoff readiness: ACP session ready: Ticket triage (running)" in text


def test_settings_ownership_record_falls_back_without_crashing():
    app = _build_test_app()
    screen = SettingsScreen(app)
    screen._ownership_by_category_cache = {}

    record = screen._ownership_record(SettingsCategoryId.OVERVIEW)

    assert record.category is SettingsCategoryId.OVERVIEW
    assert not record.writes_allowed
    assert "Ownership record missing" in record.read_only_reason


@pytest.mark.asyncio
async def test_settings_overview_renders_ownership_contract_boundaries():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Settings owns persisted defaults and validation" in text
        assert "Console owns live chat/run state" in text
        assert "MCP owns server and tool management" in text
        assert "ACP owns runtime/session setup" in text
        assert "Sync and workspace handoff defaults are read-only until source contracts exist" in text


@pytest.mark.asyncio
async def test_settings_provider_inspector_excludes_console_sampling_ownership():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Affected config: provider, model, endpoint, and credential source defaults" in text
        assert "Sampling and transport defaults are routed to Console Defaults" in text
        assert "streaming, and temperature" not in text


@pytest.mark.asyncio
async def test_settings_provider_category_lists_console_supported_catalog():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Provider catalog" in text
        for provider in ("openai", "anthropic", "custom", "custom_2", "llama_cpp", "local_vllm"):
            assert provider in text
        assert "Choose a catalog provider, or use Manual / custom provider for aliases." in text


@pytest.mark.asyncio
async def test_settings_provider_model_defaults_appear_before_reference_copy():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        card = screen.query_one("#settings-providers-models-card")
        title = card.query_one("#settings-selected-model-defaults-title", Static)
        temperature = card.query_one("#settings-model-profile-temperature", Input)
        catalog = card.query_one("#settings-provider-catalog", Static)
        widgets = list(card.query("*"))

        assert str(title.renderable) == "Selected model defaults"
        assert widgets.index(title) < widgets.index(catalog)
        assert widgets.index(temperature) < widgets.index(catalog)


@pytest.mark.asyncio
async def test_settings_provider_text_inputs_do_not_trigger_footer_shortcuts(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    saved = []

    class FakeAdapter:
        def save_values(self, section, values):
            saved.append((section, values))
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        model_input = screen.query_one("#settings-model-value", Input)
        model_input.value = "gpt-shortcut-check"
        screen.handle_model_value_changed(Input.Changed(model_input, model_input.value))
        screen._provider_test_result = None
        model_input.focus()
        await pilot.pause()

        assert screen.app.focused is model_input

        await pilot.press("s", "r", "t")
        await pilot.pause()

        assert model_input.value == "srt"
        assert saved == []
        assert screen._provider_test_result is None
        assert screen._settings_drafts

        screen.action_settings_save_category()
        screen.action_settings_test_category()
        screen.action_settings_revert_category()

        assert saved == []
        assert model_input.value == "srt"
        assert screen._provider_test_result is None
        assert screen._settings_drafts


@pytest.mark.asyncio
async def test_settings_category_selection_updates_detail_and_inspector():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Console Behavior" in text
        assert "Collapse large pasted chunks" in text
        assert "Control guide" in text


@pytest.mark.asyncio
async def test_settings_console_behavior_inspector_explains_visible_controls():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Control guide" in text
        assert (
            "Streaming: Global fallback for streaming responses when no Console session "
            "or provider+model profile overrides it"
        ) in text
        assert "Temperature: Creativity fallback, 0.0 is focused and 2.0 is exploratory" in text
        assert "Top P: Probability cutoff fallback; lower values narrow token choices" in text
        assert "Max tokens: Optional response cap for new/default Console sends" in text
        assert (
            "Paste collapse: Only pasted chunks over the threshold become compact placeholders; "
            "typed text stays literal"
        ) in text
        assert "Threshold: Minimum pasted chunk size before collapse" in text
        assert "Affected config: chat_defaults fallbacks plus Console composer paste behavior" not in text
        assert "Mutation replay: disabled" not in text


@pytest.mark.asyncio
async def test_settings_category_navigation_is_grouped_for_scan():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        for group_title in (
            "Core",
            "Interface",
            "Data & Privacy",
            "Troubleshooting",
            "Expert",
        ):
            assert group_title in text


@pytest.mark.asyncio
async def test_settings_active_category_uses_explicit_nav_marker():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        active = screen.query_one("#settings-category-advanced-config")
        inactive = screen.query_one("#settings-category-diagnostics")

        assert str(active.label) == "> Advanced Config"
        assert str(inactive.label) == "  Diagnostics"
        assert active.has_class("settings-active-section")


def test_settings_active_category_focus_style_keeps_label_readable():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"Button\.settings-category-button\.settings-active-section:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "reverse" not in body
    assert "text-style: bold underline;" in body


def test_settings_action_button_focus_style_keeps_label_readable():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"\.settings-action-row Button:focus,\s*#settings-impact-pane Button:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "reverse" not in body
    assert "text-style: bold underline;" in body
    assert "outline: none;" in body


def test_settings_shell_button_focus_does_not_use_heavy_outline():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"#settings-shell Button:focus,\s*#settings-shell Button:hover:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "outline: none;" in body
    assert "text-style: bold underline;" in body


def test_settings_invalid_compact_fields_keep_focused_text_readable():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"\.settings-compact-input\.settings-invalid-input:focus,\s*"
        r"\.settings-compact-select\.settings-invalid-input:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "color: $ds-text-primary;" in body
    assert "text-opacity: 1;" in body
    assert "background: $ds-surface-raised;" in body
    assert "outline: none;" in body
    assert "outline: heavy" not in body


@pytest.mark.asyncio
async def test_settings_detail_shows_state_banner_and_structured_rows():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        banner = screen.query_one("#settings-category-state-banner")
        detail_rows = list(screen.query(".settings-detail-row"))

        assert "State:" in str(banner.renderable)
        assert banner.has_class("settings-state-banner")
        assert len(detail_rows) >= 5


@pytest.mark.asyncio
async def test_settings_long_detail_and_inspector_panes_are_scrollable_containers():
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        assert isinstance(screen.query_one("#settings-detail-pane"), VerticalScroll)
        assert isinstance(screen.query_one("#settings-impact-pane"), VerticalScroll)

        await pilot.click("#settings-category-providers-models")
        detail_pane = screen.query_one("#settings-detail-pane", VerticalScroll)
        test_provider = screen.query_one("#settings-test-provider", Button)

        assert detail_pane.max_scroll_y > 0
        detail_pane.scroll_to_widget(
            test_provider,
            animate=False,
            immediate=True,
            top=True,
            force=True,
        )
        assert detail_pane.scroll_y > 0


@pytest.mark.asyncio
async def test_settings_inspector_uses_category_specific_guidance():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Affected config: config file path, local database paths, media storage roots" in text
        assert (
            "Recovery: Validate paths, save the config-only change, then restart Chatbook "
            "to activate new storage defaults."
        ) in text
        assert "MCP and tool-control settings live under MCP" not in text


@pytest.mark.asyncio
async def test_settings_inspector_boundary_is_structured_without_duplicate_copy():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        boundary = "Boundary: save is blocked until the exact current text validates"
        assert str(screen.query_one("#settings-boundary-note").renderable) == boundary
        assert text.count(boundary) == 1


@pytest.mark.asyncio
async def test_settings_tab_focus_and_enter_select_categories():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        screen = _active_destination_screen(host)

        assert "Providers & Models" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_keyboard_category_focus_survives_selection_recompose():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("enter")
        screen = _active_destination_screen(host)

        assert "Appearance" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_category_search_filters_and_enter_opens_first_match():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-search")

        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)
        search = screen.query_one("#settings-category-search", Input)

        assert search.has_focus

        await pilot.press(*"priv")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Filter: priv | 2 matches | Enter opens Privacy & Security",
        )

        assert screen.query_one("#settings-category-privacy-security").display
        assert not screen.query_one("#settings-category-providers-models").display

        await pilot.press("enter")
        await _wait_for_settings_text(screen, pilot, "Privacy posture")

        assert screen.active_category == SettingsCategoryId.PRIVACY_SECURITY.value
        assert "Privacy & Security" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_category_search_reports_ranked_matches_and_enter_target():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-search")
        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)

        await pilot.press(*"priv")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Filter: priv | 2 matches | Enter opens Privacy & Security",
        )

        visible_text = _visible_text(screen)
        assert "Filter: priv | 2 matches | Enter opens Privacy & Security" in visible_text
        assert screen.query_one("#settings-category-privacy-security").has_class(
            "settings-primary-search-match"
        )
        assert screen.query_one("#settings-category-overview").has_class(
            "settings-secondary-search-match"
        )


@pytest.mark.asyncio
async def test_settings_category_search_uses_plain_standard_input_widgets():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)

        search = screen.query_one("#settings-category-search", Input)
        assert type(search) is Input
        assert not screen.query_one("#settings-category-search-status", Static)._render_markup
        assert not screen.query_one("#settings-category-search-empty", Static)._render_markup


def test_settings_category_search_normalizes_oversized_control_input():
    screen = SettingsScreen(_build_test_app())

    normalized = screen._sanitize_category_search_query("[" + ("x" * 120) + "\x00")

    assert len(normalized) == 80
    assert normalized == "[" + ("x" * 79)
    assert "\x00" not in normalized


@pytest.mark.asyncio
async def test_settings_category_search_escape_clears_filter():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-search")

        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)

        await pilot.press(*"zzz")
        await _wait_for_settings_text(screen, pilot, "No Settings categories match: zzz")

        assert "No Settings categories match" in _visible_text(screen)
        assert not any(button.display for button in screen.query(".settings-category-button"))

        await pilot.press("escape")
        await pilot.pause()

        search = screen.query_one("#settings-category-search", Input)
        assert search.value == ""
        assert sum(1 for button in screen.query(".settings-category-button") if button.display) == len(
            screen._category_summaries()
        )


@pytest.mark.asyncio
async def test_settings_overview_paste_summary_updates_after_toggle(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        await pilot.click("#settings-category-overview")
        screen = _active_destination_screen(host)

        await _wait_for_settings_text(
            screen,
            pilot,
            "Console paste collapse: Disabled: collapse large pastes",
        )


@pytest.mark.asyncio
async def test_settings_paste_toggle_keeps_keyboard_focus_after_refresh(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        toggle = screen.query_one("#settings-console-collapse-large-pastes-toggle")
        toggle.focus()

        await pilot.press("enter")
        await pilot.pause()
        assert host.focused is toggle
        assert str(toggle.label) == "Disabled"
        assert "Unsaved" in _visible_text(screen)

        await pilot.press("enter")
        await pilot.pause()

        assert host.focused is toggle
        assert str(toggle.label) == "Enabled"
        assert "No unsaved changes" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_console_behavior_clean_state_does_not_show_staged_feedback():
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "No unsaved changes" in text
        assert "Console behavior settings staged." not in text


@pytest.mark.asyncio
async def test_settings_console_behavior_default_undo_does_not_show_staged_feedback():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"temperature": 0.7}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        temperature = screen.query_one("#settings-console-default-temperature", Input)

        temperature.value = "0.8"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))
        assert "Unsaved" in _visible_text(screen)

        temperature.value = "0.7"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))
        text = _visible_text(screen)

        assert "No unsaved changes" in text
        assert "Console behavior settings staged." not in text


@pytest.mark.asyncio
async def test_settings_console_behavior_clean_staged_feedback_shows_workbench_warning():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "matrix",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        screen._console_behavior_result = "Console behavior settings staged."
        screen._set_static_text(
            "#settings-console-behavior-result",
            screen._console_behavior_result_text(),
        )
        text = _visible_text(screen)

        assert "No unsaved changes" in text
        assert "Workbench scope is not available in this build; using Transcript scope." in text
        assert "Console behavior settings staged." not in text


@pytest.mark.asyncio
async def test_settings_console_behavior_stages_save_and_revert(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        screen = _active_destination_screen(host)

        assert "Unsaved" in _visible_text(screen)
        assert app.app_config["console"]["collapse_large_pastes"] is True

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [{"console": {"collapse_large_pastes": False}}]
    assert app.app_config["console"]["collapse_large_pastes"] is False


@pytest.mark.asyncio
async def test_settings_console_behavior_saves_paste_threshold(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        threshold = screen.query_one("#settings-console-paste-collapse-threshold", Input)

        assert threshold.restrict == r"^[0-9]*$"
        threshold.value = "120"
        screen.handle_console_paste_threshold_changed(Input.Changed(threshold, threshold.value))
        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [{"console": {"paste_collapse_threshold": 120}}]
    assert app.app_config["console"]["paste_collapse_threshold"] == 120


@pytest.mark.asyncio
async def test_settings_console_behavior_renders_global_default_controls():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "streaming": False,
        "temperature": 0.33,
        "top_p": 0.81,
        "min_p": 0.04,
        "top_k": 42,
        "max_tokens": 2048,
        "seed": 123,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
        "verbosity": "medium",
        "thinking_effort": "low",
        "thinking_budget_tokens": 4096,
    }
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-console-default-streaming", Input).value == "false"
        assert screen.query_one("#settings-console-default-temperature", Input).value == "0.33"
        assert screen.query_one("#settings-console-default-top-p", Input).value == "0.81"
        assert screen.query_one("#settings-console-default-min-p", Input).value == "0.04"
        assert screen.query_one("#settings-console-default-top-k", Input).value == "42"
        assert screen.query_one("#settings-console-default-max-tokens", Input).value == "2048"
        assert screen.query_one("#settings-console-default-seed", Input).value == "123"
        assert screen.query_one("#settings-console-default-presence-penalty", Input).value == "0.2"
        assert screen.query_one("#settings-console-default-frequency-penalty", Input).value == "0.3"
        assert screen.query_one("#settings-console-default-reasoning-effort", Input).value == "high"
        assert screen.query_one("#settings-console-default-reasoning-summary", Input).value == "auto"
        assert screen.query_one("#settings-console-default-verbosity", Input).value == "medium"
        assert screen.query_one("#settings-console-default-thinking-effort", Input).value == "low"
        assert (
            screen.query_one("#settings-console-default-thinking-budget-tokens", Input).value
            == "4096"
        )
        assert screen.query_one("#settings-console-paste-collapse-threshold", Input).value == "50"
        text = _visible_text(screen)
        assert "Used when no provider+model profile or active Console session overrides them." in text
        assert "chat_defaults.streaming is canonical; enable_streaming is read as fallback only." in text


@pytest.mark.asyncio
async def test_settings_console_behavior_renders_background_effect_controls():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": False,
            "effect": "none",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-console-background-effect-enabled", Button)
        assert screen.query_one("#settings-console-background-effect-type", Select)
        assert screen.query_one("#settings-console-background-effect-scope", Select)
        assert screen.query_one("#settings-console-background-effect-intensity", Select)
        assert screen.query_one("#settings-console-background-effect-fps", Input)
        assert "Transcript (recommended)" in _visible_text(screen)


def test_settings_console_behavior_owns_background_effect_settings():
    app = _build_test_app()
    screen = SettingsScreen(app)
    ownership = screen._ownership_record(SettingsCategoryId.CONSOLE_BEHAVIOR)

    assert "console.background_effects.*" in ownership.owns_config_sections


@pytest.mark.asyncio
async def test_settings_console_background_effects_save_nested_config(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "background_effects": {
            "enabled": False,
            "effect": "none",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        },
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        enabled = screen.query_one("#settings-console-background-effect-enabled", Button)
        effect = screen.query_one("#settings-console-background-effect-type", Select)
        fps = screen.query_one("#settings-console-background-effect-fps", Input)

        enabled.press()
        effect.value = "matrix"
        fps.value = "10"
        screen.handle_console_background_effect_fps_changed(Input.Changed(fps, fps.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [
        {
            "console": {
                "background_effects": {
                    "enabled": True,
                    "effect": "matrix",
                    "scope": "transcript",
                    "intensity": "low",
                    "fps": 10,
                }
            }
        }
    ]
    assert app.app_config["console"]["background_effects"]["enabled"] is True
    assert app.app_config["console"]["background_effects"]["effect"] == "matrix"


@pytest.mark.asyncio
async def test_settings_console_background_fps_rejects_out_of_range_save(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        },
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _select_settings_category(
            screen,
            pilot,
            SettingsCategoryId.CONSOLE_BEHAVIOR,
            selector="#settings-console-background-effect-fps",
        )
        fps = screen.query_one("#settings-console-background-effect-fps", Input)

        fps.value = "610"
        screen.handle_console_background_effect_fps_changed(Input.Changed(fps, fps.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Frame rate must be a whole number between 1 and 12.",
        )

        assert "Unsaved changes" in _visible_text(screen)

    assert saved == []
    assert app.app_config["console"]["background_effects"]["fps"] == 6


@pytest.mark.asyncio
async def test_settings_console_background_workbench_scope_falls_back_to_transcript(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        scope = screen.query_one("#settings-console-background-effect-scope", Select)

        scope.value = "workbench"
        screen.handle_console_background_effect_scope_changed(Select.Changed(scope, scope.value))

        await _wait_for_settings_text(
            screen,
            pilot,
            "Workbench scope is not available in this build; using Transcript scope.",
        )
        assert scope.value == "transcript"

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved[0]["console"]["background_effects"]["scope"] == "transcript"
    assert app.app_config["console"]["background_effects"]["scope"] == "transcript"


@pytest.mark.asyncio
async def test_settings_console_background_workbench_loaded_scope_save_shows_fallback(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        }
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        fps = screen.query_one("#settings-console-background-effect-fps", Input)

        fps.value = "10"
        screen.handle_console_background_effect_fps_changed(Input.Changed(fps, fps.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Workbench scope is not available in this build; using Transcript scope.",
        )

    assert saved[0]["console"]["background_effects"]["scope"] == "transcript"
    assert app.app_config["console"]["background_effects"]["scope"] == "transcript"


@pytest.mark.asyncio
async def test_settings_console_background_workbench_loaded_scope_unrelated_save_falls_back(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config["console"] = {
        "paste_collapse_threshold": 50,
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        },
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        threshold = screen.query_one("#settings-console-paste-collapse-threshold", Input)

        threshold.value = "120"
        screen.handle_console_paste_threshold_changed(Input.Changed(threshold, threshold.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Workbench scope is not available in this build; using Transcript scope.",
        )

    assert saved[0]["console"]["paste_collapse_threshold"] == 120
    assert saved[0]["console"]["background_effects"]["scope"] == "transcript"
    assert app.app_config["console"]["paste_collapse_threshold"] == 120
    assert app.app_config["console"]["background_effects"]["scope"] == "transcript"


def test_settings_console_background_workbench_raw_scope_unrelated_save_includes_fallback():
    app = _build_test_app()
    app.app_config["console"] = {
        "paste_collapse_threshold": 50,
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        },
    }
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.CONSOLE_BEHAVIOR.value
    draft = SettingsDraft(category=SettingsCategoryId.CONSOLE_BEHAVIOR)
    draft.set_value("paste_collapse_threshold", 50, 120)
    screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR] = draft
    saved_args = []

    def fake_worker(console_values, chat_default_values, workbench_scope_fallback=False):
        saved_args.append((console_values, chat_default_values, workbench_scope_fallback))

    screen._settings_save_console_behavior_worker = fake_worker

    screen.action_settings_save_category()

    console_values, chat_default_values, workbench_scope_fallback = saved_args[0]
    assert chat_default_values == {}
    assert console_values["paste_collapse_threshold"] == 120
    assert console_values["background_effects"]["scope"] == "transcript"
    assert workbench_scope_fallback is True


def test_settings_on_key_treats_detached_screen_focus_as_absent():
    app = _build_test_app()
    screen = SettingsScreen(app)

    screen.on_key(Key("slash", "/"))

    assert screen.category_search_query == ""


@pytest.mark.asyncio
async def test_settings_console_background_workbench_loaded_scope_mounts_as_transcript():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        },
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        scope = screen.query_one("#settings-console-background-effect-scope", Select)

        assert scope.value == "transcript"
        await _wait_for_settings_text(
            screen,
            pilot,
            "Workbench scope is not available in this build; using Transcript scope.",
        )


@pytest.mark.asyncio
async def test_settings_console_behavior_saves_global_defaults(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        streaming = screen.query_one("#settings-console-default-streaming", Input)
        temperature = screen.query_one("#settings-console-default-temperature", Input)
        top_p = screen.query_one("#settings-console-default-top-p", Input)
        min_p = screen.query_one("#settings-console-default-min-p", Input)
        top_k = screen.query_one("#settings-console-default-top-k", Input)
        max_tokens = screen.query_one("#settings-console-default-max-tokens", Input)
        seed = screen.query_one("#settings-console-default-seed", Input)
        presence_penalty = screen.query_one("#settings-console-default-presence-penalty", Input)
        frequency_penalty = screen.query_one("#settings-console-default-frequency-penalty", Input)
        reasoning_effort = screen.query_one("#settings-console-default-reasoning-effort", Input)
        reasoning_summary = screen.query_one("#settings-console-default-reasoning-summary", Input)
        verbosity = screen.query_one("#settings-console-default-verbosity", Input)
        thinking_effort = screen.query_one("#settings-console-default-thinking-effort", Input)
        thinking_budget = screen.query_one("#settings-console-default-thinking-budget-tokens", Input)

        streaming.value = "false"
        screen.handle_console_default_streaming_changed(Input.Changed(streaming, streaming.value))
        temperature.value = "0.33"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))
        top_p.value = "0.81"
        screen.handle_console_default_top_p_changed(Input.Changed(top_p, top_p.value))
        min_p.value = "0.04"
        screen.handle_console_default_min_p_changed(Input.Changed(min_p, min_p.value))
        top_k.value = "42"
        screen.handle_console_default_top_k_changed(Input.Changed(top_k, top_k.value))
        max_tokens.value = "2048"
        screen.handle_console_default_max_tokens_changed(Input.Changed(max_tokens, max_tokens.value))
        seed.value = "123"
        screen.handle_console_default_seed_changed(Input.Changed(seed, seed.value))
        presence_penalty.value = "0.2"
        screen.handle_console_default_presence_penalty_changed(
            Input.Changed(presence_penalty, presence_penalty.value)
        )
        frequency_penalty.value = "0.3"
        screen.handle_console_default_frequency_penalty_changed(
            Input.Changed(frequency_penalty, frequency_penalty.value)
        )
        reasoning_effort.value = "high"
        screen.handle_console_default_reasoning_effort_changed(
            Input.Changed(reasoning_effort, reasoning_effort.value)
        )
        reasoning_summary.value = "auto"
        screen.handle_console_default_reasoning_summary_changed(
            Input.Changed(reasoning_summary, reasoning_summary.value)
        )
        verbosity.value = "medium"
        screen.handle_console_default_verbosity_changed(Input.Changed(verbosity, verbosity.value))
        thinking_effort.value = "low"
        screen.handle_console_default_thinking_effort_changed(
            Input.Changed(thinking_effort, thinking_effort.value)
        )
        thinking_budget.value = "4096"
        screen.handle_console_default_thinking_budget_tokens_changed(
            Input.Changed(thinking_budget, thinking_budget.value)
        )

        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [
        {
            "chat_defaults": {
                "streaming": False,
                "temperature": 0.33,
                "top_p": 0.81,
                "min_p": 0.04,
                "top_k": 42,
                "max_tokens": 2048,
                "seed": 123,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.3,
                "reasoning_effort": "high",
                "reasoning_summary": "auto",
                "verbosity": "medium",
                "thinking_effort": "low",
                "thinking_budget_tokens": 4096,
            }
        }
    ]
    assert app.app_config["chat_defaults"] == {
        "streaming": False,
        "temperature": 0.33,
        "top_p": 0.81,
        "min_p": 0.04,
        "top_k": 42,
        "max_tokens": 2048,
        "seed": 123,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
        "verbosity": "medium",
        "thinking_effort": "low",
        "thinking_budget_tokens": 4096,
    }


@pytest.mark.asyncio
async def test_settings_console_behavior_uses_batched_save_adapter(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 50,
    }
    app.app_config["chat_defaults"] = {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    legacy_calls = []
    batched_calls = []

    class FakeAdapter:
        def save_sections(self, section_values):
            batched_calls.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    monkeypatch.setattr(
        settings_screen_module,
        "save_setting_to_cli_config",
        lambda *args, **kwargs: legacy_calls.append(args) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        threshold = screen.query_one("#settings-console-paste-collapse-threshold", Input)
        streaming = screen.query_one("#settings-console-default-streaming", Input)

        threshold.value = "120"
        screen.handle_console_paste_threshold_changed(Input.Changed(threshold, threshold.value))
        streaming.value = "false"
        screen.handle_console_default_streaming_changed(Input.Changed(streaming, streaming.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert legacy_calls == []
    assert batched_calls == [
        {
            "console": {"paste_collapse_threshold": 120},
            "chat_defaults": {"streaming": False},
        }
    ]
    assert app.app_config["console"]["paste_collapse_threshold"] == 120
    assert app.app_config["chat_defaults"]["streaming"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_id", "handler_name", "value", "message"),
    (
        (
            "#settings-console-default-streaming",
            "handle_console_default_streaming_changed",
            "tru",
            "Streaming must be true or false.",
        ),
        (
            "#settings-console-default-temperature",
            "handle_console_default_temperature_changed",
            "2.1",
            "Temperature must be between 0.0 and 2.0.",
        ),
        (
            "#settings-console-default-top-p",
            "handle_console_default_top_p_changed",
            "1.1",
            "Top P must be between 0.0 and 1.0.",
        ),
        (
            "#settings-console-default-max-tokens",
            "handle_console_default_max_tokens_changed",
            "0",
            "Max tokens must be a whole number of at least 1.",
        ),
        (
            "#settings-console-default-min-p",
            "handle_console_default_min_p_changed",
            "1.1",
            "Min P must be between 0.0 and 1.0.",
        ),
        (
            "#settings-console-default-thinking-budget-tokens",
            "handle_console_default_thinking_budget_tokens_changed",
            "128",
            "Thinking budget tokens must be a whole number of at least 1024.",
        ),
    ),
)
async def test_settings_console_behavior_rejects_invalid_global_defaults(
    monkeypatch,
    field_id,
    handler_name,
    value,
    message,
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        field = screen.query_one(field_id, Input)
        field.value = value
        getattr(screen, handler_name)(Input.Changed(field, value))

        await pilot.click("#settings-save-category")

        assert message in _visible_text(screen)

    assert saved == []
    assert app.app_config["chat_defaults"] == {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
    }


@pytest.mark.asyncio
async def test_settings_console_behavior_revert_restores_global_defaults(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        temperature = screen.query_one("#settings-console-default-temperature", Input)
        temperature.value = "0.33"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))

        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-revert-category")

        assert screen.query_one("#settings-console-default-temperature", Input).value == "0.7"
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "No unsaved changes" in _visible_text(screen)

    assert saved == []
    assert app.app_config["chat_defaults"] == {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
    }


@pytest.mark.asyncio
async def test_settings_console_behavior_revert_button_works_with_input_focus(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "streaming": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        temperature = screen.query_one("#settings-console-default-temperature", Input)
        temperature.focus()
        temperature.value = "0.33"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))
        monkeypatch.setattr(screen, "_settings_text_entry_has_focus", lambda: True)

        screen.handle_revert_category(
            Button.Pressed(screen.query_one("#settings-revert-category", Button))
        )

        assert screen.query_one("#settings-console-default-temperature", Input).value == "0.7"
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert "No unsaved changes" in _visible_text(screen)

    assert saved == []
    assert app.app_config["chat_defaults"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_settings_console_behavior_revert_discards_draft(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        screen = _active_destination_screen(host)

        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-revert-category")
        assert "No unsaved changes" in _visible_text(screen)

    assert saved == []
    assert app.app_config["console"]["collapse_large_pastes"] is True


@pytest.mark.asyncio
async def test_settings_non_editable_categories_disable_guided_save_revert():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: choose Providers or Console." in _visible_text(screen)

        await _select_settings_category(
            screen,
            pilot,
            SettingsCategoryId.PRIVACY_SECURITY,
            expected_text="Guided edits: use Check Privacy.",
        )
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: use Check Privacy." in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_console_guided_save_revert_enable_only_when_dirty():
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: change a field first." in _visible_text(screen)

        await pilot.click("#settings-console-collapse-large-pastes-toggle")

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert screen.query_one("#settings-revert-category", Button).disabled is False
        assert "Guided edits: Save or Revert changes." in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_provider_category_uses_effective_console_source():
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "llama_cpp" in text
        assert "qwen" in text
        assert "Provider source: Saved chat defaults" in text
        assert "Model source: Saved chat defaults" in text
        assert screen.query_one("#settings-provider-value", Select).value == "llama_cpp"
        assert screen.query_one("#settings-model-value", Input).value == "qwen"


@pytest.mark.asyncio
async def test_settings_provider_category_renders_catalog_select_with_visible_value():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        provider = screen.query_one("#settings-provider-value", Select)

        assert provider.value == "llama_cpp"
        assert provider.has_class("-textual-compact")
        assert "Provider catalog" in _visible_text(screen)
        assert "llama.cpp" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_navigation_context_can_preselect_provider_category_target():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "huggingface",
                "model": "meta-llama/test-model",
            }
        )
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.PROVIDERS_MODELS.value
        assert screen.query_one("#settings-provider-value", Select).value == "huggingface"
        assert screen.query_one("#settings-model-value", Input).value == "meta-llama/test-model"
        assert "HUGGINGFACE_API_KEY" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_provider_navigation_context_focuses_api_key_field():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "openai",
                "model": "gpt-4.1",
                "field": "api_key",
            }
        )
        await pilot.pause()

        api_key = screen.query_one("#settings-provider-api-key", Input)
        assert api_key.has_focus


@pytest.mark.asyncio
async def test_settings_navigation_context_preselection_does_not_create_provider_draft():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "huggingface",
                "model": "meta-llama/test-model",
            }
        )
        await pilot.pause()

        assert screen.query_one("#settings-provider-value", Select).value == "huggingface"
        assert screen.query_one("#settings-model-value", Input).value == "meta-llama/test-model"
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS)


@pytest.mark.asyncio
async def test_settings_navigation_context_preserves_existing_provider_draft_values():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        draft = SettingsDraft(category=SettingsCategoryId.PROVIDERS_MODELS)
        draft.set_value("endpoint", "https://api.example/v1", "https://draft.example/v1")
        draft.set_value("credential_env_var", "OPENAI_API_KEY", "DRAFT_PROVIDER_KEY")
        draft.set_value("model_profile_temperature", "", 0.4)
        screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS] = draft

        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "huggingface",
                "model": "meta-llama/test-model",
            }
        )
        await pilot.pause()

        preserved_draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert preserved_draft.values["endpoint"] == "https://draft.example/v1"
        assert preserved_draft.values["credential_env_var"] == "DRAFT_PROVIDER_KEY"
        assert preserved_draft.values["model_profile_temperature"] == 0.4


@pytest.mark.asyncio
async def test_settings_navigation_provider_context_tolerates_missing_provider_values(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await pilot.pause()
        monkeypatch.setattr(screen, "_provider_setting_values", lambda: None)

        screen._apply_navigation_provider_context("huggingface")
        await pilot.pause()

        assert screen.query_one("#settings-provider-value", Select).value == "huggingface"
        assert screen.query_one("#settings-model-value", Input).value == ""


@pytest.mark.asyncio
async def test_settings_provider_keyless_local_provider_does_not_report_missing_env_var():
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "api_key_env_var": "LLAMA_CPP_API_KEY",
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _select_settings_category(
            screen,
            pilot,
            SettingsCategoryId.PROVIDERS_MODELS,
            expected_text="API key: not required for this provider",
        )
        text = _visible_text(screen)

        assert "API key: not required for this provider" in text
        assert "LLAMA_CPP_API_KEY=missing" not in text


@pytest.mark.asyncio
async def test_settings_provider_openai_endpoint_placeholder_uses_provider_context():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"}
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)

        assert endpoint.value == ""
        assert endpoint.placeholder == "https://api.openai.com/v1"
        assert "127.0.0.1:9099" not in endpoint.placeholder


def test_settings_endpoint_display_breaks_browser_autolinks_without_mutating_value():
    endpoint = "http://localhost:8000/v1/chat/completions"

    display_value = settings_screen_module._textual_web_safe_url_display(endpoint)

    assert "http://" not in display_value
    assert display_value.replace("\u200b", "") == endpoint


def test_settings_endpoint_display_index_maps_url_break_boundaries_to_visible_colon():
    assert settings_screen_module._textual_web_safe_url_display_index(
        "http://localhost:8000", len("http")
    ) == len("http") + 1
    assert settings_screen_module._textual_web_safe_url_display_index(
        "https://api.example.com", len("https")
    ) == len("https") + 1


@pytest.mark.asyncio
async def test_settings_provider_endpoint_uses_url_safe_input_for_url_values():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "local_llm", "model": "local-model"}
    app.app_config["api_settings"] = {
        "local_llm": {"api_url": "http://localhost:8000/v1/chat/completions"}
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)

        assert isinstance(endpoint, settings_screen_module.SettingsURLInput)
        assert endpoint.value == "http://localhost:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_settings_provider_guided_save_revert_enable_only_when_dirty():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: change a field first." in _visible_text(screen)

        model = screen.query_one("#settings-model-value", Input)
        model.value = "gpt-4.1-mini"
        screen.handle_model_value_changed(Input.Changed(model, model.value))

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert screen.query_one("#settings-revert-category", Button).disabled is False
        assert "Guided edits: Save or Revert changes." in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_provider_test_redacts_secrets(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-token")
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Provider test")
        text = _visible_text(screen)

        assert "Provider test" in text
        assert "OPENAI_API_KEY=<redacted>" in text
        assert "sk-" not in text


@pytest.mark.asyncio
async def test_settings_provider_category_saves_provider_defaults_without_sampling(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    assert app.chat_api_provider_value == "OpenAI"
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "llama_cpp"))
        screen.query_one("#settings-model-value", Input).value = "qwen"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("chat_defaults", "provider", "llama_cpp"),
        ("chat_defaults", "model", "qwen"),
    ]
    assert app.app_config["chat_defaults"] == {
        "provider": "llama_cpp",
        "model": "qwen",
        "streaming": True,
        "temperature": 0.7,
    }
    assert app.chat_api_provider_value == "llama_cpp"


@pytest.mark.asyncio
async def test_settings_provider_category_saves_selected_model_profile(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {
            "model_defaults": {
                "gpt-4.1": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "streaming": True,
                },
            },
        },
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-profile-temperature", Input).value = "0.2"
        screen.query_one("#settings-model-profile-top-p", Input).value = "0.88"
        screen.query_one("#settings-model-profile-streaming", Input).value = "false"
        assert "Global fallbacks live under Console Defaults" in _visible_text(screen)

        await pilot.click("#settings-save-category")

    assert saved == [
        (
            "api_settings.openai",
            "model_defaults",
            {
                "gpt-4.1": {
                    "temperature": 0.2,
                    "top_p": 0.88,
                    "streaming": False,
                },
            },
        ),
    ]
    assert app.app_config["chat_defaults"] == {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    assert app.app_config["api_settings"]["openai"]["model_defaults"]["gpt-4.1"] == {
        "temperature": 0.2,
        "top_p": 0.88,
        "streaming": False,
    }


@pytest.mark.asyncio
async def test_settings_provider_category_saves_openai_generation_profile(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "o3"}
    app.app_config["api_settings"] = {"openai": {"model_defaults": {"o3": {}}}}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        values = {
            "#settings-model-profile-temperature": "0.3",
            "#settings-model-profile-top-p": "0.86",
            "#settings-model-profile-min-p": "0.04",
            "#settings-model-profile-top-k": "42",
            "#settings-model-profile-max-tokens": "2048",
            "#settings-model-profile-seed": "123",
            "#settings-model-profile-presence-penalty": "0.2",
            "#settings-model-profile-frequency-penalty": "0.3",
            "#settings-model-profile-reasoning-effort": "high",
            "#settings-model-profile-reasoning-summary": "auto",
            "#settings-model-profile-verbosity": "medium",
            "#settings-model-profile-thinking-effort": "high",
            "#settings-model-profile-thinking-budget-tokens": "4096",
            "#settings-model-profile-streaming": "false",
        }
        for selector, value in values.items():
            screen.query_one(selector, Input).value = value

        text = _visible_text(screen)
        assert "Thinking unavailable for OpenAI" in text
        assert screen.query_one("#settings-model-profile-thinking-effort", Input).disabled is True
        assert screen.query_one("#settings-model-profile-thinking-budget-tokens", Input).disabled is True

        await pilot.click("#settings-save-category")

    assert saved == [
        (
            "api_settings.openai",
            "model_defaults",
            {
                "o3": {
                    "temperature": 0.3,
                    "top_p": 0.86,
                    "min_p": 0.04,
                    "top_k": 42,
                    "max_tokens": 2048,
                    "seed": 123,
                    "presence_penalty": 0.2,
                    "frequency_penalty": 0.3,
                    "reasoning_effort": "high",
                    "reasoning_summary": "auto",
                    "verbosity": "medium",
                    "streaming": False,
                },
            },
        ),
    ]


def test_settings_generation_controls_allow_openai_none_reasoning_effort():
    screen = SettingsScreen(_build_test_app())

    assert screen._normalise_model_profile_reasoning_effort("none") == "none"
    assert (
        "none"
        in settings_screen_module.MODEL_PROFILE_INPUT_PLACEHOLDERS[
            "model_profile_reasoning_effort"
        ]
    )


def test_settings_generation_controls_allow_anthropic_max_thinking_effort():
    screen = SettingsScreen(_build_test_app())

    assert screen._normalise_model_profile_thinking_effort("max") == "max"
    assert (
        "max"
        in settings_screen_module.MODEL_PROFILE_INPUT_PLACEHOLDERS[
            "model_profile_thinking_effort"
        ]
    )


@pytest.mark.asyncio
async def test_settings_provider_category_saves_anthropic_thinking_profile(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "Anthropic", "model": "claude-opus-4-7"}
    app.app_config["api_settings"] = {
        "anthropic": {"model_defaults": {"claude-opus-4-7": {}}},
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        values = {
            "#settings-model-profile-max-tokens": "12000",
            "#settings-model-profile-reasoning-effort": "high",
            "#settings-model-profile-reasoning-summary": "auto",
            "#settings-model-profile-verbosity": "medium",
            "#settings-model-profile-thinking-effort": "xhigh",
            "#settings-model-profile-thinking-budget-tokens": "4096",
            "#settings-model-profile-streaming": "false",
        }
        for selector, value in values.items():
            screen.query_one(selector, Input).value = value

        text = _visible_text(screen)
        assert "Reasoning unavailable for Anthropic" in text
        assert screen.query_one("#settings-model-profile-reasoning-effort", Input).disabled is True
        assert screen.query_one("#settings-model-profile-reasoning-summary", Input).disabled is True
        assert screen.query_one("#settings-model-profile-verbosity", Input).disabled is True

        await pilot.click("#settings-save-category")

    assert saved == [
        (
            "api_settings.anthropic",
            "model_defaults",
            {
                "claude-opus-4-7": {
                    "max_tokens": 12000,
                    "thinking_effort": "xhigh",
                    "thinking_budget_tokens": 4096,
                    "streaming": False,
                },
            },
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_id", "value", "message"),
    (
        ("#settings-model-profile-temperature", "2.1", "Temperature must be between 0.0 and 2.0."),
        ("#settings-model-profile-top-p", "1.1", "Top P must be between 0.0 and 1.0."),
    ),
)
async def test_settings_provider_category_rejects_out_of_range_model_profile(
    monkeypatch,
    field_id,
    value,
    message,
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"model_defaults": {}}}
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one(field_id, Input).value = value

        await pilot.click("#settings-save-category")

        assert message in _visible_text(screen)

    assert saved == []
    assert app.app_config["api_settings"]["openai"]["model_defaults"] == {}


@pytest.mark.asyncio
async def test_settings_provider_category_rejects_invalid_streaming_profile(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"model_defaults": {}}}
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-profile-streaming", Input).value = "tru"

        await pilot.click("#settings-save-category")

        assert "Streaming must be true or false." in _visible_text(screen)

    assert saved == []
    assert app.app_config["api_settings"]["openai"]["model_defaults"] == {}


@pytest.mark.asyncio
async def test_settings_provider_model_switch_loads_selected_model_profile():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-model-profile-temperature", Input).value == "0.2"
        model = screen.query_one("#settings-model-value", Input)
        model.value = "gpt-4.1-mini"
        screen.handle_model_value_changed(Input.Changed(model, "gpt-4.1-mini"))

        assert screen.query_one("#settings-model-profile-temperature", Input).value == "0.45"
        assert screen.query_one("#settings-model-profile-top-p", Input).value == "0.9"
        assert screen.query_one("#settings-model-profile-streaming", Input).value == "false"


@pytest.mark.asyncio
async def test_settings_provider_model_profile_none_values_render_as_blank_inputs():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "model_defaults": {
                "gpt-4.1": {"temperature": None, "top_p": None, "streaming": None},
            },
        },
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-model-profile-temperature", Input).value == ""
        assert screen.query_one("#settings-model-profile-top-p", Input).value == ""
        assert screen.query_one("#settings-model-profile-streaming", Input).value == ""


@pytest.mark.asyncio
async def test_settings_provider_model_switch_does_not_save_unedited_profile(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        model = screen.query_one("#settings-model-value", Input)
        model.value = "gpt-4.1-mini"
        screen.handle_model_value_changed(Input.Changed(model, "gpt-4.1-mini"))

        await pilot.click("#settings-save-category")

    assert saved == [("chat_defaults", "model", "gpt-4.1-mini")]


@pytest.mark.asyncio
async def test_settings_provider_category_does_not_save_unedited_effective_defaults(monkeypatch):
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-save-category")

    assert saved == []
    assert app.app_config["chat_defaults"] == {}


@pytest.mark.asyncio
async def test_settings_provider_category_saves_only_dirty_provider_fields(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-value", Input).value = "gpt-4.1-mini"

        await pilot.click("#settings-save-category")

    assert saved == [("chat_defaults", "model", "gpt-4.1-mini")]
    assert app.app_config["chat_defaults"] == {
        "provider": "OpenAI",
        "model": "gpt-4.1-mini",
        "streaming": True,
        "temperature": 0.7,
    }


@pytest.mark.asyncio
async def test_settings_provider_category_saves_llamacpp_endpoint(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == "http://127.0.0.1:8080/v1"
        endpoint.value = "http://127.0.0.1:9099/v1"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.llama_cpp", "api_url", "http://127.0.0.1:9099/v1"),
    ]
    assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == "http://127.0.0.1:9099/v1"


@pytest.mark.asyncio
async def test_settings_provider_category_preserves_existing_endpoint_key(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == "https://api.openai.com/v1"
        endpoint.value = "https://proxy.example.com/v1"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.openai", "api_base_url", "https://proxy.example.com/v1"),
    ]
    assert app.app_config["api_settings"]["openai"]["api_base_url"] == "https://proxy.example.com/v1"


@pytest.mark.asyncio
async def test_settings_provider_save_button_works_with_endpoint_input_focus(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        endpoint.value = "https://proxy.example.com/v1"
        monkeypatch.setattr(screen, "_settings_text_entry_has_focus", lambda: True)

        screen.handle_save_category(
            Button.Pressed(screen.query_one("#settings-save-category", Button))
        )

    assert saved == [
        ("api_settings.openai", "api_base_url", "https://proxy.example.com/v1"),
    ]
    assert app.app_config["api_settings"]["openai"]["api_base_url"] == "https://proxy.example.com/v1"


@pytest.mark.asyncio
async def test_settings_provider_category_saves_credential_env_var(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
    }
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        assert env_var.value == "OPENAI_API_KEY"
        env_var.value = "CHATBOOK_OPENAI_API_KEY"
        screen.handle_provider_credential_env_var_changed(
            Input.Changed(env_var, env_var.value)
        )

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.openai", "api_key_env_var", "CHATBOOK_OPENAI_API_KEY"),
    ]
    assert app.app_config["api_settings"]["openai"]["api_key_env_var"] == (
        "CHATBOOK_OPENAI_API_KEY"
    )


@pytest.mark.asyncio
async def test_settings_provider_category_renders_local_api_key_setup_without_revealing_secret():
    app = _build_test_app()
    fake_key = "sk-test-visible-redaction-source"
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
    }
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY", "api_key": fake_key}
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        api_key = screen.query_one("#settings-provider-api-key", Input)
        assert getattr(api_key, "password", False) is True
        assert api_key.value == ""
        text = _visible_text(screen)
        assert "API key" in text
        assert "local config key saved" in text.lower()
        assert fake_key not in text


@pytest.mark.asyncio
async def test_settings_provider_category_saves_and_clears_local_api_key(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
    }
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        api_key = screen.query_one("#settings-provider-api-key", Input)
        api_key.value = "sk-test-new-local-key"
        screen.handle_provider_api_key_changed(Input.Changed(api_key, api_key.value))
        await pilot.click("#settings-save-category")

        assert ("api_settings.openai", "api_key", "sk-test-new-local-key") in saved
        assert app.app_config["api_settings"]["openai"]["api_key"] == "sk-test-new-local-key"

        await pilot.click("#settings-provider-api-key-clear")
        await pilot.click("#settings-save-category")

    assert ("api_settings.openai", "api_key", "") in saved
    assert app.app_config["api_settings"]["openai"]["api_key"] == ""


def test_settings_provider_api_key_validation_rejects_placeholder_values():
    assert input_validation_module.validate_provider_api_key("<API_KEY_HERE>") is False
    assert input_validation_module.validate_provider_api_key("sk-test-real-key") is True
    assert SettingsScreen._validate_provider_api_key("<API_KEY_HERE>") == (
        "API key looks like a placeholder; paste a real key."
    )


@pytest.mark.asyncio
async def test_settings_provider_category_rejects_invalid_credential_env_var(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
    }
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        env_var.value = "OPENAI_API_KEY;rm"
        screen.handle_provider_credential_env_var_changed(
            Input.Changed(env_var, env_var.value)
        )

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Credential env var must use environment variable syntax" in text

    assert saved == []
    assert app.app_config["api_settings"]["openai"]["api_key_env_var"] == "OPENAI_API_KEY"


@pytest.mark.asyncio
async def test_settings_provider_category_updates_existing_non_normalized_provider_section(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
    }
    app.app_config["api_settings"] = {
        "OpenAI": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        }
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == "https://api.openai.com/v1"
        endpoint.value = "https://proxy.example.com/v1"

        env_var = screen.query_one("#settings-provider-credential-env-var", Input)
        assert env_var.value == "OPENAI_API_KEY"
        env_var.value = "CHATBOOK_OPENAI_API_KEY"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.OpenAI", "api_base_url", "https://proxy.example.com/v1"),
        ("api_settings.OpenAI", "api_key_env_var", "CHATBOOK_OPENAI_API_KEY"),
    ]
    assert "openai" not in app.app_config["api_settings"]
    assert app.app_config["api_settings"]["OpenAI"] == {
        "api_base_url": "https://proxy.example.com/v1",
        "api_key_env_var": "CHATBOOK_OPENAI_API_KEY",
    }


@pytest.mark.asyncio
async def test_settings_provider_endpoint_validation_blocks_bad_url(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1"}
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-provider-endpoint-value", Input).value = "javascript:alert(1)"

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Endpoint must start with http:// or https://" in text

    assert saved == []
    assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == "http://127.0.0.1:8080/v1"


@pytest.mark.asyncio
async def test_settings_provider_endpoint_save_blocks_blank_provider(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        manual_provider = screen.query_one("#settings-provider-manual-value", Input)
        provider.value = "__manual__"
        manual_provider.value = ""
        screen.handle_provider_value_changed(Select.Changed(provider, "__manual__"))
        screen.handle_provider_manual_value_changed(
            Input.Changed(manual_provider, ""),
        )
        await pilot.pause()
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:9099/v1"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Provider is required." in text

    assert saved == []
    assert app.app_config["api_settings"] == {}


@pytest.mark.asyncio
async def test_settings_provider_category_blocks_empty_manual_provider_save(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        manual_provider = screen.query_one("#settings-provider-manual-value", Input)

        provider.value = "__manual__"
        manual_provider.value = ""
        screen.handle_provider_value_changed(Select.Changed(provider, "__manual__"))
        screen.handle_provider_manual_value_changed(
            Input.Changed(manual_provider, ""),
        )

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Provider is required." in text

    assert saved == []
    assert app.app_config["chat_defaults"]["provider"] == "OpenAI"


@pytest.mark.asyncio
async def test_settings_provider_blank_select_value_is_not_treated_as_provider():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)

        screen.handle_provider_value_changed(Select.Changed(provider, Select.BLANK))

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider"] is None
        assert "Select.BLANK" not in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_provider_revert_restores_provider_dependent_placeholders():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        },
        "llama_cpp": {},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        credential = screen.query_one("#settings-provider-credential-env-var", Input)

        assert endpoint.placeholder == "https://api.openai.com/v1"
        assert credential.placeholder == "OPENAI_API_KEY"

        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "llama_cpp"))
        assert endpoint.placeholder == "http://127.0.0.1:9099/v1"
        assert credential.placeholder == "No credential required"

        await pilot.click("#settings-revert-category")

        assert endpoint.placeholder == "https://api.openai.com/v1"
        assert credential.placeholder == "OPENAI_API_KEY"


@pytest.mark.asyncio
async def test_settings_provider_switch_does_not_save_stale_endpoint(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"},
        "llama_cpp": {},
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "llama_cpp"))

        assert screen.query_one("#settings-provider-endpoint-value", Input).value == ""
        credential = screen.query_one("#settings-provider-credential-env-var", Input)
        assert credential.value == ""
        assert credential.placeholder == "No credential required"
        await pilot.click("#settings-save-category")
        await pilot.click("#settings-save-category")

    assert ("api_settings.llama_cpp", "api_url", "https://api.openai.com/v1") not in saved
    assert saved == [("chat_defaults", "provider", "llama_cpp")]
    assert app.app_config["api_settings"]["llama_cpp"] == {}


@pytest.mark.asyncio
async def test_settings_provider_switch_updates_inspector_readiness():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"},
        "ollama": {"api_url": "http://localhost:11434/v1/chat/completions"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)

        provider.value = "ollama"
        screen.handle_provider_value_changed(Select.Changed(provider, "ollama"))
        await pilot.pause()

        assert screen.query_one("#settings-model-value", Input).value == ""
        assert (
            str(screen.query_one("#settings-provider-inspector-readiness", Static).renderable)
            == "Provider readiness: Ollama / not selected"
        )


@pytest.mark.asyncio
async def test_settings_provider_switch_selects_provider_default_model():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
        "ollama": {
            "api_url": "http://localhost:11434/v1/chat/completions",
            "model": "llama3",
        },
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)

        provider.value = "ollama"
        screen.handle_provider_value_changed(Select.Changed(provider, "ollama"))
        await pilot.pause()

        assert screen.query_one("#settings-model-value", Input).value == "llama3"
        text = _visible_text(screen)
        assert "Provider readiness: Ollama / llama3" in text
        assert "Provider source: Unsaved Settings draft" in text
        assert "Model source: Unsaved Settings draft" in text
        assert "Provider readiness: Ollama / gpt-4o" not in text


@pytest.mark.asyncio
async def test_settings_provider_switch_resets_staged_model_for_each_provider_transition():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
        "ollama": {
            "api_url": "http://localhost:11434/v1/chat/completions",
            "model": "llama3",
        },
        "anthropic": {"api_base_url": "https://api.anthropic.com/v1"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        model_input = screen.query_one("#settings-model-value", Input)

        provider.value = "ollama"
        screen.handle_provider_value_changed(Select.Changed(provider, "ollama"))
        await pilot.pause()

        assert model_input.value == "llama3"

        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()

        assert model_input.value == "gpt-4o"
        assert (
            str(screen.query_one("#settings-provider-inspector-readiness", Static).renderable)
            == "Provider readiness: OpenAI / gpt-4o"
        )

        provider.value = "anthropic"
        screen.handle_provider_value_changed(Select.Changed(provider, "anthropic"))
        await pilot.pause()

        assert model_input.value == ""
        assert (
            str(screen.query_one("#settings-provider-readiness", Static).renderable)
            == "Readiness: Anthropic / not selected"
        )
        assert (
            str(screen.query_one("#settings-provider-inspector-readiness", Static).renderable)
            == "Provider readiness: Anthropic / not selected"
        )


@pytest.mark.asyncio
async def test_settings_provider_detail_shows_field_guidance_and_readable_draft_state():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
        "ollama": {
            "api_url": "http://localhost:11434/v1/chat/completions",
            "model": "llama3",
        },
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)

        provider.value = "ollama"
        screen.handle_provider_value_changed(Select.Changed(provider, "ollama"))
        await pilot.pause()

        text = _visible_text(screen)
        assert (
            str(screen.query_one("#settings-category-providers-models", Button).label)
            == "> Providers & Models *"
        )
        assert "Provider source: Unsaved Settings draft" in text
        assert "Model source: Unsaved Settings draft" in text
        assert "Source: settings_draft" not in text
        assert "Model source: settings_draft" not in text
        assert "Endpoint key: api_settings.ollama.api_url" in text
        assert "Endpoint: http://localhost:11434/v1/chat/completions" in text
        assert "Endpoint: api_settings.ollama.api_url=http://localhost:11434" not in text
        assert (
            "No discovered models yet. Use Discover models after endpoint is configured."
            in text
        )

        screen.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()

        text = _visible_text(screen)
        assert "Focused setting: Endpoint" in text
        assert "Controls the provider endpoint used by Console generation." in text
        assert "Saved as: api_settings.ollama.api_url" in text
        assert "Validation: must start with http:// or https:// when set" in text


@pytest.mark.asyncio
async def test_settings_provider_custom_value_uses_manual_field_for_unknown_provider():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAi Typo", "model": "fake-model"}
    app.app_config["api_settings"] = {}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _select_settings_category(
            screen,
            pilot,
            SettingsCategoryId.PROVIDERS_MODELS,
            selector="#settings-provider-value",
        )

        assert screen.query_one("#settings-provider-value", Select).value == "__manual__"
        assert screen.query_one("#settings-provider-manual-value", Input).value == "OpenAi Typo"


@pytest.mark.asyncio
async def test_settings_provider_manual_entry_promotes_known_provider_to_catalog_select():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAi Typo", "model": "fake-model"}
    app.app_config["api_settings"] = {}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        manual_provider = screen.query_one("#settings-provider-manual-value", Input)

        manual_provider.value = "openai"
        screen.handle_provider_manual_value_changed(
            Input.Changed(manual_provider, "openai"),
        )

        assert provider.value == "openai"
        assert manual_provider.disabled is True
        assert manual_provider.value == ""


@pytest.mark.asyncio
async def test_settings_provider_test_blocks_unknown_provider():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAi Typo", "model": "fake-model"}
    app.app_config["api_settings"] = {}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Unknown provider")
        text = _visible_text(screen)

        assert "Unknown provider" in text
        assert "status=blocked" in text


@pytest.mark.asyncio
async def test_settings_provider_test_uses_api_settings_env_var_without_secret_leak(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "Groq", "model": "llama-3.3-70b-versatile"}
    app.app_config["api_settings"] = {
        "groq": {
            "api_key_env_var": "GROQ_API_KEY",
        }
    }
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-token")
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "GROQ_API_KEY=<redacted>")
        text = _visible_text(screen)

        assert "env:GROQ_API_KEY" in text
        assert "GROQ_API_KEY=<redacted>" in text
        assert "gsk-secret-token" not in text


@pytest.mark.asyncio
async def test_settings_provider_model_discovery_controls_render_for_eligible_provider():
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        },
    }
    app.llm_provider_catalog_scope_service = FakeSettingsModelDiscoveryScope(
        result=ModelDiscoveryResult(
            provider="openai",
            provider_list_key="openai",
            endpoint_fingerprint="fp-test",
            status="success",
            models=(_discovered_model("gpt-4o-mini"),),
        ),
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        discover = screen.query_one("#settings-discover-provider-models", Button)

        assert discover.disabled is False
        assert "Discover models from configured endpoint" in _visible_text(screen)
        assert (
            "Capabilities unknown until saved or verified; text chat is assumed."
            in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_settings_provider_model_discovery_saves_selected_runtime_models():
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        },
    }
    scope = FakeSettingsModelDiscoveryScope(
        result=ModelDiscoveryResult(
            provider="openai",
            provider_list_key="openai",
            endpoint_fingerprint="fp-test",
            status="success",
            models=(
                _discovered_model("gpt-4o-mini"),
                _discovered_model("gpt-4.1-nano"),
            ),
        ),
        persistence_result=PersistenceResult(
            provider="openai",
            provider_list_key="openai",
            status="saved",
            saved_model_ids=("gpt-4o-mini",),
            message="Saved 1 discovered model(s) to openai.",
        ),
    )
    app.llm_provider_catalog_scope_service = scope
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await _click_scrolled_settings_button(screen, pilot, "#settings-discover-provider-models")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Discovered 2 model(s) from openai.",
        )

        discovered_models = screen.query_one(
            "#settings-discovered-models-list",
            SelectionList,
        )
        discovered_models.select("gpt-4o-mini")

        await _click_scrolled_settings_button(screen, pilot, "#settings-save-discovered-provider-models")
        await _wait_for_settings_text(
            screen,
            pilot,
            "Saved 1 discovered model(s) to openai.",
        )

    assert scope.discover_calls == [
        {
            "mode": "local",
            "provider": "openai",
            "staged_settings": {
                "api_settings": {
                    "openai": {
                        "api_base_url": "https://api.openai.com/v1",
                        "api_key_env_var": "OPENAI_API_KEY",
                    },
                },
            },
        },
    ]
    assert scope.persist_calls == [
        {
            "mode": "local",
            "provider": "openai",
            "model_ids": ["gpt-4o-mini"],
        },
    ]
    assert app.providers_models["openai"] == ["gpt-4.1", "gpt-4o-mini"]


@pytest.mark.asyncio
async def test_settings_provider_model_discovery_shows_ambiguous_provider_recovery():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4.1"], "openai": ["gpt-4.1-mini"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "OpenAI": {"api_base_url": "https://api.openai.com/v1"},
        "openai": {"api_base_url": "https://proxy.example.com/v1"},
    }
    app.llm_provider_catalog_scope_service = FakeSettingsModelDiscoveryScope(
        result=ModelDiscoveryResult(
            provider="openai",
            provider_list_key=None,
            endpoint_fingerprint=None,
            status="error",
            error=ModelDiscoveryError(
                kind="ambiguous_provider_key",
                message="Multiple provider setting blocks match this provider.",
                recovery_hint=(
                    "Keep only one normalized api_settings block for this provider before "
                    "discovering models."
                ),
            ),
        ),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await _click_scrolled_settings_button(screen, pilot, "#settings-discover-provider-models")
        await _wait_for_settings_text(
            screen,
            pilot,
            (
                "Multiple provider entries match this provider. Rename or remove "
                "duplicates before saving discovered models."
            ),
        )
        status_text = str(
            screen.query_one("#settings-model-discovery-status", Static).renderable
        )

    assert "https://api.openai.com/v1" not in status_text
    assert "https://proxy.example.com/v1" not in status_text


@pytest.mark.asyncio
async def test_settings_provider_test_does_not_depend_on_console_sampling_defaults():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "Ollama",
        "model": "llama3",
        "streaming": True,
        "temperature": "not-a-number",
    }
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-profile-temperature", Input).value = "not-a-number"

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Provider test")
        text = _visible_text(screen)

        assert "Provider test" in text
        assert "status=ready" in text


def test_settings_provider_catalog_entries_do_not_import_chat_functions(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tldw_chatbook.Chat.Chat_Functions":
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    screen = SettingsScreen(_app(defaults={"provider": "openai", "model": "gpt-4.1"}))

    entries = screen._provider_catalog_entries()

    assert entries
    assert {entry.readiness_key for entry in entries} >= {"openai", "llama_cpp"}


@pytest.mark.parametrize(
    ("button_id", "expected"),
    [
        ("#settings-category-appearance", "Global visual defaults"),
        ("#settings-category-storage", "Config path"),
        ("#settings-category-privacy-security", "Encryption"),
        ("#settings-category-diagnostics", "Validate config"),
        ("#settings-category-advanced-config", "Raw TOML"),
    ],
)
@pytest.mark.asyncio
async def test_settings_first_slice_categories_have_real_content(button_id, expected):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        category_value = button_id.removeprefix("#settings-category-")
        await _select_settings_category(
            screen,
            pilot,
            category_value,
            expected_text=expected,
        )

        assert expected in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_storage_privacy_diagnostics_label_unsupported_mutations_as_wip():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        for button_id, expected in (
            ("#settings-category-privacy-security", "Credential mutation: unavailable/WIP"),
            ("#settings-category-diagnostics", "Diagnostics writes: unavailable/WIP"),
        ):
            await pilot.click(button_id)
            screen = _active_destination_screen(host)
            text = _visible_text(screen)

            assert expected in text
            assert screen.query_one("#settings-save-category", Button).disabled is True
            assert screen.query_one("#settings-revert-category", Button).disabled is True


@pytest.mark.asyncio
async def test_settings_privacy_security_renders_guided_redacted_posture(monkeypatch):
    app = _build_test_app()
    app.app_config.update(
        {
            "encryption": {"enabled": False},
            "api_settings": {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "api_key": DUMMY_REDACTION_CONFIG_VALUE,
                },
                "groq": {"api_key_env_var": "GROQ_API_KEY"},
            },
            "tldw_api": {"auth_token": DUMMY_REDACTION_SERVER_VALUE},
        }
    )
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_REDACTION_ENV_VALUE)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-privacy-security")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Privacy posture" in text
        assert "Credential sources" in text
        assert "Data boundary" in text
        assert "Config encryption: disabled" in text
        assert "Sensitive config fields: 2 present" in text
        assert "Provider env vars: 1 present / 1 missing / 2 configured" in text
        assert "Provider config secrets: 1 present" in text
        assert "Preferred source: environment variables" in text
        assert "Credential mutation: unavailable/WIP - password-gated flow required" in text
        assert "Open Providers & Models" in text
        assert "Open Advanced Config" in text
        assert "Environment variables are preferred for provider credentials." in text
        assert DUMMY_REDACTION_ENV_VALUE not in text
        assert DUMMY_REDACTION_CONFIG_VALUE not in text
        assert DUMMY_REDACTION_SERVER_VALUE not in text
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True


@pytest.mark.asyncio
async def test_settings_privacy_security_recovery_actions_navigate_to_existing_categories():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-privacy-security")
        screen = _active_destination_screen(host)

        await pilot.click("#settings-open-provider-credentials")
        assert screen.active_category == SettingsCategoryId.PROVIDERS_MODELS.value

        await pilot.click("#settings-category-privacy-security")
        await pilot.click("#settings-open-advanced-config")
        assert screen.active_category == SettingsCategoryId.ADVANCED_CONFIG.value


@pytest.mark.asyncio
async def test_settings_diagnostics_validate_and_reload_config_actions():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-diagnostics")
        screen = _active_destination_screen(host)
        await pilot.click("#settings-validate-config")
        await pilot.click("#settings-reload-config")
        text = _visible_text(screen)

        assert "Config validation: valid" in text
        assert "Config reload: loaded" in text


@pytest.mark.asyncio
async def test_settings_diagnostics_test_shortcut_runs_validate_and_reload():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-diagnostics")
        await pilot.press("t")
        screen = _active_destination_screen(host)
        await _wait_for_settings_text(screen, pilot, "Config reload: loaded")
        text = _visible_text(screen)

        assert "Config validation: valid" in text
        assert "Config reload: loaded" in text
        assert "No test action is available" not in text


def test_settings_diagnostics_combined_helper_validates_once(monkeypatch, tmp_path):
    class FakeAdapter:
        validate_calls = 0
        load_calls = 0

        def validate_config_file(self, path):
            FakeAdapter.validate_calls += 1
            return SettingsValidationResult(True, "valid once")

        def load(self, *, force_reload: bool = False):
            FakeAdapter.load_calls += 1
            return {"chat_defaults": {"provider": "OpenAI"}}

    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults]\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)

    screen = SettingsScreen(_build_test_app())
    validation_result, reload_result, loaded_config = screen._diagnostics_validation_and_reload_results()

    assert FakeAdapter.validate_calls == 1
    assert FakeAdapter.load_calls == 1
    assert "Config validation: valid - valid once" in validation_result
    assert f"Config source: {config_path}" in validation_result
    assert "Config reload: loaded" in reload_result
    assert f"Config source: {config_path}" in reload_result
    assert loaded_config == {"chat_defaults": {"provider": "OpenAI"}}


def test_settings_diagnostics_combined_helper_skips_reload_when_invalid(monkeypatch, tmp_path):
    class FakeAdapter:
        validate_calls = 0
        load_calls = 0

        def validate_config_file(self, path):
            FakeAdapter.validate_calls += 1
            return SettingsValidationResult(False, "broken TOML")

        def load(self, *, force_reload: bool = False):
            FakeAdapter.load_calls += 1
            return {"chat_defaults": {"provider": "OpenAI"}}

    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)

    screen = SettingsScreen(_build_test_app())
    validation_result, reload_result, loaded_config = screen._diagnostics_validation_and_reload_results()

    assert FakeAdapter.validate_calls == 1
    assert FakeAdapter.load_calls == 0
    assert "Config validation: invalid - broken TOML" in validation_result
    assert f"Config source: {config_path}" in validation_result
    assert "Config reload: failed - broken TOML" in reload_result
    assert f"Config source: {config_path}" in reload_result
    assert loaded_config is None


def test_settings_diagnostics_results_include_config_source_and_redact_errors(
    monkeypatch,
    tmp_path,
):
    class FakeAdapter:
        def validate_config_file(self, path):
            return SettingsValidationResult(
                False,
                f"OPENAI_API_KEY={DUMMY_REDACTION_CONFIG_VALUE} parse failure",
            )

        def load(self, *, force_reload: bool = False):
            raise AssertionError("invalid config must not reload")

    config_path = tmp_path / "config.toml"
    config_path.write_text("OPENAI_API_KEY='raw'\n[broken", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)

    screen = SettingsScreen(_build_test_app())
    validation_result, reload_result, loaded_config = (
        screen._diagnostics_validation_and_reload_results()
    )

    assert f"Config source: {config_path}" in validation_result
    assert f"Config source: {config_path}" in reload_result
    assert DUMMY_REDACTION_CONFIG_VALUE not in validation_result
    assert DUMMY_REDACTION_CONFIG_VALUE not in reload_result
    assert "OPENAI_API_KEY=<redacted>" in validation_result
    assert "OPENAI_API_KEY=<redacted>" in reload_result
    assert loaded_config is None


def test_settings_diagnostics_invalid_config_source_does_not_duplicate_error(
    monkeypatch,
):
    screen = SettingsScreen(_build_test_app())

    def raise_invalid_config_path():
        raise ValueError(f"OPENAI_API_KEY={DUMMY_REDACTION_CONFIG_VALUE} path failure")

    monkeypatch.setattr(screen, "_config_path", raise_invalid_config_path)

    validation_result, reload_result, loaded_config = (
        screen._diagnostics_validation_and_reload_results()
    )

    assert validation_result.endswith("\nConfig source: invalid")
    assert reload_result.endswith("\nConfig source: invalid")
    assert "Config source: invalid - OPENAI_API_KEY=<redacted>" not in validation_result
    assert "Config source: invalid - OPENAI_API_KEY=<redacted>" not in reload_result
    assert DUMMY_REDACTION_CONFIG_VALUE not in validation_result
    assert DUMMY_REDACTION_CONFIG_VALUE not in reload_result
    assert loaded_config is None


def test_settings_diagnostics_unexpected_config_path_errors_are_not_masked(
    monkeypatch,
):
    screen = SettingsScreen(_build_test_app())

    def raise_unexpected_config_path_error():
        raise AssertionError("programming regression")

    monkeypatch.setattr(screen, "_config_path", raise_unexpected_config_path_error)

    with pytest.raises(AssertionError, match="programming regression"):
        screen._diagnostics_validation_and_reload_results()


def test_settings_diagnostics_strictly_reports_corrupt_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    screen = SettingsScreen(app)

    assert "Config validation: invalid" in screen._validate_current_config()
    assert "Config reload: failed" in screen._reload_current_config()


def test_settings_storage_check_reports_path_readiness(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "config.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = SimpleNamespace(
        app_config={},
        user_data_dir=data_dir,
        notifications_db_path=data_dir / "notifications.db",
        subscriptions_db_path=data_dir / "watchlists.db",
        workspaces_db_path=data_dir / "workspaces.db",
    )
    screen = SettingsScreen(app)

    result = screen._storage_check_results()

    assert result[0] == "Storage check: complete"
    assert "Config path parent: missing - parent writable" in result
    assert "User data directory: writable" in result
    assert "Notifications DB parent: writable" in result
    assert "Watchlists DB parent: writable" in result
    assert "Workspaces DB parent: writable" in result


def test_settings_storage_check_reports_invalid_config_path(monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    result = screen._storage_check_results()

    assert result[0] == "Storage check: complete"
    assert any(row.startswith("Config path parent: invalid") for row in result)


def test_settings_storage_check_includes_configured_fallback_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    result = screen._storage_check_results()

    assert any(row.startswith("User data directory:") for row in result)
    assert any(row.startswith("Notifications DB parent:") for row in result)
    assert any(row.startswith("Watchlists DB parent:") for row in result)
    assert any(row.startswith("Workspaces DB parent:") for row in result)


def test_settings_storage_check_does_not_create_missing_config(monkeypatch, tmp_path):
    config_path = tmp_path / "missing" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    result = screen._storage_check_results()

    assert result[0] == "Storage check: complete"
    assert not config_path.exists()
    assert not config_path.parent.exists()


def test_settings_storage_check_reports_file_targets_as_invalid(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    file_ancestor = tmp_path / "not-a-dir"
    file_ancestor.write_text("not a directory", encoding="utf-8")
    existing_file = tmp_path / "data-dir"
    existing_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    file_parent_result = screen._storage_path_status(
        "Workspaces DB parent",
        file_ancestor / "workspaces.db",
        directory=False,
    )
    directory_result = screen._storage_path_status(
        "User data directory",
        existing_file,
        directory=True,
    )

    assert file_parent_result == "Workspaces DB parent: invalid - expected directory"
    assert directory_result == "User data directory: invalid - expected directory"


def test_settings_storage_check_reports_empty_path_as_unconfigured(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    assert screen._storage_path_status("User data directory", None, directory=True) == (
        "User data directory: not configured"
    )
    assert screen._storage_path_status("User data directory", "", directory=True) == (
        "User data directory: not configured"
    )


def test_settings_storage_check_reports_missing_path_with_parent_readiness(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    missing_dir = tmp_path / "missing-data-dir"
    missing_db = tmp_path / "missing-db" / "chatbook.db"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    directory_status = screen._storage_path_status(
        "User data directory",
        missing_dir,
        directory=True,
    )
    file_status = screen._storage_path_status(
        "Notifications DB parent",
        missing_db,
        directory=False,
    )

    assert directory_status == "User data directory: missing - parent writable"
    assert file_status == "Notifications DB parent: missing - parent writable"


def test_settings_privacy_check_reports_redacted_secret_status(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_REDACTION_ENV_VALUE)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = SimpleNamespace(
        app_config={
            "encryption": {"enabled": False},
            "api_settings": {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "api_key": DUMMY_REDACTION_CONFIG_VALUE,
                },
                "groq": {"api_key_env_var": "GROQ_API_KEY"},
            },
            "chat_defaults": {"max_tokens": 4096},
            "tldw_api": {"auth_token": DUMMY_REDACTION_SERVER_VALUE},
        }
    )
    screen = SettingsScreen(app)

    result = screen._privacy_check_results()
    text = "\n".join(result)

    assert result[0] == "Privacy check: complete"
    assert "Config encryption: disabled" in result
    assert "Sensitive config fields: 2 present" in result
    assert "Provider env vars: 1 present / 1 missing / 2 configured" in result
    assert "Redaction: active; raw secret values hidden" in result
    assert DUMMY_REDACTION_ENV_VALUE not in text
    assert DUMMY_REDACTION_CONFIG_VALUE not in text
    assert DUMMY_REDACTION_SERVER_VALUE not in text


def test_settings_privacy_check_reports_key_sources_and_data_boundaries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_REDACTION_ENV_VALUE)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = SimpleNamespace(
        app_config={
            "encryption": {"enabled": True},
            "api_settings": {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "api_key": DUMMY_REDACTION_CONFIG_VALUE,
                },
                "groq": {"api_key_env_var": "GROQ_API_KEY"},
            },
            "tldw_api": {"auth_token": DUMMY_REDACTION_SERVER_VALUE},
        }
    )
    screen = SettingsScreen(app)

    result = screen._privacy_check_results()
    text = "\n".join(result)

    assert "Sensitive config fields: 2 present" in result
    assert (
        "Provider key source: environment 1 present / 1 missing; "
        "provider config secrets 1 present"
    ) in result
    assert "Data boundary: local data stays local unless explicit server handoff or sync is enabled" in result
    assert "Server boundary: server tokens are reported as configured/missing only" in result
    assert DUMMY_REDACTION_ENV_VALUE not in text
    assert DUMMY_REDACTION_CONFIG_VALUE not in text
    assert DUMMY_REDACTION_SERVER_VALUE not in text


def test_settings_privacy_secret_count_ignores_non_secret_numeric_token_limits():
    app = SimpleNamespace(
        app_config={
            "api_settings": {
                "openai": {
                    "api_key": DUMMY_REDACTION_CONFIG_VALUE,
                    "max_tokens": 4096,
                },
            },
            "chat_defaults": {
                "max_tokens": 2048,
                "token_budget": 512,
            },
        }
    )
    screen = SettingsScreen(app)

    result = screen._privacy_check_results()

    assert "Sensitive config fields: 1 present" in result


@pytest.mark.asyncio
async def test_settings_storage_test_shortcut_runs_safety_check(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "config.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    app.user_data_dir = data_dir
    app.notifications_db_path = data_dir / "notifications.db"
    app.subscriptions_db_path = data_dir / "watchlists.db"
    app.workspaces_db_path = data_dir / "workspaces.db"
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-check-storage")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        await pilot.press("t")
        await _wait_for_settings_text(screen, pilot, "Storage check: complete")
        text = _visible_text(screen)

        assert "Storage defaults are valid. Changes apply on next app launch." in text
        assert "Base data directory:" in text
        assert "Workspaces DB:" in text
        assert "No test action is available" not in text
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True


@pytest.mark.asyncio
async def test_settings_privacy_security_test_shortcut_runs_privacy_check(monkeypatch):
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "api_key": DUMMY_REDACTION_CONFIG_VALUE,
        }
    }
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_REDACTION_ENV_VALUE)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-privacy-security")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-check-privacy")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        await pilot.press("t")
        await _wait_for_settings_text(screen, pilot, "Privacy check: complete")
        text = _visible_text(screen)

        assert "Provider env vars: 1 present / 0 missing / 1 configured" in text
        assert "Sensitive config fields: 1 present" in text
        assert DUMMY_REDACTION_ENV_VALUE not in text
        assert DUMMY_REDACTION_CONFIG_VALUE not in text
        assert "No test action is available" not in text
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True


@pytest.mark.asyncio
async def test_settings_privacy_shortcut_passes_stable_config_snapshot_to_worker():
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "api_key": DUMMY_REDACTION_CONFIG_VALUE,
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-privacy-security")
        screen = _active_destination_screen(host)
        captured = {}

        def fake_privacy_worker(app_config_snapshot):
            captured["snapshot"] = app_config_snapshot

        screen._privacy_check_worker = fake_privacy_worker
        screen.action_settings_test_category()

        snapshot = captured["snapshot"]
        assert snapshot == app.app_config
        assert snapshot is not app.app_config
        assert snapshot["api_settings"] is not app.app_config["api_settings"]
        app.app_config["api_settings"]["openai"]["api_key"] = "changed-after-dispatch"
        assert snapshot["api_settings"]["openai"]["api_key"] == DUMMY_REDACTION_CONFIG_VALUE


def test_settings_config_path_validates_env_override(monkeypatch):
    app = _build_test_app()
    screen = SettingsScreen(app)
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")

    with pytest.raises(ValueError):
        screen._config_path()


def test_settings_overview_config_path_label_hides_local_directory(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    value = screen._config_path_overview_value()

    assert "Override config: config.toml" in value
    assert str(config_path) not in value
    assert str(tmp_path) not in value


def test_settings_advanced_config_save_reports_invalid_env_override(monkeypatch):
    app = SimpleNamespace(app_config={})
    screen = SettingsScreen(app)
    text = "[chat_defaults]\nprovider = \"Ollama\"\n"
    screen._advanced_config_validated_text = text
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")

    result = screen._save_advanced_config_text(text)

    assert "Advanced config save: failed" in result
    assert "dangerous pattern" in result


@pytest.mark.asyncio
async def test_settings_advanced_config_shows_raw_editor_and_safety_actions():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Raw TOML bypasses guided validation" in text
        assert screen.query_one("#settings-advanced-config-editor", TextArea)
        assert screen.query_one("#settings-advanced-validate-config")
        save_button = screen.query_one("#settings-advanced-save-config")
        assert save_button.disabled
        assert "Last validated: not validated" in text
        assert "Save blocked until the current text validates" in text


@pytest.mark.asyncio
async def test_settings_advanced_config_keeps_safety_actions_before_raw_editor():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-advanced-config-actions")
        await _wait_for_selector(screen, pilot, "#settings-advanced-config-editor")
        actions = screen.query_one("#settings-advanced-config-actions")
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)

        assert actions.region.height > 0
        assert actions.region.width > 0
        assert actions.region.y < editor.region.y


@pytest.mark.asyncio
async def test_settings_advanced_config_uses_editor_owned_scroll_region():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-advanced-config-editor")
        detail_pane = screen.query_one("#settings-detail-pane")
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)

        assert not isinstance(detail_pane, VerticalScroll)
        assert editor.region.height > 0


def test_settings_pane_widths_are_owned_by_stylesheet_not_inline_python():
    source = inspect.getsource(SettingsScreen.compose_content)
    css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text(encoding="utf-8")

    assert ".styles.width" not in source
    for selector, expected_width in (
        ("#settings-category-pane", "3fr"),
        ("#settings-detail-pane", "6fr"),
        ("#settings-impact-pane", "2fr"),
    ):
        marker = f"{selector} {{"
        block_start = css.index(marker)
        block_end = css.index("}", block_start)
        block = css[block_start:block_end]

        assert f"width: {expected_width};" in block

    card_start = css.index("#settings-advanced-config-card {")
    card_end = css.index("}", card_start)
    card_block = css[card_start:card_end]
    assert "height: 1fr;" in card_block
    assert "min-height: 0;" in card_block


@pytest.mark.asyncio
async def test_settings_advanced_config_blocks_invalid_toml_and_redacts_secret():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "OPENAI_API_KEY=sk-secret-token\n[broken"

        await pilot.click("#settings-advanced-validate-config")
        await _wait_for_settings_text(screen, pilot, "Advanced config validation: invalid")
        text = _visible_text(screen)

        assert "Advanced config validation: invalid" in text
        assert "sk-secret-token" not in text


@pytest.mark.asyncio
async def test_settings_advanced_config_blocks_non_mapping_toml_on_save():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "42"

        save_button = screen.query_one("#settings-advanced-save-config")

        assert save_button.disabled
        assert "top-level TOML value must be a table" in screen._save_advanced_config_text("42")


@pytest.mark.asyncio
async def test_settings_advanced_config_saves_atomically_with_backup(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults]\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"

        assert screen.query_one("#settings-advanced-save-config").disabled

        await pilot.click("#settings-advanced-validate-config")
        await _wait_for_settings_text(screen, pilot, "Advanced config validation: valid")
        assert not screen.query_one("#settings-advanced-save-config").disabled
        await pilot.click("#settings-advanced-save-config")
        await _wait_for_settings_text(screen, pilot, "Advanced config save: saved")
        text = _visible_text(screen)

        assert "Advanced config save: saved" in text
        assert "Last validated: current text" in text

    assert config_path.read_text(encoding="utf-8") == (
        "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"
    )
    assert config_path.with_suffix(".toml.bak").exists()
    assert app.app_config["chat_defaults"]["provider"] == "Ollama"
    assert app.app_config["chat_defaults"]["model"] == "llama3"


@pytest.mark.asyncio
async def test_settings_advanced_config_loads_backup_preview_without_saving(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    backup_path = tmp_path / "config.toml.bak"
    current_text = "[chat_defaults]\nprovider = \"OpenAI\"\n"
    backup_text = "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"
    config_path.write_text(current_text, encoding="utf-8")
    backup_path.write_text(backup_text, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)

        assert editor.text == current_text

        await pilot.click("#settings-advanced-load-backup")
        await _wait_for_settings_text(screen, pilot, "Advanced config recovery: loaded backup preview")
        text = _visible_text(screen)

        assert editor.text == backup_text
        assert config_path.read_text(encoding="utf-8") == current_text
        assert screen.query_one("#settings-advanced-save-config").disabled
        assert "validate before save" in text


def test_settings_advanced_config_backup_preview_handles_config_path_errors(monkeypatch):
    screen = SettingsScreen(_build_test_app())

    def raise_config_path_error():
        raise RuntimeError(f"OPENAI_API_KEY={DUMMY_REDACTION_CONFIG_VALUE} path failure")

    monkeypatch.setattr(screen, "_config_path", raise_config_path_error)

    result = screen._load_advanced_backup_preview()

    assert result.startswith("Advanced config recovery: failed")
    assert "OPENAI_API_KEY=<redacted>" in result
    assert DUMMY_REDACTION_CONFIG_VALUE not in result


@pytest.mark.asyncio
async def test_settings_advanced_config_load_backup_reports_decode_failure(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.toml"
    backup_path = tmp_path / "config.toml.bak"
    current_text = "[chat_defaults]\nprovider = \"OpenAI\"\n"
    config_path.write_text(current_text, encoding="utf-8")
    backup_path.write_bytes(b"\xff\xfe\xfa")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)

        await pilot.click("#settings-advanced-load-backup")
        await _wait_for_settings_text(screen, pilot, "Advanced config recovery: failed")

        assert editor.text == current_text
        assert "invalid start byte" in screen._advanced_config_result
        assert screen.query_one("#settings-advanced-save-config").disabled


def test_settings_advanced_config_load_backup_handler_uses_worker(monkeypatch):
    screen = SettingsScreen(_build_test_app())
    calls = []

    def fail_direct_load():
        raise AssertionError("backup loading should not run in the button handler")

    def fake_worker():
        calls.append("worker")

    monkeypatch.setattr(screen, "_load_advanced_backup_preview", fail_direct_load)
    monkeypatch.setattr(screen, "_advanced_load_backup_worker", fake_worker, raising=False)

    event = SimpleNamespace(stop=lambda: calls.append("stop"))

    screen.handle_advanced_load_backup(event)

    assert calls == ["stop", "worker"]


@pytest.mark.asyncio
async def test_settings_advanced_config_guided_path_buttons_escape_raw_toml():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-advanced-open-providers-models", Button)
        assert screen.query_one("#settings-advanced-open-console-behavior", Button)
        assert screen.query_one("#settings-advanced-open-diagnostics", Button)

        await pilot.click("#settings-advanced-open-providers-models")
        await _wait_for_settings_text(screen, pilot, "Provider catalog")

        assert screen.active_category == SettingsCategoryId.PROVIDERS_MODELS.value
        assert "Selected category: Providers & Models" in _visible_text(screen)


def test_settings_advanced_config_new_file_save_reports_no_backup(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = SimpleNamespace(app_config={})
    screen = SettingsScreen(app)
    text = "[chat_defaults]\nprovider = \"Ollama\"\n"
    screen._advanced_config_validated_text = text

    result = screen._save_advanced_config_text(text)

    assert "Advanced config save: saved" in result
    assert "backup: none (new file)" in result
    assert config_path.exists()
    assert not config_path.with_suffix(".toml.bak").exists()
