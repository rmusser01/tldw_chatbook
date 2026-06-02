import inspect
import re
import time
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select, Static, TextArea

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
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
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Workspaces.display_state import LIBRARY_WORKSPACE_VISIBILITY_COPY
from tldw_chatbook.Workspaces.models import (
    WorkspaceAuthority,
    WorkspaceRecord,
    WorkspaceSyncStatus,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

DUMMY_REDACTION_ENV_VALUE = "redaction-fixture-env-value"
DUMMY_REDACTION_CONFIG_VALUE = "redaction-fixture-config-value"
DUMMY_REDACTION_SERVER_VALUE = "redaction-fixture-server-value"


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


async def _wait_for_settings_text(screen, pilot, expected_text: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected_text in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {expected_text!r}. Visible text: {_visible_text(screen)}")


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


@pytest.mark.asyncio
async def test_settings_defaults_to_overview_category():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
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
    assert not records_by_category[SettingsCategoryId.STORAGE].writes_allowed
    assert records_by_category[SettingsCategoryId.STORAGE].read_only_reason

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


def test_settings_domain_category_contracts_are_explicit_and_read_only():
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
        assert contract.settings_can_mutate is False
        assert contract.follow_up

    library_contract = contracts[SettingsCategoryId.LIBRARY_RAG]
    library_copy = " ".join(
        (
            *(value for _label, value in library_contract.rows),
            library_contract.follow_up,
        )
    )
    assert "citations" in library_copy
    assert "snippets" in library_copy


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
        assert not record.writes_allowed
        assert record.read_only_reason
        assert "Open" in record.recovery_copy
        assert record.runtime_owner


@pytest.mark.asyncio
async def test_settings_domain_category_renders_read_only_owner_contract():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await pilot.click("#settings-category-library-rag")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Library & RAG" in text
        assert "Owner destination: Library" in text
        assert "Settings mode: read-only defaults/status contract" in text
        assert "Citation/snippet defaults" in text
        assert "follow-up" in text
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        search = screen.query_one("#settings-category-search", Input)
        search.value = "mcp"
        search.focus()
        await pilot.press("enter")
        await pilot.click("#settings-category-mcp-defaults")
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


def test_settings_manual_sync_run_worker_uses_main_event_loop_async_worker():
    worker = SettingsScreen.__dict__["_manual_sync_run_worker"]
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
        assert "Manual provider entry remains available for custom/local aliases." in text


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
async def test_settings_inspector_uses_category_specific_guidance():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Affected config: config file path, local database paths, media storage roots" in text
        assert "Recovery: verify paths, reload config, then restart only if storage roots changed" in text
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
        await pilot.press("/")
        screen = _active_destination_screen(host)
        search = screen.query_one("#settings-category-search", Input)

        assert search.has_focus

        await pilot.press(*"priv")
        await pilot.pause()

        assert screen.query_one("#settings-category-privacy-security").display
        assert not screen.query_one("#settings-category-providers-models").display

        await pilot.press("enter")
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.PRIVACY_SECURITY.value
        assert "Privacy & Security" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_category_search_reports_ranked_matches_and_enter_target():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("/")
        screen = _active_destination_screen(host)

        await pilot.press(*"priv")
        await pilot.pause()

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
        await pilot.press("/")
        screen = _active_destination_screen(host)

        await pilot.press(*"zzz")
        await pilot.pause()

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

        assert "Console paste collapse: Disabled: collapse large pastes" in _visible_text(screen)


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
        "max_tokens": 2048,
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
        assert screen.query_one("#settings-console-default-max-tokens", Input).value == "2048"
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
        max_tokens = screen.query_one("#settings-console-default-max-tokens", Input)

        streaming.value = "false"
        screen.handle_console_default_streaming_changed(Input.Changed(streaming, streaming.value))
        temperature.value = "0.33"
        screen.handle_console_default_temperature_changed(Input.Changed(temperature, temperature.value))
        top_p.value = "0.81"
        screen.handle_console_default_top_p_changed(Input.Changed(top_p, top_p.value))
        max_tokens.value = "2048"
        screen.handle_console_default_max_tokens_changed(Input.Changed(max_tokens, max_tokens.value))

        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [
        {
            "chat_defaults": {
                "streaming": False,
                "temperature": 0.33,
                "top_p": 0.81,
                "max_tokens": 2048,
            }
        }
    ]
    assert app.app_config["chat_defaults"] == {
        "streaming": False,
        "temperature": 0.33,
        "top_p": 0.81,
        "max_tokens": 2048,
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

        await pilot.click("#settings-category-storage")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: Storage is read-only." in _visible_text(screen)


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
        assert "Source: chat_defaults" in text
        assert screen.query_one("#settings-provider-value", Input).value == "llama_cpp"
        assert screen.query_one("#settings-model-value", Input).value == "qwen"


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
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
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
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
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
        screen.query_one("#settings-provider-value", Input).value = "llama_cpp"
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
        provider = screen.query_one("#settings-provider-value", Input)
        provider.value = ""
        screen.handle_provider_value_changed(Input.Changed(provider, ""))
        await pilot.pause()
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:9099/v1"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Provider is required before saving an endpoint." in text

    assert saved == []
    assert app.app_config["api_settings"] == {}


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
        provider = screen.query_one("#settings-provider-value", Input)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Input.Changed(provider, "llama_cpp"))

        assert screen.query_one("#settings-provider-endpoint-value", Input).value == ""
        await pilot.click("#settings-save-category")
        await pilot.click("#settings-save-category")

    assert ("api_settings.llama_cpp", "api_url", "https://api.openai.com/v1") not in saved
    assert saved == [("chat_defaults", "provider", "llama_cpp")]
    assert app.app_config["api_settings"]["llama_cpp"] == {}


@pytest.mark.asyncio
async def test_settings_provider_test_blocks_unknown_provider():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAi Typo", "model": "fake-model"}
    app.app_config["api_settings"] = {}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
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
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "env:GROQ_API_KEY" in text
        assert "GROQ_API_KEY=<redacted>" in text
        assert "gsk-secret-token" not in text


@pytest.mark.asyncio
async def test_settings_provider_test_does_not_depend_on_console_sampling_defaults():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "Ollama",
        "model": "llama3",
        "streaming": True,
        "temperature": "not-a-number",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-profile-temperature", Input).value = "not-a-number"

        await pilot.click("#settings-test-provider")
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
        ("#settings-category-appearance", "Open Appearance"),
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
        await pilot.click(button_id)
        screen = _active_destination_screen(host)

        assert expected in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_storage_privacy_diagnostics_label_unsupported_mutations_as_wip():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        for button_id, expected in (
            ("#settings-category-storage", "Storage mutation: unavailable/WIP"),
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

        assert "Config path parent: writable" in text
        assert "User data directory: writable" in text
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
        await _wait_for_settings_text(screen, pilot, "Provider Model Endpoint")

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
