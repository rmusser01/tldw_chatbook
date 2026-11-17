"""Original private-profile composition and cleanup for the Console backend."""

import sqlite3
from contextlib import ExitStack
from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test


def _settled_profile():
    from tldw_chatbook import config
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.app_services import (
        open_chachanotes_db,
        open_prompts_db,
        open_workspaces_db,
    )
    from tldw_chatbook.runtime_policy.bootstrap import (
        default_runtime_policy_path,
        synchronize_runtime_source_state_with_app_config,
    )
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

    configuration = config.load_settings()
    config_path = config.get_cli_config_path()
    data_directory = config.get_user_data_dir()
    policy_store = RuntimeSourceStateStore(
        default_runtime_policy_path(),
        application_owned_directory=config.application_owned_config_directory(
            config_path
        ),
    )
    state = policy_store.load()
    synchronized = synchronize_runtime_source_state_with_app_config(
        state, configuration
    )
    if synchronized != state:
        policy_store.save(synchronized)
    targets = ConfiguredServerTargetStore(data_directory / "mcp_server_targets.json")
    targets.upsert_legacy_config_target(configuration)
    open_chachanotes_db(
        config.get_chachanotes_db_path(),
        client_id=config.CLI_APP_CLIENT_ID,
        seed_builtin_content=True,
    ).close()
    open_prompts_db(
        config.get_prompts_db_path(), client_id="tldw_tui_client_v1"
    ).close()
    parent_db = open_workspaces_db(
        config.get_workspaces_db_path(), client_id=config.CLI_APP_CLIENT_ID
    )
    try:
        LocalWorkspaceRegistryService(parent_db).ensure_default_workspace()
    finally:
        parent_db.close()
    return config_path, data_directory, policy_store, synchronized, parent_db


@private_profile_test
@pytest.mark.asyncio
async def test_backend_profile_reuses_original_owners_on_the_selected_files(request):
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    config_path, data_directory, _store, state, parent_db = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=config_path, expected_data_directory=data_directory
    )
    try:
        storage = profile.storage
        assert profile.configuration_identity == config.current_config_identity()
        assert profile.policy_context.state == state
        assert profile.policy_enforcer.current_state() is profile.policy_context.state
        assert storage.chachanotes.db_path == config.get_chachanotes_db_path()
        assert storage.chachanotes.client_id == config.CLI_APP_CLIENT_ID
        assert storage.prompts.db_path == config.get_prompts_db_path()
        assert storage.prompts.client_id == "tldw_tui_client_v1"
        assert storage.workspaces is profile.workspace_registry.db
        assert storage.workspaces is not parent_db
        assert storage.workspaces.db_path == parent_db.db_path
        assert (
            profile.workspace_registry.tool_profile_guard is profile.tool_profile_guard
        )
        assert profile.tool_profile_guard.active_guard is None
        assert (
            profile.conversation_scope_service.local_service.db is storage.chachanotes
        )
        assert (
            profile.conversation_scope_service.sync_scope_service is profile.sync_scope
        )
        assert profile.citation_repository.db is storage.chachanotes
        assert profile.prompt_scope.local_service.prompt_db is storage.prompts
        assert profile.skills_scope.local_service is profile.local_skills
        assert profile.local_skills._trust_service_factory is None
        assert profile.local_skills._plugin_service_factory is None
        assert profile.local_skills._builtin_disabled_loader is None
        control_plane = profile.control_plane
        assert control_plane.local_service.policy_enforcer is profile.policy_enforcer
        assert (
            control_plane.local_service.runtime_delegate._storage_provider() is storage
        )
        permission = control_plane.permission_store
        assert type(permission) is MCPPermissionStore
        assert permission is control_plane._permission_store
        assert permission.path == data_directory / "mcp_permissions.json"
        assert profile.server_context._cached_client is None
    finally:
        await profile.aclose()


@private_profile_test
@pytest.mark.asyncio
async def test_backend_profile_refuses_path_disagreement_and_unsynchronized_policy(
    request,
):
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )

    config_path, data_directory, policy_store, state, _parent_db = _settled_profile()
    with pytest.raises(RuntimeError, match="config selection changed"):
        open_console_backend_profile(
            expected_config_path=config_path.with_name("other.toml"),
            expected_data_directory=data_directory,
        )
    with pytest.raises(RuntimeError, match="data directory selection changed"):
        open_console_backend_profile(
            expected_config_path=config_path,
            expected_data_directory=data_directory.with_name("other-profile"),
        )

    stale = replace(state, last_known_server_label="stale selection")
    policy_store.save(stale)
    with pytest.raises(RuntimeError, match="runtime policy setup is not current"):
        open_console_backend_profile(
            expected_config_path=config_path, expected_data_directory=data_directory
        )
    assert policy_store.load() == stale


@private_profile_test
@pytest.mark.asyncio
async def test_backend_profile_closes_original_databases_on_partial_acquisition(
    request, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.participants import core_operation
    from tldw_chatbook.Chat import console_execution_profile as profile_module

    config_path, data_directory, _store, _state, _parent_db = _settled_profile()
    acquired = []
    borrower = ExitStack()
    open_chat = profile_module.open_chachanotes_db
    open_prompts = profile_module.open_prompts_db

    def track_chat(*args, **kwargs):
        database = open_chat(*args, **kwargs)
        acquired.append((database, database.get_connection()))
        return database

    def track_prompts(*args, **kwargs):
        database = open_prompts(*args, **kwargs)
        acquired.append((database, database.get_connection()))
        return database

    refusal = RuntimeError("prompt scope acquisition failed")

    def fail_prompt_scope(**_kwargs):
        borrower.enter_context(core_operation(acquired[1][0]))
        raise refusal

    monkeypatch.setattr(profile_module, "open_chachanotes_db", track_chat)
    monkeypatch.setattr(profile_module, "open_prompts_db", track_prompts)
    monkeypatch.setattr(profile_module, "build_prompt_scope_service", fail_prompt_scope)
    try:
        with borrower:
            with pytest.raises(
                profile_module.ConsoleBackendProfileOpenFailure
            ) as caught:
                profile_module.open_console_backend_profile(
                    expected_config_path=config_path,
                    expected_data_directory=data_directory,
                )
            failure = caught.value
            assert failure.acquisition_error is refusal
            (chat_db, chat_handle), (prompts_db, prompts_handle) = acquired
            assert failure.storage.chachanotes is chat_db
            assert failure.storage.prompts is prompts_db
            assert (
                type(failure.cleanup_error)
                is profile_module.ConsoleBackendProfileCleanupUncertain
            )
            assert failure.cleanup_error.repository is prompts_db
            assert failure.cleanup_error.connection is prompts_handle
            assert sqlite3.Connection.in_transaction.__get__(prompts_handle) is False
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                chat_handle.execute("SELECT 1")
        prompts_db.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            prompts_handle.execute("SELECT 1")
    finally:
        for database, _ in acquired:
            database.close()
