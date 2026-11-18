"""Exercise a bound project's prepared turn under its original file-scoped producer."""

import sys
from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test


def _commit_workspace_prompt(project):
    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import _acceptance
    from Tests.uat_project_chat_support import MODEL
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.participants import core_operation
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleDispatchCheckpointState, ConsoleEgressClass
    from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve, ConsoleLibraryPolicyCandidate
    from tldw_chatbook.Chat.console_prompt_mixin_models import MixinAttemptSnapshot, MixinAttemptWrite
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.profile_writer import start_profile_writer
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.app_services import open_workspaces_db

    host = start_profile_writer(config.get_user_data_dir())
    database = CharactersRAGDB(config.get_chachanotes_db_path(), config.CLI_APP_CLIENT_ID)
    workspaces = open_workspaces_db(config.get_workspaces_db_path(), client_id=config.CLI_APP_CLIENT_ID)
    try:
        registry = LocalWorkspaceRegistryService(workspaces)
        workspace = registry.create_workspace(workspace_id="writer-uat-project", name="UAT project")
        binding = registry.add_folder_binding(workspace.workspace_id, project)
        assert binding.metadata["access"] == "ro"
        registry.set_active_workspace(workspace.workspace_id)
        service = ChatPersistenceService(database, workspace_registry=registry)
        acceptance = _acceptance("writer-bound-chat")
        authority = acceptance.frozen_authority
        policy = replace(authority.policy, auto_retrieve=ConsoleAutoRetrieve.NEVER, source="new_session", policy_revision=None)
        endpoint = "http://127.0.0.1:8765/v1/chat/completions"
        acceptance = replace(
            acceptance,
            user_content="Inspect this project's README. Summarize what it does and include its verification code.",
            frozen_authority=replace(authority, policy=policy, provider_intent=replace(
                authority.provider_intent, provider="llama_cpp", model=MODEL, endpoint=endpoint,
            )),
            resolved_destination=replace(
                acceptance.resolved_destination, provider="llama_cpp", model=MODEL,
                endpoint_identity=endpoint, egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
            contributions=(MixinAttemptContribution(MixinAttemptWrite(
                acceptance.attempt_id, acceptance.user_message_id, acceptance.assistant_message_id, MixinAttemptSnapshot(),
            )),),
        )
        # _invoke_plan_native supplies both scopes in the real external Console.
        with operation_owned_connection(database), core_operation(database):
            checkpoint = service.commit_durable_turn(
                acceptance=acceptance,
                policy_candidate=ConsoleLibraryPolicyCandidate(policy.auto_retrieve, policy.assistant_access),
                conversation_kwargs={"scope_type": "workspace", "workspace_id": workspace.workspace_id},
            )
        assert checkpoint.state is ConsoleDispatchCheckpointState.ACCEPTED
        assert database.get_message_by_id(acceptance.user_message_id)["content"] == acceptance.user_content
        assert database.get_conversation_by_id(acceptance.conversation_id)["workspace_id"] == workspace.workspace_id
    finally:
        database.close_connection()
        workspaces.close()
        host.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_prepared_workspace_turn_preserves_original_database_scope(request, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _commit_workspace_prompt(project)
