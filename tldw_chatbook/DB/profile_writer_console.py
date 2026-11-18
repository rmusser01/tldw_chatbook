"""Data contracts and stock repository construction for Console transactions."""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from functools import cache

from ..Chat.chat_persistence_service import ChatPersistenceService
from ..Chat.console_dispatch_repository import ConsoleDispatchRepository


@cache
def value_types():
    """Only these installed, inert records can cross a Console write boundary."""
    from ..Chat import console_chat_models as chat
    from ..Chat import console_context_policy as context
    from ..Chat import console_dispatch_checkpoint as turn
    from ..Chat import console_library_policy as library
    from ..Chat import console_prompt_mixin_models as mixins
    from ..Chat import library_activity as activity
    from ..Chat import library_preparation as preparation
    from ..Chat.console_context_repository import ContextPolicyWriteResult, ContextPolicyWriteStatus
    from ..Chat.console_endpoint_provenance import ConsoleEndpointProvenance
    from ..Chat.console_prompt_mixin_service import MixinAttemptContribution
    from ..Chat.console_speech_preferences import ConsoleSpeechPreferences
    from ..Chat.message_metadata import AgentChatStartMetadata, AgentHandoffLaunchMetadata
    from ..Prompt_Management.prompt_artifact_models import LiteralMixinDefinition

    classes = (
        turn.ConsoleDurableTurnAcceptance, turn.ConsoleDispatchCheckpoint, turn.ConsoleDispatchTransition,
        turn.ConsoleAssistantSettlement, turn.ConsoleContinuationHandoff, turn.ConsoleDispatchWriteResult,
        turn.ConsoleDispatchCheckpointState, turn.ConsoleDispatchResultStatus, turn.ConsoleEgressClass,
        turn.ConsoleLibraryItemScopeSnapshot, turn.ConsoleProviderIntent, turn.ConsoleTurnLibraryAuthority,
        turn.ConsoleResolvedDestination, turn.ConsoleDispatchReconstructability, ConsoleEndpointProvenance,
        library.ConsoleLibraryPolicySnapshot, library.ConsoleLibraryPolicyCandidate, library.ConsoleAutoRetrieve,
        library.ConsoleAssistantLibraryAccess, AgentChatStartMetadata, AgentHandoffLaunchMetadata, ConsoleSpeechPreferences,
        context.ConsoleContextPolicyOverrides, context.ContextBudgetMode, context.ContextCompactionMode,
        context.ContextCompactionRepresentation, context.CompactionFailureBehavior, context.ContextCarryForwardMode,
        ContextPolicyWriteResult, ContextPolicyWriteStatus,
        preparation.LibraryPreparationEvent, preparation.LibraryPreparationContribution,
        activity.LibraryActivitySourceRef, activity.LibraryActivityEvent, activity.LibraryActivityContributionItem,
        activity.LibraryActivityContribution, MixinAttemptContribution, mixins.MixinAttemptWrite,
        mixins.MixinAttemptSnapshot, mixins.MixinSnapshot, mixins.MixinSourceRef, LiteralMixinDefinition,
        mixins.MixinPreferences, mixins.MixinSelection,
        chat.ConsoleDispatchRecoveryState, chat.ConsoleDispatchRecoveryAction,
        chat.ConsoleDispatchRecoveryKind, chat.ConsoleDispatchRecoveryActionId,
    )
    return {cls.__name__: cls for cls in classes}


def source_database(kind, name, owner):
    expected = ConsoleDispatchRepository if kind == "dispatch" else ChatPersistenceService
    if type(owner) is not expected:
        return None
    if name == "update_conversation_context_policy":
        from ..Chat.console_context_repository import ConsoleContextRepository

        if type(owner.context_repository) is not ConsoleContextRepository or owner.context_repository.db is not owner.db:
            return None
    return owner.db


def citation_source(repository, database):
    """Identify the original deferred citation repository without opening a credential store."""
    if repository is None:
        return None
    from ..Chat.citation_trace_identity import KeyringCitationFingerprintKeyProvider
    from ..Chat.citation_trace_repository import CitationTraceRepository

    if (
        type(repository) is not CitationTraceRepository or repository.db is not database
        or type(repository._key_provider) is not KeyringCitationFingerprintKeyProvider
        or repository._key_provider._keyring_backend is not None
        or repository._key_acquisition is not None
        or repository.fingerprint_key_state not in ("pending", "writes_disabled")
    ):
        return False
    identity = repository.identity_context
    return {
        "state": repository.fingerprint_key_state,
        "identity": None if identity is None else identity.model_dump(mode="json"),
    }


def workspace_source(kind, owner, parameters, profile):
    """Identify a stock Workspace validator and retain caller-owned transaction views."""
    if kind != "console":
        return None
    conversation = parameters.get("conversation_kwargs", {})
    if (conversation.get("scope_type") or "").strip().lower() != "workspace" or owner.workspace_registry is None:
        return None
    from ..Backup_Recovery.profile_paths import database_leaf
    from ..Workspaces.registry_service import LocalWorkspaceRegistryService
    from .Workspace_DB import WorkspaceDB

    registry = owner.workspace_registry
    if type(registry) is not LocalWorkspaceRegistryService or type(registry.db) is not WorkspaceDB:
        return False
    if registry.db.db_path != profile / database_leaf("workspaces_db_path"):
        return False
    connection = getattr(registry.db._thread_local, "conn", None)
    if connection is not None:
        try:
            if sqlite3.Connection.in_transaction.__get__(connection):
                return False
        except sqlite3.ProgrammingError:
            pass
    owner.validate_workspace_target(**conversation)
    return registry.db.client_id


def execute_console_operation(kind, function, database, profile, parameters):
    workspace_client = parameters["workspace_client_id"]
    arguments = parameters["call"]
    with ExitStack() as resources:
        if kind == "dispatch":
            if workspace_client is not None:
                raise ValueError("Dispatch operation has no Workspace repository.")
            owner = ConsoleDispatchRepository(database)
        else:
            registry = None
            if workspace_client is not None:
                from ..app_services import open_workspaces_db
                from ..Backup_Recovery.profile_paths import database_leaf
                from ..config import get_workspaces_db_path
                from ..Workspaces.registry_service import LocalWorkspaceRegistryService

                path = profile / database_leaf("workspaces_db_path")
                if type(workspace_client) is not str or not workspace_client or get_workspaces_db_path() != path:
                    raise ValueError("Console Workspace destination changed.")
                workspace_db = open_workspaces_db(path, client_id=workspace_client)
                resources.callback(workspace_db.close)
                registry = LocalWorkspaceRegistryService(workspace_db)
            citations = None
            stamp = parameters.get("citation_source")
            if stamp is not None:
                from ..Chat.citation_service_factory import build_local_citation_conversation_service
                from ..Chat.citation_trace_repository import CitationPersistenceUnavailable

                _, citations, _ = build_local_citation_conversation_service(
                    database, sidecar_path=profile / "tldw_chatbook_chat_rag_context.json", defer_key=True,
                )
                if type(stamp) is not dict or citation_source(citations, database) != stamp:
                    raise CitationPersistenceUnavailable("citation_repository_source_changed")
            owner = ChatPersistenceService(database, workspace_registry=registry, citation_repository=citations)
        return function(owner, **arguments)
