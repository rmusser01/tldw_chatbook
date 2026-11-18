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
    from ..Chat.console_endpoint_provenance import ConsoleEndpointProvenance
    from ..Chat.console_prompt_mixin_service import MixinAttemptContribution
    from ..Chat.console_speech_preferences import ConsoleSpeechPreferences
    from ..Chat.message_metadata import AgentChatStartMetadata
    from ..Prompt_Management.prompt_artifact_models import LiteralMixinDefinition

    classes = (
        turn.ConsoleDurableTurnAcceptance, turn.ConsoleDispatchCheckpoint, turn.ConsoleDispatchTransition,
        turn.ConsoleAssistantSettlement, turn.ConsoleContinuationHandoff, turn.ConsoleDispatchWriteResult,
        turn.ConsoleDispatchCheckpointState, turn.ConsoleDispatchResultStatus, turn.ConsoleEgressClass,
        turn.ConsoleLibraryItemScopeSnapshot, turn.ConsoleProviderIntent, turn.ConsoleTurnLibraryAuthority,
        turn.ConsoleResolvedDestination, turn.ConsoleDispatchReconstructability, ConsoleEndpointProvenance,
        library.ConsoleLibraryPolicySnapshot, library.ConsoleLibraryPolicyCandidate, library.ConsoleAutoRetrieve,
        library.ConsoleAssistantLibraryAccess, AgentChatStartMetadata, ConsoleSpeechPreferences,
        context.ConsoleContextPolicyOverrides, context.ContextBudgetMode, context.ContextCompactionMode,
        context.ContextCompactionRepresentation, context.CompactionFailureBehavior, context.ContextCarryForwardMode,
        preparation.LibraryPreparationEvent, preparation.LibraryPreparationContribution,
        activity.LibraryActivitySourceRef, activity.LibraryActivityEvent, activity.LibraryActivityContributionItem,
        activity.LibraryActivityContribution, MixinAttemptContribution, mixins.MixinAttemptWrite,
        mixins.MixinAttemptSnapshot, mixins.MixinSnapshot, mixins.MixinSourceRef, LiteralMixinDefinition,
        mixins.MixinPreferences, mixins.MixinSelection,
        chat.ConsoleDispatchRecoveryState, chat.ConsoleDispatchRecoveryAction,
        chat.ConsoleDispatchRecoveryKind, chat.ConsoleDispatchRecoveryActionId,
    )
    return {cls.__name__: cls for cls in classes}


def source_database(kind, owner):
    expected = ConsoleDispatchRepository if kind == "dispatch" else ChatPersistenceService
    return owner.db if type(owner) is expected else None


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
            owner = ChatPersistenceService(database, workspace_registry=registry)
        return function(owner, **arguments)
