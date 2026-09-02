"""Structural census of canonical-message and semantic-sidecar writers."""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"
INVENTORY_PATH = REPO_ROOT / "Docs/Development/console-semantic-mutation-inventory.md"

CLASSIFICATIONS = {
    "model-visible",
    "visibility/ownership-only",
    "presentation-only",
}

# Keys deliberately omit line numbers so harmless source movement does not
# churn the contract. A route is one function/table/verb sink or one
# function/boundary-method call; repeated calls inside that function collapse.
DIRECT_SQL_ROUTE_CLASSIFICATION: dict[str, str] = {
    **dict.fromkeys(
        {
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._reconcile_checkpoint_row_uncoordinated::sql:update:messages",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._cas_state_uncoordinated::sql:update:messages",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._handoff_to_provider_continuation_uncoordinated::sql:update:messages",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository.insert_with_messages::sql:insert:message_attachments",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository.insert_with_messages::sql:insert:messages",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._normalize_provider_continuation_owner_uncoordinated::sql:update:messages",
            "tldw_chatbook/Chat/console_dispatch_repository.py::ConsoleDispatchRepository._settle_with_assistant_uncoordinated::sql:update:messages",
            "tldw_chatbook/Chat/console_semantic_revision.py::SemanticRevisionCoordinator._mutate_message::sql:delete:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._messages_insert_statement::sql:insert:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_message_uncoordinated::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_chat_message.apply_sync_message::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:insert:message_attachments",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_assistant_with_continuation::sql:insert:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_message_variant::sql:insert:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._replace_assistant_generation_projection_uncoordinated::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_attachments_uncoordinated::sql:delete:message_attachments",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_attachments_uncoordinated::sql:insert:message_attachments",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:message_attachments",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_provider_continuation_uncoordinated::sql:update:messages",
        },
        "model-visible",
    ),
    **dict.fromkeys(
        {
            "tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.create_message_variant::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.select_message_variant::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.soft_delete_message::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.soft_delete_message_subtree::sql:update:messages",
        },
        "visibility/ownership-only",
    ),
    **dict.fromkeys(
        {
            "tldw_chatbook/Chat/library_activity.py::LibraryActivityContribution.write::sql:insert:message_trajectory_metadata",
            "tldw_chatbook/Chat/library_preparation.py::LibraryPreparationContribution.write::sql:insert:message_trajectory_metadata",
            "tldw_chatbook/Chat/console_trace_maintenance.py::LegacyTraceMaintenance.run_batch::sql:delete:message_exchanges",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_attachment_with_metadata.append_attachment::sql:insert:message_generation_metadata",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._set_message_feedback_uncoordinated::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_message_exchanges_local::sql:insert:message_exchanges",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.delete_full_exchanges_for_conversation::sql:delete:message_exchanges",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.set_message_generation_metadata::sql:delete:message_generation_metadata",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.set_message_generation_metadata::sql:insert:message_generation_metadata",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.swap_message_attachment_with_scalar.swap_attachment::sql:update:message_generation_metadata",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_metadata_local::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_usage_local::sql:update:messages",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.upsert_trajectory_rows::sql:insert:message_trajectory_metadata",
        },
        "presentation-only",
    ),
}

BOUNDARY_CALL_ROUTE_CLASSIFICATION: dict[str, str] = {
    **dict.fromkeys(
        {
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::add_message_to_conversation::call:db:add_message",
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::create_conversation::call:db:add_message",
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::edit_message_content::call:db:update_message",
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::load_chat_history_from_file_and_save_to_db::call:db:add_message",
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::post_message_to_conversation::call:db:add_message",
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::start_new_chat_session::call:db:add_message",
            "tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.create_character_chat_message::call:db:add_message",
            "tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.update_character_chat_message::call:db:update_message",
            "tldw_chatbook/Chat/Chat_Functions.py::save_chat_history_to_db_wrapper::call:persistence:save_history",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.append_message_attachment::call:db:append_message_attachment_with_metadata",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.commit_durable_turn::call:dispatch:insert_with_messages",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.create_message::call:db:add_message_with_semantic_sidecars",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.fork_console_conversation_bundle::call:persistence:create_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.keep_message_attachment::call:db:swap_message_attachment_with_scalar",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.promote_console_conversation_bundle::call:persistence:create_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.replace_assistant_generation_projection::call:db:replace_assistant_generation_projection",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:persistence:create_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:persistence:update_message_content",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_content.coordinated_update::call:db:update_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_content.coordinated_update::call:db:update_message_with_attachments",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._create_terminal_message::call:persistence:create_message",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._discard_provider_continuation::call:db:update_provider_continuation",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._normalize_restored_provider_continuation::call:dispatch:normalize_provider_continuation_owner",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_existing_message::call:persistence:update_message_content",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_generation_variant::call:persistence:replace_assistant_generation_projection",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_new_message::call:persistence:create_message",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._promote_ephemeral_session_atomically::call:persistence:promote_console_conversation_bundle",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._settle_dispatch_recovery::call:dispatch:settle_with_assistant",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._append_generation_variant::call:persistence:append_message_attachment",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_durable_turn::call:persistence:commit_durable_turn",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._keep_generation_variant::call:persistence:keep_message_attachment",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:dispatch:handoff_to_provider_continuation",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:create_assistant_with_continuation",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:update_provider_continuation",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_roleplay_projection_plan::call:persistence:update_message_content",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.transition_dispatch_recovery_for_retry::call:dispatch:cas_state",
            "tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::call:db:add_message",
            "tldw_chatbook/Chatbooks/chatbook_importer.py::ChatbookImporter._import_conversations::call:db:add_message_with_semantic_sidecars",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.append_chat_message::call:db:add_message",
            "tldw_chatbook/Research_Interop/chat_handoff.py::insert_research_completion_message::call:db:add_message",
            "tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:append_chat_message",
            "tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:delete_chat_message",
            "tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.append_chat_message::call:db:append_chat_message",
            "tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.delete_chat_message::call:db:delete_chat_message",
            "tldw_chatbook/UI/Console_Modules/session.py::ConsoleSessionController._commit_durable_console_chat_fork::call:persistence:fork_console_conversation_bundle",
        },
        "model-visible",
    ),
    **dict.fromkeys(
        {
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::remove_message_from_conversation::call:db:soft_delete_message",
            "tldw_chatbook/Character_Chat/local_character_persona_service.py::LocalCharacterPersonaService.delete_character_chat_message::call:db:soft_delete_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.delete_message_subtree::call:db:soft_delete_message_subtree",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.save_history::call:db:soft_delete_message",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._delete_message::call:persistence:delete_message_subtree",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.delete_chat_message::call:db:soft_delete_message",
        },
        "visibility/ownership-only",
    ),
    **dict.fromkeys(
        {
            "tldw_chatbook/Character_Chat/Character_Chat_Lib.py::set_message_ranking::call:db:update_message",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.append_message_exchanges::call:db:append_message_exchanges_local",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_message_with_semantic_sidecars::call:db:set_message_generation_metadata",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.delete_full_exchanges_for_conversation::call:db:delete_full_exchanges_for_conversation",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_metadata::call:db:update_message_metadata_local",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.update_message_usage::call:db:update_message_usage_local",
            "tldw_chatbook/Chat/chat_persistence_service.py::ChatPersistenceService.write_trajectory_rows::call:db:upsert_trajectory_rows",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_exchanges_only_locked::call:persistence:append_message_exchanges",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_metadata_only::call:persistence:update_message_metadata",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_usage_only::call:persistence:update_message_usage",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_full_capture_purge::call:persistence:delete_full_exchanges_for_conversation",
            "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.write_trajectory_rows::call:persistence:write_trajectory_rows",
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB.update_message_feedback::call:db:update_message",
        },
        "presentation-only",
    ),
}

_SQL_IDENTIFIER = r'(?:[A-Za-z_]\w*|"[^"]+"|`[^`]+`|\[[^\]]+\])'
_MESSAGE_TABLE = (
    r"(?:messages|message_attachments|message_generation_metadata|"
    r"message_exchanges|message_trajectory_metadata|"
    r'"(?:messages|message_attachments|message_generation_metadata|'
    r'message_exchanges|message_trajectory_metadata)"|'
    r"`(?:messages|message_attachments|message_generation_metadata|"
    r"message_exchanges|message_trajectory_metadata)`|"
    r"\[(?:messages|message_attachments|message_generation_metadata|"
    r"message_exchanges|message_trajectory_metadata)\])"
)
_MUTATION_SQL = re.compile(
    rf"\b(?P<verb>insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    rf"(?:{_SQL_IDENTIFIER}\s*\.\s*)?(?P<table>{_MESSAGE_TABLE})(?=\s|\(|$)",
    re.IGNORECASE,
)
_CONVERSATION_HARD_DELETE = re.compile(
    rf"\bdelete\s+from\s+(?:{_SQL_IDENTIFIER}\s*\.\s*)?"
    r'(?:conversations|"conversations"|`conversations`|\[conversations\])'
    r"(?=\s|$)",
    re.IGNORECASE,
)

_DB_MUTATORS = {
    "add_message",
    "add_message_with_semantic_sidecars",
    "append_chat_message",
    "append_message_attachment_with_metadata",
    "append_message_exchanges_local",
    "create_assistant_with_continuation",
    "create_message_variant",
    "delete_chat_message",
    "delete_full_exchanges_for_conversation",
    "replace_assistant_generation_projection",
    "select_message_variant",
    "set_message_attachments",
    "set_message_generation_metadata",
    "soft_delete_message",
    "soft_delete_message_subtree",
    "swap_message_attachment_with_scalar",
    "update_message",
    "update_message_with_attachments",
    "update_message_metadata_local",
    "update_message_usage_local",
    "update_provider_continuation",
    "upsert_trajectory_rows",
}
_PERSISTENCE_MUTATORS = {
    "append_message_exchanges",
    "append_message_attachment",
    "commit_durable_turn",
    "create_message",
    "delete_full_exchanges_for_conversation",
    "delete_message_subtree",
    "fork_console_conversation_bundle",
    "keep_message_attachment",
    "promote_console_conversation_bundle",
    "replace_assistant_generation_projection",
    "save_history",
    "update_message_content",
    "update_message_metadata",
    "update_message_usage",
    "write_trajectory_rows",
}
_DISPATCH_MUTATORS = {
    "cas_state",
    "handoff_to_provider_continuation",
    "insert_with_messages",
    "normalize_provider_continuation_owner",
    "settle_with_assistant",
}

# Purpose-built carrier contract for the frozen character-greeting projection.
# This is intentionally not a general object-attribute dataflow engine.
_CARRIED_WRITER_SPECS = {
    (
        "_RoleplayMessageProjectionWrite",
        "writer",
    ): (
        "ConsoleRoleplayProjectionPersistencePlan",
        "message_writes",
    ),
}

_BOUND_METHOD_RUNNERS = {
    (
        "ConsoleSessionController._commit_durable_console_chat_fork",
        "_run_fork_io",
    )
}
_CHAT_SYNC_HELPER_MUTATORS = {
    "append_chat_message",
    "delete_chat_message",
}

_SQL_RETURN_HELPERS = {
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._messages_insert_statement"
}


@dataclass(frozen=True, order=True)
class _Route:
    path: str
    qualname: str
    action: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}::{self.action}"


@dataclass(frozen=True)
class _DynamicSqlReview:
    domain: str
    exact_targets: frozenset[str]
    evidence_fingerprint: str = ""


@dataclass(frozen=True)
class _LiteralCallTarget:
    helper: str
    positional_index: int
    keyword: str | None = None


@dataclass(frozen=True)
class _CallTargetEvidence:
    calls: tuple[_LiteralCallTarget, ...]


@dataclass(frozen=True)
class _ContainerTargetEvidence:
    symbol: str
    mode: str
    tuple_index: int | None = None
    function_qualname: str | None = None


@dataclass(frozen=True)
class _ImportedConstantTargetEvidence:
    names: tuple[str, ...]


@dataclass(frozen=True)
class _DerivedTargetEvidence:
    targets: frozenset[str]
    fingerprint: str


@dataclass(frozen=True)
class _UnresolvedDynamicSqlSite:
    executor: str
    occurrence: int
    template: str
    source: str

    def key(self, function_identity: str) -> str:
        return (
            f"{function_identity}::dynamic-sql:{self.executor}:"
            f"{self.occurrence}:{self.template} <= {self.source}"
        )


@dataclass(frozen=True)
class _SqlFunctionScan:
    actions: frozenset[str]
    unresolved_sites: tuple[_UnresolvedDynamicSqlSite, ...]


_DYNAMIC_TARGET_EVIDENCE: dict[
    str,
    _CallTargetEvidence | _ContainerTargetEvidence | _ImportedConstantTargetEvidence,
] = {
    "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.undelete_media": (
        _ContainerTargetEvidence(
            symbol="child_tables",
            mode="sequence-tuples",
            tuple_index=0,
            function_qualname="MediaDatabase.undelete_media",
        )
    ),
    "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.soft_delete_media": (
        _ContainerTargetEvidence(
            symbol="child_tables",
            mode="sequence-tuples",
            tuple_index=0,
            function_qualname="MediaDatabase.soft_delete_media",
        )
    ),
    "tldw_chatbook/DB/Evals_DB.py::EvalsDB.delete_probe_annotations_for_run_groups": (
        _ContainerTargetEvidence(
            symbol="_PROBE_ANNOTATION_CASCADE_TABLES",
            mode="sequence-values",
        )
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_generic_item": (
        _CallTargetEvidence(
            calls=(_LiteralCallTarget("_add_generic_item", 0, "table_name"),)
        )
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_generic_item": (
        _CallTargetEvidence(
            calls=(_LiteralCallTarget("_update_generic_item", 0, "table_name"),)
        )
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::"
    "CharactersRAGDB._soft_delete_generic_item": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_soft_delete_generic_item", 0, "table_name"),)
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._manage_link": (
        _CallTargetEvidence(calls=(_LiteralCallTarget("_manage_link", 0),))
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::"
    "CharactersRAGDB._repair_missing_notes_organization_sync_ids": (
        _ContainerTargetEvidence(
            symbol="_NOTES_ORGANIZATION_SYNC_ID_TABLES",
            mode="sequence-values",
        )
    ),
    "tldw_chatbook/Chat/console_trace_maintenance.py::"
    "TraceGarbageCollector._sweep_unmarked": _ContainerTargetEvidence(
        symbol="statements",
        mode="sequence-tuples",
        tuple_index=0,
        function_qualname="TraceGarbageCollector._sweep_unmarked",
    ),
    "tldw_chatbook/Notes/notes_organization_repository.py::"
    "NotesOrganizationRepository.apply_resolved_inventory_merge": (
        _ContainerTargetEvidence(
            symbol="_RESOURCE_TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        )
    ),
    "tldw_chatbook/Notes/notes_organization_repository.py::"
    "NotesOrganizationRepository._materialize_keyword_link": (
        _ContainerTargetEvidence(
            symbol="_KEYWORD_LINK_TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        )
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository._commit_local_body": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_commit_local_body", 0, "table"),)
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository.apply_reviewed_link": _ContainerTargetEvidence(
        symbol="_REBASELINE_TABLES",
        mode="sequence-values",
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository.destroy_profile_content": _ContainerTargetEvidence(
        symbol="_PROFILE_CONTENT_TABLES",
        mode="sequence-values",
    ),
    "tldw_chatbook/Sync_Interop/conflict_review.py::"
    "SyncV2ConflictReviewService._merge_notes_organization_identity": (
        _ContainerTargetEvidence(
            symbol="_NOTES_ORGANIZATION_RESOURCE_TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        )
    ),
    "tldw_chatbook/Sync_Interop/notes_organization_sync_service.py::"
    "NotesOrganizationSyncService._resource_sync_id": _ContainerTargetEvidence(
        symbol="_RESOURCE_SYNC_ID_TABLES",
        mode="sequence-values",
    ),
    "tldw_chatbook/Notifications/event_state_repository.py::"
    "EventStateRepository._upsert_cursor": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_upsert_cursor", 2, "table"),)
    ),
    "tldw_chatbook/Notifications/event_state_repository.py::"
    "EventStateRepository._delete_scoped_rows": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_delete_scoped_rows", 1),)
    ),
    "tldw_chatbook/Research_Interop/local_research_service.py::"
    "LocalResearchService._update_row": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_update_row", 0, "table"),)
    ),
    "tldw_chatbook/Research_Interop/local_research_service.py::"
    "LocalResearchService._soft_delete": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_soft_delete", 0),)
    ),
    "tldw_chatbook/TTS/migrations/v3_to_v4.py::migrate": (
        _ImportedConstantTargetEvidence(
            names=("REFERENCE_TABLE", "_V3_REFERENCE_TABLE")
        )
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_put_reference": _ImportedConstantTargetEvidence(
        names=("REFERENCE_TABLE",)
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_set_reference.set_exact": (
        _ImportedConstantTargetEvidence(names=("REFERENCE_TABLE",))
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_remove_reference.remove_exact": (
        _ImportedConstantTargetEvidence(names=("REFERENCE_TABLE",))
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService._update_row": _CallTargetEvidence(
        calls=(
            _LiteralCallTarget("_update_row", 0, "table"),
            _LiteralCallTarget("_update_aux_row", 0, "table"),
        )
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService._soft_delete": _CallTargetEvidence(
        calls=(_LiteralCallTarget("_soft_delete", 0),)
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService.restore_trash": _ContainerTargetEvidence(
        symbol="_ENTITY_TABLES",
        mode="mapping-value-tuples",
        tuple_index=0,
    ),
}

# A changed producer/caller/import structure requires an explicit review-record
# update, even when its derived target set happens to stay the same.
_DYNAMIC_TARGET_EVIDENCE_FINGERPRINTS = {
    "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.undelete_media": (
        "d14417739a14ae3501a6ae067b1e446704156f44bd2f26b68c8dd39e1054d595"
    ),
    "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.soft_delete_media": (
        "d14417739a14ae3501a6ae067b1e446704156f44bd2f26b68c8dd39e1054d595"
    ),
    "tldw_chatbook/DB/Evals_DB.py::EvalsDB.delete_probe_annotations_for_run_groups": (
        "3784b78922231ed2195c258d8a99038d668e96ffb784ff72449e2298eaf69e67"
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_generic_item": (
        "eef38c50b1451841b8055d50439db6ea284b64c130ab8e034ed2995097ad4c40"
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_generic_item": (
        "46ed924903be6042883fc7c86212212c95baad3ef575a11d0ca509eeff3d7f2d"
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._soft_delete_generic_item": (
        "99daab16a932fd45112fe1cc0217514757cf44dea01b68ecbb165cd14ed4d96a"
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._manage_link": (
        "51d2fdcade4a7cab1583461b9ec3a136432ba36500f9d657ba0072dea6e01f54"
    ),
    "tldw_chatbook/DB/ChaChaNotes_DB.py::"
    "CharactersRAGDB._repair_missing_notes_organization_sync_ids": (
        "a009e05fc58874191347343d0c26d9a299c03d3d906645b4e02584df60eb7a63"
    ),
    "tldw_chatbook/Chat/console_trace_maintenance.py::"
    "TraceGarbageCollector._sweep_unmarked": (
        "45c1c48440ae2fbe9927a8e42048df436e47f36effe4c4dc734e983893da5544"
    ),
    "tldw_chatbook/Notes/notes_organization_repository.py::"
    "NotesOrganizationRepository.apply_resolved_inventory_merge": (
        "25b01efcb85223465ab1223aefe8a0d79648e3e8ecc6aaa0144865898570c5e1"
    ),
    "tldw_chatbook/Notes/notes_organization_repository.py::"
    "NotesOrganizationRepository._materialize_keyword_link": (
        "584697035ed0463e93b57b5db947e6039b6de113a5dc3603e6122f325ec1e414"
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository._commit_local_body": (
        "642432fe97af07beed72ae9d24c9217f0fc13d3dba5d6a0ede81b7b28670697d"
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository.apply_reviewed_link": (
        "eec077333d68af58bc5dfea02dce73bc5725e2d1a1c7824610fc1eff821526c5"
    ),
    "tldw_chatbook/Personal_Context/repository.py::"
    "PersonalContextRepository.destroy_profile_content": (
        "3cb3487843269fd2f388279fbe799f492ed2f58200709b3bd588e64754e79100"
    ),
    "tldw_chatbook/Sync_Interop/conflict_review.py::"
    "SyncV2ConflictReviewService._merge_notes_organization_identity": (
        "558d0bb73c4d9b3ae748a7f8242c5b7950ba9111f3a133f51122aeb4ff3fb9db"
    ),
    "tldw_chatbook/Sync_Interop/notes_organization_sync_service.py::"
    "NotesOrganizationSyncService._resource_sync_id": (
        "656b61911928f5e97fe9396cf11163222e80a647c9e938f44b3293ab2eaeab5b"
    ),
    "tldw_chatbook/Notifications/event_state_repository.py::"
    "EventStateRepository._upsert_cursor": (
        "02e2d1f0011e37c36491aac4268644e073258a0c42e15596a9d04d4fb23c9f03"
    ),
    "tldw_chatbook/Notifications/event_state_repository.py::"
    "EventStateRepository._delete_scoped_rows": (
        "4bd1dd34e5f88db405ec2db7194bbbad27bf427c0d48e1cec8b308c9f826219a"
    ),
    "tldw_chatbook/Research_Interop/local_research_service.py::"
    "LocalResearchService._update_row": (
        "ff2907c96262d84b82041802082cfba1b0346d3ea4dcbc440934ff783e33b3d0"
    ),
    "tldw_chatbook/Research_Interop/local_research_service.py::"
    "LocalResearchService._soft_delete": (
        "65aeaaa1af3f6111ccfadddc531ae0dbe0a34148524c1d4f79f24da3ea0b0ae1"
    ),
    "tldw_chatbook/TTS/migrations/v3_to_v4.py::migrate": (
        "51fab7d60895c09fcd23fdaf420717ec273594748acf7095dae98a11e914a901"
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_put_reference": (
        "090e00e7953db5554181bd8f73ce45300326fa9693834ba5b9ccaa8ef6e1151d"
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_set_reference.set_exact": (
        "090e00e7953db5554181bd8f73ce45300326fa9693834ba5b9ccaa8ef6e1151d"
    ),
    "tldw_chatbook/TTS/profile_repository.py::"
    "TTSProfileRepository._worker_remove_reference.remove_exact": (
        "090e00e7953db5554181bd8f73ce45300326fa9693834ba5b9ccaa8ef6e1151d"
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService._update_row": (
        "6b60ac8520d36669e81deefab83a6187c46aee77c571c3540fd95911aa08dce9"
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService._soft_delete": (
        "776ee681023b926b12b92a8320e0ea81dbcb2fd5eff7c20fdc76a767b49e4f65"
    ),
    "tldw_chatbook/Writing_Interop/local_writing_service.py::"
    "LocalWritingService.restore_trash": (
        "a0cabdc74d34c32ae2ed059024f0d152e69dd5b6378a4f05abc8e257a569961b"
    ),
}


def _reviewed_dynamic_sql_site(
    function_identity: str,
    template: str,
    source: str,
    *,
    domain: str,
    exact_targets: frozenset[str],
) -> tuple[str, _DynamicSqlReview]:
    site = _UnresolvedDynamicSqlSite(
        executor="execute",
        occurrence=1,
        template=template,
        source=source,
    )
    return site.key(function_identity), _DynamicSqlReview(
        domain,
        exact_targets,
        _DYNAMIC_TARGET_EVIDENCE_FINGERPRINTS[function_identity],
    )


# Each exception pins one executor call, its normalized template/source, and
# the exact non-chat tables proven by the producer/caller contract.
_PROVEN_NON_SEMANTIC_DYNAMIC_SQL = dict(
    [
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.undelete_media",
            "UPDATE {expression} SET deleted = 0, last_modified = ?, version = ?, "
            "client_id = ? WHERE id = ? AND version = ?",
            "f'UPDATE {table} SET deleted = 0, last_modified = ?, version = ?, "
            "client_id = ? WHERE id = ? AND version = ?'",
            domain="media cascade children",
            exact_targets=frozenset(
                {
                    "Transcripts",
                    "MediaChunks",
                    "UnvectorizedMediaChunks",
                    "DocumentVersions",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/Evals_DB.py::EvalsDB.delete_probe_annotations_for_run_groups",
            "DELETE FROM {expression} WHERE run_group_id IN ({expression})",
            "f'DELETE FROM {table} WHERE run_group_id IN ({placeholders})'",
            domain="evaluation probe annotations",
            exact_targets=frozenset(
                {"eval_probe_turn_annotations", "eval_probe_review_state"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_generic_item",
            "INSERT INTO {expression} ( {expression} ) VALUES ({expression})",
            "query",
            domain="keyword library records",
            exact_targets=frozenset({"keywords", "keyword_collections"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._update_generic_item",
            "UPDATE {expression} SET {expression} WHERE {expression} = ? AND "
            "version = ? AND deleted = 0",
            "query",
            domain="keyword library records",
            exact_targets=frozenset({"keywords", "keyword_collections"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::"
            "CharactersRAGDB._soft_delete_generic_item",
            "UPDATE {expression} SET deleted = 1, last_modified = ?, version = ?, "
            "client_id = ? WHERE {expression} = ? AND version = ? AND deleted = 0",
            "query",
            domain="keyword library records",
            exact_targets=frozenset({"keywords", "keyword_collections"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._manage_link",
            "INSERT OR IGNORE INTO {expression} ({expression}, {expression}, "
            "created_at) VALUES (?, ?, ?)",
            "query",
            domain="keyword association records",
            exact_targets=frozenset(
                {"conversation_keywords", "collection_keywords", "note_keywords"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._manage_link",
            "DELETE FROM {expression} WHERE {expression} = ? AND {expression} = ?",
            "query",
            domain="keyword association records",
            exact_targets=frozenset(
                {"conversation_keywords", "collection_keywords", "note_keywords"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Chat/console_trace_maintenance.py::"
            "TraceGarbageCollector._sweep_unmarked",
            "DELETE FROM {expression} WHERE NOT EXISTS ( SELECT 1 FROM "
            "console_trace_gc_marks AS mark WHERE mark.request_id = ? AND "
            "mark.entity_kind = ? AND mark.entity_id = "
            "{expression}.{expression})",
            "f'DELETE FROM {table} WHERE NOT EXISTS (\\n SELECT 1 FROM "
            "console_trace_gc_marks AS mark\\n WHERE mark.request_id = ? AND "
            "mark.entity_kind = ?\\n AND mark.entity_id = "
            "{table}.{identity})'",
            domain="unreachable semantic trace graph rows",
            exact_targets=frozenset(
                {
                    "console_trace_artifacts",
                    "console_trace_calls",
                    "console_trace_events",
                    "console_trace_header_components",
                    "console_trace_owners",
                    "console_trace_policies",
                    "console_trace_redaction_spans",
                    "console_trace_request_headers",
                    "console_trace_response_links",
                    "console_trace_revision_bindings",
                    "console_trace_segments",
                    "console_trace_semantic_revisions",
                    "console_trace_surface_nodes",
                    "console_trace_surface_replacements",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::"
            "CharactersRAGDB._repair_missing_notes_organization_sync_ids",
            "UPDATE {expression} SET sync_id = ? WHERE id = ? AND sync_id IS NULL",
            "f'UPDATE {table} SET sync_id = ? WHERE id = ? AND sync_id IS NULL'",
            domain="notes organization portable identities",
            exact_targets=frozenset(
                {"keywords", "keyword_collections", "note_folders"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Notes/notes_organization_repository.py::"
            "NotesOrganizationRepository.apply_resolved_inventory_merge",
            "UPDATE {expression} SET sync_id = ? WHERE CAST(id AS TEXT) = ?",
            "f'UPDATE {table} SET sync_id = ? WHERE CAST(id AS TEXT) = ?'",
            domain="notes organization portable identities",
            exact_targets=frozenset(
                {"keywords", "keyword_collections", "note_folders"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Notes/notes_organization_repository.py::"
            "NotesOrganizationRepository._materialize_keyword_link",
            "DELETE FROM {expression} WHERE {expression} = ? AND keyword_id = ?",
            "f'DELETE FROM {table} WHERE {column} = ? AND keyword_id = ?'",
            domain="notes organization keyword links",
            exact_targets=frozenset({"note_keywords", "conversation_keywords"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Notes/notes_organization_repository.py::"
            "NotesOrganizationRepository._materialize_keyword_link",
            "INSERT OR IGNORE INTO {expression}({expression}, keyword_id, "
            "created_at) VALUES (?, ?, ?)",
            "f'INSERT OR IGNORE INTO {table}({column}, keyword_id, created_at) "
            "VALUES (?, ?, ?)'",
            domain="notes organization keyword links",
            exact_targets=frozenset({"note_keywords", "conversation_keywords"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Personal_Context/repository.py::"
            "PersonalContextRepository._commit_local_body",
            "INSERT INTO {expression}(scope_id, {expression}) VALUES (?, ?) ON "
            "CONFLICT(scope_id) DO UPDATE SET {expression} = "
            "excluded.{expression}",
            "f'INSERT INTO {table}(scope_id, {version_column}) VALUES (?, ?) ON "
            "CONFLICT(scope_id) DO UPDATE SET {version_column} = "
            "excluded.{version_column}'",
            domain="Personal Context local metadata heads",
            exact_targets=frozenset(
                {"local_runtime_policy", "local_scope_bindings"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Personal_Context/repository.py::"
            "PersonalContextRepository.apply_reviewed_link",
            "DELETE FROM {expression}",
            "f'DELETE FROM {table}'",
            domain="Personal Context reviewed rebaseline rows",
            exact_targets=frozenset(
                {"local_runtime_policy", "local_scope_bindings"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Personal_Context/repository.py::"
            "PersonalContextRepository.apply_reviewed_link",
            "INSERT OR REPLACE INTO {expression}({expression}) VALUES "
            "({expression})",
            "f\"INSERT OR REPLACE INTO {table}({','.join(columns)}) VALUES "
            "({','.join(('?' for _ in columns))})\"",
            domain="Personal Context reviewed rebaseline rows",
            exact_targets=frozenset(
                {"local_runtime_policy", "local_scope_bindings"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Personal_Context/repository.py::"
            "PersonalContextRepository.destroy_profile_content",
            "DELETE FROM {expression}",
            "f'DELETE FROM {table}'",
            domain="Personal Context profile content",
            exact_targets=frozenset(
                {
                    "encrypted_objects",
                    "encrypted_outbox",
                    "local_record_links",
                    "local_runtime_policy",
                    "local_scope_bindings",
                    "local_undo",
                    "local_unlinked_scopes",
                    "object_heads",
                    "quarantine",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Sync_Interop/conflict_review.py::"
            "SyncV2ConflictReviewService._merge_notes_organization_identity",
            "UPDATE {expression} SET sync_id = ? WHERE id = ?",
            "f'UPDATE {table} SET sync_id = ? WHERE id = ?'",
            domain="notes organization portable identities",
            exact_targets=frozenset(
                {"keywords", "keyword_collections", "note_folders"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Sync_Interop/notes_organization_sync_service.py::"
            "NotesOrganizationSyncService._resource_sync_id",
            "UPDATE {expression} SET sync_id = ? WHERE id = ? AND sync_id IS NULL",
            "f'UPDATE {table} SET sync_id = ? WHERE id = ? AND sync_id IS NULL'",
            domain="notes organization portable identities",
            exact_targets=frozenset({"keywords", "keyword_collections"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/Client_Media_DB_v2.py::MediaDatabase.soft_delete_media",
            "UPDATE {expression} SET deleted = 1, last_modified = ?, version = ?, "
            "client_id = ? WHERE id = ? AND version = ? AND deleted = 0",
            "update_sql",
            domain="media cascade children",
            exact_targets=frozenset(
                {
                    "Transcripts",
                    "MediaChunks",
                    "UnvectorizedMediaChunks",
                    "DocumentVersions",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/DB/ChaChaNotes_DB.py::CharactersRAGDB._add_generic_item",
            "UPDATE {expression} SET {expression} WHERE id = ? AND version = ?",
            "undelete_query",
            domain="keyword library records",
            exact_targets=frozenset({"keywords", "keyword_collections"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Notifications/event_state_repository.py::"
            "EventStateRepository._upsert_cursor",
            "INSERT INTO {expression} ( source_authority, server_profile_id, "
            "authenticated_principal_id, stream_name, stream_instance_id, cursor, "
            "updated_at ) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT( "
            "source_authority, server_profile_id, authenticated_principal_id, "
            "stream_name, stream_instance_id ) DO UPDATE SET cursor = "
            "excluded.cursor, updated_at = excluded.updated_at",
            "f'\\n INSERT INTO {table} (\\n source_authority,\\n server_profile_id,\\n "
            "authenticated_principal_id,\\n stream_name,\\n stream_instance_id,\\n "
            "cursor,\\n updated_at\\n )\\n VALUES (?, ?, ?, ?, ?, ?, ?)\\n ON CONFLICT(\\n "
            "source_authority,\\n server_profile_id,\\n authenticated_principal_id,\\n "
            "stream_name,\\n stream_instance_id\\n )\\n DO UPDATE SET cursor = "
            "excluded.cursor, updated_at = excluded.updated_at\\n '",
            domain="notification cursor state",
            exact_targets=frozenset(
                {"event_processed_cursors", "event_presented_high_water"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Notifications/event_state_repository.py::"
            "EventStateRepository._delete_scoped_rows",
            "DELETE FROM {expression} WHERE {expression}",
            "f'DELETE FROM {table} WHERE {table_filter}'",
            domain="notification scoped state",
            exact_targets=frozenset(
                {
                    "event_processed_cursors",
                    "event_presented_high_water",
                    "event_observer_status",
                    "event_retention_policies",
                    "event_replay_windows",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Research_Interop/local_research_service.py::"
            "LocalResearchService._update_row",
            "UPDATE {expression} SET {expression} WHERE id = ?",
            "f'UPDATE {table} SET {assignments} WHERE id = ?'",
            domain="local research records",
            exact_targets=frozenset({"research_sessions"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Research_Interop/local_research_service.py::"
            "LocalResearchService._soft_delete",
            "UPDATE {expression} SET deleted = 1, updated_at = ?, version = ? "
            "WHERE id = ?",
            "f'UPDATE {table} SET deleted = 1, updated_at = ?, version = ? WHERE id = ?'",
            domain="local research records",
            exact_targets=frozenset({"research_sessions", "research_runs"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/TTS/migrations/v3_to_v4.py::migrate",
            "INSERT INTO {expression} ( profile_id, reference_id, wav_bytes, "
            "reference_text, sha256, byte_length, duration_ms, sample_rate_hz, "
            "channels, sample_encoding, created_at, updated_at, recipe_id, "
            "recipe_revision ) SELECT profile_id, reference_id, wav_bytes, "
            "reference_text, sha256, byte_length, duration_ms, sample_rate_hz, "
            "channels, sample_encoding, created_at, updated_at, NULL, NULL FROM "
            "{expression}",
            "f'\\n INSERT INTO {REFERENCE_TABLE} (\\n profile_id, reference_id, "
            "wav_bytes, reference_text, sha256,\\n byte_length, duration_ms, "
            "sample_rate_hz, channels, sample_encoding,\\n created_at, updated_at, "
            "recipe_id, recipe_revision\\n )\\n SELECT profile_id, reference_id, "
            "wav_bytes, reference_text, sha256,\\n byte_length, duration_ms, "
            "sample_rate_hz, channels, sample_encoding,\\n created_at, updated_at, "
            "NULL, NULL\\n FROM {_V3_REFERENCE_TABLE}\\n '",
            domain="TTS clone-reference migration",
            exact_targets=frozenset(
                {"tts_profile_clone_references", "_tts_profile_clone_references_v3"}
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/TTS/profile_repository.py::"
            "TTSProfileRepository._worker_put_reference",
            "INSERT INTO {expression} ( profile_id, reference_id, wav_bytes, "
            "reference_text, sha256, byte_length, duration_ms, sample_rate_hz, "
            "channels, sample_encoding, created_at, updated_at, recipe_id, "
            "recipe_revision ) VALUES (?, ?, zeroblob(?), ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?)",
            "f'\\n INSERT INTO {REFERENCE_TABLE} (\\n profile_id, reference_id, "
            "wav_bytes, reference_text, sha256,\\n byte_length, duration_ms, "
            "sample_rate_hz, channels,\\n sample_encoding, created_at, updated_at, "
            "recipe_id,\\n recipe_revision\\n ) VALUES (?, ?, zeroblob(?), ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?)\\n '",
            domain="TTS clone references",
            exact_targets=frozenset({"tts_profile_clone_references"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/TTS/profile_repository.py::"
            "TTSProfileRepository._worker_set_reference.set_exact",
            "INSERT INTO {expression} ( profile_id, reference_id, wav_bytes, "
            "reference_text, sha256, byte_length, duration_ms, sample_rate_hz, "
            "channels, sample_encoding, created_at, updated_at, recipe_id, "
            "recipe_revision ) VALUES (?, ?, zeroblob(?), ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?) ON CONFLICT(profile_id) DO UPDATE SET reference_id = "
            "excluded.reference_id, wav_bytes = excluded.wav_bytes, reference_text "
            "= excluded.reference_text, sha256 = excluded.sha256, byte_length = "
            "excluded.byte_length, duration_ms = excluded.duration_ms, sample_rate_hz "
            "= excluded.sample_rate_hz, channels = excluded.channels, "
            "sample_encoding = excluded.sample_encoding, created_at = "
            "excluded.created_at, updated_at = excluded.updated_at, recipe_id = "
            "excluded.recipe_id, recipe_revision = excluded.recipe_revision",
            "f'\\n INSERT INTO {REFERENCE_TABLE} (\\n profile_id,\\n reference_id,\\n "
            "wav_bytes,\\n reference_text,\\n sha256,\\n byte_length,\\n duration_ms,\\n "
            "sample_rate_hz,\\n channels,\\n sample_encoding,\\n created_at,\\n "
            "updated_at,\\n recipe_id,\\n recipe_revision\\n ) VALUES (?, ?, "
            "zeroblob(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\\n ON "
            "CONFLICT(profile_id) DO UPDATE SET\\n reference_id = "
            "excluded.reference_id,\\n wav_bytes = excluded.wav_bytes,\\n "
            "reference_text = excluded.reference_text,\\n sha256 = excluded.sha256,\\n "
            "byte_length = excluded.byte_length,\\n duration_ms = "
            "excluded.duration_ms,\\n sample_rate_hz = excluded.sample_rate_hz,\\n "
            "channels = excluded.channels,\\n sample_encoding = "
            "excluded.sample_encoding,\\n created_at = excluded.created_at,\\n "
            "updated_at = excluded.updated_at,\\n recipe_id = excluded.recipe_id,\\n "
            "recipe_revision = excluded.recipe_revision\\n '",
            domain="TTS clone references",
            exact_targets=frozenset({"tts_profile_clone_references"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/TTS/profile_repository.py::"
            "TTSProfileRepository._worker_remove_reference.remove_exact",
            "DELETE FROM {expression} WHERE profile_id = ?",
            "f'DELETE FROM {REFERENCE_TABLE} WHERE profile_id = ?'",
            domain="TTS clone references",
            exact_targets=frozenset({"tts_profile_clone_references"}),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Writing_Interop/local_writing_service.py::"
            "LocalWritingService._update_row",
            "UPDATE {expression} SET {expression} WHERE id = ?",
            "f'UPDATE {table} SET {assignments} WHERE id = ?'",
            domain="local writing records",
            exact_targets=frozenset(
                {
                    "writing_projects",
                    "writing_manuscripts",
                    "writing_chapters",
                    "writing_scenes",
                    "writing_characters",
                    "writing_world_info",
                    "writing_plot_lines",
                    "writing_plot_events",
                    "writing_plot_holes",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Writing_Interop/local_writing_service.py::"
            "LocalWritingService._soft_delete",
            "UPDATE {expression} SET deleted = 1, last_modified = ?, version = ? "
            "WHERE id = ?",
            "f'UPDATE {table} SET deleted = 1, last_modified = ?, version = ? WHERE id = ?'",
            domain="local writing records",
            exact_targets=frozenset(
                {
                    "writing_projects",
                    "writing_manuscripts",
                    "writing_chapters",
                    "writing_scenes",
                    "writing_characters",
                    "writing_relationships",
                    "writing_world_info",
                    "writing_plot_lines",
                    "writing_plot_events",
                    "writing_plot_holes",
                    "writing_citations",
                }
            ),
        ),
        _reviewed_dynamic_sql_site(
            "tldw_chatbook/Writing_Interop/local_writing_service.py::"
            "LocalWritingService.restore_trash",
            "UPDATE {expression} SET deleted = 0, last_modified = ?, version = ? "
            "WHERE id = ?",
            "f'UPDATE {table} SET deleted = 0, last_modified = ?, version = ? WHERE id = ?'",
            domain="local writing trash restoration",
            exact_targets=frozenset(
                {
                    "writing_projects",
                    "writing_manuscripts",
                    "writing_chapters",
                    "writing_scenes",
                }
            ),
        ),
    ]
)


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _function_nodes(
    tree: ast.AST,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            found.append((".".join([*self.scope, node.name]), node))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    Visitor().visit(tree)
    return found


def _function_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    """Return a function's own nodes, excluding its docstring and nested scopes."""

    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if node is function:
                for index, statement in enumerate(node.body):
                    if (
                        index == 0
                        and isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)
                    ):
                        continue
                    self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    Visitor().visit(function)
    return nodes


def _string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else "{expression}"
            for value in node.values
        )
    return None


def _literal_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_value(node.left)
        right = _literal_string_value(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(value, ast.Constant) and isinstance(value.value, str)
        for value in node.values
    ):
        return "".join(value.value for value in node.values)  # type: ignore[union-attr]
    return None


def _module_string_constants(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        name = _assigned_name(statement)
        value = (
            statement.value
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            else None
        )
        if name is None or value is None:
            continue
        literal = _literal_string_value(value)
        if literal is not None:
            constants[name] = literal
        else:
            constants.pop(name, None)
    return tuple(sorted(constants.items()))


def _live_python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


@cache
def _production_modules() -> tuple[
    tuple[Path, ast.Module, tuple[tuple[str, str], ...]], ...
]:
    modules: list[tuple[Path, ast.Module, tuple[tuple[str, str], ...]]] = []
    for path in _live_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules.append((path, tree, _module_string_constants(tree)))
    return tuple(modules)


@cache
def _production_functions() -> tuple[
    tuple[Path, str, ast.FunctionDef | ast.AsyncFunctionDef], ...
]:
    functions: list[tuple[Path, str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path, tree, _constants in _production_modules():
        functions.extend(
            (path, qualname, function) for qualname, function in _function_nodes(tree)
        )
    return tuple(functions)


def _evidence_fingerprint(parts: list[str]) -> str:
    normalized = "\n".join(sorted(parts))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _literal_call_argument(
    call: ast.Call,
    target: _LiteralCallTarget,
) -> ast.expr | None:
    keyword = next(
        (
            item.value
            for item in call.keywords
            if target.keyword is not None and item.arg == target.keyword
        ),
        None,
    )
    if keyword is not None:
        return keyword
    if len(call.args) > target.positional_index:
        return call.args[target.positional_index]
    return None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _assert_direct_helper_references(
    nodes: list[ast.AST],
    helpers: frozenset[str],
) -> None:
    direct_reference_ids = {
        id(node.func)
        for node in nodes
        if isinstance(node, ast.Call) and _call_name(node) in helpers
    }
    for node in nodes:
        referenced_helper: str | None = None
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            referenced_helper = node.attr
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced_helper = node.id
        if referenced_helper in helpers and id(node) not in direct_reference_ids:
            raise AssertionError(
                f"reviewed dynamic SQL helper {referenced_helper!r} was forwarded "
                "instead of called directly"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
        ):
            reflected_name = _literal_string_value(node.args[1])
            if reflected_name is None:
                raise AssertionError(
                    "runtime-dynamic helper reflection cannot exclude a reviewed "
                    "dynamic SQL helper"
                )
            if reflected_name not in helpers:
                continue
            raise AssertionError(
                f"reviewed dynamic SQL helper {reflected_name!r} was forwarded "
                "through getattr()"
            )


def _module_scope_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    for statement in tree.body:
        Visitor().visit(statement)
    return nodes


def _target_binds_name(node: ast.AST, symbol: str) -> bool:
    return any(
        isinstance(item, ast.Name)
        and item.id == symbol
        and isinstance(item.ctx, (ast.Store, ast.Del))
        for item in ast.walk(node)
    )


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _function_scope_declarations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: str,
) -> tuple[bool, bool, bool]:
    declares_global = False
    declares_nonlocal = False
    binds_local = symbol in _argument_names(function.args)

    class Visitor(ast.NodeVisitor):
        def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
            nonlocal declares_global
            declares_global = declares_global or symbol in node.names

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
            nonlocal declares_nonlocal
            declares_nonlocal = declares_nonlocal or symbol in node.names

        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
            nonlocal binds_local
            if node.id == symbol and isinstance(node.ctx, (ast.Store, ast.Del)):
                binds_local = True

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            nonlocal binds_local
            binds_local = binds_local or node.name == symbol

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            nonlocal binds_local
            binds_local = binds_local or node.name == symbol

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            nonlocal binds_local
            binds_local = binds_local or node.name == symbol

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
            self._visit_comprehension(node, (node.elt,))

        def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
            self._visit_comprehension(node, (node.elt,))

        def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
            self._visit_comprehension(node, (node.key, node.value))

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
            self._visit_comprehension(node, (node.elt,))

        def _visit_comprehension(
            self,
            node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
            results: tuple[ast.expr, ...],
        ) -> None:
            for generator in node.generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for result in results:
                self.visit(result)

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    return declares_global, declares_nonlocal, binds_local


def _function_definition_expressions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.expr]:
    expressions = [*function.decorator_list]
    expressions.extend(function.args.defaults)
    expressions.extend(
        default for default in function.args.kw_defaults if default is not None
    )
    expressions.extend(
        argument.annotation
        for argument in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if argument.annotation is not None
    )
    if function.args.vararg is not None and function.args.vararg.annotation is not None:
        expressions.append(function.args.vararg.annotation)
    if function.args.kwarg is not None and function.args.kwarg.annotation is not None:
        expressions.append(function.args.kwarg.annotation)
    if function.returns is not None:
        expressions.append(function.returns)
    expressions.extend(getattr(function, "type_params", ()))
    return expressions


def _lambda_binds_name(lambda_node: ast.Lambda, symbol: str) -> bool:
    if symbol in _argument_names(lambda_node.args):
        return True
    binds_name = False

    class Visitor(ast.NodeVisitor):
        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
            nonlocal binds_name
            binds_name = binds_name or _target_binds_name(node.target, symbol)
            self.visit(node.value)

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    Visitor().visit(lambda_node.body)
    return binds_name


def _class_statement_binds_name(statement: ast.stmt, symbol: str) -> bool:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return statement.name == symbol
    if isinstance(statement, ast.Assign):
        return any(_target_binds_name(target, symbol) for target in statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return statement.value is not None and _target_binds_name(
            statement.target, symbol
        )
    if isinstance(statement, ast.AugAssign):
        return _target_binds_name(statement.target, symbol)
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.split(".", 1)[0]) == symbol
            for alias in statement.names
        )
    return False


def _class_statement_deletes_name(statement: ast.stmt, symbol: str) -> bool:
    return isinstance(statement, ast.Delete) and any(
        _target_binds_name(target, symbol) for target in statement.targets
    )


_COMPOUND_CLASS_STATEMENT_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


def _compound_class_statement_mentions_name(statement: ast.stmt, symbol: str) -> bool:
    """Return whether a compound class statement ambiguously uses ``symbol``."""
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and node.id == symbol:
            return True
        if isinstance(node, ast.arg) and node.arg == symbol:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and symbol in node.names:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == symbol:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == symbol:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest == symbol:
            return True
        if isinstance(node, ast.alias):
            bound_name = node.asname or node.name.split(".", 1)[0]
            if bound_name == symbol:
                return True
    return False


def _module_container_name_ids(tree: ast.Module, symbol: str) -> frozenset[int]:
    relevant: set[int] = set()

    def visit(
        node: ast.AST,
        resolves_module: bool,
        *,
        collect_stores: bool = True,
    ) -> None:
        if isinstance(node, ast.Name):
            if (
                node.id == symbol
                and resolves_module
                and (isinstance(node.ctx, ast.Load) or collect_stores)
            ):
                relevant.add(id(node))
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(node, resolves_module, resolves_module)
            return
        if isinstance(node, ast.Lambda):
            lambda_local = _lambda_binds_name(node, symbol)
            visit(node.body, resolves_module and not lambda_local)
            return
        if isinstance(node, ast.ClassDef):
            visit_class(node, resolves_module)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            visit_comprehension(node, node.elt, resolves_module)
            return
        if isinstance(node, ast.DictComp):
            visit_comprehension(node, (node.key, node.value), resolves_module)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, resolves_module, collect_stores=collect_stores)

    def visit_function(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        header_resolves_module: bool,
        body_enclosing_resolves_module: bool,
    ) -> None:
        for expression in _function_definition_expressions(function):
            visit(expression, header_resolves_module)
        declares_global, declares_nonlocal, binds_local = _function_scope_declarations(
            function, symbol
        )
        if declares_global:
            body_resolves_module = True
        elif declares_nonlocal or binds_local:
            body_resolves_module = False
        else:
            body_resolves_module = body_enclosing_resolves_module
        for statement in function.body:
            visit(statement, body_resolves_module)

    def visit_class(class_node: ast.ClassDef, enclosing_resolves_module: bool) -> None:
        for expression in [
            *class_node.decorator_list,
            *class_node.bases,
            *(keyword.value for keyword in class_node.keywords),
            *getattr(class_node, "type_params", ()),
        ]:
            visit(expression, enclosing_resolves_module)
        class_bound = False
        for statement in class_node.body:
            class_resolves_module = enclosing_resolves_module and not class_bound
            if isinstance(statement, _COMPOUND_CLASS_STATEMENT_TYPES) and (
                _compound_class_statement_mentions_name(statement, symbol)
            ):
                raise AssertionError(
                    f"dynamic SQL evidence producer {symbol!r} appears in a "
                    "compound class statement whose namespace transition is "
                    "intentionally not modeled"
                )
            if (
                isinstance(statement, ast.AugAssign)
                and _target_binds_name(statement.target, symbol)
                and class_resolves_module
            ):
                raise AssertionError(
                    f"dynamic SQL evidence producer {symbol!r} was read before "
                    "binding by class AugAssign"
                )
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_function(
                    statement,
                    class_resolves_module,
                    enclosing_resolves_module,
                )
            elif isinstance(statement, ast.ClassDef):
                visit_class(statement, enclosing_resolves_module)
            else:
                visit(statement, class_resolves_module, collect_stores=False)
            if _class_statement_deletes_name(statement, symbol):
                class_bound = False
            else:
                class_bound = class_bound or _class_statement_binds_name(
                    statement, symbol
                )

    def visit_comprehension(
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        result: ast.expr | tuple[ast.expr, ast.expr],
        enclosing_resolves_module: bool,
    ) -> None:
        comprehension_resolves_module = enclosing_resolves_module
        for generator in node.generators:
            visit(generator.iter, comprehension_resolves_module)
            if _target_binds_name(generator.target, symbol):
                comprehension_resolves_module = False
            for condition in generator.ifs:
                visit(condition, comprehension_resolves_module)
        results = result if isinstance(result, tuple) else (result,)
        for expression in results:
            visit(expression, comprehension_resolves_module)

    for statement in tree.body:
        visit(statement, True)
    return frozenset(relevant)


def _container_assignment(
    tree: ast.Module,
    evidence: _ContainerTargetEvidence,
) -> ast.Assign | ast.AnnAssign:
    if evidence.function_qualname is None:
        nodes = _module_scope_nodes(tree)
        audit_nodes = list(ast.walk(tree))
        relevant_name_ids = _module_container_name_ids(tree, evidence.symbol)
    else:
        function = next(
            (
                node
                for qualname, node in _function_nodes(tree)
                if qualname == evidence.function_qualname
            ),
            None,
        )
        assert function is not None, (
            f"dynamic SQL evidence function disappeared: {evidence.function_qualname}"
        )
        nodes = _function_body_nodes(function)
        audit_nodes = nodes
        relevant_name_ids = frozenset(
            id(node)
            for node in nodes
            if isinstance(node, ast.Name) and node.id == evidence.symbol
        )
    assignments = [
        node
        for node in nodes
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assigned_name(node) == evidence.symbol
    ]
    assert len(assignments) == 1, (
        f"dynamic SQL evidence producer {evidence.symbol!r} changed: "
        f"found {len(assignments)} assignments"
    )
    assignment = assignments[0]
    _assert_immutable_container_producer(
        audit_nodes,
        evidence.symbol,
        assignment,
        relevant_name_ids,
    )
    return assignment


_READ_ONLY_CONTAINER_METHODS = frozenset(
    {"copy", "count", "get", "index", "items", "keys", "values"}
)


def _root_name_node(node: ast.AST) -> ast.Name | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node if isinstance(node, ast.Name) else None


def _assert_immutable_container_producer(
    nodes: list[ast.AST],
    symbol: str,
    assignment: ast.Assign | ast.AnnAssign,
    relevant_name_ids: frozenset[int],
) -> None:
    node_ids = {id(node) for node in nodes}
    parents = {
        id(child): parent
        for parent in nodes
        for child in ast.iter_child_nodes(parent)
        if id(child) in node_ids
    }
    allowed_store_ids = {
        id(node)
        for node in ast.walk(assignment)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == symbol
    }
    for node in nodes:
        if (
            isinstance(node, ast.Name)
            and node.id == symbol
            and id(node) in relevant_name_ids
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and id(node) not in allowed_store_ids
        ):
            raise AssertionError(
                f"dynamic SQL evidence producer {symbol!r} was mutated or rebound"
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (root := _root_name_node(node)) is not None
            and root.id == symbol
            and id(root) in relevant_name_ids
        ):
            raise AssertionError(
                f"dynamic SQL evidence producer {symbol!r} was mutated by subscript"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (root := _root_name_node(node.func.value)) is not None
            and root.id == symbol
            and id(root) in relevant_name_ids
            and node.func.attr not in _READ_ONLY_CONTAINER_METHODS
        ):
            raise AssertionError(
                f"dynamic SQL evidence producer {symbol!r} was mutated or "
                f"forwarded through {node.func.attr}()"
            )
        if (
            isinstance(node, ast.Name)
            and node.id == symbol
            and id(node) in relevant_name_ids
            and isinstance(node.ctx, ast.Load)
        ):
            parent = parents.get(id(node))
            grandparent = parents.get(id(parent)) if parent is not None else None
            is_loop_iterator = (
                isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension))
                and parent.iter is node
            )
            is_read_subscript = (
                isinstance(parent, ast.Subscript)
                and parent.value is node
                and isinstance(parent.ctx, ast.Load)
            )
            is_read_method_call = (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in _READ_ONLY_CONTAINER_METHODS
                and isinstance(grandparent, ast.Call)
                and grandparent.func is parent
            )
            is_membership_read = isinstance(parent, ast.Compare) and any(
                comparator is node
                and isinstance(parent.ops[index], (ast.In, ast.NotIn))
                for index, comparator in enumerate(parent.comparators)
            )
            if not (
                is_loop_iterator
                or is_read_subscript
                or is_read_method_call
                or is_membership_read
            ):
                raise AssertionError(
                    f"dynamic SQL evidence producer {symbol!r} has an "
                    "unapproved load context"
                )


def _literal_string(node: ast.AST) -> str:
    assert isinstance(node, ast.Constant) and isinstance(node.value, str), (
        "dynamic SQL target evidence must remain a literal string"
    )
    return node.value


def _container_targets(
    assignment: ast.Assign | ast.AnnAssign,
    evidence: _ContainerTargetEvidence,
) -> frozenset[str]:
    value = assignment.value
    if evidence.mode == "sequence-values":
        assert isinstance(value, (ast.List, ast.Tuple))
        return frozenset(_literal_string(item) for item in value.elts)
    if evidence.mode == "sequence-tuples":
        assert isinstance(value, (ast.List, ast.Tuple))
        assert evidence.tuple_index is not None
        targets: set[str] = set()
        for item in value.elts:
            assert isinstance(item, (ast.List, ast.Tuple))
            targets.add(_literal_string(item.elts[evidence.tuple_index]))
        return frozenset(targets)
    if evidence.mode == "mapping-value-tuples":
        assert isinstance(value, ast.Dict)
        assert evidence.tuple_index is not None
        targets = set()
        for item in value.values:
            assert isinstance(item, (ast.List, ast.Tuple))
            targets.add(_literal_string(item.elts[evidence.tuple_index]))
        return frozenset(targets)
    raise AssertionError(f"unknown dynamic SQL evidence mode: {evidence.mode}")


def _assigned_module_value(tree: ast.Module, name: str) -> ast.expr | None:
    module_nodes = _module_scope_nodes(tree)
    stores = [
        node
        for node in module_nodes
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    imports = [
        (statement, alias)
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if (alias.asname or alias.name) == name
    ]
    bindings = len(stores) + len(imports)
    assert bindings == 1, (
        f"dynamic SQL target constant {name!r} must have exactly one immutable "
        f"binding; found {bindings} bindings"
    )
    if imports:
        return None
    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and _assigned_name(statement) == name
    ]
    assert len(assignments) == 1, (
        f"dynamic SQL target constant {name!r} has unsupported bindings"
    )
    return assignments[0].value


def _resolve_string_expression(
    node: ast.AST,
    tree: ast.Module,
    imported_modules: Mapping[str, ast.Module],
    trace: list[str],
    seen: set[tuple[int, str]],
) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return _resolve_string_name(
            tree,
            node.id,
            imported_modules,
            trace,
            seen,
        )
    if isinstance(node, ast.FormattedValue):
        return _resolve_string_expression(
            node.value,
            tree,
            imported_modules,
            trace,
            seen,
        )
    if isinstance(node, ast.JoinedStr):
        return "".join(
            _resolve_string_expression(
                value,
                tree,
                imported_modules,
                trace,
                seen,
            )
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _resolve_string_expression(
            node.left,
            tree,
            imported_modules,
            trace,
            seen,
        ) + _resolve_string_expression(
            node.right,
            tree,
            imported_modules,
            trace,
            seen,
        )
    raise AssertionError("imported dynamic SQL target stopped resolving to a string")


def _resolve_string_name(
    tree: ast.Module,
    name: str,
    imported_modules: Mapping[str, ast.Module],
    trace: list[str],
    seen: set[tuple[int, str]],
) -> str:
    marker = (id(tree), name)
    assert marker not in seen, f"cyclic dynamic SQL target constant: {name}"
    seen.add(marker)
    assigned = _assigned_module_value(tree, name)
    if assigned is not None:
        trace.append(f"assign:{name}:{ast.dump(assigned, include_attributes=False)}")
        result = _resolve_string_expression(
            assigned,
            tree,
            imported_modules,
            trace,
            seen,
        )
        seen.remove(marker)
        return result
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.module is None:
            continue
        for alias in statement.names:
            bound_name = alias.asname or alias.name
            if bound_name != name:
                continue
            imported_tree = imported_modules.get(statement.module)
            assert imported_tree is not None, (
                f"dynamic SQL target import is outside reviewed modules: "
                f"{statement.module}.{alias.name}"
            )
            trace.append(
                f"import:{statement.module}:{alias.name}:"
                f"{ast.dump(statement, include_attributes=False)}"
            )
            result = _resolve_string_name(
                imported_tree,
                alias.name,
                imported_modules,
                trace,
                seen,
            )
            seen.remove(marker)
            return result
    raise AssertionError(f"dynamic SQL target constant disappeared: {name}")


def _derive_target_evidence(
    tree: ast.Module,
    evidence: object,
    *,
    imported_modules: Mapping[str, ast.Module] | None = None,
) -> _DerivedTargetEvidence:
    modules = imported_modules or {}
    fingerprint_parts: list[str] = []
    if isinstance(evidence, _CallTargetEvidence):
        targets: set[str] = set()
        helper_names = frozenset(target.helper for target in evidence.calls)
        _assert_direct_helper_references(list(ast.walk(tree)), helper_names)
        for qualname, function in _function_nodes(tree):
            function_nodes = _function_body_nodes(function)
            for node in function_nodes:
                if not isinstance(node, ast.Call):
                    continue
                for target in evidence.calls:
                    if _call_name(node) != target.helper:
                        continue
                    argument = _literal_call_argument(node, target)
                    assert argument is not None, (
                        f"reviewed helper call lacks target argument: {target.helper}"
                    )
                    fingerprint_parts.append(
                        f"call:{qualname}:{ast.dump(node, include_attributes=False)}"
                    )
                    if isinstance(argument, ast.Constant) and isinstance(
                        argument.value, str
                    ):
                        targets.add(argument.value)
        assert fingerprint_parts, "reviewed dynamic SQL helper has no call evidence"
        return _DerivedTargetEvidence(
            frozenset(targets),
            _evidence_fingerprint(fingerprint_parts),
        )
    if isinstance(evidence, _ContainerTargetEvidence):
        assignment = _container_assignment(tree, evidence)
        fingerprint_parts.append(
            f"container:{ast.dump(assignment, include_attributes=False)}"
        )
        return _DerivedTargetEvidence(
            _container_targets(assignment, evidence),
            _evidence_fingerprint(fingerprint_parts),
        )
    if isinstance(evidence, _ImportedConstantTargetEvidence):
        targets = {
            _resolve_string_name(tree, name, modules, fingerprint_parts, set())
            for name in evidence.names
        }
        return _DerivedTargetEvidence(
            frozenset(targets),
            _evidence_fingerprint(fingerprint_parts),
        )
    raise AssertionError(f"unknown dynamic SQL target evidence: {evidence!r}")


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


@cache
def _derived_production_target_evidence(
    function_identity: str,
) -> _DerivedTargetEvidence:
    relative_path, _qualname = function_identity.split("::", 1)
    path = REPO_ROOT / relative_path
    modules_by_path = {
        module_path: tree for module_path, tree, _constants in _production_modules()
    }
    modules_by_name = {
        _module_name(module_path): tree
        for module_path, tree, _constants in _production_modules()
    }
    evidence = _DYNAMIC_TARGET_EVIDENCE.get(function_identity)
    assert evidence is not None, (
        f"dynamic SQL review lacks target evidence: {function_identity}"
    )
    return _derive_target_evidence(
        modules_by_path[path],
        evidence,
        imported_modules=modules_by_name,
    )


def _assert_review_target_evidence(
    site_key: str,
    review: _DynamicSqlReview,
    derived: _DerivedTargetEvidence,
) -> None:
    assert derived.targets == review.exact_targets, (
        f"derived dynamic SQL targets changed: {site_key}\n"
        f"expected={sorted(review.exact_targets)}\nactual={sorted(derived.targets)}"
    )
    assert derived.fingerprint == review.evidence_fingerprint, (
        f"dynamic SQL target evidence fingerprint changed: {site_key}\n"
        f"expected={review.evidence_fingerprint}\nactual={derived.fingerprint}"
    )


def _sql_actions_in_text(value: str) -> set[str]:
    value = _SQL_COMMENT.sub(" ", value)
    actions: set[str] = set()
    for match in _MUTATION_SQL.finditer(value):
        verb = match.group("verb").lower()
        if verb.startswith("insert"):
            verb = "insert"
        elif verb.startswith("delete"):
            verb = "delete"
        table = match.group("table").strip('"`[]').lower()
        actions.add(f"sql:{verb}:{table}")
    if _CONVERSATION_HARD_DELETE.search(value):
        actions.add("sql:delete:conversations(cascades-messages)")
    return actions


def _sql_template(
    node: ast.AST,
    constants: Mapping[str, str],
) -> str | None:
    literal = _literal_string_value(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Name):
        return constants.get(node.id, "{expression}")
    if isinstance(node, ast.FormattedValue):
        resolved = _literal_string_value(node.value)
        if resolved is None and isinstance(node.value, ast.Name):
            resolved = constants.get(node.value.id)
        return resolved if resolved is not None else "{expression}"
    if isinstance(node, ast.JoinedStr):
        parts = [_sql_template(value, constants) for value in node.values]
        return None if any(part is None for part in parts) else "".join(parts)  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _sql_template(node.left, constants)
        right = _sql_template(node.right, constants)
        return None if left is None or right is None else left + right
    return None


_DYNAMIC_MUTATION_TARGET = re.compile(
    r"(?:insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"(?:\{expression\}|[^\s]+\.\{expression\})",
    re.IGNORECASE,
)
_SQL_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*(?:\n|$)", re.DOTALL)


def _normalize_site_text(value: str) -> str:
    return " ".join(value.split())


def _scan_sql_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_constants: Mapping[str, str] | None = None,
    *,
    scan_return_values: bool = False,
) -> _SqlFunctionScan:
    constants = dict(module_constants or {})
    actions: set[str] = set()
    unresolved_sites: list[_UnresolvedDynamicSqlSite] = []
    occurrences: dict[tuple[str, str, str], int] = {}
    for node in _function_body_nodes(function):
        assigned_name = _assigned_name(node)
        if assigned_name is not None:
            assigned_value = (
                node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            )
            literal = (
                _sql_template(assigned_value, constants)
                if assigned_value is not None
                else None
            )
            if literal is None:
                constants.pop(assigned_name, None)
            else:
                constants[assigned_name] = literal
        if (
            scan_return_values
            and isinstance(node, ast.Return)
            and node.value is not None
        ):
            return_values = (
                node.value.elts if isinstance(node.value, ast.Tuple) else (node.value,)
            )
            for return_value in return_values:
                template = _sql_template(return_value, constants)
                if template is not None:
                    actions.update(_sql_actions_in_text(template))
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execute", "executemany"}
            and node.args
        ):
            continue
        template = _sql_template(node.args[0], constants)
        if template is None:
            continue
        actions.update(_sql_actions_in_text(template))
        executable_template = _SQL_COMMENT.sub(" ", template)
        if not _DYNAMIC_MUTATION_TARGET.search(executable_template):
            continue
        normalized_template = _normalize_site_text(template)
        source = _normalize_site_text(ast.unparse(node.args[0]))
        occurrence_key = (node.func.attr, normalized_template, source)
        occurrence = occurrences.get(occurrence_key, 0) + 1
        occurrences[occurrence_key] = occurrence
        unresolved_sites.append(
            _UnresolvedDynamicSqlSite(
                executor=node.func.attr,
                occurrence=occurrence,
                template=normalized_template,
                source=source,
            )
        )
    return _SqlFunctionScan(
        actions=frozenset(actions),
        unresolved_sites=tuple(unresolved_sites),
    )


def _sql_actions_in_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_constants: Mapping[str, str] | None = None,
) -> set[str]:
    scan = _scan_sql_function(function, module_constants)
    if scan.unresolved_sites:
        raise AssertionError(
            "unresolved dynamic mutation SQL target; use a literal local/module "
            "constant or extend the census contract"
        )
    return set(scan.actions)


_CANONICAL_SQL_TARGETS = {
    "conversations",
    "messages",
    "message_attachments",
    "message_exchanges",
    "message_generation_metadata",
    "message_trajectory_metadata",
}


def _assert_exact_dynamic_sql_reviews(
    function_identity: str,
    sites: tuple[_UnresolvedDynamicSqlSite, ...],
    reviews: Mapping[str, _DynamicSqlReview],
    *,
    derived_target_evidence: _DerivedTargetEvidence | None = None,
) -> set[str]:
    prefix = f"{function_identity}::dynamic-sql:"
    actual = {site.key(function_identity) for site in sites}
    expected = {key for key in reviews if key.startswith(prefix)}
    unreviewed = sorted(actual - expected)
    stale = sorted(expected - actual)
    assert not unreviewed, f"unreviewed dynamic SQL sites: {unreviewed}"
    assert not stale, f"stale dynamic SQL reviews: {stale}"
    if actual and derived_target_evidence is None:
        assert function_identity in _DYNAMIC_TARGET_EVIDENCE, (
            f"dynamic SQL review lacks checked target evidence: {function_identity}"
        )
        derived_target_evidence = _derived_production_target_evidence(function_identity)
    for key in actual:
        review = reviews[key]
        assert derived_target_evidence is not None
        _assert_review_target_evidence(
            key,
            review,
            derived_target_evidence,
        )
        assert review.domain.strip(), f"dynamic SQL review lacks target domain: {key}"
        assert review.exact_targets, f"dynamic SQL review lacks exact targets: {key}"
        normalized_targets = {
            target.rsplit(".", 1)[-1].strip('"`[]').casefold()
            for target in review.exact_targets
        }
        assert not normalized_targets & _CANONICAL_SQL_TARGETS, (
            f"dynamic SQL review cannot exempt canonical targets: {key}"
        )
    return actual


@cache
def _direct_sql_routes() -> frozenset[str]:
    routes: set[_Route] = set()
    seen_dynamic_reviews: set[str] = set()
    constants_by_path = {
        path: dict(constants) for path, _tree, constants in _production_modules()
    }
    for path, qualname, function in _production_functions():
        if function.name.startswith("_migrate_from_"):
            continue
        identity = f"{_relative(path)}::{qualname}"
        scan = _scan_sql_function(
            function,
            constants_by_path[path],
            scan_return_values=identity in _SQL_RETURN_HELPERS,
        )
        seen_dynamic_reviews.update(
            _assert_exact_dynamic_sql_reviews(
                identity,
                scan.unresolved_sites,
                _PROVEN_NON_SEMANTIC_DYNAMIC_SQL,
            )
        )
        for action in scan.actions:
            routes.add(_Route(_relative(path), qualname, action))
    assert seen_dynamic_reviews == _PROVEN_NON_SEMANTIC_DYNAMIC_SQL.keys(), (
        "reviewed non-semantic dynamic SQL sites changed; remove stale entries or "
        "review and classify new sites\n"
        f"missing={sorted(_PROVEN_NON_SEMANTIC_DYNAMIC_SQL.keys() - seen_dynamic_reviews)}"
        f"\nunreviewed={sorted(seen_dynamic_reviews - _PROVEN_NON_SEMANTIC_DYNAMIC_SQL.keys())}"
    )
    return frozenset(route.key for route in routes)


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return None


def _is_require_db_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _attribute_chain(node.func.value) == ("self",)
        and node.func.attr == "_require_db"
    )


def _boundary_action_for_receiver(
    method: str,
    receiver: ast.AST,
    qualname: str,
    database_aliases: AbstractSet[str] = frozenset(),
) -> str | None:
    chain = _attribute_chain(receiver)

    if method in _DB_MUTATORS:
        if (
            chain in {("db",), ("self", "db")}
            or _is_require_db_call(receiver)
            or (isinstance(receiver, ast.Name) and receiver.id in database_aliases)
        ):
            return f"call:db:{method}"
        if chain == ("self",) and qualname.startswith("CharactersRAGDB."):
            return f"call:db:{method}"
        if (
            chain == ("self", "store")
            and qualname.startswith("_ContinuationValidatingChatStore.")
            and method in _CHAT_SYNC_HELPER_MUTATORS
        ):
            return f"call:db:{method}"

    if method in _PERSISTENCE_MUTATORS:
        if chain and chain[-1] in {"persistence", "persistence_service"}:
            return f"call:persistence:{method}"
        if chain == ("self",) and qualname.startswith("ChatPersistenceService."):
            return f"call:persistence:{method}"

    if (
        method in _DISPATCH_MUTATORS
        and chain
        and chain[-1]
        in {
            "console_dispatch_repository",
            "repository",
        }
    ):
        return f"call:dispatch:{method}"
    return None


def _boundary_action(
    call: ast.Call,
    qualname: str,
    database_aliases: AbstractSet[str] = frozenset(),
) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    return _boundary_action_for_receiver(
        call.func.attr,
        call.func.value,
        qualname,
        database_aliases,
    )


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _database_handle_alias(node: ast.AST) -> str | None:
    alias = _assigned_name(node)
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    if (
        isinstance(value, ast.IfExp)
        and isinstance(value.orelse, ast.Constant)
        and value.orelse.value is None
    ):
        value = value.body
    if (
        alias is None
        or not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "getattr"
        or len(value.args) < 2
        or not isinstance(value.args[1], ast.Constant)
        or value.args[1].value != "db"
    ):
        return None
    receiver_chain = _attribute_chain(value.args[0])
    if receiver_chain in {
        ("persistence",),
        ("persistence_service",),
        ("self", "persistence"),
        ("self", "persistence_service"),
    }:
        return alias
    return None


def _literal_getattr_boundary_alias(
    node: ast.AST,
    qualname: str,
    database_aliases: AbstractSet[str] = frozenset(),
) -> tuple[str, str] | None:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    value = node.value
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "getattr"
        or len(value.args) < 2
        or not isinstance(value.args[1], ast.Constant)
        or not isinstance(value.args[1].value, str)
    ):
        return None
    alias = _assigned_name(node)
    if alias is None:
        return None
    action = _boundary_action_for_receiver(
        value.args[1].value,
        value.args[0],
        qualname,
        database_aliases,
    )
    if action is None:
        return None
    return alias, action


def _literal_chat_sync_dispatch_action(
    call: ast.Call,
    qualname: str,
) -> str | None:
    if (
        qualname != "ChatSyncAdapter.apply"
        or not isinstance(call.func, ast.Name)
        or call.func.id != "call_if_present"
        or len(call.args) < 2
        or not isinstance(call.args[0], ast.Name)
        or call.args[0].id != "local_store"
        or not isinstance(call.args[1], ast.Constant)
        or call.args[1].value not in _CHAT_SYNC_HELPER_MUTATORS
    ):
        return None
    return f"call:db:{call.args[1].value}"


def _bound_method_argument_actions(
    call: ast.Call,
    qualname: str,
) -> set[str]:
    if (
        not isinstance(call.func, ast.Attribute)
        or (
            qualname,
            call.func.attr,
        )
        not in _BOUND_METHOD_RUNNERS
    ):
        return set()
    actions: set[str] = set()
    for argument in (
        *call.args,
        *(keyword.value for keyword in call.keywords if keyword.arg is not None),
    ):
        if not isinstance(argument, ast.Attribute):
            continue
        action = _boundary_action_for_receiver(
            argument.attr,
            argument.value,
            qualname,
        )
        if action is not None:
            actions.add(action)
    return actions


def _carried_writer_actions_in_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
) -> Mapping[tuple[str, str, str], frozenset[str]]:
    actions: dict[tuple[str, str, str], set[str]] = {}
    database_aliases: set[str] = set()
    writer_aliases: dict[str, str] = {}
    for node in _function_body_nodes(function):
        alias = _assigned_name(node)
        if alias is not None:
            if _database_handle_alias(node) == alias:
                database_aliases.add(alias)
            else:
                database_aliases.discard(alias)
            alias_action = _literal_getattr_boundary_alias(
                node,
                qualname,
                database_aliases,
            )
            if alias_action is None:
                writer_aliases.pop(alias, None)
            else:
                writer_aliases[alias_action[0]] = alias_action[1]
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        for (
            carrier_name,
            writer_field,
        ), consumer_spec in _CARRIED_WRITER_SPECS.items():
            if node.func.id != carrier_name:
                continue
            writer_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == writer_field),
                None,
            )
            if writer_keyword is None or not isinstance(writer_keyword.value, ast.Name):
                continue
            action = writer_aliases.get(writer_keyword.value.id)
            if action is not None:
                actions.setdefault((*consumer_spec, writer_field), set()).add(action)
    return MappingProxyType(
        {consumer_call: frozenset(values) for consumer_call, values in actions.items()}
    )


@cache
def _carried_writer_actions() -> Mapping[tuple[str, str, str], frozenset[str]]:
    actions: dict[tuple[str, str, str], set[str]] = {}
    for _path, qualname, function in _production_functions():
        for consumer_call, values in _carried_writer_actions_in_function(
            function,
            qualname,
        ).items():
            actions.setdefault(consumer_call, set()).update(values)
    return MappingProxyType(
        {consumer_call: frozenset(values) for consumer_call, values in actions.items()}
    )


def _annotation_name(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None


def _carried_boundary_actions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    nodes: list[ast.AST],
    carried_writer_actions: Mapping[tuple[str, str, str], frozenset[str]],
) -> set[str]:
    annotated_arguments = {
        argument.arg: annotation_name
        for argument in (*function.args.posonlyargs, *function.args.args)
        if (annotation_name := _annotation_name(argument.annotation)) is not None
    }
    carrier_aliases: dict[str, tuple[str, str]] = {}
    for node in nodes:
        if not (
            isinstance(node, (ast.For, ast.AsyncFor))
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Attribute)
            and isinstance(node.iter.value, ast.Name)
        ):
            continue
        consumer_spec = (
            annotated_arguments.get(node.iter.value.id),
            node.iter.attr,
        )
        if any(
            action_key[:2] == consumer_spec for action_key in carried_writer_actions
        ):
            carrier_aliases[node.target.id] = consumer_spec

    actions: set[str] = set()
    for node in nodes:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            continue
        consumer_spec = carrier_aliases.get(node.func.value.id)
        if consumer_spec is None:
            continue
        consumer_call = (*consumer_spec, node.func.attr)
        if consumer_call in carried_writer_actions:
            actions.update(carried_writer_actions[consumer_call])
    return actions


def _function_boundary_actions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
    carried_writer_actions: Mapping[tuple[str, str, str], frozenset[str]] | None = None,
) -> set[str]:
    nodes = _function_body_nodes(function)
    database_aliases: set[str] = set()
    aliases: dict[str, str] = {}
    actions = _carried_boundary_actions(
        function,
        nodes,
        (
            carried_writer_actions
            if carried_writer_actions is not None
            else _carried_writer_actions()
        ),
    )
    for node in nodes:
        alias = _assigned_name(node)
        if alias is not None:
            if _database_handle_alias(node) == alias:
                database_aliases.add(alias)
            else:
                database_aliases.discard(alias)
            alias_action = _literal_getattr_boundary_alias(
                node,
                qualname,
                database_aliases,
            )
            if alias_action is None:
                aliases.pop(alias, None)
            else:
                aliases[alias_action[0]] = alias_action[1]
        if not isinstance(node, ast.Call):
            continue
        action = _boundary_action(node, qualname, database_aliases)
        if action is not None:
            actions.add(action)
        helper_action = _literal_chat_sync_dispatch_action(node, qualname)
        if helper_action is not None:
            actions.add(helper_action)
        actions.update(_bound_method_argument_actions(node, qualname))
        if isinstance(node.func, ast.Name) and node.func.id in aliases:
            actions.add(aliases[node.func.id])
    return actions


@cache
def _boundary_call_routes() -> frozenset[str]:
    routes: set[_Route] = set()
    carried_writer_actions = _carried_writer_actions()
    for path, qualname, function in _production_functions():
        for action in _function_boundary_actions(
            function,
            qualname,
            carried_writer_actions,
        ):
            routes.add(_Route(_relative(path), qualname, action))
    return frozenset(route.key for route in routes)


def _assert_exact_routes(
    actual: AbstractSet[str],
    classified: Mapping[str, str],
) -> None:
    assert set(classified.values()) <= CLASSIFICATIONS
    missing = sorted(actual - classified.keys())
    stale = sorted(classified.keys() - actual)
    assert not missing and not stale, (
        "semantic mutation inventory changed; classify every new route and remove "
        f"stale rows\nunclassified={missing}\nstale={stale}"
    )


_INVENTORY_SECTION_HEADERS = {
    "## Exact live SQL sink census": "sql",
    "## Exact boundary-call census": "boundary",
}
_INVENTORY_ROW = re.compile(r"^- `(?P<route>[^`]+)` — (?P<classification>[^\s]+)$")


def _parse_inventory_census(inventory: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"sql": {}, "boundary": {}}
    active: str | None = None
    for line in inventory.splitlines():
        if line in _INVENTORY_SECTION_HEADERS:
            active = _INVENTORY_SECTION_HEADERS[line]
            continue
        if line.startswith("## "):
            active = None
            continue
        if active is None or not line.startswith("- `"):
            continue
        match = _INVENTORY_ROW.fullmatch(line)
        assert match is not None, f"malformed exact-census row: {line!r}"
        route = match.group("route")
        assert route not in sections[active], (
            f"duplicate exact-census route in {active}: {route}"
        )
        sections[active][route] = match.group("classification")
    return sections


def test_live_sql_mutation_sites_are_classified() -> None:
    _assert_exact_routes(_direct_sql_routes(), DIRECT_SQL_ROUTE_CLASSIFICATION)


def test_public_mutation_boundary_calls_are_classified() -> None:
    _assert_exact_routes(_boundary_call_routes(), BOUNDARY_CALL_ROUTE_CLASSIFICATION)


def test_boundary_scanner_resolves_console_soft_delete_getattr_alias() -> None:
    assert (
        "tldw_chatbook/Chat/console_chat_store.py::"
        "ConsoleChatStore._delete_message::call:persistence:delete_message_subtree"
        in _boundary_call_routes()
    )


def test_boundary_scanner_resolves_roleplay_projection_writer_carrier() -> None:
    assert (
        "tldw_chatbook/Chat/console_chat_store.py::"
        "ConsoleChatStore.persist_roleplay_projection_plan::"
        "call:persistence:update_message_content" in _boundary_call_routes()
    )


def test_boundary_scanner_resolves_reviewed_residual_writer_routes() -> None:
    expected = {
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._discard_provider_continuation::call:db:update_provider_continuation",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_exchanges_only_locked::call:persistence:append_message_exchanges",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_metadata_only::call:persistence:update_message_metadata",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._persist_usage_only::call:persistence:update_message_usage",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore._promote_ephemeral_session_atomically::call:persistence:promote_console_conversation_bundle",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_durable_turn::call:persistence:commit_durable_turn",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.commit_full_capture_purge::call:persistence:delete_full_exchanges_for_conversation",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:create_assistant_with_continuation",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.persist_provider_continuation_event::call:db:update_provider_continuation",
        "tldw_chatbook/Chat/console_chat_store.py::ConsoleChatStore.write_trajectory_rows::call:persistence:write_trajectory_rows",
        "tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:append_chat_message",
        "tldw_chatbook/Sync_Interop/domain_adapters/chat.py::ChatSyncAdapter.apply::call:db:delete_chat_message",
        "tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.append_chat_message::call:db:append_chat_message",
        "tldw_chatbook/Sync_Interop/envelope_applier.py::_ContinuationValidatingChatStore.delete_chat_message::call:db:delete_chat_message",
        "tldw_chatbook/UI/Console_Modules/session.py::ConsoleSessionController._commit_durable_console_chat_fork::call:persistence:fork_console_conversation_bundle",
    }
    missing = expected - _boundary_call_routes()
    assert not missing, f"reviewed mutation routes remain uncensused: {sorted(missing)}"


def test_inventory_document_exists_and_names_the_contract() -> None:
    inventory = INVENTORY_PATH.read_text(encoding="utf-8")
    assert "ADR-097" in inventory
    assert "Hard deletion" in inventory
    assert "Generated and dynamic SQL" in inventory
    assert "39 live SQL sink identities and 64 boundary call" in inventory
    all_classifications = (
        *DIRECT_SQL_ROUTE_CLASSIFICATION.values(),
        *BOUNDARY_CALL_ROUTE_CLASSIFICATION.values(),
    )
    assert {
        classification: all_classifications.count(classification)
        for classification in CLASSIFICATIONS
    } == {
        "model-visible": 66,
        "visibility/ownership-only": 11,
        "presentation-only": 26,
    }
    assert "66 model-visible, 11 visibility/ownership-only, and 26" in inventory
    for phrase in (
        "generation settlement",
        "Edit and regeneration replacement",
        "Import",
        "Sync create/update",
        "Attachment mutation",
        "soft delete",
        "hard-delete route",
    ):
        assert phrase in inventory
    for classified in (
        DIRECT_SQL_ROUTE_CLASSIFICATION,
        BOUNDARY_CALL_ROUTE_CLASSIFICATION,
    ):
        for route, classification in classified.items():
            assert f"`{route}` — {classification}" in inventory


def test_scanner_ignores_docstrings_and_nested_functions() -> None:
    tree = ast.parse(
        '''
def outer():
    """UPDATE messages SET content = 'not SQL'"""
    def nested():
        return "DELETE FROM messages"
    return "SELECT 1"
'''
    )
    outer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "outer"
    )
    values = {
        value
        for node in _function_body_nodes(outer)
        if (value := _string_value(node)) is not None
    }
    assert not any(_MUTATION_SQL.search(value) for value in values)


def test_boundary_scanner_rejects_unrelated_add_message_receivers() -> None:
    calls = [
        node
        for node in ast.walk(
            ast.parse(
                "db.add_message({}); self.db.update_message('x', {}, 1); "
                "widget.add_message('hello')"
            )
        )
        if isinstance(node, ast.Call)
    ]
    actions = {_boundary_action(call, "Owner.write") for call in calls}
    assert actions == {"call:db:add_message", "call:db:update_message", None}


def test_getattr_alias_scanner_is_receiver_and_literal_method_scoped() -> None:
    tree = ast.parse(
        """
def write():
    persisted_delete = getattr(self.persistence, "delete_message_subtree", None)
    widget_writer = getattr(widget, "add_message", None)
    dynamic_writer = getattr(self.persistence, method_name, None)
    persisted_delete(message_id="message-1")
    widget_writer("hello")
    dynamic_writer(message_id="message-1")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    assert _function_boundary_actions(function, "Owner.write") == {
        "call:persistence:delete_message_subtree"
    }


def test_carried_writer_scanner_is_plan_annotation_and_field_scoped() -> None:
    tree = ast.parse(
        """
def expected(plan: ConsoleRoleplayProjectionPersistencePlan):
    for renamed_write in plan.message_writes:
        renamed_write.writer()

def unrelated_plan(plan: UnrelatedPlan):
    for item in plan.message_writes:
        item.writer()

def unrelated_collection(plan: ConsoleRoleplayProjectionPersistencePlan):
    for item in plan.other_writes:
        item.writer()

def unrelated_field(plan: ConsoleRoleplayProjectionPersistencePlan):
    for item in plan.message_writes:
        item.callback()
"""
    )
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    carried_actions = {
        (
            "ConsoleRoleplayProjectionPersistencePlan",
            "message_writes",
            "writer",
        ): frozenset({"call:persistence:update_message_content"})
    }

    assert _carried_boundary_actions(
        functions["expected"],
        _function_body_nodes(functions["expected"]),
        carried_actions,
    ) == {"call:persistence:update_message_content"}
    for name in ("unrelated_plan", "unrelated_collection", "unrelated_field"):
        assert not _carried_boundary_actions(
            functions[name],
            _function_body_nodes(functions[name]),
            carried_actions,
        )


def test_carried_writer_producer_aliases_are_lexical_and_immutable() -> None:
    tree = ast.parse(
        """
def prepare():
    writer = getattr(self.persistence, "create_message", None)
    _RoleplayMessageProjectionWrite(writer=writer)
    writer = getattr(self.persistence, "update_message_content", None)
    _RoleplayMessageProjectionWrite(writer=writer)
    writer = unrelated_callback
    _RoleplayMessageProjectionWrite(writer=writer)
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )

    actions = _carried_writer_actions_in_function(function, "Owner.prepare")

    assert actions == {
        (
            "ConsoleRoleplayProjectionPersistencePlan",
            "message_writes",
            "writer",
        ): frozenset(
            {
                "call:persistence:create_message",
                "call:persistence:update_message_content",
            }
        )
    }


def test_residual_indirection_scanner_rejects_unrelated_shapes() -> None:
    tree = ast.parse(
        """
def unrelated():
    database = getattr(widget, "db", None)
    updater = getattr(database, "update_provider_continuation", None)
    updater()
    call_if_present(local_store, "append_chat_message")
    self._run_fork_io(persistence.fork_console_conversation_bundle)

def computed_method():
    database = getattr(self.persistence, "db", None) if self.persistence else None
    updater = getattr(database, method_name, None)
    updater()

def wrong_sync_helper():
    dispatch_if_present(local_store, "append_chat_message")
    call_if_present(local_store, "update_message")
    call_if_present(widget, "append_chat_message")
"""
    )
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert not _function_boundary_actions(functions["unrelated"], "Other.write")
    assert not _function_boundary_actions(
        functions["computed_method"],
        "Other.computed_method",
    )
    assert not _function_boundary_actions(
        functions["wrong_sync_helper"],
        "ChatSyncAdapter.apply",
    )


def test_boundary_identity_collapses_repeated_calls_in_one_function() -> None:
    tree = ast.parse(
        """
def flush():
    writer = getattr(self.persistence, "update_message_usage", None)
    writer(message_id="message-1")
    writer(message_id="message-2")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "flush"
    )
    assert _function_boundary_actions(function, "Owner.flush") == {
        "call:persistence:update_message_usage"
    }


def test_getattr_alias_scanner_tracks_reassignment_at_each_call() -> None:
    tree = ast.parse(
        """
def write():
    writer = getattr(self.persistence, "create_message", None)
    writer(message_id="message-1")
    writer = getattr(self.persistence, "update_message_content", None)
    writer(message_id="message-1")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    assert _function_boundary_actions(function, "Owner.write") == {
        "call:persistence:create_message",
        "call:persistence:update_message_content",
    }


def test_getattr_alias_scanner_invalidates_stale_binding_on_reassignment() -> None:
    tree = ast.parse(
        """
def write():
    writer = getattr(self.persistence, "delete_message_subtree", None)
    writer = unrelated_callback
    writer(message_id="message-1")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    assert not _function_boundary_actions(function, "Owner.write")


def test_database_handle_alias_is_lexical_and_invalidated_on_reassignment() -> None:
    tree = ast.parse(
        """
def before_binding():
    updater = getattr(database, "update_provider_continuation", None)
    updater()
    database = getattr(self.persistence, "db", None)

def after_invalidation():
    database = getattr(self.persistence, "db", None)
    database = widget
    database.update_message("message-1", {}, 1)
"""
    )
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert not _function_boundary_actions(
        functions["before_binding"],
        "Owner.before_binding",
    )
    assert not _function_boundary_actions(
        functions["after_invalidation"],
        "Owner.after_invalidation",
    )


def test_bound_method_runner_scans_positional_and_keyword_values() -> None:
    tree = ast.parse(
        """
def commit():
    self._run_fork_io(
        persistence.fork_console_conversation_bundle,
        writer=persistence.promote_console_conversation_bundle,
    )
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "commit"
    )

    assert _function_boundary_actions(
        function,
        "ConsoleSessionController._commit_durable_console_chat_fork",
    ) == {
        "call:persistence:fork_console_conversation_bundle",
        "call:persistence:promote_console_conversation_bundle",
    }


def test_mutation_sql_regex_accepts_quoted_and_schema_qualified_targets() -> None:
    statements = {
        'UPDATE "messages" SET content = ?': ("update", '"messages"'),
        "UPDATE main.messages SET content = ?": ("update", "messages"),
        "DELETE FROM [main].[message_attachments] WHERE message_id = ?": (
            "delete from",
            "[message_attachments]",
        ),
        "INSERT INTO `main`.`message_exchanges` (message_id) VALUES (?)": (
            "insert into",
            "`message_exchanges`",
        ),
    }

    for statement, expected in statements.items():
        match = _MUTATION_SQL.search(statement)
        assert match is not None, statement
        assert (match.group("verb").lower(), match.group("table")) == expected


def test_boundary_scanner_rejects_unknown_nested_db_receiver() -> None:
    calls = [
        node
        for node in ast.walk(
            ast.parse(
                "widget.db.update_message('x', {}, 1); "
                "widget._require_db().update_message('x', {}, 1)"
            )
        )
        if isinstance(node, ast.Call)
    ]

    assert {_boundary_action(call, "Owner.write") for call in calls} == {None}


def test_sql_scanner_resolves_module_constant_used_by_executor() -> None:
    tree = ast.parse(
        """
MODULE_SQL = 'UPDATE main."messages" SET content = ? WHERE id = ?'
UNUSED_SQL = "DELETE FROM message_attachments WHERE message_id = ?"

def write(cursor):
    cursor.execute(MODULE_SQL, ("hello", "message-1"))
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    assert _sql_actions_in_function(
        function,
        {
            "MODULE_SQL": 'UPDATE main."messages" SET content = ? WHERE id = ?',
            "UNUSED_SQL": "DELETE FROM message_attachments WHERE message_id = ?",
        },
    ) == {"sql:update:messages"}


def test_sql_scanner_resolves_local_prefix_before_rejecting_dynamic_target() -> None:
    tree = ast.parse(
        """
def write(cursor, table_name):
    mutation_prefix = "UPDATE "
    cursor.execute(mutation_prefix + table_name + " SET content = ?", ("hello",))
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    with pytest.raises(AssertionError, match="unresolved dynamic mutation SQL target"):
        _sql_actions_in_function(function)


def test_dynamic_sql_review_is_site_exact_and_preserves_literal_actions() -> None:
    tree = ast.parse(
        """
def write(cursor, table_name):
    cursor.execute("UPDATE messages SET content = ?", ("hello",))
    cursor.execute(f"UPDATE {table_name} SET deleted = 1")

def write_fixture_rows(cursor):
    write(cursor, "fixture_rows")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    identity = "fixture.py::Owner.write"

    scan = _scan_sql_function(function)
    site = scan.unresolved_sites[0]
    derived = _derive_target_evidence(
        tree,
        _CallTargetEvidence(calls=(_LiteralCallTarget("write", positional_index=1),)),
    )
    reviews = {
        site.key(identity): _DynamicSqlReview(
            domain="fixture non-chat rows",
            exact_targets=frozenset({"fixture_rows"}),
            evidence_fingerprint=derived.fingerprint,
        )
    }

    _assert_exact_dynamic_sql_reviews(
        identity,
        scan.unresolved_sites,
        reviews,
        derived_target_evidence=derived,
    )
    assert scan.actions == frozenset({"sql:update:messages"})


def test_dynamic_sql_review_rejects_a_second_unresolved_site() -> None:
    tree = ast.parse(
        """
def write(cursor, first_table, second_table):
    cursor.execute(f"UPDATE {first_table} SET deleted = 1")
    cursor.execute(f"DELETE FROM {second_table} WHERE deleted = 1")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    identity = "fixture.py::Owner.write"
    scan = _scan_sql_function(function)
    first = scan.unresolved_sites[0]
    reviews = {
        first.key(identity): _DynamicSqlReview(
            domain="fixture non-chat rows",
            exact_targets=frozenset({"fixture_rows"}),
        )
    }

    with pytest.raises(AssertionError, match="unreviewed dynamic SQL sites"):
        _assert_exact_dynamic_sql_reviews(identity, scan.unresolved_sites, reviews)


def test_dynamic_sql_review_rejects_retargeting_to_messages() -> None:
    dynamic_tree = ast.parse(
        """
def write(cursor, table_name):
    cursor.execute(f"UPDATE {table_name} SET deleted = 1")
"""
    )
    retargeted_tree = ast.parse(
        """
def write(cursor):
    cursor.execute("UPDATE messages SET deleted = 1")
"""
    )
    dynamic_function = next(
        node
        for node in ast.walk(dynamic_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    retargeted_function = next(
        node
        for node in ast.walk(retargeted_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    identity = "fixture.py::Owner.write"
    dynamic_scan = _scan_sql_function(dynamic_function)
    reviews = {
        dynamic_scan.unresolved_sites[0].key(identity): _DynamicSqlReview(
            domain="fixture non-chat rows",
            exact_targets=frozenset({"fixture_rows"}),
        )
    }
    retargeted_scan = _scan_sql_function(retargeted_function)

    with pytest.raises(AssertionError, match="stale dynamic SQL reviews"):
        _assert_exact_dynamic_sql_reviews(
            identity,
            retargeted_scan.unresolved_sites,
            reviews,
        )
    assert retargeted_scan.actions == frozenset({"sql:update:messages"})


def test_dynamic_target_evidence_checks_literal_helper_call_arguments() -> None:
    baseline = ast.parse(
        """
def create_keyword():
    self._add_generic_item("keywords", "keyword")

def create_collection():
    self._add_generic_item(table_name="keyword_collections", unique_col_name="name")
"""
    )
    retargeted = ast.parse(
        """
def create_keyword():
    self._add_generic_item("keywords", "keyword")

def create_collection():
    self._add_generic_item(table_name="messages", unique_col_name="name")
"""
    )
    evidence = _CallTargetEvidence(
        calls=(
            _LiteralCallTarget(
                helper="_add_generic_item",
                positional_index=0,
                keyword="table_name",
            ),
        )
    )
    baseline_result = _derive_target_evidence(baseline, evidence)
    review = _DynamicSqlReview(
        domain="keyword library records",
        exact_targets=frozenset({"keywords", "keyword_collections"}),
        evidence_fingerprint=baseline_result.fingerprint,
    )

    _assert_review_target_evidence("fixture::add", review, baseline_result)
    with pytest.raises(AssertionError, match="derived dynamic SQL targets changed"):
        _assert_review_target_evidence(
            "fixture::add",
            review,
            _derive_target_evidence(retargeted, evidence),
        )


def test_dynamic_target_evidence_derives_literal_container_producers() -> None:
    cases = [
        (
            """
def write():
    child_tables = [("Transcripts", "media_id"), ("MediaChunks", "media_id")]
""",
            _ContainerTargetEvidence(
                symbol="child_tables",
                mode="sequence-tuples",
                tuple_index=0,
                function_qualname="write",
            ),
            frozenset({"Transcripts", "MediaChunks"}),
        ),
        (
            'TABLES = ("eval_probe_turn_annotations", "eval_probe_review_state")',
            _ContainerTargetEvidence(symbol="TABLES", mode="sequence-values"),
            frozenset({"eval_probe_turn_annotations", "eval_probe_review_state"}),
        ),
        (
            'TABLES = {"project": ("writing_projects", "project"), '
            '"scene": ("writing_scenes", "scene")}',
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
            frozenset({"writing_projects", "writing_scenes"}),
        ),
    ]

    for source, evidence, expected in cases:
        result = _derive_target_evidence(ast.parse(source), evidence)
        assert result.targets == expected
        assert result.fingerprint


def test_dynamic_target_evidence_rejects_messages_added_to_media_children() -> None:
    evidence = _ContainerTargetEvidence(
        symbol="child_tables",
        mode="sequence-tuples",
        tuple_index=0,
        function_qualname="write",
    )
    baseline = _derive_target_evidence(
        ast.parse(
            """
def write():
    child_tables = [("Transcripts", "media_id"), ("MediaChunks", "media_id")]
"""
        ),
        evidence,
    )
    review = _DynamicSqlReview(
        domain="media cascade children",
        exact_targets=baseline.targets,
        evidence_fingerprint=baseline.fingerprint,
    )
    mutated = _derive_target_evidence(
        ast.parse(
            """
def write():
    child_tables = [
        ("Transcripts", "media_id"),
        ("MediaChunks", "media_id"),
        ("messages", "media_id"),
    ]
"""
        ),
        evidence,
    )

    with pytest.raises(AssertionError, match="derived dynamic SQL targets changed"):
        _assert_review_target_evidence("fixture::media", review, mutated)


@pytest.mark.parametrize(
    "mutation",
    [
        'child_tables.append(("messages", "media_id"))',
        'child_tables.extend([("messages", "media_id")])',
        'child_tables[0] = ("messages", "media_id")',
    ],
)
def test_dynamic_target_evidence_rejects_container_mutation(
    mutation: str,
) -> None:
    tree = ast.parse(
        f"""
def write():
    child_tables = [("Transcripts", "media_id")]
    {mutation}
"""
    )

    with pytest.raises(AssertionError, match="producer.*mutated"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="child_tables",
                mode="sequence-tuples",
                tuple_index=0,
                function_qualname="write",
            ),
        )


def test_dynamic_target_evidence_rejects_mapping_subscript_mutation() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}
TABLES["message"] = ("messages", "message")
"""
    )

    with pytest.raises(AssertionError, match="producer.*mutated"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


@pytest.mark.parametrize(
    "forwarding",
    [
        "alias = child_tables\n    alias.append(('messages', 'media_id'))",
        "mutate = child_tables.append\n    mutate(('messages', 'media_id'))",
        "list.append(child_tables, ('messages', 'media_id'))",
    ],
)
def test_dynamic_target_evidence_rejects_container_load_forwarding(
    forwarding: str,
) -> None:
    tree = ast.parse(
        f"""
def write():
    child_tables = [("Transcripts", "media_id")]
    {forwarding}
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="child_tables",
                mode="sequence-tuples",
                tuple_index=0,
                function_qualname="write",
            ),
        )


def test_dynamic_target_evidence_allows_proven_container_read_contexts() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}
FIRST = TABLES["project"]

def read_tables():
    for item in TABLES:
        consume(item)
    return TABLES.get("project"), TABLES.items()
"""
    )

    result = _derive_target_evidence(
        tree,
        _ContainerTargetEvidence(
            symbol="TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        ),
    )

    assert result.targets == frozenset({"writing_projects"})


def test_dynamic_target_evidence_rejects_global_container_rebinding() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

def retarget():
    global TABLES
    TABLES = {"message": ("messages", "message")}
"""
    )

    with pytest.raises(AssertionError, match="producer.*rebound"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


@pytest.mark.parametrize(
    "definition",
    [
        "def mutate(alias=TABLES):\n    alias['message'] = ('messages', 'message')",
        "def mutate(*, alias=TABLES):\n    alias['message'] = ('messages', 'message')",
        "@decorate(TABLES)\ndef mutate():\n    pass",
        "def mutate(alias: TABLES):\n    pass",
        "def mutate() -> TABLES:\n    pass",
    ],
)
def test_dynamic_target_evidence_rejects_definition_time_container_capture(
    definition: str,
) -> None:
    tree = ast.parse(
        f"""
TABLES = {{"project": ("writing_projects", "project")}}
{definition}
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_rejects_class_body_container_capture() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    alias = TABLES

Holder.alias["message"] = ("messages", "message")
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_comprehension_target_does_not_mask_module_load() -> (
    None
):
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

def mutate():
    [None for TABLES in ()]
    alias = TABLES
    alias["message"] = ("messages", "message")
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


@pytest.mark.parametrize(
    "local_scope",
    [
        """
def local_function():
    TABLES = {"message": ("messages", "message")}
    return TABLES.get("message")
""",
        "LOCAL_LOOKUP = lambda TABLES: TABLES.get('message')",
        """
LOCAL_LOOKUP = lambda: (
    (TABLES := {"message": ("messages", "message")}),
    TABLES.get("message"),
)
""",
        """
class Holder:
    TABLES = {"message": ("messages", "message")}
    alias = TABLES
""",
    ],
)
def test_dynamic_target_evidence_allows_true_local_container_shadowing(
    local_scope: str,
) -> None:
    tree = ast.parse(
        f"""
TABLES = {{"project": ("writing_projects", "project")}}
{local_scope}
"""
    )

    result = _derive_target_evidence(
        tree,
        _ContainerTargetEvidence(
            symbol="TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        ),
    )

    assert result.targets == frozenset({"writing_projects"})


def test_dynamic_target_evidence_allows_progressive_comprehension_scope() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

def read_tables():
    return [TABLES for TABLES in TABLES]
"""
    )

    result = _derive_target_evidence(
        tree,
        _ContainerTargetEvidence(
            symbol="TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        ),
    )

    assert result.targets == frozenset({"writing_projects"})


def test_dynamic_target_evidence_rejects_class_capture_before_local_binding() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    alias = TABLES
    TABLES = {"message": ("messages", "message")}
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_class_annotation_does_not_mask_module_load() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    TABLES: dict
    alias = TABLES
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_rejects_class_augassign_read_before_bind() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    TABLES |= {"message": ("messages", "message")}
"""
    )

    with pytest.raises(AssertionError, match="producer.*read before binding"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_class_delete_restores_module_fallback() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    TABLES = {"message": ("messages", "message")}
    del TABLES
    consume(TABLES)
"""
    )

    with pytest.raises(AssertionError, match="producer.*unapproved load"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_rejects_nested_class_if_augassign() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    if enabled:
        TABLES |= {"message": ("messages", "message")}
"""
    )

    with pytest.raises(AssertionError, match="producer.*compound class statement"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_rejects_nested_class_delete_fallback() -> None:
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

class Holder:
    TABLES = {"message": ("messages", "message")}
    if enabled:
        del TABLES
    consume(TABLES)
"""
    )

    with pytest.raises(AssertionError, match="producer.*compound class statement"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


@pytest.mark.parametrize(
    "compound_statement",
    [
        """
try:
    TABLES = {"message": ("messages", "message")}
except LookupError:
    pass
""",
        """
with container_scope():
    del TABLES
""",
        """
match event:
    case "mutate":
        TABLES |= {"message": ("messages", "message")}
""",
    ],
    ids=("try-binding", "with-delete", "match-read-write"),
)
def test_dynamic_target_evidence_rejects_reviewed_symbol_in_compound_class_statement(
    compound_statement: str,
) -> None:
    indented_statement = textwrap.indent(compound_statement.strip(), "    ")
    tree = ast.parse(
        f"""
TABLES = {{"project": ("writing_projects", "project")}}

class Holder:
{indented_statement}
"""
    )

    with pytest.raises(AssertionError, match="producer.*compound class statement"):
        _derive_target_evidence(
            tree,
            _ContainerTargetEvidence(
                symbol="TABLES",
                mode="mapping-value-tuples",
                tuple_index=0,
            ),
        )


def test_dynamic_target_evidence_allows_comprehension_namedexpr_function_local() -> (
    None
):
    tree = ast.parse(
        """
TABLES = {"project": ("writing_projects", "project")}

def local_function(rows):
    [(TABLES := row) for row in rows]
    consume(TABLES)
"""
    )

    result = _derive_target_evidence(
        tree,
        _ContainerTargetEvidence(
            symbol="TABLES",
            mode="mapping-value-tuples",
            tuple_index=0,
        ),
    )

    assert result.targets == frozenset({"writing_projects"})


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "def save(self, conn, event):\n"
            '    self._upsert_cursor(conn, event, table="event_processed_cursors")',
            _LiteralCallTarget("_upsert_cursor", 2, "table"),
        ),
        (
            "def save(self):\n"
            '    self._update_row(table="research_sessions", row_id="id")',
            _LiteralCallTarget("_update_row", 0, "table"),
        ),
        (
            "def save(self):\n"
            '    self._update_row(table="writing_projects", row_id="id")',
            _LiteralCallTarget("_update_row", 0, "table"),
        ),
    ],
)
def test_dynamic_target_evidence_rejects_messages_in_service_helper_calls(
    source: str,
    target: _LiteralCallTarget,
) -> None:
    evidence = _CallTargetEvidence(calls=(target,))
    baseline = _derive_target_evidence(ast.parse(source), evidence)
    review = _DynamicSqlReview(
        domain="service-owned rows",
        exact_targets=baseline.targets,
        evidence_fingerprint=baseline.fingerprint,
    )
    mutated = _derive_target_evidence(
        ast.parse(source.replace(next(iter(baseline.targets)), "messages")),
        evidence,
    )

    with pytest.raises(AssertionError, match="derived dynamic SQL targets changed"):
        _assert_review_target_evidence("fixture::service", review, mutated)


def test_dynamic_target_evidence_checks_imported_table_constants() -> None:
    consumer = ast.parse(
        """
from fixture.constants import REFERENCE_TABLE
ARCHIVE_TABLE = f"_{REFERENCE_TABLE}_v3"
"""
    )
    evidence = _ImportedConstantTargetEvidence(
        names=("REFERENCE_TABLE", "ARCHIVE_TABLE")
    )
    baseline = _derive_target_evidence(
        consumer,
        evidence,
        imported_modules={
            "fixture.constants": ast.parse(
                'REFERENCE_TABLE = "tts_profile_clone_references"'
            )
        },
    )
    review = _DynamicSqlReview(
        domain="TTS clone-reference migration",
        exact_targets=frozenset(
            {"tts_profile_clone_references", "_tts_profile_clone_references_v3"}
        ),
        evidence_fingerprint=baseline.fingerprint,
    )

    _assert_review_target_evidence("fixture::migrate", review, baseline)
    with pytest.raises(AssertionError, match="derived dynamic SQL targets changed"):
        _assert_review_target_evidence(
            "fixture::migrate",
            review,
            _derive_target_evidence(
                consumer,
                evidence,
                imported_modules={
                    "fixture.constants": ast.parse('REFERENCE_TABLE = "messages"')
                },
            ),
        )


@pytest.mark.parametrize(
    ("consumer_source", "constant_source"),
    [
        (
            "REFERENCE_TABLE = 'tts_profile_clone_references'\n"
            "REFERENCE_TABLE = 'messages'",
            None,
        ),
        (
            "from fixture.constants import REFERENCE_TABLE",
            "REFERENCE_TABLE = 'tts_profile_clone_references'\n"
            "REFERENCE_TABLE = 'messages'",
        ),
        (
            "from fixture.constants import REFERENCE_TABLE\n"
            "REFERENCE_TABLE = 'messages'",
            "REFERENCE_TABLE = 'tts_profile_clone_references'",
        ),
    ],
)
def test_dynamic_target_evidence_rejects_rebound_string_constants(
    consumer_source: str,
    constant_source: str | None,
) -> None:
    imported_modules = (
        {"fixture.constants": ast.parse(constant_source)}
        if constant_source is not None
        else None
    )

    with pytest.raises(AssertionError, match="constant.*bindings"):
        _derive_target_evidence(
            ast.parse(consumer_source),
            _ImportedConstantTargetEvidence(names=("REFERENCE_TABLE",)),
            imported_modules=imported_modules,
        )


@pytest.mark.parametrize(
    "forwarding",
    [
        "writer = self._update_row\n    writer('messages')",
        "run_writer(self._update_row, 'messages')",
        "writers = {'message': self._update_row}",
        "writer = getattr(self, '_update_row')\n    writer('messages')",
        "writer = getattr(self, f'_update_row')\n    writer('messages')",
        "writer = getattr(self, '_update_' + 'row')\n    writer('messages')",
    ],
)
def test_dynamic_target_evidence_rejects_bound_helper_forwarding(
    forwarding: str,
) -> None:
    tree = ast.parse(
        f"""
def update_project(self):
    self._update_row(table="writing_projects", row_id="project-id")

def forwarded(self):
    {forwarding}
"""
    )

    with pytest.raises(AssertionError, match="helper.*forwarded"):
        _derive_target_evidence(
            tree,
            _CallTargetEvidence(calls=(_LiteralCallTarget("_update_row", 0, "table"),)),
        )


def test_dynamic_target_evidence_rejects_module_bound_helper_forwarding() -> None:
    tree = ast.parse(
        """
FORWARDED_WRITER = Service._update_row

def update_project():
    Service._update_row(table="writing_projects", row_id="project-id")
"""
    )

    with pytest.raises(AssertionError, match="helper.*forwarded"):
        _derive_target_evidence(
            tree,
            _CallTargetEvidence(calls=(_LiteralCallTarget("_update_row", 0, "table"),)),
        )


def test_dynamic_target_evidence_rejects_runtime_dynamic_helper_reflection() -> None:
    tree = ast.parse(
        """
def update_project(self):
    self._update_row(table="writing_projects", row_id="project-id")

def forwarded(self, method_name):
    writer = getattr(self, method_name)
    writer("messages")
"""
    )

    with pytest.raises(AssertionError, match="runtime-dynamic.*reflection"):
        _derive_target_evidence(
            tree,
            _CallTargetEvidence(calls=(_LiteralCallTarget("_update_row", 0, "table"),)),
        )


def test_dynamic_target_evidence_rejects_runtime_only_claimed_metadata() -> None:
    tree = ast.parse(
        """
def public_caller(table_name):
    self._write_dynamic_table(table_name)
"""
    )
    evidence = _CallTargetEvidence(
        calls=(
            _LiteralCallTarget(
                helper="_write_dynamic_table",
                positional_index=0,
            ),
        )
    )
    result = _derive_target_evidence(tree, evidence)
    review = _DynamicSqlReview(
        domain="claimed fixture rows",
        exact_targets=frozenset({"fixture_rows"}),
        evidence_fingerprint=result.fingerprint,
    )

    with pytest.raises(AssertionError, match="derived dynamic SQL targets changed"):
        _assert_review_target_evidence("fixture::write", review, result)


def test_all_dynamic_sql_reviews_match_derived_production_targets() -> None:
    for site_key, review in _PROVEN_NON_SEMANTIC_DYNAMIC_SQL.items():
        function_identity = site_key.split("::dynamic-sql:", 1)[0]
        derived = _derived_production_target_evidence(function_identity)
        _assert_review_target_evidence(site_key, review, derived)


@pytest.mark.parametrize(
    "expression",
    [
        '"/* audit */ UPDATE " + table_name + " SET deleted = 1"',
        '"WITH candidates AS (SELECT id FROM fixture_rows) UPDATE " '
        '+ table_name + " SET deleted = 1"',
        'f"UPDATE {schema}.messages SET content = ?"',
    ],
)
def test_sql_scanner_rejects_dynamic_targets_after_comments_ctes_or_schema(
    expression: str,
) -> None:
    tree = ast.parse(
        f"""
def write(cursor, table_name, schema):
    cursor.execute({expression}, ("hello",))
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    with pytest.raises(AssertionError, match="unresolved dynamic mutation SQL target"):
        _sql_actions_in_function(function)


def test_sql_scanner_ignores_unexecuted_diagnostic_string() -> None:
    tree = ast.parse(
        """
def diagnose(logger):
    logger.warning("UPDATE messages SET content = ? is the expected statement")
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "diagnose"
    )

    assert _sql_actions_in_function(function) == set()


@pytest.mark.parametrize(
    "expression",
    [
        'f"UPDATE {table_name} SET content = ?"',
        '"DELETE FROM " + table_name + " WHERE id = ?"',
    ],
)
def test_sql_scanner_rejects_unresolved_dynamic_mutation_target(
    expression: str,
) -> None:
    tree = ast.parse(
        f"""
def write(cursor, table_name):
    cursor.execute({expression}, ("message-1",))
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    with pytest.raises(AssertionError, match="unresolved dynamic mutation SQL target"):
        _sql_actions_in_function(function)


def test_sql_scanner_allows_dynamic_values_after_literal_target() -> None:
    tree = ast.parse(
        """
def write(cursor, assignments):
    cursor.execute(f"UPDATE messages SET {assignments} WHERE id = ?", ("message-1",))
"""
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )

    assert _sql_actions_in_function(function) == {"sql:update:messages"}


def test_document_exact_census_is_bidirectionally_synchronized() -> None:
    census = _parse_inventory_census(INVENTORY_PATH.read_text(encoding="utf-8"))

    assert census == {
        "sql": DIRECT_SQL_ROUTE_CLASSIFICATION,
        "boundary": BOUNDARY_CALL_ROUTE_CLASSIFICATION,
    }


def test_document_exact_census_rejects_duplicate_or_contradictory_rows() -> None:
    duplicated = """
## Exact live SQL sink census

- `path.py::Owner.write::sql:update:messages` — model-visible
- `path.py::Owner.write::sql:update:messages` — presentation-only

## Exact boundary-call census
"""

    with pytest.raises(AssertionError, match="duplicate exact-census route"):
        _parse_inventory_census(duplicated)


def test_public_owner_table_is_labeled_manual_and_names_required_owners() -> None:
    inventory = INVENTORY_PATH.read_text(encoding="utf-8")
    assert "Public-owner table is manual guidance" in inventory
    for owner in (
        "ConsoleChatStore.commit_durable_turn",
        "ConsoleChatStore.promote_ephemeral_session",
        "ConsoleSessionController._commit_durable_console_chat_fork",
        "ConsoleChatStore.discard_provider_continuation",
        "ConsoleChatStore.persist_provider_continuation_event",
        "SyncEnvelopeApplier.apply",
        "ChatSyncAdapter.apply",
        "ConsoleChatStore.commit_full_capture_purge",
        "ConsoleChatStore.write_trajectory_rows",
        "ConsoleChatStore.set_message_usage",
        "ConsoleChatStore.attach_message_exchanges",
        "ConsoleChatStore.set_message_metadata",
    ):
        assert f"`{owner}`" in inventory


def test_repository_scan_results_are_cached_and_immutable() -> None:
    assert hasattr(_direct_sql_routes, "cache_info")
    assert hasattr(_boundary_call_routes, "cache_info")
    assert isinstance(_direct_sql_routes(), frozenset)
    assert isinstance(_boundary_call_routes(), frozenset)
    assert isinstance(_carried_writer_actions(), MappingProxyType)
