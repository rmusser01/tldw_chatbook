"""Original, caller-bound transactions on the stock ChaChaNotes database."""

from __future__ import annotations

import inspect
import sqlite3
import sys


# Each entry names the method containing the actual outer SQLite transaction.
# No callback, SQL, file path, or provider request is represented on the wire.
_SITES = {
    ("tldw_chatbook.Chat.console_trace_runtime", "ConsoleTraceBoundaryFactory"): {
        "__call__": "trace_boundary_factory",
        "_recover_owned_boundary": "trace_recovery",
    },
    ("tldw_chatbook.Chat.console_trace_service", "ConsoleTraceService"): {
        "bind_and_mark_dispatch": "trace_dispatch",
        "reserve_call": "trace_reserve",
    },
    ("tldw_chatbook.Chat.console_trace_service", "ConsoleTraceCallBoundary"): {
        "mark_not_dispatched": "trace_cancel",
        "mark_dispatch_unknown": "trace_unknown",
    },
    ("tldw_chatbook.Chat.console_trace_settlement", "ConsoleTraceSettlementCoordinator"): {
        "_write_prepared": "trace_settlement",
        "_mark_response_started": "trace_response_started",
        "recover_open_calls": "trace_recover_open_calls",
    },
    ("tldw_chatbook.Chat.chat_persistence_service", "ChatPersistenceService"): {
        "create_message": "citation_message_create",
        "update_message_content": "citation_message_update",
        "promote_console_conversation_bundle": "conversation_promotion",
        "fork_console_conversation_bundle": "conversation_fork",
        "commit_completed_voice_pair": "conversation_voice_pair",
    },
    ("tldw_chatbook.Notes.Notes_Library", "NotesInteropService"): {
        "save_note_with_organization": "notes_save_organization",
    },
    ("tldw_chatbook.Sync_Interop.notes_organization_sync_service", "NotesOrganizationSyncService"): {
        name: "notes_organization_" + name.lstrip("_")
        for name in (
            "create_folder attach_folder_link detach_folder_link mutate_managed_folder_links sync_subject_keywords "
            "create_keyword mutate_keyword create_keyword_collection mutate_keyword_collection "
            "set_collection_keyword_link _mutate_folder resolve_placement_review "
            "finalize_pending_note_organization_receipts drain_pending_note_intents "
            "_record_server_enrollment_status _set_enrollment_local_state drain_pending_intents "
            "_bind_predecessor_base_if_ready reconcile_acknowledgements"
        ).split()
    },
}
CHACHA_LOCAL_GRANT_KINDS = frozenset(
    f"chacha_local.{kind}" for sites in _SITES.values() for kind in sites.values()
)
_methods = {}
_note_transaction = None


def register_chacha_grant_sources(cls):
    """Capture installed transaction bodies and their published descriptors."""
    global _note_transaction
    for name, kind in _SITES[(cls.__module__, cls.__name__)].items():
        descriptor = vars(cls)[name]
        original = inspect.unwrap(descriptor)
        _methods[original.__code__] = f"chacha_local.{kind}", cls, name, descriptor, original
    if cls.__name__ == "NotesInteropService":
        entry = vars(cls)["note_transaction"]
        _note_transaction = entry, entry.__code__
    return cls


def local_chacha_transaction(transaction):
    """Select an original transaction at its installed database entry point."""
    from ..Backup_Recovery.bootstrap import RecoveryRequired
    from .ChaChaNotes_DB import CharactersRAGDB, _PROFILE_TRANSACTION_ENTRY

    db = transaction.db
    entry_frame = sys._getframe(1)
    entry, code = _PROFILE_TRANSACTION_ENTRY
    if (type(db) is not CharactersRAGDB or entry_frame.f_code is not code
        or entry.__code__ is not code or vars(CharactersRAGDB).get("transaction") is not entry
        or entry_frame.f_globals is not entry.__globals__ or entry_frame.f_locals.get("self") is not db
        or "transaction" in vars(db)):
        raise RecoveryRequired("profile_writer_grant_source_changed")
    frame = entry_frame.f_back
    # The Notes service exposes note_transaction() publicly. Only its original
    # complete save can use that forwarding entry to obtain a grant.
    forward = None
    if _note_transaction is not None and frame.f_code is _note_transaction[1]:
        forward = frame
        frame = forward.f_back
    site = _methods.get(frame.f_code)
    if site is None:
        return transaction
    kind, cls, name, descriptor, original = site
    owner = frame.f_locals.get("self")

    def require_source():
        if (type(owner) is not cls or name in getattr(owner, "__dict__", {})
            or vars(cls).get(name) is not descriptor or inspect.unwrap(descriptor) is not original
            or original.__code__ is not frame.f_code or frame.f_globals is not original.__globals__
            or frame.f_globals.get(cls.__name__) is not cls
            or vars(CharactersRAGDB).get("transaction") is not entry or entry.__code__ is not entry_frame.f_code
            or "transaction" in vars(db)):
            raise RecoveryRequired("profile_writer_grant_source_changed")
        if forward is not None:
            method, code = _note_transaction
            valid = (kind == "chacha_local.notes_save_organization" and method.__code__ is code
                     and forward.f_globals is method.__globals__ and forward.f_locals.get("self") is owner
                     and vars(cls).get("note_transaction") is method and "note_transaction" not in vars(owner)
                     and frame.f_locals.get("db") is db
                     and owner._db_instances.get(frame.f_locals["user_id"].strip()) is db)
        elif cls.__name__ == "NotesOrganizationSyncService":
            valid = owner.notes_repository.db is db
        elif cls.__name__ == "ChatPersistenceService":
            valid = owner.db is db
        elif cls.__name__ in ("ConsoleTraceCallBoundary", "ConsoleTraceBoundaryFactory"):
            valid = owner.database is db
        else:
            valid = frame.f_locals.get("database") is db
        if not valid:
            raise RecoveryRequired("profile_writer_grant_source_changed")

    require_source()

    # A native transaction begun by an outer owner keeps its original
    # commit/rollback and connection. A grant may only bind before BEGIN.
    connection = getattr(db._local, "conn", None)
    if connection is not None:
        try:
            if sqlite3.Connection.in_transaction.__get__(connection):
                from .profile_writer_selection import writer_file_identity

                if db._local.profile_writer_file_identity != writer_file_identity(db.db_path):
                    raise RecoveryRequired("profile_writer_database_changed")
                return transaction
        except sqlite3.ProgrammingError:
            pass
    from .profile_writer_chacha_grant import granted_transaction

    return granted_transaction(transaction, kind, require_source)
