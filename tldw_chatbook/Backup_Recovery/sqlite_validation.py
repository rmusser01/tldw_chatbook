"""Bounded inspection/migration of disposable candidates using installed SQL.

Cross-owner file locators require the executor's staged dependency map. This
entry checks SQLite content and owned BLOBs; it never opens those external paths.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

from .models import OwnerAdapter
from tldw_chatbook.DB.recovery_sqlite import _split_statistics, _statistics_metadata

_PROGRESS_INTERVAL = 1000
_STEP_BUDGET = 5_000_000
_SECONDS = 30.0
_CATALOG_LIMIT = 10_000
_CATALOG_BYTES = 8 * 1024**2


_active_restrictions = ContextVar("recovery_sqlite_restrictions", default=None)


@contextmanager
def _recovery_restriction_scope(connection, restrictions):
    """Expose only this connection's original budget to nested owner checks."""
    token = _active_restrictions.set((connection, restrictions))
    try:
        yield
    finally:
        _active_restrictions.reset(token)


def _current_restrictions(connection):
    active = _active_restrictions.get()
    return active[1] if active is not None and active[0] is connection else None


def _installed_owner(owner_id):
    # Factories contain frozen declarations only: no config/profile/repository
    # construction and no archive-supplied policy or migration authority.
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_operations import recovery_adapters as operations
    from tldw_chatbook.DB.recovery_dreams_guardian import recovery_adapters as dreams_guardian
    from tldw_chatbook.Evals.recovery import recovery_adapters as evals
    from tldw_chatbook.Kanban_Interop.recovery import recovery_adapters as kanban
    from tldw_chatbook.Notes.recovery import recovery_adapters as notes
    from tldw_chatbook.Notifications.recovery import recovery_adapters as notifications
    from tldw_chatbook.Plugins.backup import recovery_adapters as plugins
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters as research
    from tldw_chatbook.Scheduling.recovery import recovery_adapters as scheduling
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters as study
    from tldw_chatbook.Sync_Interop.recovery import recovery_adapters as sync
    from tldw_chatbook.TTS.recovery import recovery_adapters as tts
    from tldw_chatbook.Writing_Interop.recovery import recovery_adapters as writing

    from .config_adapter import recovery_adapters as config
    from .rag_indexing import recovery_adapters as rag_indexing
    from .recovered_media import recovery_adapters as recovered

    for factory in (
        core_adapters,
        operations,
        dreams_guardian,
        research,
        writing,
        evals,
        study,
        notes,
        kanban,
        scheduling,
        notifications,
        sync,
        tts,
        recovered,
        config,
        rag_indexing,
        plugins,
    ):
        for owner in factory():
            if owner.owner_id == owner_id:
                policy = owner.schema_policy()
                if policy is not None and policy.schema_sql:
                    return owner
    raise ValueError("unsupported_sqlite_owner")


@contextmanager
def _open_installed_candidate(owner_id, path, *, cancel=None, _custody=None):
    """Retain the native restricted read of an installed owner's candidate."""
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    installed = _installed_owner(owner_id)
    with open_recovery_validation(
        installed.owner_id,
        path,
        writable=False,
        with_restrictions=True,
        cancel=cancel,
        _custody=_custody,
    ) as (connection, restrictions):
        yield connection, restrictions, installed


#: The five Evals tables whose inert ``version`` column the v5 -> v6 step drops.
_EVALS_VERSION_COLUMN_TABLES = frozenset(
    {"eval_tasks", "eval_datasets", "eval_models", "eval_runs", "ab_tests"}
)


class _Restrictions:
    def __init__(self, connection, cancel=None):
        self.cancel = cancel
        self.deadline = monotonic() + _SECONDS
        self.steps = 0
        self.migrating = False
        self.migration_owner = None
        self.shipped_checkpoint_migration = False
        self.shipped_checkpoint_rename = False
        self.fleet_progress_migration = False
        self.browse_order_migration = False
        self.response_rules_migration = False
        self.prompt_mixins_migration = False
        self.canvas_sibling_migration = False
        self.portrait_migration = None
        self.archive_chronology_migration = False
        self.workflow_statistics = ()
        self.dropping_workflow_phase = False
        self.preparing_response_rules = False
        self.sargable_migration = False
        self.canvas_schema = False
        self.changing_schema_trust = False
        self.reading_fts_metadata = False
        connection.set_authorizer(self.authorize)
        connection.set_progress_handler(self.progress, _PROGRESS_INTERVAL)

    def progress(self):
        self.steps += _PROGRESS_INTERVAL
        return int(self.expired())

    def expired(self):
        return (
            (self.cancel is not None and self.cancel.is_set())
            or self.steps > _STEP_BUDGET
            or monotonic() >= self.deadline
        )

    def _evals_drop_column_step(self, action, first, second, database):
        """Whether this is part of the Evals v5 -> v6 ``DROP COLUMN version`` step.

        TASK-19566 F8 declares that step so a v5 Evals backup can be restored.
        ``ALTER TABLE ... DROP COLUMN`` reports the dropped COLUMN in the
        authorizer's database slot and rewrites the stored CREATE text in the
        temp schema as well as the main one, so every one of these fails the
        main-database-only rule. Only the five declared tables, only the
        ``version`` column, and only while this owner's migration gate is open.

        Args:
            action: The SQLite authorizer action code.
            first: The action's first argument.
            second: The action's second argument.
            database: The authorizer's database slot.

        Returns:
            ``True`` for exactly the operations that step performs.
        """
        if not (self.migrating and self.migration_owner == "db.evals"):
            return False
        if action == sqlite3.SQLITE_ALTER_TABLE:
            return (
                first == "main"
                and second in _EVALS_VERSION_COLUMN_TABLES
                and database == "version"
            )
        if database != "temp" or first != "sqlite_temp_master":
            return False
        if action == sqlite3.SQLITE_READ:
            return second in {"type", "name", "sql"}
        return action == sqlite3.SQLITE_UPDATE and second == "sql"

    def _shipped_checkpoint_step(self, action, first, second, database, source):
        """Admit only the measured installed v76 checkpoint rebuild callbacks."""
        if not (
            self.shipped_checkpoint_migration
            and self.migrating
            and self.migration_owner == "db.chachanotes.primary"
            and source is None
        ):
            return False
        if action == sqlite3.SQLITE_ALTER_TABLE:
            return self.shipped_checkpoint_rename and (first, second, database) == (
                "main",
                "console_dispatch_checkpoints_v77",
                None,
            )
        if action == sqlite3.SQLITE_FUNCTION:
            return self.shipped_checkpoint_rename and (first, second, database) == (
                None,
                "sqlite_rename_table",
                None,
            )
        # RENAME compiles these internal updates even with no temporary tables
        # or AUTOINCREMENT on the renamed table (observed on SQLite 3.49.1).
        if database == "temp" and first == "sqlite_temp_master":
            return self.shipped_checkpoint_rename and (
                action == sqlite3.SQLITE_READ
                and second in {"type", "name", "sql", "tbl_name"}
                or action == sqlite3.SQLITE_UPDATE
                and second in {"sql", "tbl_name"}
            )
        if database != "main":
            return False
        indexes = {
            "idx_console_dispatch_checkpoint_conversation",
            "idx_console_dispatch_checkpoints_user_message",
        }
        return (
            action == sqlite3.SQLITE_CREATE_TABLE
            and first == "console_dispatch_checkpoints_v77"
            or action == sqlite3.SQLITE_INSERT
            and first in {"sqlite_master", "console_dispatch_checkpoints_v77"}
            or action == sqlite3.SQLITE_DROP_TABLE
            and first == "console_dispatch_checkpoints"
            or action == sqlite3.SQLITE_DELETE
            and first in {"sqlite_master", "console_dispatch_checkpoints"}
            or action == sqlite3.SQLITE_CREATE_INDEX
            and (
                first in indexes
                and second == "console_dispatch_checkpoints"
                or first
                in {
                    "sqlite_autoindex_console_dispatch_checkpoints_v77_1",
                    "sqlite_autoindex_console_dispatch_checkpoints_v77_2",
                    "sqlite_autoindex_console_dispatch_checkpoints_v77_3",
                }
                and second == "console_dispatch_checkpoints_v77"
            )
            or action == sqlite3.SQLITE_REINDEX
            and first in indexes
            or action == sqlite3.SQLITE_UPDATE
            and (
                self.shipped_checkpoint_rename
                and first == "sqlite_sequence"
                and second == "name"
                or first == "sqlite_master"
                and second in {"type", "name", "tbl_name", "rootpage", "sql"}
            )
        )

    def authorize(self, action, first, second, database, source):
        if self._shipped_checkpoint_step(action, first, second, database, source):
            return sqlite3.SQLITE_OK
        if self._evals_drop_column_step(action, first, second, database):
            return sqlite3.SQLITE_OK
        if (self.migrating and self.migration_owner == "db.dreams"
            and action in {sqlite3.SQLITE_READ, sqlite3.SQLITE_UPDATE}
            and first == "sqlite_temp_master" and second in {"type", "name", "tbl_name", "sql"}
            and database == "temp" and source is None):
            # SQLite's fixed table rename inspects its empty temp catalog.
            return sqlite3.SQLITE_OK
        if (
            self.migrating
            and self.migration_owner == "db.prompts.primary"
            and database == "temp"
            and source is None
            and first == "sqlite_temp_master"
            and action in {sqlite3.SQLITE_READ, sqlite3.SQLITE_UPDATE}
            and second in {"type", "name", "tbl_name", "sql"}
        ):
            return sqlite3.SQLITE_OK
        if (
            self.migrating
            and self.migration_owner == "db.agent_runs"
            and source is None
        ):
            if (
                self.dropping_workflow_phase
                and action == sqlite3.SQLITE_DELETE
                and first in self.workflow_statistics
                and second is None
                and database == "main"
            ):
                return sqlite3.SQLITE_OK
            if (
                database == "temp"
                and first == "sqlite_temp_master"
                and (
                    action == sqlite3.SQLITE_READ
                    and second in {"type", "name", "sql", "tbl_name"}
                    or action == sqlite3.SQLITE_UPDATE
                    and second in {"sql", "tbl_name"}
                )
            ):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_ALTER_TABLE and (first, second, database) == (
                "main",
                "automatic_wake_attempts",
                "old_attempt_kind",
            ):
                return sqlite3.SQLITE_OK
        if database not in (None, "main"):
            return sqlite3.SQLITE_DENY
        if action in (
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_RECURSIVE,
        ):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            allowed = {
                "length",
                "typeof",
                "max",
                "min",
                "count",
                "sum",
                "avg",
                "coalesce",
                "ifnull",
                "nullif",
                "abs",
                "round",
                "lower",
                "upper",
                "substr",
                "substring",
                "like",
                "glob",
                "match",
                "hex",
            }
            if self.canvas_schema:
                allowed.add("canvas_revision_payload_valid")
            if (
                self.preparing_response_rules
                or self.response_rules_migration
                or self.migrating
                and self.migration_owner == "db.chachanotes.primary"
                and self.portrait_migration in (82, 83)
            ):
                allowed.add("json_valid")
            if self.migrating:
                allowed |= {"printf", "sqlite_rename_test", "sqlite_rename_quotefix"}
                if self.migration_owner == "db.prompts.primary":
                    allowed |= {"trim", "sqlite_rename_table"}
                if self.migration_owner == "db.subscriptions":
                    allowed |= {"json_valid", "json_type", "json_extract", "json_set"}
                if self.migration_owner == "db.dreams":
                    allowed |= {"sqlite_rename_table", "sqlite_rename_column"}
                if self.migration_owner == "db.agent_runs":
                    allowed |= {
                        "sqlite_rename_table",
                        "sqlite_rename_column",
                        "sqlite_drop_column",
                    }
                if self.migration_owner == "db.evals":
                    allowed.add("sqlite_drop_column")
                # ADR-224 v78->v79: normalization UPDATEs (julianday/strftime),
                # the partial json_extract index, and the functions compiled
                # from every conversations/flashcards trigger body the UPDATE
                # statement code-generates (current_timestamp, json_*).
                if (
                    self.sargable_migration
                    and self.migration_owner == "db.chachanotes.primary"
                ):
                    allowed |= {
                        "julianday",
                        "strftime",
                        "json_extract",
                        "json_valid",
                        "json_object",
                        "json_type",
                        "replace",
                        "current_timestamp",
                    }
            return sqlite3.SQLITE_OK if second in allowed else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            reads = {
                "trusted_schema",
                "user_version",
                "quick_check",
                "foreign_key_check",
                "data_version",
            }
            metadata = {
                "table_xinfo",
                "index_list",
                "index_xinfo",
                "table_list",
                "foreign_key_list",
            }
            permitted = (first in reads and second is None) or first in metadata
            permitted |= first == "trusted_schema" and second in ("OFF", "0")
            permitted |= (
                self.changing_schema_trust
                and first == "trusted_schema"
                and second in ("ON", "1")
            )
            permitted |= self.migrating and first == "user_version"
            # SQLite validates existing rows when ADD COLUMN adds a CHECK.
            permitted |= (
                self.migrating
                and self.migration_owner == "db.agent_runs"
                and first == "quick_check"
                and second
                in {
                    "automatic_wake_attempts",
                    "workflow_phase_attempts",
                    "workflow_runs",
                }
            )
            permitted |= (
                self.migrating and self.migration_owner == "db.library_collections"
                and first == "quick_check" and second == "collection_capture_items"
            )
            permitted |= (
                self.migrating
                and self.migration_owner == "db.workspaces"
                and first == "quick_check"
                and second == "workspace_records"
            )
            permitted |= (
                self.migrating and self.migration_owner == "db.guardian"
                and first == "quick_check" and second == "guardian_alerts"
            )
            permitted |= (
                self.migrating
                and self.migration_owner == "db.chachanotes.primary"
                and self.portrait_migration == 83
                and first == "quick_check"
                and second == "character_portrait_performances"
            )
            return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
        if (
            self.reading_fts_metadata
            and action == sqlite3.SQLITE_UPDATE
            and first == "sqlite_master"
            and second in {"type", "name", "tbl_name", "rootpage", "sql"}
            and database == "main"
            and source is None
        ):
            # SQLite 3.37 compiles, then discards, this declaration UPDATE
            # while connecting an existing FTS5 table for table_xinfo.
            return sqlite3.SQLITE_OK
        if (self.preparing_response_rules and database == "main" and source is None
            and action == sqlite3.SQLITE_UPDATE and (
                first == "console_response_rule_bindings" and second in {"state", "binding_revision"}
                or first == "console_response_rule_assessments" and second in {"state", "assessment_json"}
            )
        ):
            return sqlite3.SQLITE_OK
        if self.preparing_response_rules and action == sqlite3.SQLITE_TRANSACTION:
            return sqlite3.SQLITE_OK
        if self.migrating:
            if action == sqlite3.SQLITE_TRANSACTION:
                return sqlite3.SQLITE_OK
            if self.migration_owner in {
                "db.chachanotes.primary",
                "db.workspaces",
                "db.agent_runs",
            }:
                tables = {"mobile_consolidation_batches"}
                if self.migration_owner != "db.chachanotes.primary":
                    tables.add("mobile_import_authority_holds")
                if (
                    database == "main"
                    and source is None
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first in tables
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and (first, second)
                        in {(f"sqlite_autoindex_{table}_1", table) for table in tables}
                        or action == sqlite3.SQLITE_INSERT
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                        or self.migration_owner == "db.workspaces"
                        and action == sqlite3.SQLITE_INSERT
                        and first == "schema_version"
                    )
                ):
                    return sqlite3.SQLITE_OK
            if self.migration_owner == "db.chachanotes.primary":
                allowed = (
                    action == sqlite3.SQLITE_UPDATE
                    and first == "db_schema_version"
                    and second == "version"
                    and database == "main"
                    and source is None
                )
                if self.prompt_mixins_migration and source is None:
                    tables = {
                        "console_prompt_mixin_preferences",
                        "console_prompt_mixin_revisions",
                        "console_prompt_mixin_attempts",
                        "console_prompt_mixin_attempt_items",
                    }
                    indexes = {
                        "idx_console_mixin_revision_digest": "console_prompt_mixin_revisions",
                        "idx_console_mixin_attempt_user": "console_prompt_mixin_attempts",
                        "idx_console_mixin_attempt_assistant": "console_prompt_mixin_attempts",
                        **{f"sqlite_autoindex_{table}_1": table for table in tables},
                    }
                    triggers = {
                        "console_prompt_mixin_revision_immutable": "console_prompt_mixin_revisions",
                        "console_prompt_mixin_attempt_anchor_insert": "console_prompt_mixin_attempts",
                        "console_prompt_mixin_attempt_immutable": "console_prompt_mixin_attempts",
                        "console_prompt_mixin_item_immutable": "console_prompt_mixin_attempt_items",
                        "console_prompt_mixin_message_marker_preserve": "messages",
                        "conversations_ad": "conversations",
                    }
                    allowed |= (
                        action == sqlite3.SQLITE_ALTER_TABLE
                        and first == "main"
                        and second == "console_dispatch_checkpoints"
                        or database == "main"
                        and (
                            action == sqlite3.SQLITE_CREATE_TABLE
                            and first in tables
                            or action == sqlite3.SQLITE_DROP_TRIGGER
                            and first == "conversations_ad"
                            and second == "conversations"
                            or action == sqlite3.SQLITE_CREATE_TRIGGER
                            and triggers.get(first) == second
                            or action == sqlite3.SQLITE_CREATE_INDEX
                            and indexes.get(first) == second
                            or action == sqlite3.SQLITE_REINDEX
                            and first in indexes
                            or action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}
                            and first == "sqlite_master"
                            or action == sqlite3.SQLITE_UPDATE
                            and first == "sqlite_master"
                            and second
                            in {"type", "name", "tbl_name", "rootpage", "sql"}
                        )
                    )
                if self.archive_chronology_migration and source is None:
                    allowed |= (
                        action == sqlite3.SQLITE_ALTER_TABLE
                        and (first, second, database) == ("main", "conversations", None)
                        or database == "main"
                        and (
                            action
                            in {
                                sqlite3.SQLITE_CREATE_TRIGGER,
                                sqlite3.SQLITE_DROP_TRIGGER,
                            }
                            and (first, second)
                            in {
                                ("conversations_sync_update", "conversations"),
                                ("sync_log_prune_conversations", "sync_log"),
                            }
                            or action == sqlite3.SQLITE_CREATE_INDEX
                            and (first, second)
                            == ("idx_conversations_archive_chronology", "conversations")
                            or action == sqlite3.SQLITE_REINDEX
                            and first == "idx_conversations_archive_chronology"
                            or action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}
                            and first == "sqlite_master"
                            or action == sqlite3.SQLITE_UPDATE
                            and first == "sqlite_master"
                            and second
                            in {"type", "name", "tbl_name", "rootpage", "sql"}
                        )
                    )
                if self.portrait_migration in (82, 83) and source is None:
                    tables = (
                        {
                            "character_portrait_performances",
                            "character_portrait_performance_frames",
                        }
                        if self.portrait_migration == 82
                        else {"character_portrait_regions"}
                    )
                    indexes = {f"sqlite_autoindex_{table}_1": table for table in tables}
                    allowed |= database == "main" and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first in tables
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and indexes.get(first) == second
                        or action == sqlite3.SQLITE_REINDEX
                        and first in indexes
                        or action == sqlite3.SQLITE_INSERT
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                    allowed |= (
                        self.portrait_migration == 83
                        and action == sqlite3.SQLITE_ALTER_TABLE
                        and (first, second, database)
                        == ("main", "character_portrait_performances", None)
                    )
                if (
                    self.response_rules_migration
                    and database == "main"
                    and source is None
                ):
                    tables = {
                        "console_response_rule_revisions",
                        "console_response_rule_bindings",
                        "console_response_rule_validations",
                        "console_response_rule_drafts",
                        "console_response_rule_fixtures",
                        "console_response_rule_assessments",
                        "console_machine_followup_receipts",
                    }
                    indexes = {
                        "idx_response_rule_drafts_scope": "console_response_rule_drafts",
                        "idx_response_rule_assessments_source": "console_response_rule_assessments",
                        "idx_machine_followup_receipts_conversation": "console_machine_followup_receipts",
                        **{f"sqlite_autoindex_{table}_1": table for table in tables},
                    }
                    allowed |= (
                        action == sqlite3.SQLITE_CREATE_TABLE and first in tables
                        or action == sqlite3.SQLITE_CREATE_TRIGGER and first == "console_response_rules_conversation_cleanup" and second == "conversations"
                        or action == sqlite3.SQLITE_CREATE_INDEX and indexes.get(first) == second
                        or action == sqlite3.SQLITE_REINDEX and first in indexes
                        or action == sqlite3.SQLITE_INSERT and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE and first == "sqlite_master" and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                if (
                    self.canvas_sibling_migration
                    and database == "main"
                    and source is None
                ):
                    indexes = {
                        "idx_canvas_revisions_canvas_sequence",
                        "uq_canvas_revisions_single_root",
                    }
                    allowed |= (
                        action == sqlite3.SQLITE_DROP_INDEX
                        and first == "idx_canvas_revisions_canvas_sequence"
                        and second == "canvas_revisions"
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and first in indexes
                        and second == "canvas_revisions"
                        or action == sqlite3.SQLITE_REINDEX
                        and first in indexes
                        or action == sqlite3.SQLITE_INSERT
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_DELETE
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                allowed |= (
                    self.fleet_progress_migration
                    and database == "main"
                    and source is None
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first == "fleet_progress_messages"
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and first in {
                            "idx_fleet_progress_conversation_sequence",
                            "sqlite_autoindex_fleet_progress_messages_1",
                        }
                        and second == "fleet_progress_messages"
                        or action == sqlite3.SQLITE_REINDEX
                        and first == "idx_fleet_progress_conversation_sequence"
                        or action == sqlite3.SQLITE_INSERT
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                )
                allowed |= (
                    self.browse_order_migration
                    and database == "main"
                    and source is None
                    and (
                        action == sqlite3.SQLITE_CREATE_INDEX
                        and first
                        in {
                            "idx_conversations_last_modified",
                            "idx_conversations_archived_browse_order",
                        }
                        and second == "conversations"
                        or action == sqlite3.SQLITE_REINDEX
                        and first
                        in {
                            "idx_conversations_last_modified",
                            "idx_conversations_archived_browse_order",
                        }
                        or action == sqlite3.SQLITE_INSERT
                        and first == "sqlite_master"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                )
                # ADR-224 v78->v79: the two normalization UPDATEs, the two
                # new indexes, and the two conversations triggers the
                # migration drops and recreates verbatim (sync update;
                # search-projection dirty marker -- both suppressed so a
                # format-only change emits no events and no wall-clock
                # stamps). Unlike the fleet step, ``source`` is NOT required
                # to be None: the UPDATE statements still compile every
                # other conversations/flashcards trigger body (FTS mirrors,
                # sync journals), so trigger-sourced writes of exactly those
                # shapes must pass. The admitted set was recorded from the
                # statements the installed .sql file actually executes.
                allowed |= (
                    self.sargable_migration
                    and database == "main"
                    and (
                        action == sqlite3.SQLITE_UPDATE
                        and (
                            first == "conversations"
                            and second == "last_modified"
                            or first == "flashcards"
                            and second == "next_review"
                            or first == "sqlite_master"
                            and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                        )
                        or action == sqlite3.SQLITE_INSERT
                        and first
                        in {
                            "sqlite_master",
                            "sync_log",
                            "conversations_fts",
                            "conversations_fts_data",
                            "conversations_fts_docsize",
                            "conversations_fts_idx",
                            "flashcards_fts",
                            "flashcards_fts_data",
                            "flashcards_fts_docsize",
                            "flashcards_fts_idx",
                            "conversation_dictionary_attachments",
                            "conversation_dictionary_unresolved",
                        }
                        or action == sqlite3.SQLITE_DELETE
                        and first
                        in {
                            "sqlite_master",
                            "sync_log",
                            "conversations_fts_docsize",
                            "flashcards_fts_docsize",
                            "conversation_dictionary_attachments",
                            "conversation_dictionary_unresolved",
                        }
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and first
                        in {"idx_conv_char_lm", "idx_character_cards_visible_name"}
                        or action == sqlite3.SQLITE_REINDEX
                        and first
                        in {"idx_conv_char_lm", "idx_character_cards_visible_name"}
                        or action == sqlite3.SQLITE_CREATE_TRIGGER
                        and first
                        in {
                            "conversations_sync_update",
                            "character_conversation_search_conversations_au",
                        }
                        or action == sqlite3.SQLITE_DROP_TRIGGER
                        and first
                        in {
                            "conversations_sync_update",
                            "character_conversation_search_conversations_au",
                        }
                    )
                )
                return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
            if self.migration_owner == "db.workspaces" and source is None:
                allowed = (
                    action == sqlite3.SQLITE_ALTER_TABLE
                    and (first, second, database) == ("main", "workspace_records", None)
                    or database == "main"
                    and (
                        action == sqlite3.SQLITE_INSERT
                        and first in {"schema_version", "sqlite_master"}
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    )
                )
                if allowed:
                    return sqlite3.SQLITE_OK
            # Fixed Dreams predecessor migrations only; candidate catalog matched
            # before this gate. No writes to story/profile payloads outside copy.
            if self.migration_owner == "db.dreams":
                tables = {"dream_tracked_items", "dream_track_runs", "dream_interest_profile_v3", "sqlite_sequence"}
                indexes = {"idx_dream_tracked_status", "idx_dream_tracked_origin_story", "idx_dream_track_runs_item", "sqlite_autoindex_dream_interest_profile_v3_1"}
                allowed = source is None and database == "main" and (
                    action == sqlite3.SQLITE_CREATE_TABLE and first in tables
                    or action in {sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_REINDEX} and first in indexes
                    or action == sqlite3.SQLITE_DROP_TABLE and first in {"dream_interest_profile", "dream_interest_profile_v3"}
                    or action == sqlite3.SQLITE_INSERT and first in {"schema_version", "sqlite_master", "dream_interest_profile_v3", "sqlite_sequence"}
                    or action == sqlite3.SQLITE_DELETE and first in {"sqlite_master", "dream_interest_profile", "dream_interest_profile_v3"}
                    or action == sqlite3.SQLITE_UPDATE and first == "sqlite_master"
                    and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                    or action == sqlite3.SQLITE_UPDATE and first == "sqlite_sequence" and second == "name"
                )
                allowed |= (action == sqlite3.SQLITE_ALTER_TABLE
                    and first == "main" and second in {"dream_interest_profile_v3", "dream_stories", "dream_tracked_items"}
                    and database is None and source is None)
                if allowed:
                    return sqlite3.SQLITE_OK
            # ADR-207: installed provenance steps only. The candidate SQL was
            # matched to a complete historical catalog before this gate opens.
            if self.migration_owner in {"db.library_collections", "db.subscriptions", "db.guardian"}:
                allowed = database == "main" and source is None and (
                    action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}
                    and first == "schema_version"
                    or self.migration_owner == "db.subscriptions"
                    and action == sqlite3.SQLITE_UPDATE
                    and first == "subscriptions" and second == "processing_options"
                )
                allowed |= (
                    self.migration_owner == "db.library_collections"
                    and action == sqlite3.SQLITE_ALTER_TABLE
                    and (first, second, database, source)
                    == ("main", "collection_capture_items", None, None)
                )
                allowed |= (
                    self.migration_owner == "db.subscriptions"
                    and source is None and database == "main"
                    and (
                        action in {sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_DROP_TRIGGER}
                        and (first, second) == ("update_subscriptions_timestamp", "subscriptions")
                        or action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}
                        and first == "sqlite_master"
                    )
                )
                allowed |= (
                    self.migration_owner == "db.guardian"
                    and action == sqlite3.SQLITE_ALTER_TABLE
                    and (first, second, database, source) == ("main", "guardian_alerts", None, None)
                )
                if allowed:
                    return sqlite3.SQLITE_OK
            # Research keeps its existing ADD COLUMN authority. Only fixed
            # installed statements execute while this temporary gate is open.
            if (
                action == sqlite3.SQLITE_ALTER_TABLE
                and first == "main"
                and second == "research_runs"
            ):
                return sqlite3.SQLITE_OK
            if self.migration_owner == "db.agent_runs":
                if (
                    database == "main"
                    and source is None
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first
                        in {
                            "workflow_checkpoints",
                            "workflow_resume_commands",
                            "workflow_change_windows",
                            "workflow_change_window_refs",
                        }
                        or action
                        in {sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_REINDEX}
                        and (first, second)
                        in {
                            (
                                "sqlite_autoindex_workflow_change_windows_1",
                                "workflow_change_windows",
                            ),
                            (
                                "sqlite_autoindex_workflow_change_windows_2",
                                "workflow_change_windows",
                            ),
                            (
                                "sqlite_autoindex_workflow_change_window_refs_1",
                                "workflow_change_window_refs",
                            ),
                            (
                                "sqlite_autoindex_workflow_change_window_refs_2",
                                "workflow_change_window_refs",
                            ),
                            (
                                "sqlite_autoindex_workflow_checkpoints_1",
                                "workflow_checkpoints",
                            ),
                            (
                                "sqlite_autoindex_workflow_resume_commands_1",
                                "workflow_resume_commands",
                            ),
                        }
                        or action == sqlite3.SQLITE_CREATE_TRIGGER
                        and (first, second)
                        in {
                            ("workflow_checkpoint_immutable", "workflow_checkpoints"),
                            ("workflow_recovery_review_immutable", "workflow_runs"),
                            (
                                "workflow_repair_input_immutable",
                                "workflow_phase_attempts",
                            ),
                            (
                                "workflow_change_window_identity",
                                "workflow_change_windows",
                            ),
                            (
                                "workflow_change_window_anchor",
                                "workflow_change_windows",
                            ),
                        }
                    )
                ):
                    return sqlite3.SQLITE_OK
                allowed_workflow_admission = source is None and (
                    database == "main"
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first in {"workflow_phase_attempts", "workflow_task_claims"}
                        or action == sqlite3.SQLITE_DROP_TABLE
                        and first == "workflow_phase_attempts_v26"
                        or action == sqlite3.SQLITE_INSERT
                        and first in {"workflow_phase_attempts", "sqlite_master"}
                        or action == sqlite3.SQLITE_DELETE
                        and first in {"workflow_phase_attempts_v26", "sqlite_master"}
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_sequence"
                        and second == "name"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "automatic_wake_attempts"
                        and second == "attempt_kind"
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                        or action
                        in {sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_REINDEX}
                        and first
                        in {
                            "idx_workflow_open_interval",
                            "idx_workflow_task_phase",
                            "sqlite_autoindex_workflow_phase_attempts_4",
                            "sqlite_autoindex_workflow_task_claims_1",
                            "sqlite_autoindex_workflow_task_claims_2",
                            "sqlite_autoindex_workflow_task_claims_3",
                        }
                    )
                    or action == sqlite3.SQLITE_ALTER_TABLE
                    and first == "main"
                    and second
                    in {
                        "workflow_phase_attempts",
                        "workflow_active_intervals",
                        "workflow_runs",
                    }
                )
                if allowed_workflow_admission:
                    return sqlite3.SQLITE_OK
                from tldw_chatbook.DB.recovery_goal_schema import GOAL_AUTO_INDEXES, GOAL_TABLES
                from tldw_chatbook.DB.recovery_review_boundary_schema import (
                    REVIEW_BOUNDARY_AUTO_INDEXES,
                )
                from tldw_chatbook.DB.recovery_workflow_schema import (
                    WORKFLOW_INDEXES,
                    WORKFLOW_TABLES,
                    WORKFLOW_TRIGGERS,
                )

                if database == "main" and source is None and (
                    action == sqlite3.SQLITE_CREATE_TABLE and first in WORKFLOW_TABLES
                    or action == sqlite3.SQLITE_CREATE_INDEX and (first, second) in WORKFLOW_INDEXES
                    or action == sqlite3.SQLITE_CREATE_TRIGGER and (first, second) in WORKFLOW_TRIGGERS
                    or action == sqlite3.SQLITE_REINDEX and first in {name for name, _ in WORKFLOW_INDEXES}
                ):
                    return sqlite3.SQLITE_OK

                if (
                    database == "main"
                    and source is None
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first in GOAL_TABLES
                        or action == sqlite3.SQLITE_CREATE_TABLE
                        and first == "automatic_review_boundary_claims"
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and (first, second)
                        in GOAL_AUTO_INDEXES | REVIEW_BOUNDARY_AUTO_INDEXES
                        or action == sqlite3.SQLITE_CREATE_TRIGGER
                        and first == "goal_launch_immutable"
                        and second == "goal_runs"
                        or action == sqlite3.SQLITE_CREATE_TABLE
                        and first == "console_command_receipts"
                        or action == sqlite3.SQLITE_CREATE_INDEX
                        and first == "sqlite_autoindex_console_command_receipts_1"
                        and second == "console_command_receipts"
                    )
                ):
                    return sqlite3.SQLITE_OK
                allowed = (
                    action == sqlite3.SQLITE_CREATE_TABLE
                    and first in {"agent_worktrees", "automatic_progress_wake_claims"}
                    or action == sqlite3.SQLITE_CREATE_INDEX
                    and first
                    in {
                        "idx_agent_worktrees_scope",
                        "sqlite_autoindex_agent_worktrees_1",
                    }
                    and second == "agent_worktrees"
                    or action == sqlite3.SQLITE_CREATE_INDEX
                    and first in {"idx_automatic_progress_claims_attempt", "sqlite_autoindex_automatic_progress_wake_claims_1"}
                    and second == "automatic_progress_wake_claims"
                    or action == sqlite3.SQLITE_REINDEX
                    and first == "idx_automatic_progress_claims_attempt"
                    or action == sqlite3.SQLITE_REINDEX
                    and first == "idx_agent_worktrees_scope"
                    or action == sqlite3.SQLITE_ALTER_TABLE
                    and first == "main"
                    and second in {"agent_definitions", "agent_runs", "automatic_wake_attempts"}
                    or action == sqlite3.SQLITE_INSERT
                    and first in {"sqlite_master", "schema_version"}
                )
                # Fixed ADR-219 migration objects; unrelated DDL and domain
                # row writes remain refused under this temporary owner gate.
                allowed |= (
                    action == sqlite3.SQLITE_ALTER_TABLE
                    and first == "main"
                    and second == "automatic_work_chains"
                    or action == sqlite3.SQLITE_CREATE_TABLE
                    and first == "automatic_chat_start_attempts"
                    or action == sqlite3.SQLITE_CREATE_INDEX
                    and (
                        first == "idx_automatic_chains_allowance_root"
                        and second == "automatic_work_chains"
                        or first
                        in {
                            "idx_automatic_chat_start_conversation_active",
                            "sqlite_autoindex_automatic_chat_start_attempts_1",
                            "sqlite_autoindex_automatic_chat_start_attempts_2",
                            "sqlite_autoindex_automatic_chat_start_attempts_3",
                        }
                        and second == "automatic_chat_start_attempts"
                    )
                    or action == sqlite3.SQLITE_REINDEX
                    and first
                    in {
                        "idx_automatic_chains_allowance_root",
                        "idx_automatic_chat_start_conversation_active",
                    }
                    or action == sqlite3.SQLITE_DROP_TRIGGER
                    and first == "automatic_chain_identity_immutable"
                    and second == "automatic_work_chains"
                    or action == sqlite3.SQLITE_CREATE_TRIGGER
                    and (
                        first
                        in {
                            "automatic_chain_identity_immutable",
                            "automatic_chain_root_insert",
                            "automatic_chain_root_update",
                        }
                        and second == "automatic_work_chains"
                        or first == "automatic_chat_start_identity_immutable"
                        and second == "automatic_chat_start_attempts"
                    )
                    or action == sqlite3.SQLITE_DELETE
                    and first == "sqlite_master"
                    and source is None
                )
                if allowed:
                    return sqlite3.SQLITE_OK
            if self.migration_owner == "db.evals":
                allowed = (
                    action == sqlite3.SQLITE_CREATE_INDEX
                    and first == "idx_eval_results_run_created"
                    and second == "eval_results"
                    or action == sqlite3.SQLITE_INSERT
                    and first == "sqlite_master"
                    or action == sqlite3.SQLITE_REINDEX
                    and first == "idx_eval_results_run_created"
                )
                if allowed:
                    return sqlite3.SQLITE_OK
            if self.migration_owner == "db.prompts.primary":
                prompt_indexes = {
                    "idx_prompts_name",
                    "idx_prompts_author",
                    "idx_prompts_uuid",
                    "idx_prompts_last_modified",
                    "idx_prompts_deleted",
                }
                allowed_rebuild = source is None and (
                    database == "main"
                    and (
                        action == sqlite3.SQLITE_CREATE_TABLE
                        and first
                        in {"Prompts_mixin_migration", "Prompts_sequence_migration"}
                        or action == sqlite3.SQLITE_DROP_TABLE
                        and first in {"Prompts", "Prompts_sequence_migration"}
                        or action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}
                        and first
                        in {
                            "sqlite_master",
                            "Prompts",
                            "Prompts_mixin_migration",
                            "Prompts_sequence_migration",
                            "sqlite_sequence",
                        }
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_master"
                        and second in {"type", "name", "tbl_name", "rootpage", "sql"}
                        or action == sqlite3.SQLITE_UPDATE
                        and first == "sqlite_sequence"
                        and second in {"name", "seq"}
                        or action
                        in {sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_REINDEX}
                        and (
                            first in prompt_indexes
                            or first
                            in {
                                "sqlite_autoindex_Prompts_mixin_migration_1",
                                "sqlite_autoindex_Prompts_mixin_migration_2",
                            }
                        )
                        or action
                        in {sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_DROP_TRIGGER}
                        and first == "prompts_validate_sync_update"
                    )
                    or action == sqlite3.SQLITE_ALTER_TABLE
                    and (first, second, database)
                    == ("main", "Prompts_mixin_migration", None)
                )
                allowed = allowed_rebuild or (
                    action == sqlite3.SQLITE_CREATE_TABLE
                    and first == "LocalPromptDrafts"
                    or action == sqlite3.SQLITE_CREATE_INDEX
                    and first == "idx_local_prompt_drafts_updated"
                    and second == "LocalPromptDrafts"
                    or action == sqlite3.SQLITE_REINDEX
                    and first == "idx_local_prompt_drafts_updated"
                    or action == sqlite3.SQLITE_INSERT
                    and first == "sqlite_master"
                    or action == sqlite3.SQLITE_UPDATE
                    and first == "schema_version"
                    and second == "version"
                )
                if allowed:
                    return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_UPDATE and first == "sqlite_master":
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY


def _restrict_connection(connection, cancel=None):
    """Install and verify mandatory primitives before any candidate query."""
    try:
        disable_extensions = getattr(connection, "enable_load_extension", None)
        if disable_extensions is None:
            option = sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION
            connection.setconfig(option, False)
            if connection.getconfig(option) is not False:
                raise ValueError("sqlite_security_unavailable")
        else:
            disable_extensions(False)
        connection.execute("PRAGMA trusted_schema=OFF")
        if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
            raise ValueError("sqlite_security_unavailable")
        for category, bound in (
            (sqlite3.SQLITE_LIMIT_LENGTH, 64 * 1024**2),
            (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 1024**2),
            (sqlite3.SQLITE_LIMIT_COLUMN, 2048),
            (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 1000),
            (sqlite3.SQLITE_LIMIT_VDBE_OP, 250_000),
            (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
            (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 1000),
            (sqlite3.SQLITE_LIMIT_TRIGGER_DEPTH, 100),
            (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 50),
            (sqlite3.SQLITE_LIMIT_WORKER_THREADS, 0),
        ):
            connection.setlimit(category, bound)
            if connection.getlimit(category) > bound:
                raise ValueError("sqlite_security_unavailable")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA temp_store=MEMORY")
        return _Restrictions(connection, cancel)
    except (AttributeError, NotImplementedError, sqlite3.Error) as error:
        raise ValueError("sqlite_security_unavailable") from error


def _catalog(connection, *, _schemas=None):
    from .mobile_preview_budget import catalog_cost, current, literal_catalog

    if _schemas is not None and current() is not None:
        return literal_catalog(connection, _schemas)
    catalog_cost(connection)
    rows = []
    size = 0
    for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
    ):
        size += sum(len(value.encode("utf-8")) for value in row if value is not None)
        if len(rows) >= _CATALOG_LIMIT or size > _CATALOG_BYTES:
            raise ValueError("sqlite_resource_limit")
        rows.append(tuple(row))
    return tuple(rows)


def _metadata(connection, catalog, restrictions=None):
    from .mobile_preview_budget import pragma_cost, reserve

    reserve(256 + 96 * len(catalog))

    def quote(name):
        return '"' + name.replace('"', '""') + '"'

    result = []
    for kind, name, _, sql in catalog:
        if kind in ("table", "view"):
            if restrictions is not None:
                restrictions.reading_fts_metadata = (
                    kind == "table"
                    and sql is not None
                    and sql.startswith("CREATE VIRTUAL TABLE")
                    and "USING fts5(" in sql
                )
            try:
                pragma_cost(connection, "table_xinfo", name)
                columns = tuple(
                    tuple(row)
                    for row in connection.execute(f"PRAGMA table_xinfo({quote(name)})")
                )
            finally:
                if restrictions is not None:
                    restrictions.reading_fts_metadata = False
            # Sequence reflects creation order, not index semantics.
            pragma_cost(connection, "index_list", name)
            indexes = tuple(
                sorted(
                    row[1:]
                    for row in connection.execute(f"PRAGMA index_list({quote(name)})")
                )
            )
            result.append((name, columns, indexes))
        elif kind == "index":
            pragma_cost(connection, "index_xinfo", name)
            result.append(
                (
                    name,
                    tuple(
                        tuple(row)
                        for row in connection.execute(
                            f"PRAGMA index_xinfo({quote(name)})"
                        )
                    ),
                )
            )
    return tuple(result)


@contextmanager
def _canvas_schema_access(connection, schema, restrictions=None):
    """Run the shipped pure CHECK only for an exact installed Canvas catalog.

    Candidate callers must first compare their entire catalog. Reference and
    reconstruction callers execute only this same frozen installed SQL. Python
    cannot mark a SQLite UDF innocuous, so trust is enabled for this bounded
    scope and removed before the restricted connection returns to its caller.
    """
    from tldw_chatbook.DB.canvas_payload_validation import (
        CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION,
        install_canvas_revision_payload_validator,
    )
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_DICTIONARY_UPDATE_SCHEMA,
        CHACHANOTES_V76_NATIVE_SCHEMAS,
        CHACHANOTES_V76_SHIPPED_SCHEMAS,
        CHACHANOTES_V77_SCHEMAS,
        CHACHANOTES_V78_SCHEMAS,
        CHACHANOTES_V79_SCHEMAS,
        CHACHANOTES_V80_SCHEMAS,
        CHACHANOTES_V81_SCHEMAS,
        CHACHANOTES_V82_SCHEMAS,
        CHACHANOTES_V83_SCHEMAS,
        CHACHANOTES_V84_SCHEMAS,
        CHACHANOTES_V85_SCHEMAS,
        CHACHANOTES_V86_SCHEMAS,
        CHACHANOTES_V87_SCHEMAS,
        CORE_SCHEMAS,
    )

    function = CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION
    if not any(function in sql for sql in schema):
        yield
        return
    from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_SCHEMA

    installed = next(
        sql for owner, _, sql in CORE_SCHEMAS if owner == "db.chachanotes.primary"
    )
    frozen = (
        installed,
        CHACHANOTES_DICTIONARY_UPDATE_SCHEMA,
        *CHACHANOTES_V76_SHIPPED_SCHEMAS,
        *CHACHANOTES_V76_NATIVE_SCHEMAS,
        *CHACHANOTES_V77_SCHEMAS,
        *CHACHANOTES_V78_SCHEMAS,
        *CHACHANOTES_V79_SCHEMAS,
        *CHACHANOTES_V80_SCHEMAS,
        *CHACHANOTES_V81_SCHEMAS,
        *CHACHANOTES_V82_SCHEMAS,
        *CHACHANOTES_V83_SCHEMAS,
        *CHACHANOTES_V84_SCHEMAS,
        *CHACHANOTES_V85_SCHEMAS,
        *CHACHANOTES_V86_SCHEMAS,
        *CHACHANOTES_V87_SCHEMAS,
        *(sql for _, sql in _SUBSCRIPTIONS_SCHEMA),
    )
    if schema not in frozen:
        raise ValueError("unsupported_schema")
    if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
        raise ValueError("sqlite_security_unavailable")
    install_canvas_revision_payload_validator(connection)
    try:
        if restrictions is not None:
            restrictions.canvas_schema = True
            restrictions.changing_schema_trust = True
        connection.execute("PRAGMA trusted_schema=ON")
        if connection.execute("PRAGMA trusted_schema").fetchone() != (1,):
            raise ValueError("sqlite_security_unavailable")
        if restrictions is not None:
            restrictions.changing_schema_trust = False
        yield
    finally:
        # An expired validation deadline must not interrupt removal of trust.
        if restrictions is not None:
            connection.set_progress_handler(None, 0)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
                raise ValueError("sqlite_security_unavailable")
        finally:
            if restrictions is not None:
                restrictions.canvas_schema = False
                restrictions.changing_schema_trust = False
                connection.set_progress_handler(
                    restrictions.progress, _PROGRESS_INTERVAL
                )
            connection.create_function(function, 3, None)


def _canvas_payload_issues(connection, restrictions):
    """Check payloads explicitly after the installed CHECK function is enabled.

    SQLite may have parsed the CHECK before this connection registered the UDF;
    its cached schema can then omit it from quick_check. A direct expression
    evaluates the exact shipped validator under the same limits and progress
    handler, regardless of that schema-cache state.
    """
    if restrictions is None or not restrictions.canvas_schema:
        return ()
    if (
        connection.execute(
            "SELECT 1 FROM canvas_revisions WHERE typeof(html) != 'text' OR "
            "canvas_revision_payload_valid(CAST(html AS BLOB),content_sha256,html_bytes) != 1 LIMIT 1"
        ).fetchone()
        is not None
    ):
        return ("invalid_sqlite_integrity",)
    return ()


def _reference(schema, *, _scope=None):
    from tldw_chatbook.DB.recovery_workflow_schema import open_installed_schema

    from .mobile_preview_budget import current, reserve

    preview_scope = current()
    if preview_scope is not None:
        if _scope is not None and _scope is not preview_scope:
            raise ValueError("mobile_preview_scope_changed")
        _scope = preview_scope
        # Reference schema is installed text. These are new statement/catalog
        # containers and fixed scalar query results, not a second data copy.
        reserve(4096 + 1024 * len(schema))

    with ExitStack() as resources:
        reference = open_installed_schema()
        if _scope is None:
            resources.enter_context(closing(reference))
        else:
            from .storage_admission import _CaptureLease

            lease = _CaptureLease(_scope)
            lease.attach(reference)
            resources.callback(lease.retire)
        reference.execute("PRAGMA trusted_schema=OFF")
        with _canvas_schema_access(reference, schema):
            # Execute only the installed catalog, never the candidate's SQL text.
            tables = [
                sql
                for sql in schema
                if sql[:20].upper().startswith("CREATE TABLE")
                or sql[:20].upper().startswith("CREATE VIRTUAL TABLE")
            ]
            others = [sql for sql in schema if sql not in tables]
            for sql in tables + others:
                # AUTOINCREMENT and FTS create their own internal/shadow tables.
                # Keep the same bounds without copying the growing catalog
                # into Python before every installed statement.
                count, size, existing = reference.execute(
                    "SELECT COUNT(*), COALESCE(SUM("
                    "length(CAST(COALESCE(type,'') AS BLOB)) + "
                    "length(CAST(COALESCE(name,'') AS BLOB)) + "
                    "length(CAST(COALESCE(tbl_name,'') AS BLOB)) + "
                    "length(CAST(COALESCE(sql,'') AS BLOB))),0), "
                    "COALESCE(MAX(sql = ?),0) FROM sqlite_schema",
                    (sql,),
                ).fetchone()
                if count > _CATALOG_LIMIT or size > _CATALOG_BYTES:
                    raise ValueError("sqlite_resource_limit")
                if existing:
                    continue
                reference.execute(sql)
            catalog = (
                _catalog(reference, _schemas=((0, schema),))
                if preview_scope is not None
                else _catalog(reference)
            )
            return catalog, _metadata(reference, catalog)


def _version(connection, owner_id):
    from .mobile_preview_budget import scalar_cost, table_cost

    scalar_cost(connection, 2)
    if owner_id in {
        "db.chachanotes.primary",
        "study.local",
        "quiz.local",
        "notes.sync_bindings",
        "chat.attachments",
    }:
        query = "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
    elif owner_id in {
        "db.media.primary",
        "db.prompts.primary",
        "db.library_ingest_jobs",
        "db.guardian",
    }:
        query = "SELECT version FROM schema_version"
    elif owner_id in {
        "db.library_collections",
        "db.workspaces",
        "db.agent_runs",
        "db.subscriptions",
        "db.scheduled_tasks",
        "notifications.client",
        "runtime.event_state",
        "runtime.sync_state",
        "db.dreams",
    }:
        query = "SELECT MAX(version) FROM schema_version"
    elif owner_id == "kanban.local":
        query = "SELECT CAST(value AS INTEGER) FROM local_kanban_schema_meta WHERE key='schema_version'"
    elif owner_id == "notes.file_notes":
        query = "SELECT 0"
    else:
        query = "PRAGMA user_version"
    if owner_id == "db.dreams":
        table_cost(connection, "schema_version", ("version",))
        history = [
            row[0] for row in connection.execute("SELECT version FROM schema_version")
        ]
        if not history or any(
            type(version) is not int or version not in range(1, 7)
            for version in history
        ):
            return None
    rows = connection.execute(query).fetchmany(2)
    return rows[0][0] if len(rows) == 1 else None


def _check_details(connection, owner, policy, restrictions):
    from .mobile_preview_budget import current, integrity_cost, reserve

    if restrictions.expired():
        raise ValueError("sqlite_resource_limit")
    actual = (
        _catalog(connection, _schemas=policy.schema_sql)
        if current() is not None
        else _catalog(connection)
    )
    reserve(256 + 64 * (len(actual) + len(policy.schema_sql)))
    if owner.owner_id == "db.agent_runs":
        reserve(128 + 32 * len(actual))
        base, stats = _split_statistics(actual)
    else:
        base, stats = actual, ()
    actual_sql = tuple(row[3] for row in base if row[3] is not None)
    matched = tuple(
        (version, schema)
        for version, schema in policy.schema_sql
        if schema == actual_sql
    )
    if not matched:
        return ("unsupported_schema",), None, None
    # Only after exact catalog match may metadata/version/domain queries prepare
    # expressions contained in the imported schema.
    version = _version(connection, owner.owner_id)
    if version not in policy.versions or not any(v == version for v, _ in matched):
        return ("unsupported_schema_version",), None, None
    with _canvas_schema_access(connection, matched[0][1], restrictions):
        reference_catalog, metadata = _reference(
            matched[0][1], _scope=getattr(restrictions, "capture_scope", None)
        )
        if base != reference_catalog:
            return ("unsupported_schema",), None, None
        actual_metadata = _metadata(connection, actual, restrictions)
        if stats:
            reserve(128 + 32 * (len(metadata) + len(stats)))
        if actual_metadata != _statistics_metadata(metadata, stats):
            return ("unsupported_schema",), None, None
        integrity_cost(connection, actual)
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return ("invalid_domain_reference",), None, None
        payload_issues = _canvas_payload_issues(connection, restrictions)
        if payload_issues:
            return payload_issues, None, None
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            return ("invalid_sqlite_integrity",), None, None
        if owner.owner_id == "db.subscriptions" and any(
            row[1] == "db_schema_version" for row in actual
        ):
            from tldw_chatbook.DB.recovery_operations import (
                _subscriptions_chachanotes_stamps,
            )

            expected_stamps = _subscriptions_chachanotes_stamps(actual_sql)
            if (
                connection.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
                ).fetchone()
                not in expected_stamps
            ):
                return ("unsupported_schema_version",), None, None
        checker = getattr(owner, "_validate_connection", None)
        if checker is not None:
            issues = checker(connection)
            if issues:
                return issues, None, None
        if restrictions.expired():
            raise ValueError("sqlite_resource_limit")
        if owner.owner_id == "db.agent_runs":
            restrictions.workflow_statistics = tuple(row[1] for row in stats)
        return (), version, (actual, actual_metadata)


def _check(connection, owner, policy, restrictions):
    issues, version, _ = _check_details(connection, owner, policy, restrictions)
    return issues, version


def _write_normalized_candidate(connection, destination, restrictions, cancel):
    from tldw_chatbook.DB.private_sqlite import (
        _backup_pages,
        _connect_recovery_candidate,
        _restore_destination_mode,
    )

    from .native_files import create_private_file

    def progress_guard():
        if cancel.is_set():
            raise InterruptedError("cancelled")
        if restrictions.expired():
            raise ValueError("detached_sqlite_limit")

    with create_private_file(destination):
        pass
    with closing(_connect_recovery_candidate(destination, writable=True)) as output:
        output.enable_load_extension(False)
        output.execute("PRAGMA trusted_schema=OFF")
        _backup_pages(
            connection, output, restore=False, progress_guard=progress_guard, pages=64
        )
        _restore_destination_mode(output, "delete", restore=False)


def _validate_candidate(
    owner: OwnerAdapter,
    candidate: Path,
    cancel: Event,
    *,
    migrate: bool,
    normalize_to: Path | None = None,
) -> tuple[tuple[str, ...], int | None]:
    """Validate installed content; optionally migrate this disposable file only."""
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    restrictions = None
    if cancel.is_set():
        if normalize_to is not None:
            raise InterruptedError("cancelled")
        return (("cancelled",), None)
    try:
        installed = _installed_owner(owner.owner_id)
        policy = installed.schema_policy()
        if owner.schema_policy() != policy:
            return (("unsupported_schema_policy",), None)
        with open_recovery_validation(
            installed.owner_id,
            candidate,
            writable=migrate or normalize_to is not None,
            with_restrictions=True,
            cancel=cancel,
        ) as (connection, restrictions):
            issues, version = _check(connection, installed, policy, restrictions)
            if issues:
                return (issues, None)
            if normalize_to is not None:
                _write_normalized_candidate(
                    connection, normalize_to, restrictions, cancel
                )
                return ((), version)
            if migrate and version != max(policy.versions):
                shipped_checkpoint = False
                if installed.owner_id == "db.chachanotes.primary":
                    from tldw_chatbook.DB.recovery_core_schema import (
                        CHACHANOTES_V76_NATIVE_SCHEMAS,
                        CHACHANOTES_V76_SHIPPED_SCHEMAS,
                        CHACHANOTES_V77_SCHEMAS,
                        CHACHANOTES_V78_SCHEMAS,
                        CHACHANOTES_V79_SCHEMAS,
                        CHACHANOTES_V80_SCHEMAS,
                        CHACHANOTES_V81_SCHEMAS,
                        CHACHANOTES_V82_SCHEMAS,
                        CHACHANOTES_V83_SCHEMAS,
                        CHACHANOTES_V84_SCHEMAS,
                        CHACHANOTES_V85_SCHEMAS,
                        CHACHANOTES_V86_SCHEMAS,
                    )

                    actual_sql = tuple(
                        row[3] for row in _catalog(connection) if row[3] is not None
                    )
                    if not (
                        version == 76
                        and actual_sql
                        in (
                            *CHACHANOTES_V76_NATIVE_SCHEMAS,
                            *CHACHANOTES_V76_SHIPPED_SCHEMAS,
                        )
                        or version == 77
                        and actual_sql in CHACHANOTES_V77_SCHEMAS
                        or version == 78
                        and actual_sql in CHACHANOTES_V78_SCHEMAS
                        or version == 79
                        and actual_sql in CHACHANOTES_V79_SCHEMAS
                        or version == 80
                        and actual_sql in CHACHANOTES_V80_SCHEMAS
                        or version == 81
                        and actual_sql in CHACHANOTES_V81_SCHEMAS
                        or version == 82
                        and actual_sql in CHACHANOTES_V82_SCHEMAS
                        or version == 83
                        and actual_sql in CHACHANOTES_V83_SCHEMAS
                        or version == 84
                        and actual_sql in CHACHANOTES_V84_SCHEMAS
                        or version == 85
                        and actual_sql in CHACHANOTES_V85_SCHEMAS
                        or version == 86
                        and actual_sql in CHACHANOTES_V86_SCHEMAS
                    ):
                        return (("unsupported_schema_migration",), None)
                    shipped_checkpoint = actual_sql in CHACHANOTES_V76_SHIPPED_SCHEMAS
                restrictions.migrating = True
                restrictions.migration_owner = installed.owner_id
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    while version != max(policy.versions):
                        choices = [
                            (end, sql)
                            for start, end, sql in policy.migration_steps
                            if start == version and end > start
                        ]
                        bridge = None
                        if installed.owner_id == "db.agent_runs":
                            from tldw_chatbook.DB.recovery_operations import (
                                _legacy_workflow_upgrade,
                            )

                            bridge = _legacy_workflow_upgrade(connection)
                        if bridge is not None:
                            expected, statements = bridge
                        elif len(choices) == 1:
                            expected, statements = choices[0]
                        else:
                            return (("unsupported_schema_migration",), None)
                        # Fixed installed DDL reparses other tables' CHECKs.
                        # Authenticate the complete pre-step catalog, reuse its
                        # pure Canvas scope, then retire it before validation.
                        with _canvas_schema_access(
                            connection,
                            tuple(
                                row[3]
                                for row in _catalog(connection)
                                if row[3] is not None
                            ),
                            restrictions,
                        ):
                            if (
                                shipped_checkpoint
                                and version == 76
                                or installed.owner_id == "db.chachanotes.primary"
                                and version in (77, 78, 79, 80, 81, 84, 85)
                            ):
                                migration = (
                                    Path(__file__).resolve().parents[1]
                                    / "DB"
                                    / "migrations"
                                    / (
                                        "chachanotes_prompt_mixins.sql"
                                        if version == 85
                                        else "chachanotes_v84_to_v85_archive_chronology.sql"
                                        if version == 84
                                        else "chachanotes_v81_to_v82_canvas_sibling_sequences.sql"
                                        if version == 81
                                        else "chachanotes_v77_to_v78_fleet_progress.sql"
                                        if version == 77
                                        else "chachanotes_v78_to_v79_sargable_timestamp_normalization.sql"
                                        if version == 78
                                        else "chachanotes_v79_to_v80_conversations_browse_order_index.sql"
                                        if version == 79
                                        else "chachanotes_v80_to_v81_response_rules.sql"
                                        if version == 80
                                        else "chachanotes_v76_to_v77_agent_chat_starts.sql"
                                    )
                                )
                                restrictions.shipped_checkpoint_migration = version == 76
                                restrictions.fleet_progress_migration = version == 77
                                restrictions.sargable_migration = version == 78
                                restrictions.browse_order_migration = version == 79
                                restrictions.response_rules_migration = version == 80
                                restrictions.canvas_sibling_migration = version == 81
                                restrictions.archive_chronology_migration = version == 84
                                restrictions.prompt_mixins_migration = version == 85
                                try:
                                    pending = ""
                                    for line in migration.read_text(
                                        encoding="utf-8"
                                    ).splitlines(keepends=True):
                                        pending += line
                                        if sqlite3.complete_statement(pending):
                                            if restrictions.expired():
                                                raise InterruptedError
                                            # Only this fixed RENAME needs internal sequence/temp
                                            # updates; direct SQL cannot borrow their authority.
                                            restrictions.shipped_checkpoint_rename = (
                                                pending.strip()
                                                == (
                                                    "ALTER TABLE console_dispatch_checkpoints_v77 "
                                                    "RENAME TO console_dispatch_checkpoints;"
                                                )
                                            )
                                            try:
                                                connection.execute(pending)
                                            finally:
                                                restrictions.shipped_checkpoint_rename = (
                                                    False
                                                )
                                            pending = ""
                                    if pending.strip():
                                        raise ValueError("incomplete_installed_migration")
                                finally:
                                    restrictions.shipped_checkpoint_migration = False
                                    restrictions.fleet_progress_migration = False
                                    restrictions.sargable_migration = False
                                    restrictions.browse_order_migration = False
                                    restrictions.response_rules_migration = False
                                    restrictions.canvas_sibling_migration = False
                                    restrictions.archive_chronology_migration = False
                                    restrictions.prompt_mixins_migration = False
                            restrictions.portrait_migration = (
                                version
                                if installed.owner_id == "db.chachanotes.primary"
                                and version in (82, 83)
                                else None
                            )
                            try:
                                for statement in statements:
                                    if restrictions.expired():
                                        raise InterruptedError
                                    restrictions.dropping_workflow_phase = (
                                        installed.owner_id == "db.agent_runs"
                                        and statement
                                        == "DROP TABLE workflow_phase_attempts_v26"
                                    )
                                    try:
                                        connection.execute(statement)
                                    finally:
                                        restrictions.dropping_workflow_phase = False
                            finally:
                                restrictions.portrait_migration = None
                        restrictions.migrating = False
                        issues, version = _check(
                            connection, installed, policy, restrictions
                        )
                        if issues or version != expected:
                            return (issues or ("unsupported_schema_migration",), None)
                        restrictions.migrating = True
                    if restrictions.expired():
                        raise InterruptedError
                    connection.commit()
                finally:
                    # Cancellation cannot interrupt the rollback needed to leave
                    # this disposable candidate at its original committed state.
                    connection.set_progress_handler(None, 0)
                    restrictions.migrating = True
                    try:
                        connection.rollback()
                    finally:
                        restrictions.migrating = False
                        restrictions.migration_owner = None
            if restrictions.expired():
                return (
                    ("cancelled" if cancel.is_set() else "sqlite_resource_limit",),
                    None,
                )
            return ((), version)
    except (
        sqlite3.Error,
        OSError,
        ValueError,
        TypeError,
        MemoryError,
        InterruptedError,
    ) as error:
        if normalize_to is not None:
            raise
        if cancel.is_set():
            return (("cancelled",), None)
        if restrictions is not None and restrictions.expired():
            return (("sqlite_resource_limit",), None)
        if isinstance(error, ValueError) and str(error) in {
            "sqlite_security_unavailable",
            "sqlite_resource_limit",
            "unsupported_sqlite_owner",
        }:
            return ((str(error),), None)
        return (("sqlite_validation_unavailable",), None)


def normalize_detached_candidate(
    owner: OwnerAdapter, source: Path, destination: Path, cancel: Event
) -> int:
    """Copy a validated disposable candidate, then revalidate the closed output."""
    installed = _installed_owner(owner.owner_id)
    issues, _ = _validate_candidate(
        installed, source, cancel, migrate=False, normalize_to=destination
    )
    if issues:
        raise ValueError(issues[0])
    return validated_schema_version(installed, destination, cancel)


def validate_candidate(
    owner: OwnerAdapter, candidate: Path, cancel: Event, *, migrate: bool
) -> tuple[str, ...]:
    """Validate with the original issue-only contract on one restricted connection."""
    return _validate_candidate(owner, candidate, cancel, migrate=migrate)[0]


def validated_schema_version(
    owner: OwnerAdapter, candidate: Path, cancel: Event
) -> int:
    """Return the exact observed installed version after full restricted validation."""
    issues, version = _validate_candidate(owner, candidate, cancel, migrate=False)
    if issues or version is None:
        raise ValueError(issues[0] if issues else "sqlite_validation_unavailable")
    return version


def _validate_sqlite(
    connection: sqlite3.Connection,
    versions: tuple[int, ...],
    schemas: tuple[tuple[int, tuple[str, ...]], ...],
    *,
    version_query: str = "PRAGMA user_version",
    allow_statistics: bool = False,
) -> tuple[str, ...]:
    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import (
        current,
        integrity_cost,
        matching_schemas,
        reserve,
        scalar_cost,
    )

    restrictions = _current_restrictions(connection)
    if any(
        "canvas_revision_payload_valid" in sql
        for _, schema in schemas
        for sql in schema
    ):
        if restrictions is None:
            restrictions = _restrict_connection(connection)
    else:
        connection.execute("PRAGMA trusted_schema=OFF")
    if current() is not None and not allow_statistics:
        matches = matching_schemas(connection, schemas)
        matched = tuple(row[0] for row in matches)
        actual = matches[0][1] if matches else ()
    else:
        catalog = _catalog(connection, _schemas=schemas)
        if allow_statistics:
            reserve(128 + 32 * len(catalog))
            catalog, _stats = _split_statistics(catalog)
        actual = tuple(row[3] for row in catalog if row[3] is not None)
        matched = tuple(known for known, schema in schemas if actual == schema)
    if not matched:
        return ("unsupported_schema",)
    scalar_cost(connection)
    version = connection.execute(version_query).fetchone()[0]
    if version not in versions or version not in matched:
        return ("unsupported_schema_version",)
    with _canvas_schema_access(connection, actual, restrictions):
        integrity_cost(connection)
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return ("invalid_domain_reference",)
        payload_issues = _canvas_payload_issues(connection, restrictions)
        if payload_issues:
            return payload_issues
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            return ("invalid_sqlite_integrity",)
        return ()


def validate_goal_rows(connection):
    """Validate exact goal payloads and cross-table ownership without opening sources."""
    import hashlib
    from typing import get_args

    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import (
        connection_budget,
        scalar_cost,
        table_cost,
    )

    budget = connection_budget(connection)
    scalar_cost(connection, 2)

    from tldw_chatbook.Agents.goal_models import (
        GoalCheckpoint,
        GoalEvidence,
        GoalReport,
        GoalRequest,
        GoalSnapshot,
    )

    # Version admission runs first; still fail closed without a version.
    version = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[
        0
    ]
    if type(version) is not int:
        return ("unsupported_schema_version",)
    # Goal rows exist from v24 on (v25 adds review-boundary claims only).
    if version < 24:
        return ()
    invalid = connection.execute(
        "SELECT 1 FROM goal_runs g JOIN automatic_work_chains c ON c.id=g.chain_id "
        "WHERE c.conversation_id!=g.conversation_id OR c.allowance_root_chain_id IS NOT NULL "
        "UNION ALL SELECT 1 FROM goal_iterations i JOIN goal_runs g ON g.id=i.goal_id "
        "JOIN automatic_wake_attempts a ON a.id=i.attempt_id "
        "WHERE a.attempt_kind!='goal_iteration' OR a.chain_id!=g.chain_id OR a.conversation_id!=g.conversation_id "
        "OR (i.run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM agent_runs r WHERE r.id=i.run_id AND r.work_chain_id=g.chain_id AND r.conversation_id=g.conversation_id)) "
        "UNION ALL SELECT 1 FROM automatic_wake_attempts a WHERE a.attempt_kind='goal_iteration' "
        "AND NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=a.id) "
        "UNION ALL SELECT 1 FROM goal_reports r JOIN goal_iterations i ON i.id=r.iteration_id WHERE r.goal_id!=i.goal_id "
        "UNION ALL SELECT 1 FROM goal_checkpoints c WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=c.attempt_id AND i.goal_id=c.goal_id) "
        "UNION ALL SELECT 1 FROM goal_evidence e WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=e.attempt_id AND i.goal_id=e.goal_id) "
        "UNION ALL SELECT 1 FROM goal_payload_reservations p WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=p.attempt_id AND i.goal_id=p.goal_id) LIMIT 1"
    ).fetchone()
    if invalid is not None:
        return ("invalid_goal_ownership",)
    if budget is not None:
        for table, columns in (
            ("goal_runs", ("status", "request_json", "payload_hash")),
            ("goal_reports", ("payload_json",)),
            ("goal_checkpoints", ("attempt_id", "payload_json")),
            ("goal_evidence", ("id", "goal_id", "attempt_id", "payload_json")),
        ):
            table_cost(connection, table, columns)
        budget.reserve(2048)
        count, size = connection.execute(
            "SELECT count(*),coalesce(sum(coalesce(length(CAST(i.run_id AS BLOB)),0)),0) "
            "FROM goal_evidence e JOIN goal_iterations i ON i.attempt_id=e.attempt_id"
        ).fetchone()
        budget.reserve(256 + 128 * count + 4 * size)

    def payload_cost(payload):
        if budget is not None:
            budget.json(payload)
            budget.reserve(1024 + 4 * len(payload))

    try:
        for status, payload, digest in connection.execute(
            "SELECT status, request_json, payload_hash FROM goal_runs"
        ):
            if status not in get_args(GoalSnapshot.model_fields["status"].annotation):
                return ("invalid_goal_payload",)
            if status == "removed":
                if payload:
                    return ("invalid_goal_payload",)
                continue
            payload_cost(payload)
            GoalRequest.model_validate_json(payload)
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                return ("invalid_goal_payload",)
        for (payload,) in connection.execute("SELECT payload_json FROM goal_reports"):
            payload_cost(payload)
            GoalReport.model_validate_json(payload)
        for attempt_id, ordinal, payload in connection.execute(
            "SELECT c.attempt_id,i.ordinal,c.payload_json FROM goal_checkpoints c "
            "JOIN goal_iterations i ON i.attempt_id=c.attempt_id"
        ):
            payload_cost(payload)
            checkpoint = GoalCheckpoint.model_validate_json(payload)
            if checkpoint.id != attempt_id or checkpoint.ordinal != ordinal:
                return ("invalid_goal_ownership",)
        for evidence_id, goal_id, attempt_id, run_id, payload in connection.execute(
            "SELECT e.id,e.goal_id,e.attempt_id,i.run_id,e.payload_json FROM goal_evidence e "
            "JOIN goal_iterations i ON i.attempt_id=e.attempt_id"
        ):
            payload_cost(payload)
            evidence = GoalEvidence.model_validate_json(payload)
            if (
                evidence.id,
                evidence.goal_id,
                evidence.attempt_id,
                evidence.run_id,
            ) != (evidence_id, goal_id, attempt_id, run_id):
                return ("invalid_goal_ownership",)
    except (ValueError, TypeError):
        return ("invalid_goal_payload",)
    return ()


@contextmanager
def _open_recovery_validation(
    owner_id: "str",
    candidate: "Path",
    *,
    writable: "bool",
    with_restrictions: "bool" = False,
    cancel=None,
    _custody=None,
) -> "Iterator[sqlite3.Connection | tuple[sqlite3.Connection, Any]]":
    """Open a disposable candidate with restrictions before any imported query.

    The caller owns staging and must never pass a live destination. This authority
    deliberately does not enroll the candidate as an ordinary profile resource.
    Internal validators may request the original restriction budget alongside
    the connection so catalog inspection and later checks share one deadline.
    """
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        _installed_owner,
        _recovery_restriction_scope,
        _restrict_connection,
    )
    from tldw_chatbook.DB import private_sqlite as _sqlite

    _installed_owner(owner_id)
    with _sqlite.contextlib.ExitStack() as resources:
        if _custody is None:
            connection = resources.enter_context(
                _sqlite.contextlib.closing(
                    _sqlite._connect_registered_sqlite(
                        "recovery.validation",
                        candidate,
                        read_only=not writable,
                        must_exist=True,
                        immutable=not writable,
                        isolation_level=None,
                        cached_statements=0,
                        timeout=0,
                    )
                )
            )
        else:
            if writable:
                raise ValueError("captured_validation_is_read_only")
            from tldw_chatbook.Backup_Recovery.archive_reader import _regular
            from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureLease

            stream = resources.enter_context(_regular(candidate, _custody=_custody))
            lease = _CaptureLease(_custody.scope)
            lease.descriptor_outcome = _sqlite._SQLiteDescriptorOutcome()
            resources.callback(lease.retire)
            connection = _sqlite.connect_private_sqlite_descriptor(
                "recovery.validation",
                stream.fileno(),
                _native_outcome=lease.descriptor_outcome,
                isolation_level=None,
                cached_statements=0,
                timeout=0,
            )
            lease.attach(connection)
        restrictions = _restrict_connection(connection, cancel)
        restrictions.capture_scope = None if _custody is None else _custody.scope
        restrictions.capture_stream = None if _custody is None else stream
        with _recovery_restriction_scope(connection, restrictions):
            yield ((connection, restrictions) if with_restrictions else connection)
