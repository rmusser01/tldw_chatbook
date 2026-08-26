"""Local persistence for Console context policy, memory, and auxiliary calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping

from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextPolicyError,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class AuxiliaryAttemptStatus(str, Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


TERMINAL_AUXILIARY_ATTEMPT_STATUSES = frozenset(
    {
        AuxiliaryAttemptStatus.SUCCEEDED,
        AuxiliaryAttemptStatus.FAILED,
        AuxiliaryAttemptStatus.CANCELLED,
        AuxiliaryAttemptStatus.STALE,
    }
)

DEFAULT_ACTIVE_MEMORY_PAGE_SIZE = 100
DEFAULT_AUXILIARY_ATTEMPT_PAGE_SIZE = 50
MAX_REPOSITORY_PAGE_SIZE = 500


@dataclass(frozen=True)
class ContextPolicyReadResult:
    overrides: ConsoleContextPolicyOverrides
    revision: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class AuxiliaryPricingProvenance:
    """Content-free pricing identity captured with an auxiliary call."""

    catalog_revision: str | None = None
    source: str | None = None
    estimated: bool = True

    def __post_init__(self) -> None:
        if self.catalog_revision is not None:
            _validate_bounded_text("catalog_revision", self.catalog_revision, 200)
        if self.source is not None:
            _validate_bounded_text("source", self.source, 200)
        if type(self.estimated) is not bool:
            raise ValueError("estimated must be a boolean")

    def to_json(self) -> str:
        return json.dumps(
            {
                "catalog_revision": self.catalog_revision,
                "source": self.source,
                "estimated": self.estimated,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class AuxiliaryAttemptStart:
    operation_id: str
    conversation_id: str
    purpose: str
    provider: str
    model: str
    requested_output_cap: int
    estimated_input_tokens: int
    started_at: str

    def __post_init__(self) -> None:
        _validate_bounded_text("operation_id", self.operation_id, 200)
        _validate_bounded_text("conversation_id", self.conversation_id, 200)
        if self.purpose != "conversation_compaction":
            raise ValueError("purpose must be conversation_compaction")
        _validate_bounded_text("provider", self.provider, 120)
        _validate_bounded_text("model", self.model, 500)
        _validate_bounded_text("started_at", self.started_at, 80)
        if type(self.requested_output_cap) is not int or self.requested_output_cap <= 0:
            raise ValueError("requested_output_cap must be a positive integer")
        if (
            type(self.estimated_input_tokens) is not int
            or self.estimated_input_tokens < 0
        ):
            raise ValueError("estimated_input_tokens must be a non-negative integer")


@dataclass(frozen=True)
class ConsoleMemoryRecord:
    """Branch-provenanced derived memory; transcript rows remain authoritative."""

    memory_id: str
    conversation_id: str
    boundary_message_id: str
    captured_leaf_message_id: str
    lineage_json: str
    summary_text: str = field(repr=False)
    provider: str
    model: str
    prompt_id: str
    prompt_revision: int
    prompt_digest: str
    selected_units_json: str
    summarized_prefix_digest: str
    input_tokens: int
    output_tokens: int
    before_tokens: int
    after_tokens: int
    created_at: str
    revision: int = 1
    active: bool = True
    source_kind: str = "generated"

    def __post_init__(self) -> None:
        for name, maximum in (
            ("memory_id", 200),
            ("conversation_id", 200),
            ("boundary_message_id", 200),
            ("captured_leaf_message_id", 200),
            ("provider", 120),
            ("model", 500),
            ("prompt_id", 200),
            ("prompt_digest", 256),
            ("summarized_prefix_digest", 256),
            ("created_at", 80),
        ):
            _validate_bounded_text(name, getattr(self, name), maximum)
        if not isinstance(self.summary_text, str) or not self.summary_text.strip():
            raise ValueError("summary_text must be a non-empty string")
        for name in ("lineage_json", "selected_units_json"):
            try:
                value = json.loads(getattr(self, name))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{name} must be valid JSON") from exc
            if not isinstance(value, list):
                raise ValueError(f"{name} must encode a JSON list")
        for name in (
            "prompt_revision",
            "revision",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "input_tokens",
            "output_tokens",
            "before_tokens",
            "after_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.active) is not bool:
            raise ValueError("active must be a boolean")
        if self.source_kind not in {"generated", "legacy"}:
            raise ValueError("source_kind must be generated or legacy")


class ConsoleContextRepository:
    """Repository for local-only conversation memory ownership."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def load_policy(self, conversation_id: str) -> ContextPolicyReadResult:
        if not isinstance(conversation_id, str) or not conversation_id:
            return ContextPolicyReadResult(
                ConsoleContextPolicyOverrides(), error="invalid_conversation_id"
            )
        with self.db.transaction() as cursor:
            row = cursor.execute(
                """
            SELECT budget_mode, custom_budget_tokens, compaction_mode,
                   compaction_representation,
                   trigger_ratio, target_ratio, summary_max_tokens,
                   failure_behavior, carry_forward_mode, policy_revision
              FROM console_conversation_context_policy
             WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return ContextPolicyReadResult(ConsoleContextPolicyOverrides())
        mapping = {key: row[key] for key in row.keys() if key != "policy_revision"}
        try:
            overrides = ConsoleContextPolicyOverrides.from_mapping(mapping)
        except ContextPolicyError:
            return ContextPolicyReadResult(
                ConsoleContextPolicyOverrides(),
                revision=_positive_revision(row["policy_revision"]),
                error="invalid_persisted_context_policy",
            )
        return ContextPolicyReadResult(
            overrides,
            revision=_positive_revision(row["policy_revision"]),
        )

    def save_policy(
        self,
        conversation_id: str,
        overrides: ConsoleContextPolicyOverrides,
    ) -> int | None:
        """Upsert sparse overrides; an empty override removes the local row."""
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be a non-empty string")
        if not isinstance(overrides, ConsoleContextPolicyOverrides):
            raise TypeError("overrides must be ConsoleContextPolicyOverrides")
        if overrides.is_empty:
            with self.db.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM console_conversation_context_policy "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                )
            return None

        values = overrides.to_dict()
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO console_conversation_context_policy(
                    conversation_id, budget_mode, custom_budget_tokens,
                    compaction_mode, compaction_representation,
                    trigger_ratio, target_ratio,
                    summary_max_tokens, failure_behavior, carry_forward_mode,
                    policy_revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    budget_mode = excluded.budget_mode,
                    custom_budget_tokens = excluded.custom_budget_tokens,
                    compaction_mode = excluded.compaction_mode,
                    compaction_representation = excluded.compaction_representation,
                    trigger_ratio = excluded.trigger_ratio,
                    target_ratio = excluded.target_ratio,
                    summary_max_tokens = excluded.summary_max_tokens,
                    failure_behavior = excluded.failure_behavior,
                    carry_forward_mode = excluded.carry_forward_mode,
                    policy_revision = console_conversation_context_policy.policy_revision + 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    conversation_id,
                    values.get("budget_mode"),
                    values.get("custom_budget_tokens"),
                    values.get("compaction_mode"),
                    values.get("compaction_representation"),
                    values.get("trigger_ratio"),
                    values.get("target_ratio"),
                    values.get("summary_max_tokens"),
                    values.get("failure_behavior"),
                    values.get("carry_forward_mode"),
                ),
            )
            row = cursor.execute(
                "SELECT policy_revision FROM console_conversation_context_policy "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return _positive_revision(row[0] if row is not None else None)

    def insert_memory(self, record: ConsoleMemoryRecord) -> None:
        """Insert one immutable generated-memory revision."""
        if not isinstance(record, ConsoleMemoryRecord):
            raise TypeError("record must be ConsoleMemoryRecord")
        with self.db.transaction() as cursor:
            _insert_memory(cursor, record)

    def insert_memory_if_current(
        self,
        record: ConsoleMemoryRecord,
        *,
        expected_memory_id: str | None,
        expected_memory_revision: int | None,
    ) -> bool:
        """Commit only while the admitted active memory revision still exists."""
        if not isinstance(record, ConsoleMemoryRecord):
            raise TypeError("record must be ConsoleMemoryRecord")
        if expected_memory_id is None:
            if expected_memory_revision is not None:
                raise ValueError("A memory revision requires a memory id")
        else:
            _validate_bounded_text("expected_memory_id", expected_memory_id, 200)
            if (
                type(expected_memory_revision) is not int
                or expected_memory_revision <= 0
            ):
                raise ValueError("expected_memory_revision must be positive")
        with self.db.transaction() as cursor:
            if expected_memory_id is not None:
                row = cursor.execute(
                    """
                    SELECT revision, active
                      FROM console_conversation_memories
                     WHERE id = ? AND conversation_id = ?
                    """,
                    (expected_memory_id, record.conversation_id),
                ).fetchone()
                if (
                    row is None
                    or int(row["active"]) != 1
                    or int(row["revision"]) != expected_memory_revision
                ):
                    return False
            _insert_memory(cursor, record)
        return True

    def list_active_memories(
        self,
        conversation_id: str,
        *,
        limit: int = DEFAULT_ACTIVE_MEMORY_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[ConsoleMemoryRecord, ...]:
        """Return a bounded newest-first page of memory candidates.

        Args:
            conversation_id: Durable conversation whose generated memories are read.
            limit: Maximum number of rows returned, from 1 through 500.
            offset: Zero-based row offset for deterministic pagination.

        Returns:
            Decoded generated-memory records; corrupt derived rows are omitted.

        Raises:
            ValueError: If an identifier or pagination value is invalid.
        """
        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_page(limit=limit, offset=offset)
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                """
            SELECT *
              FROM console_conversation_memories
             WHERE conversation_id = ? AND active = 1
                   AND source_kind = 'generated'
             ORDER BY created_at DESC, rowid DESC
             LIMIT ? OFFSET ?
            """,
                (conversation_id, limit, offset),
            ).fetchall()
        records: list[ConsoleMemoryRecord] = []
        for row in rows:
            try:
                records.append(_memory_from_row(row))
            except (TypeError, ValueError, json.JSONDecodeError):
                # Corrupt derived memory is never eligible for injection.
                continue
        return tuple(records)

    def deactivate_memory(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        reset_at: str,
    ) -> bool:
        """Deactivate one exact memory revision for current-branch reset."""
        _validate_bounded_text("memory_id", memory_id, 200)
        _validate_bounded_text("reset_at", reset_at, 80)
        if type(expected_revision) is not int or expected_revision <= 0:
            raise ValueError("expected_revision must be a positive integer")
        with self.db.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE console_conversation_memories
                   SET active = 0, reset_at = ?, revision = revision + 1
                 WHERE id = ? AND active = 1 AND revision = ?
                """,
                (reset_at, memory_id, expected_revision),
            )
        return result.rowcount == 1

    def deactivate_all_memories(
        self,
        conversation_id: str,
        *,
        reset_at: str,
    ) -> int:
        """Deactivate every active branch memory after separate confirmation."""
        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_bounded_text("reset_at", reset_at, 80)
        with self.db.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE console_conversation_memories
                   SET active = 0, reset_at = ?, revision = revision + 1
                 WHERE conversation_id = ? AND active = 1
                """,
                (reset_at, conversation_id),
            )
        return result.rowcount

    def reactivate_memory(
        self,
        memory_id: str,
        *,
        expected_revision: int,
    ) -> bool:
        """Undo an exact current-branch reset when no later mutation won."""
        _validate_bounded_text("memory_id", memory_id, 200)
        if type(expected_revision) is not int or expected_revision <= 0:
            raise ValueError("expected_revision must be a positive integer")
        with self.db.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE console_conversation_memories
                   SET active = 1, reset_at = NULL, revision = revision + 1
                 WHERE id = ? AND active = 0 AND revision = ?
                """,
                (memory_id, expected_revision),
            )
        return result.rowcount == 1

    def start_auxiliary_attempt(self, attempt: AuxiliaryAttemptStart) -> None:
        """Record admission without accepting transcript or summary content."""
        if not isinstance(attempt, AuxiliaryAttemptStart):
            raise TypeError("attempt must be AuxiliaryAttemptStart")
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO console_auxiliary_attempts(
                    operation_id, conversation_id, purpose, provider, model,
                    requested_output_cap, estimated_input_tokens, status,
                    started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?)
                """,
                (
                    attempt.operation_id,
                    attempt.conversation_id,
                    attempt.purpose,
                    attempt.provider,
                    attempt.model,
                    attempt.requested_output_cap,
                    attempt.estimated_input_tokens,
                    attempt.started_at,
                ),
            )

    def finish_auxiliary_attempt(
        self,
        operation_id: str,
        *,
        status: AuxiliaryAttemptStatus,
        finished_at: str,
        elapsed_ms: int | None = None,
        usage: ProviderUsage | None = None,
        pricing: AuxiliaryPricingProvenance | None = None,
    ) -> bool:
        """Record one terminal outcome using only bounded content-free fields."""
        _validate_bounded_text("operation_id", operation_id, 200)
        _validate_bounded_text("finished_at", finished_at, 80)
        if status not in TERMINAL_AUXILIARY_ATTEMPT_STATUSES:
            raise ValueError("status must be a terminal auxiliary-attempt status")
        if elapsed_ms is not None and (type(elapsed_ms) is not int or elapsed_ms < 0):
            raise ValueError("elapsed_ms must be a non-negative integer")
        usage_json = usage.to_json() if usage is not None else None
        pricing_json = pricing.to_json() if pricing is not None else None
        with self.db.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE console_auxiliary_attempts
                   SET status = ?, finished_at = ?, elapsed_ms = ?,
                       provider_usage_json = ?, pricing_provenance_json = ?
                 WHERE operation_id = ? AND status = 'started'
                """,
                (
                    status.value,
                    finished_at,
                    elapsed_ms,
                    usage_json,
                    pricing_json,
                    operation_id,
                ),
            )
        return result.rowcount == 1

    def get_auxiliary_attempt(self, operation_id: str) -> Mapping[str, Any] | None:
        _validate_bounded_text("operation_id", operation_id, 200)
        with self.db.transaction() as cursor:
            row = cursor.execute(
                "SELECT * FROM console_auxiliary_attempts WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_auxiliary_attempts(
        self,
        conversation_id: str,
        *,
        limit: int = DEFAULT_AUXILIARY_ATTEMPT_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return a bounded page of content-free auxiliary accounting rows.

        Args:
            conversation_id: Durable conversation whose attempts are read.
            limit: Maximum number of rows returned, from 1 through 500.
            offset: Zero-based row offset for deterministic pagination.

        Returns:
            Newest-first content-free attempt mappings.

        Raises:
            ValueError: If an identifier or pagination value is invalid.
        """

        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_page(limit=limit, offset=offset)
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT operation_id, conversation_id, purpose, provider, model,
                       requested_output_cap, estimated_input_tokens, status,
                       started_at, finished_at, elapsed_ms, provider_usage_json,
                       pricing_provenance_json
                  FROM console_auxiliary_attempts
                 WHERE conversation_id = ?
                 ORDER BY started_at DESC, operation_id DESC
                 LIMIT ? OFFSET ?
                """,
                (conversation_id, limit, offset),
            ).fetchall()
        return tuple(dict(row) for row in rows)


def _validate_page(*, limit: int, offset: int) -> None:
    if type(limit) is not int or not 1 <= limit <= MAX_REPOSITORY_PAGE_SIZE:
        raise ValueError(
            f"limit must be an integer between 1 and {MAX_REPOSITORY_PAGE_SIZE}"
        )
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")


def _positive_revision(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _memory_from_row(row: Mapping[str, Any]) -> ConsoleMemoryRecord:
    """Decode one generated-memory row through the public value contract."""
    return ConsoleMemoryRecord(
        memory_id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        boundary_message_id=str(row["boundary_message_id"]),
        captured_leaf_message_id=str(row["captured_leaf_message_id"]),
        lineage_json=str(row["lineage_json"]),
        summary_text=str(row["summary_text"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        prompt_id=str(row["prompt_id"]),
        prompt_revision=int(row["prompt_revision"]),
        prompt_digest=str(row["prompt_digest"]),
        selected_units_json=str(row["selected_units_json"]),
        summarized_prefix_digest=str(row["summarized_prefix_digest"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        before_tokens=int(row["before_tokens"]),
        after_tokens=int(row["after_tokens"]),
        created_at=str(row["created_at"]),
        revision=int(row["revision"]),
        active=bool(row["active"]),
        source_kind=str(row["source_kind"]),
    )


def _insert_memory(cursor: Any, record: ConsoleMemoryRecord) -> None:
    cursor.execute(
        """
        INSERT INTO console_conversation_memories(
            id, conversation_id, boundary_message_id,
            captured_leaf_message_id, lineage_json, summary_text,
            provider, model, prompt_id, prompt_revision, prompt_digest,
            selected_units_json, summarized_prefix_digest,
            input_tokens, output_tokens, before_tokens, after_tokens,
            created_at, revision, active, source_kind
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.memory_id,
            record.conversation_id,
            record.boundary_message_id,
            record.captured_leaf_message_id,
            record.lineage_json,
            record.summary_text,
            record.provider,
            record.model,
            record.prompt_id,
            record.prompt_revision,
            record.prompt_digest,
            record.selected_units_json,
            record.summarized_prefix_digest,
            record.input_tokens,
            record.output_tokens,
            record.before_tokens,
            record.after_tokens,
            record.created_at,
            record.revision,
            int(record.active),
            record.source_kind,
        ),
    )


def _validate_bounded_text(name: str, value: object, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{name} must be a non-empty string up to {maximum} characters"
        )
