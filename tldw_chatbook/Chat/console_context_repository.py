"""Local persistence for Console context policy, memory, and auxiliary calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
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


class MemoryCoverageKind(str, Enum):
    PREFIX = "prefix"
    RANGE = "range"


class MemoryOriginKind(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL_REWIND = "manual_rewind"


class MemorySelectionKind(str, Enum):
    SELECT = "select"
    RESET = "reset"


TERMINAL_AUXILIARY_ATTEMPT_STATUSES = frozenset(
    {
        AuxiliaryAttemptStatus.SUCCEEDED,
        AuxiliaryAttemptStatus.FAILED,
        AuxiliaryAttemptStatus.CANCELLED,
        AuxiliaryAttemptStatus.STALE,
    }
)

DEFAULT_ACTIVE_MEMORY_PAGE_SIZE = 100
DEFAULT_MEMORY_SELECTION_PAGE_SIZE = 100
DEFAULT_AUXILIARY_ATTEMPT_PAGE_SIZE = 50
MAX_REPOSITORY_PAGE_SIZE = 500


@dataclass(frozen=True)
class ContextPolicyReadResult:
    overrides: ConsoleContextPolicyOverrides
    revision: int | None = None
    error: str | None = None


class ContextPolicyWriteStatus(str, Enum):
    """Outcome of a context-policy owned-revision compare-and-set."""

    WRITTEN = "written"
    CONFLICT = "conflict"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ContextPolicyWriteResult:
    """Result of writing one complete sparse context-policy snapshot."""

    status: ContextPolicyWriteStatus
    revision: int | None


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


@dataclass(frozen=True, slots=True)
class ConsoleMemoryScopeRecord:
    """Validated coverage metadata for one generated memory."""

    memory_id: str
    conversation_id: str
    coverage_kind: MemoryCoverageKind
    origin_kind: MemoryOriginKind
    selection_anchor_message_id: str | None = None

    def __post_init__(self) -> None:
        _validate_bounded_text("memory_id", self.memory_id, 200)
        _validate_bounded_text("conversation_id", self.conversation_id, 200)
        if not isinstance(self.coverage_kind, MemoryCoverageKind):
            raise ValueError("coverage_kind must be a MemoryCoverageKind")
        if not isinstance(self.origin_kind, MemoryOriginKind):
            raise ValueError("origin_kind must be a MemoryOriginKind")
        if self.origin_kind is MemoryOriginKind.AUTOMATIC:
            if (
                self.coverage_kind is not MemoryCoverageKind.PREFIX
                or self.selection_anchor_message_id is not None
            ):
                raise ValueError(
                    "automatic memory must use prefix coverage without an anchor"
                )
            return
        if self.selection_anchor_message_id is None:
            raise ValueError("manual memory requires a selection anchor")
        _validate_bounded_text(
            "selection_anchor_message_id", self.selection_anchor_message_id, 200
        )


@dataclass(frozen=True, slots=True)
class ConsoleMemorySelectionRecord:
    """One append-mostly branch memory selection event."""

    sequence: int
    selection_id: str
    conversation_id: str
    activation_message_id: str
    selected_memory_id: str | None
    event_kind: MemorySelectionKind
    suppresses_legacy: bool
    created_at: str
    revision: int = 1
    active: bool = True

    def __post_init__(self) -> None:
        for name in (
            "selection_id",
            "conversation_id",
            "activation_message_id",
        ):
            _validate_bounded_text(name, getattr(self, name), 200)
        _validate_bounded_text("created_at", self.created_at, 80)
        for name in ("sequence", "revision"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.event_kind, MemorySelectionKind):
            raise ValueError("event_kind must be a MemorySelectionKind")
        if type(self.suppresses_legacy) is not bool:
            raise ValueError("suppresses_legacy must be a boolean")
        if type(self.active) is not bool:
            raise ValueError("active must be a boolean")
        if self.event_kind is MemorySelectionKind.SELECT:
            if self.selected_memory_id is None:
                raise ValueError("select event requires a selected memory")
            _validate_bounded_text(
                "selected_memory_id", self.selected_memory_id, 200
            )
        else:
            if self.selected_memory_id is not None:
                raise ValueError("reset event cannot carry a selected memory")
            if not self.suppresses_legacy:
                raise ValueError("reset event must suppress legacy")


@dataclass(frozen=True, slots=True)
class ApplicableBranchMemoryState:
    """Newest active selection applicable to one lineage and its exact rows."""

    selection: ConsoleMemorySelectionRecord | None
    memory: ConsoleMemoryRecord | None = field(default=None, repr=False)
    scope: ConsoleMemoryScopeRecord | None = None


@dataclass(frozen=True, slots=True)
class MemorySelectionFence:
    """Exact effective/head state admitted before an atomic mutation."""

    effective_kind: str
    legacy_boundary_message_id: str | None
    legacy_summary_digest: str | None
    selection_sequence: int | None
    selection_id: str | None
    selection_revision: int | None
    memory_id: str | None
    memory_revision: int | None

    def __post_init__(self) -> None:
        _validate_bounded_text("effective_kind", self.effective_kind, 80)
        _validate_optional_pair(
            "legacy boundary",
            self.legacy_boundary_message_id,
            self.legacy_summary_digest,
        )
        _validate_optional_selection_identity(
            self.selection_sequence,
            self.selection_id,
            self.selection_revision,
        )
        _validate_optional_revision_pair(
            "memory", self.memory_id, self.memory_revision
        )


@dataclass(frozen=True, slots=True)
class PersistedLineageFenceRow:
    """Content-free exact durable message facts used by repository CAS."""

    message_id: str
    parent_message_id: str | None
    version: int
    deleted: bool
    content_digest: str
    selected_variant_id: str | None
    selected_variant_index: int | None
    attachment_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_bounded_text("message_id", self.message_id, 200)
        if self.parent_message_id is not None:
            _validate_bounded_text(
                "parent_message_id", self.parent_message_id, 200
            )
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("version must be a positive integer")
        if type(self.deleted) is not bool:
            raise ValueError("deleted must be a boolean")
        _validate_bounded_text("content_digest", self.content_digest, 256)
        if (self.selected_variant_id is None) != (
            self.selected_variant_index is None
        ):
            raise ValueError(
                "selected variant id and index must both be present or absent"
            )
        if self.selected_variant_id is not None:
            _validate_bounded_text(
                "selected_variant_id", self.selected_variant_id, 200
            )
            if (
                type(self.selected_variant_index) is not int
                or self.selected_variant_index < 0
            ):
                raise ValueError(
                    "selected_variant_index must be a non-negative integer"
                )
        if not isinstance(self.attachment_digests, tuple):
            raise ValueError("attachment_digests must be a tuple")
        for digest in self.attachment_digests:
            _validate_bounded_text("attachment digest", digest, 256)


@dataclass(frozen=True, slots=True)
class BranchMemoryCommit:
    """Immutable input for one memory/scope/selection compare-and-swap."""

    memory: ConsoleMemoryRecord
    scope: ConsoleMemoryScopeRecord
    selection: ConsoleMemorySelectionRecord
    expected_effective: MemorySelectionFence
    expected_branch_head: MemorySelectionFence
    expected_cursor: tuple[str, str | None]
    durable_lineage: tuple[PersistedLineageFenceRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.memory, ConsoleMemoryRecord):
            raise TypeError("memory must be ConsoleMemoryRecord")
        if not isinstance(self.scope, ConsoleMemoryScopeRecord):
            raise TypeError("scope must be ConsoleMemoryScopeRecord")
        if not isinstance(self.selection, ConsoleMemorySelectionRecord):
            raise TypeError("selection must be ConsoleMemorySelectionRecord")
        if not isinstance(self.expected_effective, MemorySelectionFence):
            raise TypeError("expected_effective must be MemorySelectionFence")
        if not isinstance(self.expected_branch_head, MemorySelectionFence):
            raise TypeError("expected_branch_head must be MemorySelectionFence")
        if (
            not isinstance(self.expected_cursor, tuple)
            or len(self.expected_cursor) != 2
        ):
            raise ValueError("expected_cursor must be a two-item tuple")
        _validate_bounded_text(
            "expected active leaf", self.expected_cursor[0], 200
        )
        if self.expected_cursor[1] is not None:
            _validate_bounded_text(
                "expected before message", self.expected_cursor[1], 200
            )
        if not isinstance(self.durable_lineage, tuple) or not self.durable_lineage:
            raise ValueError("durable_lineage must be a non-empty tuple")
        if any(
            not isinstance(row, PersistedLineageFenceRow)
            for row in self.durable_lineage
        ):
            raise TypeError(
                "durable_lineage must contain PersistedLineageFenceRow values"
            )


@dataclass(frozen=True, slots=True)
class _PersistedLineageState:
    fence: PersistedLineageFenceRow
    role: str
    content: str = field(repr=False)


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

    def save_policy_if_revision(
        self,
        conversation_id: str,
        overrides: ConsoleContextPolicyOverrides,
        *,
        expected_revision: int | None,
    ) -> ContextPolicyWriteResult:
        """Write one complete policy only while its owned revision matches.

        ``None`` represents an absent policy row. An empty policy therefore
        performs a revision-guarded delete and publishes ``None`` as the new
        durable revision.
        """
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be a non-empty string")
        if not isinstance(overrides, ConsoleContextPolicyOverrides):
            raise TypeError("overrides must be ConsoleContextPolicyOverrides")
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 1
        ):
            raise ValueError("expected_revision must be positive or None")

        values = overrides.to_dict()
        with self.db.transaction(immediate=True) as cursor:
            conversation = cursor.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND deleted = 0",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                return ContextPolicyWriteResult(
                    ContextPolicyWriteStatus.MISSING,
                    None,
                )
            row = cursor.execute(
                "SELECT policy_revision FROM console_conversation_context_policy "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            current_revision = _positive_revision(
                row["policy_revision"] if row is not None else None
            )
            if current_revision != expected_revision:
                return ContextPolicyWriteResult(
                    ContextPolicyWriteStatus.CONFLICT,
                    current_revision,
                )
            if overrides.is_empty:
                if current_revision is not None:
                    cursor.execute(
                        "DELETE FROM console_conversation_context_policy "
                        "WHERE conversation_id = ? AND policy_revision = ?",
                        (conversation_id, current_revision),
                    )
                    if cursor.rowcount != 1:
                        latest = cursor.execute(
                            "SELECT policy_revision FROM "
                            "console_conversation_context_policy "
                            "WHERE conversation_id = ?",
                            (conversation_id,),
                        ).fetchone()
                        return ContextPolicyWriteResult(
                            ContextPolicyWriteStatus.CONFLICT,
                            _positive_revision(
                                latest["policy_revision"]
                                if latest is not None
                                else None
                            ),
                        )
                return ContextPolicyWriteResult(
                    ContextPolicyWriteStatus.WRITTEN,
                    None,
                )

            policy_values = (
                values.get("budget_mode"),
                values.get("custom_budget_tokens"),
                values.get("compaction_mode"),
                values.get("compaction_representation"),
                values.get("trigger_ratio"),
                values.get("target_ratio"),
                values.get("summary_max_tokens"),
                values.get("failure_behavior"),
                values.get("carry_forward_mode"),
            )
            if current_revision is None:
                cursor.execute(
                    """
                    INSERT INTO console_conversation_context_policy(
                        conversation_id, budget_mode, custom_budget_tokens,
                        compaction_mode, compaction_representation,
                        trigger_ratio, target_ratio, summary_max_tokens,
                        failure_behavior, carry_forward_mode, policy_revision,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                    """,
                    (conversation_id, *policy_values),
                )
                return ContextPolicyWriteResult(
                    ContextPolicyWriteStatus.WRITTEN,
                    1,
                )

            new_revision = current_revision + 1
            cursor.execute(
                """
                UPDATE console_conversation_context_policy
                   SET budget_mode = ?, custom_budget_tokens = ?,
                       compaction_mode = ?, compaction_representation = ?,
                       trigger_ratio = ?, target_ratio = ?,
                       summary_max_tokens = ?, failure_behavior = ?,
                       carry_forward_mode = ?, policy_revision = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE conversation_id = ? AND policy_revision = ?
                """,
                (*policy_values, new_revision, conversation_id, current_revision),
            )
            if cursor.rowcount != 1:
                latest = cursor.execute(
                    "SELECT policy_revision FROM "
                    "console_conversation_context_policy "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                return ContextPolicyWriteResult(
                    ContextPolicyWriteStatus.CONFLICT,
                    _positive_revision(
                        latest["policy_revision"] if latest is not None else None
                    ),
                )
            return ContextPolicyWriteResult(
                ContextPolicyWriteStatus.WRITTEN,
                new_revision,
            )

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

    def commit_memory_selection_if_current(
        self, commit: BranchMemoryCommit
    ) -> bool:
        """Atomically append memory, scope, and selection if every fence matches.

        Args:
            commit: Immutable branch-memory commit with its expected cursor,
                lineage, active-memory, and selection fences.

        Returns:
            ``True`` when the complete commit is appended; ``False`` when any
            current durable fence no longer matches.

        Raises:
            TypeError: If ``commit`` is not a ``BranchMemoryCommit``.
            ValueError: If the commit contains inconsistent ownership or scope.
        """
        if not isinstance(commit, BranchMemoryCommit):
            raise TypeError("commit must be BranchMemoryCommit")
        _validate_branch_memory_commit_ownership(commit)
        with self.db.transaction(immediate=True) as cursor:
            persisted = _load_persisted_branch_state(
                cursor, commit.memory.conversation_id
            )
            if persisted is None:
                return False
            cursor_pair, conversation_row, lineage = persisted
            if cursor_pair != commit.expected_cursor:
                return False
            if tuple(state.fence for state in lineage) != commit.durable_lineage:
                return False
            positions = {
                state.fence.message_id: index
                for index, state in enumerate(lineage)
            }
            boundary_index = positions.get(commit.memory.boundary_message_id)
            if (
                boundary_index is None
                or _prefix_digest_from_persisted(lineage[: boundary_index + 1])
                != commit.memory.summarized_prefix_digest
            ):
                return False

            branch_head, branch_head_fence = _load_applicable_branch_head(
                cursor,
                commit.memory.conversation_id,
                frozenset(state.fence.message_id for state in lineage),
            )
            if branch_head_fence != commit.expected_branch_head:
                return False
            effective = _effective_memory_fence(
                cursor,
                commit.memory.conversation_id,
                conversation_row,
                lineage,
                branch_head,
                branch_head_fence,
            )
            if effective != commit.expected_effective:
                return False

            suppresses_legacy = (
                True
                if commit.scope.origin_kind is MemoryOriginKind.MANUAL_REWIND
                else (
                    branch_head.suppresses_legacy
                    if branch_head is not None
                    else False
                )
            )
            selection = ConsoleMemorySelectionRecord(
                sequence=commit.selection.sequence,
                selection_id=commit.selection.selection_id,
                conversation_id=commit.selection.conversation_id,
                activation_message_id=commit.selection.activation_message_id,
                selected_memory_id=commit.selection.selected_memory_id,
                event_kind=commit.selection.event_kind,
                suppresses_legacy=suppresses_legacy,
                created_at=commit.selection.created_at,
                revision=commit.selection.revision,
                active=commit.selection.active,
            )
            _insert_memory(cursor, commit.memory)
            _insert_memory_scope(cursor, commit.scope)
            _insert_memory_selection(cursor, selection)
        return True

    def append_current_branch_reset_if_current(
        self,
        reset: ConsoleMemorySelectionRecord,
        *,
        expected_effective: MemorySelectionFence,
        expected_branch_head: MemorySelectionFence,
        expected_cursor: tuple[str, str | None],
        durable_lineage: tuple[PersistedLineageFenceRow, ...],
    ) -> tuple[str, int] | None:
        """Append a current-leaf reset tombstone if every captured fence matches."""
        _validate_branch_reset_input(
            reset,
            expected_effective=expected_effective,
            expected_branch_head=expected_branch_head,
            expected_cursor=expected_cursor,
            durable_lineage=durable_lineage,
        )
        with self.db.transaction(immediate=True) as cursor:
            persisted = _load_persisted_branch_state(
                cursor, reset.conversation_id
            )
            if persisted is None:
                return None
            cursor_pair, conversation_row, lineage = persisted
            if cursor_pair != expected_cursor:
                return None
            if tuple(state.fence for state in lineage) != durable_lineage:
                return None
            branch_head, branch_head_fence = _load_applicable_branch_head(
                cursor,
                reset.conversation_id,
                frozenset(state.fence.message_id for state in lineage),
            )
            if branch_head_fence != expected_branch_head:
                return None
            effective = _effective_memory_fence(
                cursor,
                reset.conversation_id,
                conversation_row,
                lineage,
                branch_head,
                branch_head_fence,
            )
            if effective != expected_effective:
                return None
            _insert_memory_selection(cursor, reset)
        return reset.selection_id, reset.revision

    def undo_current_branch_reset_if_current(
        self,
        conversation_id: str,
        *,
        selection_id: str,
        expected_revision: int,
    ) -> bool:
        """Deactivate only an exact reset that remains this branch's head."""
        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_bounded_text("selection_id", selection_id, 200)
        if type(expected_revision) is not int or expected_revision <= 0:
            raise ValueError("expected_revision must be a positive integer")
        with self.db.transaction(immediate=True) as cursor:
            persisted = _load_persisted_branch_state(cursor, conversation_id)
            if persisted is None:
                return False
            _cursor_pair, _conversation_row, lineage = persisted
            branch_head, _branch_head_fence = _load_applicable_branch_head(
                cursor,
                conversation_id,
                frozenset(state.fence.message_id for state in lineage),
            )
            if (
                branch_head is None
                or branch_head.event_kind is not MemorySelectionKind.RESET
                or branch_head.selection_id != selection_id
                or branch_head.revision != expected_revision
            ):
                return False
            updated = cursor.execute(
                """
                UPDATE console_conversation_memory_selections
                   SET active = 0, revision = revision + 1
                 WHERE conversation_id = ? AND selection_id = ?
                   AND event_kind = 'reset' AND active = 1 AND revision = ?
                """,
                (conversation_id, selection_id, expected_revision),
            )
        return updated.rowcount == 1

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

    def insert_memory_scope(self, record: ConsoleMemoryScopeRecord) -> None:
        """Insert the immutable scope paired with one generated memory."""
        if not isinstance(record, ConsoleMemoryScopeRecord):
            raise TypeError("record must be ConsoleMemoryScopeRecord")
        with self.db.transaction() as cursor:
            _insert_memory_scope(cursor, record)

    def load_memory_scope(
        self, memory_id: str
    ) -> ConsoleMemoryScopeRecord | None:
        """Read one scope; corrupt derived metadata is ineligible."""
        _validate_bounded_text("memory_id", memory_id, 200)
        with self.db.transaction() as cursor:
            row = cursor.execute(
                "SELECT * FROM console_conversation_memory_scopes WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return _memory_scope_from_row(row)
        except (TypeError, ValueError):
            return None

    def insert_memory_selection(
        self, record: ConsoleMemorySelectionRecord
    ) -> ConsoleMemorySelectionRecord:
        """Append one event using the database-owned sequence."""
        if not isinstance(record, ConsoleMemorySelectionRecord):
            raise TypeError("record must be ConsoleMemorySelectionRecord")
        with self.db.transaction() as cursor:
            sequence = _insert_memory_selection(cursor, record)
        return ConsoleMemorySelectionRecord(
            sequence=sequence,
            selection_id=record.selection_id,
            conversation_id=record.conversation_id,
            activation_message_id=record.activation_message_id,
            selected_memory_id=record.selected_memory_id,
            event_kind=record.event_kind,
            suppresses_legacy=record.suppresses_legacy,
            created_at=record.created_at,
            revision=record.revision,
            active=record.active,
        )

    def list_active_memory_selections(
        self,
        conversation_id: str,
        *,
        limit: int = DEFAULT_MEMORY_SELECTION_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[ConsoleMemorySelectionRecord, ...]:
        """Return a bounded newest-sequence-first page of active events."""
        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_page(limit=limit, offset=offset)
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT sequence, selection_id, conversation_id,
                       activation_message_id, selected_memory_id, event_kind,
                       suppresses_legacy, created_at, revision, active
                  FROM console_conversation_memory_selections
                 WHERE conversation_id = ? AND active = 1
                 ORDER BY sequence DESC
                 LIMIT ? OFFSET ?
                """,
                (conversation_id, limit, offset),
            ).fetchall()
        records: list[ConsoleMemorySelectionRecord] = []
        for row in rows:
            try:
                records.append(_memory_selection_from_row(row))
            except (TypeError, ValueError):
                continue
        return tuple(records)

    def load_applicable_branch_memory(
        self,
        conversation_id: str,
        lineage_message_ids: frozenset[str],
    ) -> ApplicableBranchMemoryState:
        """Load the complete-stream branch head and its exact memory metadata.

        Args:
            conversation_id: Durable conversation whose branch state is read.
            lineage_message_ids: Persisted message IDs on the active lineage.

        Returns:
            The newest applicable active event plus its referenced generated
            memory and scope when both derived rows decode successfully.

        Raises:
            ValueError: If the conversation or lineage identities are invalid.
        """
        _validate_bounded_text("conversation_id", conversation_id, 200)
        if not isinstance(lineage_message_ids, frozenset):
            raise ValueError("lineage_message_ids must be a frozenset")
        for message_id in lineage_message_ids:
            _validate_bounded_text("lineage message id", message_id, 200)
        with self.db.transaction() as cursor:
            selection, _fence = _load_applicable_branch_head(
                cursor,
                conversation_id,
                lineage_message_ids,
            )
            if selection is None or selection.selected_memory_id is None:
                return ApplicableBranchMemoryState(selection=selection)
            memory_row = cursor.execute(
                """
                SELECT *
                  FROM console_conversation_memories
                 WHERE id = ? AND conversation_id = ?
                """,
                (selection.selected_memory_id, conversation_id),
            ).fetchone()
            scope_row = cursor.execute(
                """
                SELECT *
                  FROM console_conversation_memory_scopes
                 WHERE memory_id = ? AND conversation_id = ?
                """,
                (selection.selected_memory_id, conversation_id),
            ).fetchone()
        memory = None
        scope = None
        if memory_row is not None:
            try:
                memory = _memory_from_row(memory_row)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if scope_row is not None:
            try:
                scope = _memory_scope_from_row(scope_row)
            except (TypeError, ValueError):
                pass
        return ApplicableBranchMemoryState(
            selection=selection,
            memory=memory,
            scope=scope,
        )

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
        """Clear all durable conversation memory after separate confirmation."""
        _validate_bounded_text("conversation_id", conversation_id, 200)
        _validate_bounded_text("reset_at", reset_at, 80)
        with self.db.transaction(immediate=True) as cursor:
            cursor.execute(
                """
                UPDATE conversations
                   SET context_summary = NULL,
                       summary_boundary_message_id = NULL
                 WHERE id = ? AND deleted = 0
                """,
                (conversation_id,),
            )
            cursor.execute(
                """
                UPDATE console_conversation_memory_selections
                   SET active = 0, revision = revision + 1
                 WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            result = cursor.execute(
                """
                UPDATE console_conversation_memories
                   SET active = 0, reset_at = ?, revision = revision + 1
                 WHERE conversation_id = ?
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


def _validate_branch_memory_commit_ownership(commit: BranchMemoryCommit) -> None:
    memory = commit.memory
    scope = commit.scope
    selection = commit.selection
    if not (
        memory.conversation_id
        == scope.conversation_id
        == selection.conversation_id
    ):
        raise ValueError("memory, scope, and selection must share a conversation")
    if scope.memory_id != memory.memory_id:
        raise ValueError("scope must own the committed memory")
    if (
        selection.event_kind is not MemorySelectionKind.SELECT
        or selection.selected_memory_id != memory.memory_id
    ):
        raise ValueError("selection must select the committed memory")
    if memory.source_kind != "generated" or not memory.active or not selection.active:
        raise ValueError("new memory and selection must be active generated state")
    leaf_id = commit.expected_cursor[0]
    if (
        memory.captured_leaf_message_id != leaf_id
        or selection.activation_message_id != leaf_id
        or commit.durable_lineage[-1].message_id != leaf_id
    ):
        raise ValueError("memory and selection must use the captured durable leaf")
    lineage_ids = tuple(row.message_id for row in commit.durable_lineage)
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError("durable_lineage cannot contain duplicate messages")
    if commit.durable_lineage[0].parent_message_id is not None:
        raise ValueError("durable_lineage must start at a root message")
    for parent, child in zip(
        commit.durable_lineage[:-1], commit.durable_lineage[1:], strict=True
    ):
        if child.parent_message_id != parent.message_id:
            raise ValueError("durable_lineage must be one ordered parent chain")
    try:
        recorded_lineage = json.loads(memory.lineage_json)
    except json.JSONDecodeError as exc:  # pragma: no cover - record validates first
        raise ValueError("memory lineage_json must be valid JSON") from exc
    if recorded_lineage != list(lineage_ids):
        raise ValueError("memory lineage must match the durable fence")
    positions = {message_id: index for index, message_id in enumerate(lineage_ids)}
    boundary_index = positions.get(memory.boundary_message_id)
    if boundary_index is None:
        raise ValueError("memory boundary must lie on the durable lineage")
    anchor_id = scope.selection_anchor_message_id
    if scope.origin_kind is MemoryOriginKind.AUTOMATIC:
        return
    anchor_index = positions.get(anchor_id) if anchor_id is not None else None
    if anchor_index is None:
        raise ValueError("manual selection anchor must lie on the durable lineage")
    if scope.coverage_kind is MemoryCoverageKind.PREFIX:
        valid_order = boundary_index < anchor_index
    else:
        valid_order = anchor_index < boundary_index
    if not valid_order:
        raise ValueError("manual memory boundary and anchor order is invalid")


def _validate_branch_reset_input(
    reset: ConsoleMemorySelectionRecord,
    *,
    expected_effective: MemorySelectionFence,
    expected_branch_head: MemorySelectionFence,
    expected_cursor: tuple[str, str | None],
    durable_lineage: tuple[PersistedLineageFenceRow, ...],
) -> None:
    if not isinstance(reset, ConsoleMemorySelectionRecord):
        raise TypeError("reset must be ConsoleMemorySelectionRecord")
    if (
        reset.event_kind is not MemorySelectionKind.RESET
        or reset.selected_memory_id is not None
        or not reset.suppresses_legacy
        or not reset.active
    ):
        raise ValueError("reset must be an active suppressing reset event")
    if not isinstance(expected_effective, MemorySelectionFence):
        raise TypeError("expected_effective must be MemorySelectionFence")
    if expected_effective.effective_kind not in {
        "legacy_prefix",
        "generated_prefix",
        "generated_range",
    }:
        raise ValueError("current reset requires an effective memory")
    if not isinstance(expected_branch_head, MemorySelectionFence):
        raise TypeError("expected_branch_head must be MemorySelectionFence")
    if not isinstance(expected_cursor, tuple) or len(expected_cursor) != 2:
        raise ValueError("expected_cursor must be a two-item tuple")
    _validate_bounded_text("expected active leaf", expected_cursor[0], 200)
    if expected_cursor[1] is not None:
        _validate_bounded_text("expected before message", expected_cursor[1], 200)
    if not isinstance(durable_lineage, tuple) or not durable_lineage:
        raise ValueError("durable_lineage must be a non-empty tuple")
    if any(
        not isinstance(row, PersistedLineageFenceRow) for row in durable_lineage
    ):
        raise TypeError(
            "durable_lineage must contain PersistedLineageFenceRow values"
        )
    if (
        reset.activation_message_id != expected_cursor[0]
        or durable_lineage[-1].message_id != expected_cursor[0]
    ):
        raise ValueError("reset must use the captured durable leaf")


def _load_persisted_branch_state(
    cursor: Any,
    conversation_id: str,
) -> tuple[
    tuple[str, str | None],
    Mapping[str, Any],
    tuple[_PersistedLineageState, ...],
] | None:
    conversation = cursor.execute(
        """
        SELECT active_leaf_message_id, active_leaf_before_message_id,
               context_summary, summary_boundary_message_id
          FROM conversations
         WHERE id = ? AND deleted = 0
        """,
        (conversation_id,),
    ).fetchone()
    if conversation is None or conversation["active_leaf_message_id"] is None:
        return None
    leaf_id = str(conversation["active_leaf_message_id"])
    before_message_id = (
        None
        if conversation["active_leaf_before_message_id"] is None
        else str(conversation["active_leaf_before_message_id"])
    )
    reversed_lineage: list[_PersistedLineageState] = []
    seen: set[str] = set()
    message_id: str | None = leaf_id
    while message_id is not None:
        if message_id in seen:
            return None
        seen.add(message_id)
        row = cursor.execute(
            """
            SELECT id, conversation_id, parent_message_id, sender, role,
                   content, image_data, image_mime_type, version, deleted,
                   variant_of, variant_number, is_selected_variant,
                   total_variants
              FROM messages
             WHERE id = ? AND conversation_id = ?
            """,
            (message_id, conversation_id),
        ).fetchone()
        if row is None:
            return None
        state = _persisted_lineage_state(cursor, row)
        if state is None:
            return None
        reversed_lineage.append(state)
        message_id = (
            None
            if row["parent_message_id"] is None
            else str(row["parent_message_id"])
        )
    return (
        (leaf_id, before_message_id),
        conversation,
        tuple(reversed(reversed_lineage)),
    )


def _persisted_lineage_state(
    cursor: Any, row: Mapping[str, Any]
) -> _PersistedLineageState | None:
    message_id = str(row["id"])
    root_variant_id = (
        message_id if row["variant_of"] is None else str(row["variant_of"])
    )
    variants = cursor.execute(
        """
        SELECT id, role, sender, content, image_data, image_mime_type,
               variant_number, is_selected_variant, deleted
          FROM messages
         WHERE (id = ? OR variant_of = ?)
         ORDER BY variant_number, id
        """,
        (root_variant_id, root_variant_id),
    ).fetchall()
    live_variants = [variant for variant in variants if not bool(variant["deleted"])]
    selected_variant_id: str | None = None
    selected_variant_index: int | None = None
    content_row = row
    attachment_owner_id = message_id
    if len(live_variants) > 1:
        selected = [
            (index, variant)
            for index, variant in enumerate(live_variants)
            if int(variant["is_selected_variant"] or 0) == 1
        ]
        if len(selected) != 1:
            return None
        selected_variant_index, content_row = selected[0]
        selected_variant_id = str(content_row["id"])
        attachment_owner_id = selected_variant_id
    role = str(content_row["role"] or content_row["sender"])
    content = str(content_row["content"])
    attachment_digests = _persisted_attachment_digests(
        cursor,
        attachment_owner_id,
        image_data=content_row["image_data"],
        image_mime_type=content_row["image_mime_type"],
    )
    return _PersistedLineageState(
        fence=PersistedLineageFenceRow(
            message_id=message_id,
            parent_message_id=(
                None
                if row["parent_message_id"] is None
                else str(row["parent_message_id"])
            ),
            version=int(row["version"]),
            deleted=bool(row["deleted"]),
            content_digest=_digest_json(content),
            selected_variant_id=selected_variant_id,
            selected_variant_index=selected_variant_index,
            attachment_digests=attachment_digests,
        ),
        role=role,
        content=content,
    )


def _persisted_attachment_digests(
    cursor: Any,
    message_id: str,
    *,
    image_data: object,
    image_mime_type: object,
) -> tuple[str, ...]:
    digests: list[str] = []
    if image_data is not None:
        digests.append(
            persisted_attachment_digest(
                position=0,
                mime_type=str(image_mime_type or ""),
                display_name="",
                data=bytes(image_data),
            )
        )
    rows = cursor.execute(
        """
        SELECT position, data, mime_type, display_name
          FROM message_attachments
         WHERE message_id = ?
         ORDER BY position
        """,
        (message_id,),
    ).fetchall()
    for attachment in rows:
        digests.append(
            persisted_attachment_digest(
                position=int(attachment["position"]),
                mime_type=str(attachment["mime_type"]),
                display_name=str(attachment["display_name"] or ""),
                data=bytes(attachment["data"]),
            )
        )
    return tuple(digests)


def persisted_attachment_digest(
    *, position: int, mime_type: str, display_name: str, data: bytes
) -> str:
    """Digest attachment facts that survive durable Console persistence.

    Position zero is stored only as scalar image bytes and MIME, so its runtime
    display label is deliberately excluded. Positions one and above retain the
    label stored in ``message_attachments``.
    """
    data_digest = hashlib.sha256(data).hexdigest()
    durable_display_name = "" if position == 0 else display_name
    return hashlib.sha256(
        f"{position}\0{mime_type}\0{durable_display_name}\0{data_digest}".encode(
            "utf-8"
        )
    ).hexdigest()


def _load_applicable_branch_head(
    cursor: Any,
    conversation_id: str,
    lineage_ids: frozenset[str],
) -> tuple[ConsoleMemorySelectionRecord | None, MemorySelectionFence]:
    rows = cursor.execute(
        """
        SELECT sequence, selection_id, conversation_id,
               activation_message_id, selected_memory_id, event_kind,
               suppresses_legacy, created_at, revision, active
          FROM console_conversation_memory_selections
         WHERE conversation_id = ? AND active = 1
         ORDER BY sequence DESC
        """,
        (conversation_id,),
    ).fetchall()
    for row in rows:
        try:
            selection = _memory_selection_from_row(row)
        except (TypeError, ValueError):
            continue
        if selection.activation_message_id not in lineage_ids:
            continue
        memory_revision = None
        if selection.selected_memory_id is not None:
            memory_row = cursor.execute(
                """
                SELECT revision
                  FROM console_conversation_memories
                 WHERE id = ? AND conversation_id = ?
                """,
                (selection.selected_memory_id, conversation_id),
            ).fetchone()
            if memory_row is not None:
                memory_revision = int(memory_row["revision"])
        return selection, MemorySelectionFence(
            effective_kind=selection.event_kind.value,
            legacy_boundary_message_id=None,
            legacy_summary_digest=None,
            selection_sequence=selection.sequence,
            selection_id=selection.selection_id,
            selection_revision=selection.revision,
            memory_id=selection.selected_memory_id,
            memory_revision=memory_revision,
        )
    return None, _empty_memory_fence("no_head")


def _effective_memory_fence(
    cursor: Any,
    conversation_id: str,
    conversation: Mapping[str, Any],
    lineage: tuple[_PersistedLineageState, ...],
    branch_head: ConsoleMemorySelectionRecord | None,
    branch_head_fence: MemorySelectionFence,
) -> MemorySelectionFence:
    positions = {
        state.fence.message_id: index for index, state in enumerate(lineage)
    }
    legacy_summary = conversation["context_summary"]
    legacy_boundary = conversation["summary_boundary_message_id"]
    valid_legacy = (
        isinstance(legacy_summary, str)
        and bool(legacy_summary.strip())
        and isinstance(legacy_boundary, str)
        and legacy_boundary in positions
    )
    if valid_legacy and (
        branch_head is None or not branch_head.suppresses_legacy
    ):
        return MemorySelectionFence(
            effective_kind="legacy_prefix",
            legacy_boundary_message_id=legacy_boundary,
            legacy_summary_digest=_digest_json(legacy_summary),
            selection_sequence=None,
            selection_id=None,
            selection_revision=None,
            memory_id=None,
            memory_revision=None,
        )
    if branch_head is None or branch_head.event_kind is MemorySelectionKind.RESET:
        return _empty_memory_fence("raw")

    memory = cursor.execute(
        """
        SELECT memory.id, memory.conversation_id, memory.boundary_message_id,
               memory.captured_leaf_message_id,
               memory.summarized_prefix_digest, memory.revision,
               memory.active, memory.source_kind,
               scope.coverage_kind, scope.origin_kind,
               scope.selection_anchor_message_id
          FROM console_conversation_memories AS memory
          LEFT JOIN console_conversation_memory_scopes AS scope
            ON scope.memory_id = memory.id
           AND scope.conversation_id = memory.conversation_id
         WHERE memory.id = ? AND memory.conversation_id = ?
        """,
        (branch_head.selected_memory_id, conversation_id),
    ).fetchone()
    if memory is None or not _persisted_memory_is_valid(
        memory, branch_head, lineage, positions
    ):
        return _empty_memory_fence("raw")
    kind = (
        "generated_prefix"
        if memory["coverage_kind"] == MemoryCoverageKind.PREFIX.value
        else "generated_range"
    )
    return MemorySelectionFence(
        effective_kind=kind,
        legacy_boundary_message_id=None,
        legacy_summary_digest=None,
        selection_sequence=branch_head_fence.selection_sequence,
        selection_id=branch_head_fence.selection_id,
        selection_revision=branch_head_fence.selection_revision,
        memory_id=branch_head_fence.memory_id,
        memory_revision=int(memory["revision"]),
    )


def _persisted_memory_is_valid(
    memory: Mapping[str, Any],
    branch_head: ConsoleMemorySelectionRecord,
    lineage: tuple[_PersistedLineageState, ...],
    positions: dict[str, int],
) -> bool:
    if (
        int(memory["active"]) != 1
        or memory["source_kind"] != "generated"
        or memory["coverage_kind"] not in {
            MemoryCoverageKind.PREFIX.value,
            MemoryCoverageKind.RANGE.value,
        }
        or memory["origin_kind"] not in {
            MemoryOriginKind.AUTOMATIC.value,
            MemoryOriginKind.MANUAL_REWIND.value,
        }
        or memory["captured_leaf_message_id"]
        != branch_head.activation_message_id
    ):
        return False
    boundary_index = positions.get(str(memory["boundary_message_id"]))
    if boundary_index is None:
        return False
    if (
        _prefix_digest_from_persisted(lineage[: boundary_index + 1])
        != memory["summarized_prefix_digest"]
    ):
        return False
    if memory["origin_kind"] == MemoryOriginKind.AUTOMATIC.value:
        return (
            memory["coverage_kind"] == MemoryCoverageKind.PREFIX.value
            and memory["selection_anchor_message_id"] is None
        )
    if not branch_head.suppresses_legacy:
        return False
    anchor_id = memory["selection_anchor_message_id"]
    anchor_index = positions.get(str(anchor_id)) if anchor_id is not None else None
    if anchor_index is None or lineage[anchor_index].role != "user":
        return False
    if memory["coverage_kind"] == MemoryCoverageKind.RANGE.value:
        return anchor_index < boundary_index
    return boundary_index < anchor_index


def _prefix_digest_from_persisted(
    lineage: tuple[_PersistedLineageState, ...],
) -> str:
    return _digest_json(
        [
            {
                "message_id": state.fence.message_id,
                "version": state.fence.version,
                "role": state.role,
                "content": state.content,
                "selected_variant_id": state.fence.selected_variant_id,
                "selected_variant_index": state.fence.selected_variant_index,
                "attachment_digests": list(state.fence.attachment_digests),
            }
            for state in lineage
        ]
    )


def _empty_memory_fence(kind: str) -> MemorySelectionFence:
    return MemorySelectionFence(
        effective_kind=kind,
        legacy_boundary_message_id=None,
        legacy_summary_digest=None,
        selection_sequence=None,
        selection_id=None,
        selection_revision=None,
        memory_id=None,
        memory_revision=None,
    )


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


def _memory_scope_from_row(row: Mapping[str, Any]) -> ConsoleMemoryScopeRecord:
    return ConsoleMemoryScopeRecord(
        memory_id=str(row["memory_id"]),
        conversation_id=str(row["conversation_id"]),
        coverage_kind=MemoryCoverageKind(row["coverage_kind"]),
        origin_kind=MemoryOriginKind(row["origin_kind"]),
        selection_anchor_message_id=(
            None
            if row["selection_anchor_message_id"] is None
            else str(row["selection_anchor_message_id"])
        ),
    )


def _memory_selection_from_row(
    row: Mapping[str, Any],
) -> ConsoleMemorySelectionRecord:
    suppresses_legacy = row["suppresses_legacy"]
    active = row["active"]
    if type(suppresses_legacy) is not int or suppresses_legacy not in (0, 1):
        raise ValueError("invalid persisted suppresses_legacy")
    if type(active) is not int or active not in (0, 1):
        raise ValueError("invalid persisted active")
    return ConsoleMemorySelectionRecord(
        sequence=int(row["sequence"]),
        selection_id=str(row["selection_id"]),
        conversation_id=str(row["conversation_id"]),
        activation_message_id=str(row["activation_message_id"]),
        selected_memory_id=(
            None
            if row["selected_memory_id"] is None
            else str(row["selected_memory_id"])
        ),
        event_kind=MemorySelectionKind(row["event_kind"]),
        suppresses_legacy=bool(suppresses_legacy),
        created_at=str(row["created_at"]),
        revision=int(row["revision"]),
        active=bool(active),
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


def _insert_memory_scope(cursor: Any, record: ConsoleMemoryScopeRecord) -> None:
    cursor.execute(
        """
        INSERT INTO console_conversation_memory_scopes(
            memory_id, conversation_id, coverage_kind, origin_kind,
            selection_anchor_message_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            record.memory_id,
            record.conversation_id,
            record.coverage_kind.value,
            record.origin_kind.value,
            record.selection_anchor_message_id,
        ),
    )


def _insert_memory_selection(
    cursor: Any, record: ConsoleMemorySelectionRecord
) -> int:
    result = cursor.execute(
        """
        INSERT INTO console_conversation_memory_selections(
            selection_id, conversation_id, activation_message_id,
            selected_memory_id, event_kind, suppresses_legacy,
            created_at, revision, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.selection_id,
            record.conversation_id,
            record.activation_message_id,
            record.selected_memory_id,
            record.event_kind.value,
            int(record.suppresses_legacy),
            record.created_at,
            record.revision,
            int(record.active),
        ),
    )
    return int(result.lastrowid)


def _digest_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_optional_pair(
    name: str, first: str | None, second: str | None
) -> None:
    if (first is None) != (second is None):
        raise ValueError(f"{name} fields must both be present or absent")
    if first is not None:
        _validate_bounded_text(f"{name} id", first, 200)
        _validate_bounded_text(f"{name} digest", second, 256)


def _validate_optional_selection_identity(
    sequence: int | None,
    selection_id: str | None,
    revision: int | None,
) -> None:
    present = (sequence is not None, selection_id is not None, revision is not None)
    if any(present) and not all(present):
        raise ValueError("selection identity fields must all be present or absent")
    if sequence is not None and (type(sequence) is not int or sequence <= 0):
        raise ValueError("selection_sequence must be a positive integer")
    if selection_id is not None:
        _validate_bounded_text("selection_id", selection_id, 200)
    if revision is not None and (type(revision) is not int or revision <= 0):
        raise ValueError("selection_revision must be a positive integer")


def _validate_optional_revision_pair(
    name: str, identity: str | None, revision: int | None
) -> None:
    if (identity is None) != (revision is None):
        raise ValueError(f"{name} id and revision must both be present or absent")
    if identity is not None:
        _validate_bounded_text(f"{name}_id", identity, 200)
    if revision is not None and (type(revision) is not int or revision <= 0):
        raise ValueError(f"{name}_revision must be a positive integer")


def _validate_bounded_text(name: str, value: object, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{name} must be a non-empty string up to {maximum} characters"
        )
