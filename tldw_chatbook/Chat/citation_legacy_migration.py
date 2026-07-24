"""Bounded, restart-safe migration for legacy citation sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import errno
from enum import Enum
import hmac
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from tldw_chatbook.Chat.citation_evidence_models import CitationRef
from tldw_chatbook.Chat.citation_trace_adapters import (
    synthesize_legacy_citation_refs,
    synthesize_legacy_payloads,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintDomain,
)
from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttemptPayload,
    CitationTrace,
    SealedCitationWrite,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)


LEGACY_SIDECAR_BYTES_MAX = 32 * 1024 * 1024
LEGACY_MIGRATION_BATCH_SIZE = 100
LEGACY_JSON_DEPTH_MAX = 32
LEGACY_JSON_NODES_MAX = 20_001
LEGACY_MAPPING_ITEMS_MAX = 5_000
LEGACY_SEQUENCE_ITEMS_MAX = 5_000
LEGACY_FIELD_UTF8_BYTES_MAX = 64 * 1024
LEGACY_KEY_UTF8_BYTES_MAX = 256
LEGACY_REASON_CODE_MAX = 256
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class LegacyMigrationState(str, Enum):
    """Durable per-conversation migration states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    DIVERGED = "diverged"


class LegacyCitationReadState(str, Enum):
    """Bounded canonical/legacy read decisions."""

    LEGACY_FALLBACK = "legacy_fallback"
    VERIFICATION_PENDING = "verification_pending"
    CANONICAL = "canonical"
    DIVERGED = "diverged"


@dataclass(frozen=True, slots=True)
class LegacyMigrationJournal:
    """Normalized migration progress for one conversation."""

    conversation_id: str
    source_fingerprint: str
    state: LegacyMigrationState
    attempt_count: int
    started_at: str
    updated_at: str
    next_message_cursor: str | None
    reason_code: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class LegacyMigrationBatchResult:
    """One bounded idle-unit migration result."""

    state: LegacyMigrationState
    processed_messages: int = 0
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class LegacyCitationConversationRead:
    """One non-merged citation read for a conversation."""

    state: LegacyCitationReadState
    records: Mapping[str, Mapping[str, Any]]


class _LegacyInputError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code[:LEGACY_REASON_CODE_MAX]
        super().__init__(self.reason_code)


class _MigrationClaimLost(RuntimeError):
    """Abort a stale migration transaction before it can stage any rows."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_json_lexical_depth(value: str) -> None:
    """Reject hostile nesting before handing text to the recursive JSON parser."""

    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > LEGACY_JSON_DEPTH_MAX:
                raise _LegacyInputError("legacy_json_too_deep")
        elif character in "]}":
            depth -= 1


def _validate_json_bounds(value: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > LEGACY_JSON_NODES_MAX:
            raise _LegacyInputError("legacy_json_too_many_nodes")
        if depth > LEGACY_JSON_DEPTH_MAX:
            raise _LegacyInputError("legacy_json_too_deep")
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            if len(current.encode("utf-8")) > LEGACY_FIELD_UTF8_BYTES_MAX:
                raise _LegacyInputError("legacy_field_too_large")
            continue
        if isinstance(current, int):
            if not -(2**63) <= current < 2**63:
                raise _LegacyInputError("legacy_integer_out_of_range")
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise _LegacyInputError("legacy_number_not_finite")
            continue
        if isinstance(current, Mapping):
            if len(current) > LEGACY_MAPPING_ITEMS_MAX:
                raise _LegacyInputError("legacy_mapping_too_large")
            for key, nested in current.items():
                if not isinstance(key, str):
                    raise _LegacyInputError("legacy_key_invalid")
                if len(key.encode("utf-8")) > LEGACY_KEY_UTF8_BYTES_MAX:
                    raise _LegacyInputError("legacy_key_too_large")
                stack.append((nested, depth + 1))
            continue
        if isinstance(current, (list, tuple)):
            if len(current) > LEGACY_SEQUENCE_ITEMS_MAX:
                raise _LegacyInputError("legacy_sequence_too_large")
            stack.extend((nested, depth + 1) for nested in current)
            continue
        raise _LegacyInputError("legacy_value_type_invalid")


def _safe_legacy_identifier(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value)
    if not text or len(text.encode("utf-8")) > 256:
        return None
    return text


def _legacy_payload_parts(record: Mapping[str, Any]) -> tuple[Any, Any]:
    nested: Mapping[str, Any] = {}
    for key in ("rag_context", "chat_rag_context"):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            nested = candidate
            break
    bundle = record.get("evidence_bundle", nested.get("evidence_bundle"))
    citations = record.get("citations", nested.get("citations", ()))
    return bundle, citations


def _legacy_reference_shape_valid(reference: Mapping[str, Any]) -> bool:
    if any(
        _safe_legacy_identifier(reference.get(key)) is None
        for key in ("evidence_id", "source_id")
    ):
        return False
    if any(
        not isinstance(reference.get(key), str) or not reference.get(key)
        for key in ("source_type", "title", "authority_label")
    ):
        return False
    score = reference.get("score")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float))
    ):
        return False
    original_count = reference.get("original_snippet_char_count")
    if original_count is not None and (
        isinstance(original_count, bool) or not isinstance(original_count, int)
    ):
        return False
    truncated = reference.get("snippet_truncated")
    return truncated is None or isinstance(truncated, bool)


def _stable_id(
    fingerprint: Callable[..., str],
    conversation_id: str,
    message_id: str,
    label: str,
    ordinal: int = 0,
) -> str:
    digest = fingerprint(
        CitationFingerprintDomain.LEGACY_SOURCE,
        conversation_id,
        message_id,
        label,
        str(ordinal),
    )
    return f"legacy-{label}_{digest.rsplit(':', 1)[-1][:32]}"


def _remap_legacy_write(
    write: SealedCitationWrite,
    *,
    conversation_id: str,
    message_id: str,
    answer_body: str,
    fingerprint: Callable[..., str],
) -> SealedCitationWrite:
    run = write.trace.evidence_runs[0]
    prompt = write.trace.prompt_evidence_sets[0]
    attempt = write.trace.answer_attempts[0]
    run_id = _stable_id(fingerprint, conversation_id, message_id, "run")
    run_payload_id = _stable_id(fingerprint, conversation_id, message_id, "run-payload")
    prompt_id = _stable_id(fingerprint, conversation_id, message_id, "prompt")
    attempt_id = _stable_id(fingerprint, conversation_id, message_id, "attempt")
    answer_payload_id = _stable_id(
        fingerprint, conversation_id, message_id, "answer-payload"
    )
    snapshot_ids = {
        payload.payload_id: _stable_id(
            fingerprint,
            conversation_id,
            message_id,
            "snapshot",
            ordinal,
        )
        for ordinal, payload in enumerate(
            write.evidence_snapshot_payloads,
            start=1,
        )
    }
    remapped_prompt = prompt.model_copy(
        update={
            "prompt_set_id": prompt_id,
            "entries": tuple(
                entry.model_copy(
                    update={
                        "run_id": run_id,
                        "snapshot_payload_ref": snapshot_ids[
                            entry.snapshot_payload_ref
                        ],
                    }
                )
                for entry in prompt.entries
            ),
        }
    )
    remapped_attempt = attempt.model_copy(
        update={
            "attempt_id": attempt_id,
            "prompt_evidence_set_id": prompt_id,
            "answer_payload_ref": answer_payload_id,
            "occurrences": tuple(
                occurrence.model_copy(
                    update={
                        "occurrence_id": _stable_id(
                            fingerprint,
                            conversation_id,
                            message_id,
                            "occurrence",
                            ordinal,
                        )
                    }
                )
                for ordinal, occurrence in enumerate(
                    attempt.occurrences,
                    start=1,
                )
            ),
        }
    )
    remapped_trace = write.trace.model_copy(
        update={
            "trace_id": _stable_id(
                fingerprint,
                conversation_id,
                message_id,
                "trace",
            ),
            "request_id": _stable_id(
                fingerprint,
                conversation_id,
                message_id,
                "request",
            ),
            "generation_id": _stable_id(
                fingerprint,
                conversation_id,
                message_id,
                "generation",
            ),
            "evidence_runs": (
                run.model_copy(
                    update={
                        "run_id": run_id,
                        "request_id": _stable_id(
                            fingerprint,
                            conversation_id,
                            message_id,
                            "request",
                        ),
                        "payload_ref": run_payload_id,
                    }
                ),
            ),
            "prompt_evidence_sets": (remapped_prompt,),
            "answer_attempts": (remapped_attempt,),
            "selected_attempt_id": attempt_id,
        }
    )
    run_payload = write.evidence_run_payloads[0]
    remapped_run_payload = run_payload.model_copy(
        update={
            "payload_id": run_payload_id,
            "run_id": run_id,
            "candidates": tuple(
                candidate.model_copy(
                    update={
                        "candidate_id": _stable_id(
                            fingerprint,
                            conversation_id,
                            message_id,
                            "candidate",
                            ordinal,
                        )
                    }
                )
                for ordinal, candidate in enumerate(
                    run_payload.candidates,
                    start=1,
                )
            ),
        }
    )
    body_fingerprint = fingerprint(
        CitationFingerprintDomain.MESSAGE_BODY,
        answer_body,
    )
    return SealedCitationWrite(
        trace=remapped_trace,
        evidence_run_payloads=(remapped_run_payload,),
        evidence_snapshot_payloads=tuple(
            payload.model_copy(update={"payload_id": snapshot_ids[payload.payload_id]})
            for payload in write.evidence_snapshot_payloads
        ),
        answer_attempt_payloads=(
            AnswerAttemptPayload(
                payload_id=answer_payload_id,
                attempt_id=attempt_id,
                answer_body=answer_body,
                body_integrity_hmac=body_fingerprint,
            ),
        ),
    )


def synthesize_legacy_message(
    record: Mapping[str, Any],
    *,
    conversation_id: str,
    message_id: str,
    answer_body: str,
    created_at: datetime,
    fingerprint_codec: CitationFingerprintCodec | None = None,
    fingerprint: Callable[..., str] | None = None,
) -> SealedCitationWrite:
    """Adapt one bounded legacy record without granting locator authority."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise TypeError("created_at must be timezone-aware")
    if (fingerprint_codec is None) == (fingerprint is None):
        raise TypeError("provide exactly one fingerprint implementation")
    fingerprint_fn = (
        fingerprint_codec.fingerprint if fingerprint_codec is not None else fingerprint
    )
    assert fingerprint_fn is not None
    _validate_json_bounds(record)
    bundle, raw_citations = _legacy_payload_parts(record)
    citations: list[Mapping[str, Any]] = []
    valid_shape = isinstance(raw_citations, (list, tuple))
    if valid_shape:
        for item in raw_citations:
            if not isinstance(item, Mapping):
                valid_shape = False
                break
            evidence_id = _safe_legacy_identifier(item.get("evidence_id"))
            source_id = _safe_legacy_identifier(item.get("source_id"))
            if evidence_id is None or source_id is None:
                valid_shape = False
                break
            citations.append(
                {**item, "evidence_id": evidence_id, "source_id": source_id}
            )
    try:
        if isinstance(bundle, Mapping) and valid_shape:
            references = bundle.get("references")
            if not isinstance(references, list) or any(
                not isinstance(item, Mapping) or not _legacy_reference_shape_valid(item)
                for item in references
            ):
                raise ValueError("invalid legacy evidence bundle")
            base = synthesize_legacy_payloads(
                bundle,
                citations,
                answer_body=answer_body,
                created_at=created_at,
            )
        elif bundle is None and valid_shape and citations:
            refs = tuple(CitationRef.from_payload(item) for item in citations)
            base = synthesize_legacy_citation_refs(
                refs,
                answer_body=answer_body,
                created_at=created_at,
            )
        else:
            raise ValueError("legacy provenance is incomplete")
    except (TypeError, ValueError):
        base = synthesize_legacy_citation_refs(
            (),
            answer_body=answer_body,
            created_at=created_at,
        )
    return _remap_legacy_write(
        base,
        conversation_id=conversation_id,
        message_id=message_id,
        answer_body=answer_body,
        fingerprint=fingerprint_fn,
    )


class CitationLegacyMigrationService:
    """Migrate at most one bounded sidecar batch per call."""

    def __init__(
        self,
        *,
        db: Any,
        repository: CitationTraceRepository,
        sidecar_path: str | Path,
        fingerprint_codec: CitationFingerprintCodec | None = None,
        batch_size: int = LEGACY_MIGRATION_BATCH_SIZE,
        before_cutover_hook: Callable[[], None] | None = None,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= LEGACY_MIGRATION_BATCH_SIZE
        ):
            raise ValueError("batch_size must be an integer between 1 and 100")
        self.db = db
        self.repository = repository
        self.sidecar_path = Path(sidecar_path)
        self._codec = fingerprint_codec
        self.batch_size = batch_size
        self._before_cutover_hook = before_cutover_hook

    @property
    def writes_enabled(self) -> bool:
        return self.repository.canonical_writes_enabled

    @property
    def ready(self) -> bool:
        """Return whether the enabled migration has identity/key material."""

        return self.repository.legacy_migration_ready

    def _fingerprint(self, *parts: str | bytes) -> str:
        return self._domain_fingerprint(
            CitationFingerprintDomain.LEGACY_SOURCE,
            *parts,
        )

    def _domain_fingerprint(
        self,
        domain: CitationFingerprintDomain,
        *parts: str | bytes,
    ) -> str:
        if self._codec is not None:
            return self._codec.fingerprint(domain, *parts)
        return self.repository.legacy_migration_fingerprint(domain, *parts)

    def _raw_sidecar(self) -> tuple[bytes, Mapping[str, Any]]:
        try:
            path_before = os.stat(self.sidecar_path, follow_symlinks=False)
        except FileNotFoundError:
            return b"", {"version": 1, "conversations": {}}
        except OSError:
            raise _LegacyInputError("legacy_sidecar_unreadable") from None
        if stat.S_ISLNK(path_before.st_mode):
            raise _LegacyInputError("legacy_sidecar_symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.sidecar_path, flags)
        except OSError as exc:
            reason = (
                "legacy_sidecar_symlink"
                if exc.errno == errno.ELOOP
                else "legacy_sidecar_unreadable"
            )
            raise _LegacyInputError(reason) from None
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode):
                raise _LegacyInputError("legacy_sidecar_not_regular")
            if opened_before.st_size > LEGACY_SIDECAR_BYTES_MAX:
                raise _LegacyInputError("legacy_sidecar_too_large")
            chunks: list[bytes] = []
            remaining = LEGACY_SIDECAR_BYTES_MAX + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            opened_after = os.fstat(descriptor)
            try:
                path_after = os.stat(self.sidecar_path, follow_symlinks=False)
            except OSError:
                raise _LegacyInputError("legacy_sidecar_changed") from None
        finally:
            os.close(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(opened_before, field) != getattr(opened_after, field)
            for field in stable_fields
        ) or any(
            getattr(path_before, field) != getattr(opened_before, field)
            or getattr(opened_after, field) != getattr(path_after, field)
            for field in ("st_dev", "st_ino")
        ):
            raise _LegacyInputError("legacy_sidecar_changed")
        if len(raw) > LEGACY_SIDECAR_BYTES_MAX:
            raise _LegacyInputError("legacy_sidecar_too_large")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise _LegacyInputError("legacy_sidecar_invalid_json") from None
        _validate_json_lexical_depth(decoded)
        try:
            payload = json.loads(decoded) if raw else {}
        except (
            json.JSONDecodeError,
            MemoryError,
            OverflowError,
            RecursionError,
        ):
            raise _LegacyInputError("legacy_sidecar_invalid_json") from None
        if not isinstance(payload, Mapping):
            raise _LegacyInputError("legacy_sidecar_invalid_root")
        _validate_json_bounds(payload)
        return raw, payload

    @staticmethod
    def _conversation_records(
        payload: Mapping[str, Any],
        conversation_id: str,
    ) -> dict[str, Mapping[str, Any]]:
        conversations = payload.get("conversations", {})
        if not isinstance(conversations, Mapping):
            raise _LegacyInputError("legacy_conversations_invalid")
        raw_records = conversations.get(conversation_id, {})
        if not isinstance(raw_records, Mapping):
            raise _LegacyInputError("legacy_conversation_invalid")
        records: dict[str, Mapping[str, Any]] = {}
        for raw_message_id, record in raw_records.items():
            message_id = _safe_legacy_identifier(raw_message_id)
            if message_id is None:
                continue
            records[message_id] = record if isinstance(record, Mapping) else {}
        return records

    def _source_snapshot(
        self,
        conversation_id: str,
    ) -> tuple[dict[str, Mapping[str, Any]], str]:
        _, payload = self._raw_sidecar()
        records = self._conversation_records(payload, conversation_id)
        canonical = _canonical_json(records)
        return records, self._fingerprint(conversation_id, canonical)

    def _fallback_snapshot(
        self,
        conversation_id: str,
    ) -> dict[str, Mapping[str, Any]]:
        _, payload = self._raw_sidecar()
        return self._conversation_records(payload, conversation_id)

    def get_journal(self, conversation_id: str) -> LegacyMigrationJournal | None:
        identity = self.repository.identity_context
        if identity is None:
            return None
        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT conversation_id, source_fingerprint, state, attempt_count,
                       started_at, updated_at, next_message_cursor, error_code,
                       completed_at
                FROM rag_legacy_migration_journal
                WHERE profile_id = ? AND conversation_id = ?
                """,
                (identity.profile_id, conversation_id),
            )
            .fetchone()
        )
        if row is None:
            return None
        return LegacyMigrationJournal(
            conversation_id=str(row["conversation_id"]),
            source_fingerprint=str(row["source_fingerprint"]),
            state=LegacyMigrationState(row["state"]),
            attempt_count=int(row["attempt_count"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            next_message_cursor=row["next_message_cursor"],
            reason_code=row["error_code"],
            completed_at=row["completed_at"],
        )

    def _record_terminal_state(
        self,
        conversation_id: str,
        *,
        source_fingerprint: str,
        state: LegacyMigrationState,
        reason_code: str,
        expected_journal: LegacyMigrationJournal | None,
    ) -> LegacyMigrationBatchResult:
        identity = self.repository.identity_context
        if identity is None:
            return LegacyMigrationBatchResult(
                state=state,
                reason_code=reason_code,
            )
        current = expected_journal
        if current is not None:
            if current.state in {
                LegacyMigrationState.DIVERGED,
                LegacyMigrationState.FAILED,
            }:
                return LegacyMigrationBatchResult(
                    state=current.state,
                    reason_code=current.reason_code,
                )
            allowed = state is LegacyMigrationState.DIVERGED or (
                state is LegacyMigrationState.FAILED
                and current.state
                in {
                    LegacyMigrationState.PENDING,
                    LegacyMigrationState.RUNNING,
                }
            )
            if not allowed:
                return LegacyMigrationBatchResult(
                    state=current.state,
                    reason_code="legacy_terminal_transition_invalid",
                )
            now = datetime.now(UTC).isoformat()
            with self.db.transaction() as cursor:
                changed = cursor.execute(
                    """
                    UPDATE rag_legacy_migration_journal
                    SET state=?, updated_at=?, error_code=?, completed_at=NULL
                    WHERE profile_id=? AND conversation_id=?
                      AND source_fingerprint=? AND state=?
                      AND next_message_cursor IS ? AND attempt_count=?
                    """,
                    (
                        state.value,
                        now,
                        reason_code[:LEGACY_REASON_CODE_MAX],
                        identity.profile_id,
                        conversation_id,
                        current.source_fingerprint,
                        current.state.value,
                        current.next_message_cursor,
                        current.attempt_count,
                    ),
                ).rowcount
            if changed == 1:
                return LegacyMigrationBatchResult(
                    state=state,
                    reason_code=reason_code,
                )
            refreshed = self.get_journal(conversation_id)
            return LegacyMigrationBatchResult(
                state=(
                    refreshed.state
                    if refreshed is not None
                    else LegacyMigrationState.FAILED
                ),
                reason_code=(
                    refreshed.reason_code
                    if refreshed is not None
                    else "legacy_cutover_guard_failed"
                ),
            )
        now = datetime.now(UTC).isoformat()
        with self.db.transaction() as cursor:
            inserted = cursor.execute(
                """
                INSERT INTO rag_legacy_migration_journal(
                    profile_id, conversation_id, source_fingerprint, state,
                    attempt_count, started_at, updated_at, next_message_cursor,
                    error_code, completed_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, NULL, ?, NULL)
                ON CONFLICT(profile_id, conversation_id) DO NOTHING
                """,
                (
                    identity.profile_id,
                    conversation_id,
                    source_fingerprint,
                    state.value,
                    now,
                    now,
                    reason_code[:LEGACY_REASON_CODE_MAX],
                ),
            ).rowcount
        if inserted != 1:
            refreshed = self.get_journal(conversation_id)
            return LegacyMigrationBatchResult(
                state=(
                    refreshed.state
                    if refreshed is not None
                    else LegacyMigrationState.FAILED
                ),
                reason_code=(
                    refreshed.reason_code
                    if refreshed is not None
                    else "legacy_cutover_guard_failed"
                ),
            )
        return LegacyMigrationBatchResult(state=state, reason_code=reason_code)

    def _resolve_source_mismatch(
        self,
        conversation_id: str,
    ) -> LegacyMigrationBatchResult:
        """Re-read both journal and source before a monotonic divergence CAS."""

        current = self.get_journal(conversation_id)
        if current is None:
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_cutover_guard_failed",
            )
        if current.state in {
            LegacyMigrationState.DIVERGED,
            LegacyMigrationState.FAILED,
        }:
            return LegacyMigrationBatchResult(
                state=current.state,
                reason_code=current.reason_code,
            )
        try:
            _, fresh_fingerprint = self._source_snapshot(conversation_id)
        except (_LegacyInputError, CitationPersistenceUnavailable):
            fresh_fingerprint = None
        if fresh_fingerprint == current.source_fingerprint:
            return LegacyMigrationBatchResult(
                state=current.state,
                reason_code=(
                    None
                    if current.state is LegacyMigrationState.COMPLETE
                    else "legacy_cutover_guard_failed"
                ),
            )
        return self._record_terminal_state(
            conversation_id,
            source_fingerprint=current.source_fingerprint,
            state=LegacyMigrationState.DIVERGED,
            reason_code="legacy_source_changed",
            expected_journal=current,
        )

    def _resolve_unavailable_source(
        self,
        conversation_id: str,
        *,
        reason_code: str,
    ) -> LegacyMigrationBatchResult:
        """Keep terminal journals monotonic when a source read fails."""

        current = self.get_journal(conversation_id)
        if current is None:
            try:
                fallback_fingerprint = self._fingerprint(
                    conversation_id,
                    reason_code,
                )
            except CitationPersistenceUnavailable:
                fallback_fingerprint = "legacy-source-unavailable"
            return self._record_terminal_state(
                conversation_id,
                source_fingerprint=fallback_fingerprint,
                state=LegacyMigrationState.FAILED,
                reason_code=reason_code,
                expected_journal=None,
            )
        if current.state in {
            LegacyMigrationState.DIVERGED,
            LegacyMigrationState.FAILED,
        }:
            return LegacyMigrationBatchResult(
                state=current.state,
                reason_code=current.reason_code,
            )
        if current.state is LegacyMigrationState.COMPLETE:
            return self._resolve_source_mismatch(conversation_id)
        return self._record_terminal_state(
            conversation_id,
            source_fingerprint=current.source_fingerprint,
            state=LegacyMigrationState.FAILED,
            reason_code=reason_code,
            expected_journal=current,
        )

    def migrate_next_batch(
        self,
        conversation_id: str,
    ) -> LegacyMigrationBatchResult:
        """Migrate one batch without making partial conversations readable."""

        if not self.ready:
            return LegacyMigrationBatchResult(state=LegacyMigrationState.PENDING)
        if _safe_legacy_identifier(conversation_id) is None:
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_conversation_id_invalid",
            )
        try:
            records, fingerprint = self._source_snapshot(conversation_id)
        except (_LegacyInputError, CitationPersistenceUnavailable) as exc:
            reason = getattr(exc, "reason_code", "legacy_source_unavailable")
            return self._resolve_unavailable_source(
                conversation_id,
                reason_code=reason,
            )
        journal = self.get_journal(conversation_id)
        if journal is not None:
            if journal.state is LegacyMigrationState.DIVERGED:
                return LegacyMigrationBatchResult(
                    state=LegacyMigrationState.DIVERGED,
                    reason_code=journal.reason_code,
                )
            if journal.state is LegacyMigrationState.FAILED:
                return LegacyMigrationBatchResult(
                    state=LegacyMigrationState.FAILED,
                    reason_code=journal.reason_code,
                )
            if journal.source_fingerprint != fingerprint:
                return self._resolve_source_mismatch(conversation_id)
            if journal.state is LegacyMigrationState.COMPLETE:
                return LegacyMigrationBatchResult(
                    state=LegacyMigrationState.COMPLETE,
                )

        cursor_after = journal.next_message_cursor if journal is not None else None
        ordered_ids = sorted(records)
        pending_ids = [
            message_id
            for message_id in ordered_ids
            if cursor_after is None or message_id > cursor_after
        ]
        batch_ids = pending_ids[: self.batch_size]
        final_batch = len(pending_ids) <= self.batch_size

        identity = self.repository.identity_context
        if identity is None:
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="citation_identity_context_unavailable",
            )
        now = datetime.now(UTC).isoformat()
        processed = 0
        expected_state = journal.state if journal is not None else None
        expected_cursor = journal.next_message_cursor if journal is not None else None
        expected_generation = journal.attempt_count if journal is not None else 0
        claimed_generation = expected_generation + 1
        next_cursor = batch_ids[-1] if batch_ids else cursor_after
        try:
            with self.db.transaction() as cursor:
                if journal is None:
                    claimed = cursor.execute(
                        """
                        INSERT INTO rag_legacy_migration_journal(
                            profile_id, conversation_id, source_fingerprint,
                            state, attempt_count, started_at, updated_at,
                            next_message_cursor, error_code, completed_at
                        ) VALUES (?, ?, ?, 'running', 1, ?, ?, NULL, NULL, NULL)
                        ON CONFLICT(profile_id, conversation_id) DO NOTHING
                        """,
                        (
                            identity.profile_id,
                            conversation_id,
                            fingerprint,
                            now,
                            now,
                        ),
                    ).rowcount
                else:
                    claimed = cursor.execute(
                        """
                        UPDATE rag_legacy_migration_journal
                        SET state='running', attempt_count=attempt_count + 1,
                            updated_at=?, error_code=NULL, completed_at=NULL
                        WHERE profile_id=? AND conversation_id=?
                          AND source_fingerprint=? AND state=?
                          AND next_message_cursor IS ? AND attempt_count=?
                          AND attempt_count < 2147483647
                        """,
                        (
                            now,
                            identity.profile_id,
                            conversation_id,
                            fingerprint,
                            expected_state.value,
                            expected_cursor,
                            expected_generation,
                        ),
                    ).rowcount
                if claimed != 1:
                    raise _MigrationClaimLost
                for message_id in batch_ids:
                    message = cursor.execute(
                        """
                        SELECT id, conversation_id, content, version, timestamp
                        FROM messages
                        WHERE id = ? AND deleted = 0
                        """,
                        (message_id,),
                    ).fetchone()
                    if (
                        message is None
                        or str(message["conversation_id"]) != conversation_id
                    ):
                        continue
                    existing_owner = cursor.execute(
                        """
                        SELECT trace_id
                        FROM rag_message_trace_owners
                        WHERE profile_id = ? AND message_id = ?
                          AND message_revision = ? AND state = 'active'
                        LIMIT 1
                        """,
                        (identity.profile_id, message_id, message["version"]),
                    ).fetchone()
                    if existing_owner is not None:
                        continue
                    created_at = _parse_timestamp(message["timestamp"])
                    sealed = synthesize_legacy_message(
                        records[message_id],
                        conversation_id=conversation_id,
                        message_id=message_id,
                        answer_body=message["content"],
                        created_at=created_at,
                        fingerprint=self._domain_fingerprint,
                    )
                    self.repository.write_legacy_migrating(
                        cursor,
                        sealed,
                        conversation_id=conversation_id,
                        message_id=message_id,
                        message_revision=message["version"],
                        message_body=message["content"],
                    )
                    processed += 1
                advanced = cursor.execute(
                    """
                    UPDATE rag_legacy_migration_journal
                    SET next_message_cursor=?, updated_at=?,
                        completed_at=NULL, error_code=NULL
                    WHERE profile_id=? AND conversation_id=?
                      AND source_fingerprint=? AND state='running'
                      AND next_message_cursor IS ? AND attempt_count=?
                    """,
                    (
                        next_cursor,
                        now,
                        identity.profile_id,
                        conversation_id,
                        fingerprint,
                        expected_cursor,
                        claimed_generation,
                    ),
                ).rowcount
                if advanced != 1:
                    raise _MigrationClaimLost
        except _MigrationClaimLost:
            refreshed = self.get_journal(conversation_id)
            return LegacyMigrationBatchResult(
                state=(
                    refreshed.state
                    if refreshed is not None
                    else LegacyMigrationState.FAILED
                ),
                reason_code="legacy_cutover_guard_failed",
            )
        except (CitationPersistenceUnavailable, _LegacyInputError, ValueError):
            return self._record_terminal_state(
                conversation_id,
                source_fingerprint=fingerprint,
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_batch_invalid",
                expected_journal=journal,
            )

        if not final_batch:
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.RUNNING,
                processed_messages=processed,
            )

        if self._before_cutover_hook is not None:
            self._before_cutover_hook()
        try:
            _, final_fingerprint = self._source_snapshot(conversation_id)
        except (_LegacyInputError, CitationPersistenceUnavailable):
            final_fingerprint = ""
        if final_fingerprint != fingerprint:
            return self._resolve_source_mismatch(conversation_id)

        completed_at = datetime.now(UTC).isoformat()
        try:
            with self.db.transaction() as cursor:
                completed = cursor.execute(
                    """
                    UPDATE rag_legacy_migration_journal
                    SET state='complete', updated_at=?, completed_at=?,
                        error_code=NULL
                    WHERE profile_id=? AND conversation_id=?
                      AND source_fingerprint=? AND state='running'
                      AND next_message_cursor IS ? AND attempt_count=?
                    """,
                    (
                        completed_at,
                        completed_at,
                        identity.profile_id,
                        conversation_id,
                        fingerprint,
                        next_cursor,
                        claimed_generation,
                    ),
                ).rowcount
                if completed != 1:
                    raise _MigrationClaimLost
                cursor.execute(
                    """
                    UPDATE rag_citation_traces
                    SET visibility_state = 'active'
                    WHERE profile_id = ? AND origin = 'legacy_inferred'
                      AND legacy_conversation_id = ?
                      AND visibility_state = 'migrating'
                    """,
                    (identity.profile_id, conversation_id),
                )
        except _MigrationClaimLost:
            refreshed = self.get_journal(conversation_id)
            return LegacyMigrationBatchResult(
                state=(
                    refreshed.state
                    if refreshed is not None
                    else LegacyMigrationState.FAILED
                ),
                processed_messages=processed,
                reason_code="legacy_cutover_guard_failed",
            )
        return LegacyMigrationBatchResult(
            state=LegacyMigrationState.COMPLETE,
            processed_messages=processed,
        )

    def persist_package_record(
        self,
        *,
        conversation_id: str,
        message_id: str,
        record: Mapping[str, Any],
    ) -> LegacyMigrationBatchResult:
        """Adapt an existing package citation record without import authority."""

        if not self.ready:
            return LegacyMigrationBatchResult(state=LegacyMigrationState.PENDING)
        if not isinstance(record, Mapping):
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_package_record_invalid",
            )
        try:
            with self.db.transaction() as cursor:
                message = cursor.execute(
                    """
                    SELECT id, conversation_id, content, version, timestamp
                    FROM messages
                    WHERE id=? AND deleted=0
                    """,
                    (message_id,),
                ).fetchone()
                if (
                    message is None
                    or str(message["conversation_id"]) != conversation_id
                ):
                    raise CitationPersistenceUnavailable(
                        "message_row_identity_conflict"
                    )
                existing_owner = cursor.execute(
                    """
                    SELECT trace_id
                    FROM rag_message_trace_owners
                    WHERE profile_id=? AND message_id=? AND message_revision=?
                      AND state='active'
                    """,
                    (
                        self.repository.identity_context.profile_id,
                        message_id,
                        message["version"],
                    ),
                ).fetchone()
                if existing_owner is not None:
                    return LegacyMigrationBatchResult(
                        state=LegacyMigrationState.COMPLETE,
                    )
                sealed = synthesize_legacy_message(
                    record,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    answer_body=message["content"],
                    created_at=_parse_timestamp(message["timestamp"]),
                    fingerprint=self._domain_fingerprint,
                )
                self.repository.write_legacy_migrating(
                    cursor,
                    sealed,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    message_revision=message["version"],
                    message_body=message["content"],
                )
                identity = self.repository.identity_context
                if identity is None:
                    raise CitationPersistenceUnavailable(
                        "citation_identity_context_unavailable"
                    )
                cursor.execute(
                    """
                    UPDATE rag_citation_traces
                    SET visibility_state='active'
                    WHERE profile_id=? AND trace_id=?
                      AND origin='legacy_inferred'
                      AND visibility_state='migrating'
                    """,
                    (identity.profile_id, sealed.trace.trace_id),
                )
        except (CitationPersistenceUnavailable, _LegacyInputError, ValueError):
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_package_record_invalid",
            )
        return LegacyMigrationBatchResult(
            state=LegacyMigrationState.COMPLETE,
            processed_messages=1,
        )

    def migrate_idle_unit(self) -> LegacyMigrationBatchResult:
        """Run at most one conversation batch after application readiness."""

        if not self.ready:
            return LegacyMigrationBatchResult(state=LegacyMigrationState.PENDING)
        try:
            _, payload = self._raw_sidecar()
        except _LegacyInputError as exc:
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code=exc.reason_code,
            )
        conversations = payload.get("conversations", {})
        if not isinstance(conversations, Mapping):
            return LegacyMigrationBatchResult(
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_conversations_invalid",
            )
        for raw_conversation_id in sorted(conversations):
            conversation_id = _safe_legacy_identifier(raw_conversation_id)
            if conversation_id is None:
                continue
            journal = self.get_journal(conversation_id)
            if journal is not None and journal.state in {
                LegacyMigrationState.COMPLETE,
                LegacyMigrationState.DIVERGED,
                LegacyMigrationState.FAILED,
            }:
                continue
            result = self.migrate_next_batch(conversation_id)
            if result.state is LegacyMigrationState.RUNNING:
                return result
            for remaining_raw_id in sorted(conversations):
                remaining_id = _safe_legacy_identifier(remaining_raw_id)
                if remaining_id is None:
                    continue
                remaining_journal = self.get_journal(remaining_id)
                if remaining_journal is None or remaining_journal.state in {
                    LegacyMigrationState.PENDING,
                    LegacyMigrationState.RUNNING,
                }:
                    return LegacyMigrationBatchResult(
                        state=LegacyMigrationState.RUNNING,
                        processed_messages=result.processed_messages,
                        reason_code=result.reason_code,
                    )
            return result
        return LegacyMigrationBatchResult(state=LegacyMigrationState.COMPLETE)

    def read_conversation(
        self,
        conversation_id: str,
        *,
        verify_canonical: bool = False,
    ) -> LegacyCitationConversationRead:
        """Choose canonical or legacy once; never merge the two histories."""

        journal = self.get_journal(conversation_id)
        if journal is not None and journal.state is LegacyMigrationState.DIVERGED:
            return LegacyCitationConversationRead(
                state=LegacyCitationReadState.DIVERGED,
                records={},
            )
        if (
            journal is not None
            and journal.state is not LegacyMigrationState.COMPLETE
            and self._has_active_legacy_rows(conversation_id)
        ):
            return LegacyCitationConversationRead(
                state=LegacyCitationReadState.VERIFICATION_PENDING,
                records={},
            )
        canonical_records = self._canonical_records(conversation_id)
        if journal is not None and journal.state is LegacyMigrationState.COMPLETE:
            if not verify_canonical:
                return LegacyCitationConversationRead(
                    state=LegacyCitationReadState.VERIFICATION_PENDING,
                    records={},
                )
            try:
                _, fingerprint = self._source_snapshot(conversation_id)
            except (_LegacyInputError, CitationPersistenceUnavailable):
                fingerprint = ""
            if fingerprint != journal.source_fingerprint:
                resolved = self._resolve_source_mismatch(conversation_id)
                if resolved.state is LegacyMigrationState.DIVERGED:
                    return LegacyCitationConversationRead(
                        state=LegacyCitationReadState.DIVERGED,
                        records={},
                    )
                if resolved.state is not LegacyMigrationState.COMPLETE:
                    return LegacyCitationConversationRead(
                        state=LegacyCitationReadState.VERIFICATION_PENDING,
                        records={},
                    )
            return LegacyCitationConversationRead(
                state=LegacyCitationReadState.CANONICAL,
                records=canonical_records,
            )
        if canonical_records:
            return LegacyCitationConversationRead(
                state=LegacyCitationReadState.CANONICAL,
                records=canonical_records,
            )
        try:
            records = self._fallback_snapshot(conversation_id)
        except _LegacyInputError:
            records = {}
        return LegacyCitationConversationRead(
            state=LegacyCitationReadState.LEGACY_FALLBACK,
            records=records,
        )

    def _has_active_legacy_rows(self, conversation_id: str) -> bool:
        identity = self.repository.identity_context
        if identity is None:
            return False
        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT 1
                FROM rag_citation_traces
                WHERE profile_id=? AND origin='legacy_inferred'
                  AND legacy_conversation_id=? AND visibility_state='active'
                LIMIT 1
                """,
                (identity.profile_id, conversation_id),
            )
            .fetchone()
        )
        return row is not None

    def _canonical_records(
        self,
        conversation_id: str,
    ) -> dict[str, Mapping[str, Any]]:
        identity = self.repository.identity_context
        if identity is None:
            return {}
        rows = (
            self.db.get_connection()
            .execute(
                """
                SELECT trace.trace_id, trace.origin, trace.aggregate_json,
                       owner.message_id, owner.body_fingerprint,
                       message.content AS message_body,
                       refs.evidence_ordinal,
                       refs.marker_ordinal, snapshot.title,
                       snapshot.snapshot_text, snapshot.source_identity_json
                FROM rag_message_trace_owners AS owner
                JOIN messages AS message
                  ON message.id=owner.message_id
                 AND message.version=owner.message_revision
                 AND message.deleted=0
                JOIN rag_citation_traces AS trace
                  ON trace.profile_id=owner.profile_id
                 AND trace.trace_id=owner.trace_id
                LEFT JOIN rag_trace_evidence_refs AS refs
                  ON refs.profile_id=trace.profile_id
                 AND refs.trace_id=trace.trace_id
                LEFT JOIN rag_evidence_snapshots AS snapshot
                  ON snapshot.profile_id=refs.profile_id
                 AND snapshot.payload_id=refs.snapshot_payload_id
                WHERE owner.profile_id = ?
                  AND owner.state = 'active'
                  AND message.conversation_id = ?
                  AND trace.visibility_state = 'active'
                ORDER BY owner.message_id, refs.evidence_ordinal
                """,
                (identity.profile_id, conversation_id),
            )
            .fetchall()
        )
        records: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            try:
                body_fingerprint = self._domain_fingerprint(
                    CitationFingerprintDomain.MESSAGE_BODY,
                    str(row["message_body"]),
                )
            except CitationPersistenceUnavailable:
                body_fingerprint = None
            if body_fingerprint is not None and not hmac.compare_digest(
                body_fingerprint,
                str(row["body_fingerprint"]),
            ):
                continue
            message_id = str(row["message_id"])
            record = records.get(message_id)
            if record is None:
                try:
                    trace = CitationTrace.model_validate_json(row["aggregate_json"])
                    if trace.origin.value != row["origin"]:
                        continue
                    attempt = next(
                        (
                            item
                            for item in trace.answer_attempts
                            if item.attempt_id == trace.selected_attempt_id
                        ),
                        None,
                    )
                    occurrences = attempt.occurrences if attempt is not None else ()
                    citations = [
                        {
                            "evidence_id": occurrence.evidence_ordinal,
                            "raw_marker": occurrence.raw_marker,
                        }
                        for occurrence in occurrences
                    ]
                    completeness = trace.completeness_at_seal.value
                except (TypeError, ValueError):
                    continue
                record = {
                    "provenance_origin": trace.origin.value,
                    "citation_validation": {"completeness": completeness},
                    "citations": citations,
                    "evidence_bundle": {
                        "bundle_id": row["trace_id"],
                        "query": "",
                        "references": [],
                    },
                }
                records[message_id] = record
            if row["evidence_ordinal"] is None:
                continue
            try:
                source_identity = json.loads(row["source_identity_json"] or "{}")
                if not isinstance(source_identity, dict):
                    source_identity = {}
            except json.JSONDecodeError:
                source_identity = {}
            record["evidence_bundle"]["references"].append(
                {
                    "evidence_id": str(
                        row["marker_ordinal"] or row["evidence_ordinal"]
                    ),
                    "source_id": str(
                        source_identity.get("source_id") or "legacy-unavailable"
                    ),
                    "source_type": str(source_identity.get("source_type") or "legacy"),
                    "title": row["title"] or "Legacy source",
                    "snippet": row["snapshot_text"] or "",
                    "authority_label": str(
                        source_identity.get("authority_label") or "Legacy"
                    ),
                    "source_owner": str(source_identity.get("source_owner") or "local"),
                }
            )
        return records

    def retry_diverged(self, conversation_id: str) -> None:
        """Explicitly clear hidden staging before rebuilding changed legacy data."""

        identity = self.repository.identity_context
        if identity is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        with self.db.transaction() as cursor:
            row = cursor.execute(
                """
                SELECT state FROM rag_legacy_migration_journal
                WHERE profile_id=? AND conversation_id=?
                """,
                (identity.profile_id, conversation_id),
            ).fetchone()
            if row is None or row["state"] not in {"diverged", "failed"}:
                raise ValueError("legacy migration is not retryable")
            snapshot_ids = tuple(
                item["snapshot_payload_id"]
                for item in cursor.execute(
                    """
                    SELECT refs.snapshot_payload_id
                    FROM rag_trace_evidence_refs AS refs
                    JOIN rag_citation_traces AS trace
                      ON trace.profile_id=refs.profile_id
                     AND trace.trace_id=refs.trace_id
                    WHERE trace.profile_id=?
                      AND trace.origin='legacy_inferred'
                      AND trace.legacy_conversation_id=?
                    """,
                    (identity.profile_id, conversation_id),
                ).fetchall()
            )
            cursor.execute(
                """
                DELETE FROM rag_message_trace_owners
                WHERE profile_id=? AND trace_id IN (
                    SELECT trace_id FROM rag_citation_traces
                    WHERE profile_id=? AND origin='legacy_inferred'
                      AND legacy_conversation_id=?
                )
                """,
                (
                    identity.profile_id,
                    identity.profile_id,
                    conversation_id,
                ),
            )
            cursor.execute(
                """
                DELETE FROM rag_citation_traces
                WHERE profile_id=? AND origin='legacy_inferred'
                  AND legacy_conversation_id=?
                """,
                (identity.profile_id, conversation_id),
            )
            cursor.executemany(
                """
                DELETE FROM rag_evidence_snapshots
                WHERE profile_id=? AND payload_id=?
                  AND origin_namespace='legacy_inferred_v1'
                  AND NOT EXISTS(
                    SELECT 1 FROM rag_trace_evidence_refs
                    WHERE profile_id=? AND snapshot_payload_id=?
                  )
                """,
                (
                    (
                        identity.profile_id,
                        snapshot_id,
                        identity.profile_id,
                        snapshot_id,
                    )
                    for snapshot_id in snapshot_ids
                ),
            )
            cursor.execute(
                """
                DELETE FROM rag_legacy_migration_journal
                WHERE profile_id=? AND conversation_id=?
                """,
                (identity.profile_id, conversation_id),
            )


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        return _EPOCH
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _EPOCH
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "CitationLegacyMigrationService",
    "LEGACY_MIGRATION_BATCH_SIZE",
    "LEGACY_SIDECAR_BYTES_MAX",
    "LegacyCitationConversationRead",
    "LegacyCitationReadState",
    "LegacyMigrationBatchResult",
    "LegacyMigrationJournal",
    "LegacyMigrationState",
    "synthesize_legacy_message",
]
