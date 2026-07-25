"""Request-scoped construction of local citation provenance."""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .citation_source_locators import CanonicalSourceKind
from .citation_trace_identity import (
    BoundedIdentifier,
    CitationFingerprintCodec,
    CitationFingerprintDomain,
    LocalCitationIdentityContext,
    new_opaque_id,
)
from .citation_trace_models import (
    EVIDENCE_ENTRIES_PER_PROMPT_MAX,
    PROMPT_EVIDENCE_SETS_MAX,
    RETRIEVAL_CANDIDATES_PER_RUN_MAX,
    SNAPSHOT_TEXT_UTF8_BYTES_MAX,
    EvidenceRun,
    EvidenceRunPayload,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    MarkerNamespace,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    RetrievalCandidatePayload,
    RetrievalScoreKind,
    RetrievalScoreScale,
    ShortCode,
)

_SAFE_PIPELINE_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RAW_QUERY_UTF8_BYTES_MAX = 64 * 1024
_TITLE_UTF8_BYTES_MAX = 4 * 1024
_MAX_CONTEXT_CHARACTERS = 4 * 1024 * 1024
_MAX_LINEAGE_OFFSET = (2**63) - 1
_LOCAL_SOURCE_KINDS = frozenset(
    {
        CanonicalSourceKind.MEDIA_DB,
        CanonicalSourceKind.NOTES,
        CanonicalSourceKind.CHAT_HISTORY,
    }
)
_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


class _StrictFrozenCapture(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )


class _LocalBuilderHeader(_StrictFrozenCapture):
    request_id: BoundedIdentifier
    generation_id: BoundedIdentifier
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LocalRetrievalCandidateCapture(_StrictFrozenCapture):
    """Bounded local candidate data accepted by the Chat provenance layer."""

    candidate_rank: int = Field(ge=1, le=RETRIEVAL_CANDIDATES_PER_RUN_MAX)
    source_kind: CanonicalSourceKind
    source_id: BoundedIdentifier
    title: str = Field(min_length=1)
    score_kind: RetrievalScoreKind | None = None
    score_scale: RetrievalScoreScale | None = None
    score: float | None = None
    chunk_id: BoundedIdentifier | None = None
    chunk_index: int | None = Field(default=None, ge=0, le=_MAX_LINEAGE_OFFSET)
    start_char: int | None = Field(default=None, ge=0, le=_MAX_LINEAGE_OFFSET)
    end_char: int | None = Field(default=None, ge=0, le=_MAX_LINEAGE_OFFSET)

    @field_validator("source_id", "chunk_id")
    @classmethod
    def _validate_opaque_source_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            _URI_SCHEME.match(value) is not None
            or "/" in value
            or "\\" in value
            or value.startswith("~")
        ):
            raise ValueError(
                "source identifiers cannot contain an executable path or URL"
            )
        return value

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        byte_count = len(value.encode("utf-8"))
        if byte_count > _TITLE_UTF8_BYTES_MAX:
            raise ValueError(f"title exceeds {_TITLE_UTF8_BYTES_MAX} UTF-8 bytes")
        if any(ord(character) < 32 and character not in "\t" for character in value):
            raise ValueError("title must not contain control characters")
        return value

    @model_validator(mode="after")
    def _validate_score_and_lineage(self) -> LocalRetrievalCandidateCapture:
        if self.source_kind not in _LOCAL_SOURCE_KINDS:
            raise ValueError("source_kind must identify a local source family")
        score_parts = (self.score_kind, self.score_scale, self.score)
        if any(part is not None for part in score_parts) and any(
            part is None for part in score_parts
        ):
            raise ValueError("score requires kind, scale, and value together")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("start_char and end_char must be supplied together")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must not precede start_char")
        return self


class LocalRetrievalRunMetadata(_StrictFrozenCapture):
    """Allowlisted, non-executable metadata for one local retrieval run."""

    search_mode: str
    requested_top_k: int = Field(ge=1, le=RETRIEVAL_CANDIDATES_PER_RUN_MAX)
    max_context_characters: int = Field(ge=0, le=_MAX_CONTEXT_CHARACTERS)
    rerank_enabled: bool
    source_kinds: tuple[CanonicalSourceKind, ...] = Field(
        min_length=1,
        max_length=len(CanonicalSourceKind),
    )
    scope_state: Literal["unscoped", "scoped", "empty"]

    @field_validator("search_mode")
    @classmethod
    def _validate_search_mode(cls, value: str) -> str:
        if _SAFE_PIPELINE_CODE.fullmatch(value) is None:
            raise ValueError("search_mode must be a safe pipeline identifier")
        return value

    @model_validator(mode="after")
    def _validate_source_kinds(self) -> LocalRetrievalRunMetadata:
        if len(set(self.source_kinds)) != len(self.source_kinds):
            raise ValueError("source_kinds must be unique")
        if any(
            source_kind not in _LOCAL_SOURCE_KINDS for source_kind in self.source_kinds
        ):
            raise ValueError("source_kinds must identify local source families")
        return self


class LocalPromptEvidenceCapture(_StrictFrozenCapture):
    """One exact marked evidence block submitted at the prompt boundary."""

    candidate_rank: int = Field(ge=1, le=RETRIEVAL_CANDIDATES_PER_RUN_MAX)
    snapshot_text: str = Field(min_length=1)
    transformations: tuple[ShortCode, ...] = Field(default=(), max_length=32)

    @field_validator("snapshot_text")
    @classmethod
    def _validate_snapshot_text(cls, value: str) -> str:
        byte_count = len(value.encode("utf-8"))
        if byte_count > SNAPSHOT_TEXT_UTF8_BYTES_MAX:
            raise ValueError(
                f"snapshot_text exceeds {SNAPSHOT_TEXT_UTF8_BYTES_MAX} UTF-8 bytes"
            )
        return value

    @model_validator(mode="after")
    def _validate_transformations(self) -> LocalPromptEvidenceCapture:
        if len(set(self.transformations)) != len(self.transformations):
            raise ValueError("transformations must be unique")
        return self


class CitationTraceBuilder:
    """Mutable, request-scoped local citation capture.

    The builder intentionally has no persistence or sealing API. Secret
    fingerprint material remains private and is never represented.
    """

    __slots__ = (
        "_created_at",
        "_evidence_run_payloads",
        "_evidence_runs",
        "_evidence_snapshot_payloads",
        "_fingerprint_codec",
        "_generation_id",
        "_identity_context",
        "_prompt_evidence_sets",
        "_request_id",
    )

    def __init__(
        self,
        *,
        request_id: str,
        generation_id: str,
        identity_context: LocalCitationIdentityContext,
        fingerprint_codec: CitationFingerprintCodec,
        created_at: datetime,
    ) -> None:
        if not isinstance(identity_context, LocalCitationIdentityContext):
            raise TypeError("identity_context must be a LocalCitationIdentityContext")
        if not isinstance(fingerprint_codec, CitationFingerprintCodec):
            raise TypeError("fingerprint_codec must be a CitationFingerprintCodec")
        validated_identity = LocalCitationIdentityContext.model_validate(
            identity_context.model_dump(mode="python")
        )
        header = _LocalBuilderHeader(
            request_id=request_id,
            generation_id=generation_id,
            created_at=created_at,
        )
        self._request_id = header.request_id
        self._generation_id = header.generation_id
        self._identity_context = validated_identity
        self._fingerprint_codec = fingerprint_codec
        self._created_at = header.created_at
        self._evidence_runs: list[EvidenceRun] = []
        self._evidence_run_payloads: list[EvidenceRunPayload] = []
        self._prompt_evidence_sets: list[PromptEvidenceSet] = []
        self._evidence_snapshot_payloads: list[EvidenceSnapshotPayload] = []

    @classmethod
    def local(
        cls,
        *,
        request_id: str,
        generation_id: str,
        identity_context: LocalCitationIdentityContext,
        fingerprint_codec: CitationFingerprintCodec,
        created_at: datetime | None = None,
    ) -> CitationTraceBuilder:
        """Create a builder bound to one validated local authority profile."""

        return cls(
            request_id=request_id,
            generation_id=generation_id,
            identity_context=identity_context,
            fingerprint_codec=fingerprint_codec,
            created_at=created_at or datetime.now(UTC),
        )

    @property
    def request_id(self) -> str:
        """Return the provider-request identity."""

        return self._request_id

    @property
    def generation_id(self) -> str:
        """Return the generation identity."""

        return self._generation_id

    @property
    def created_at(self) -> datetime:
        """Return the builder creation timestamp."""

        return self._created_at

    @property
    def evidence_runs(self) -> tuple[EvidenceRun, ...]:
        """Return the recorded immutable run relationships."""

        return tuple(self._evidence_runs)

    @property
    def evidence_run_payloads(self) -> tuple[EvidenceRunPayload, ...]:
        """Return the governed retrieval payloads recorded so far."""

        return tuple(self._evidence_run_payloads)

    @property
    def prompt_evidence_sets(self) -> tuple[PromptEvidenceSet, ...]:
        """Return exact prompt-boundary evidence relationships."""

        return tuple(self._prompt_evidence_sets)

    @property
    def evidence_snapshot_payloads(self) -> tuple[EvidenceSnapshotPayload, ...]:
        """Return governed exact prompt evidence payloads."""

        return tuple(self._evidence_snapshot_payloads)

    def record_retrieval_run(
        self,
        *,
        stage: str,
        raw_query: str,
        candidates: tuple[LocalRetrievalCandidateCapture, ...],
        retrieval_metadata: LocalRetrievalRunMetadata,
        started_at: datetime,
        ended_at: datetime | None,
    ) -> str:
        """Record one validated local retrieval execution atomically."""

        if not isinstance(stage, str):
            raise TypeError("stage must be a string")
        if stage and _SAFE_PIPELINE_CODE.fullmatch(stage) is None:
            raise ValueError("stage must be a safe non-empty pipeline identifier")
        if not isinstance(raw_query, str):
            raise TypeError("raw_query must be a string")
        if not raw_query:
            raise ValueError("raw_query must not be empty")
        if len(raw_query.encode("utf-8")) > _RAW_QUERY_UTF8_BYTES_MAX:
            raise ValueError(
                f"raw_query exceeds {_RAW_QUERY_UTF8_BYTES_MAX} UTF-8 bytes"
            )
        if not isinstance(candidates, tuple):
            raise TypeError("candidates must be a tuple")
        if len(candidates) > RETRIEVAL_CANDIDATES_PER_RUN_MAX:
            raise ValueError(
                f"candidates exceeds {RETRIEVAL_CANDIDATES_PER_RUN_MAX} entries"
            )
        if not isinstance(retrieval_metadata, LocalRetrievalRunMetadata):
            raise TypeError("retrieval_metadata must be LocalRetrievalRunMetadata")

        metadata = LocalRetrievalRunMetadata.model_validate(
            retrieval_metadata.model_dump(mode="python")
        )
        validated_candidates: list[LocalRetrievalCandidateCapture] = []
        for candidate in candidates:
            if not isinstance(candidate, LocalRetrievalCandidateCapture):
                raise TypeError(
                    "candidates must contain LocalRetrievalCandidateCapture values"
                )
            validated_candidates.append(
                LocalRetrievalCandidateCapture.model_validate(
                    candidate.model_dump(mode="python")
                )
            )

        run_id = new_opaque_id("evidence-run")
        payload_id = new_opaque_id("run-payload")
        candidate_payloads = tuple(
            self._candidate_payload(candidate) for candidate in validated_candidates
        )
        run_payload = EvidenceRunPayload(
            payload_id=payload_id,
            run_id=run_id,
            raw_query=None,
            query_fingerprint=self._fingerprint_codec.fingerprint(
                CitationFingerprintDomain.RAW_QUERY,
                raw_query,
            ),
            authority_id=self._identity_context.local_authority_id,
            retrieval_metadata=metadata.model_dump(mode="json"),
            candidates=candidate_payloads,
        )
        run = EvidenceRun(
            run_id=run_id,
            request_id=self._request_id,
            run_ordinal=len(self._evidence_runs) + 1,
            stage=stage,
            payload_ref=payload_id,
            started_at=started_at,
            ended_at=ended_at,
        )

        self._evidence_runs.append(run)
        self._evidence_run_payloads.append(run_payload)
        return run_id

    @staticmethod
    def _candidate_payload(
        candidate: LocalRetrievalCandidateCapture,
    ) -> RetrievalCandidatePayload:
        lineage = {
            key: value
            for key, value in (
                ("chunk_id", candidate.chunk_id),
                ("chunk_index", candidate.chunk_index),
                ("start_char", candidate.start_char),
                ("end_char", candidate.end_char),
            )
            if value is not None
        }
        return RetrievalCandidatePayload(
            candidate_id=new_opaque_id("retrieval-candidate"),
            rank=candidate.candidate_rank,
            source_identity={
                "source_kind": candidate.source_kind.value,
                "source_id": candidate.source_id,
            },
            title=candidate.title,
            locator={},
            lineage=lineage,
            score_kind=candidate.score_kind,
            score_scale=candidate.score_scale,
            score=candidate.score,
        )

    def record_prompt_evidence_set(
        self,
        *,
        run_id: str,
        evidence: tuple[LocalPromptEvidenceCapture, ...],
        created_at: datetime,
    ) -> str:
        """Record one exact local prompt-evidence set atomically."""

        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if len(self._prompt_evidence_sets) >= PROMPT_EVIDENCE_SETS_MAX:
            raise ValueError(
                f"prompt evidence sets exceeds {PROMPT_EVIDENCE_SETS_MAX} entries"
            )
        if not isinstance(evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if not evidence:
            raise ValueError("evidence must not be empty")
        if len(evidence) > EVIDENCE_ENTRIES_PER_PROMPT_MAX:
            raise ValueError(
                f"evidence entries exceeds {EVIDENCE_ENTRIES_PER_PROMPT_MAX} entries"
            )

        matching_runs = [run for run in self._evidence_runs if run.run_id == run_id]
        matching_payloads = [
            payload
            for payload in self._evidence_run_payloads
            if payload.run_id == run_id
        ]
        if not matching_runs or not matching_payloads:
            raise ValueError("unknown evidence run")
        if len(matching_runs) != 1 or len(matching_payloads) != 1:
            raise ValueError("duplicate evidence run reference")
        run_payload = matching_payloads[0]
        candidates_by_rank = {
            candidate.rank: candidate for candidate in run_payload.candidates
        }

        validated_evidence: list[LocalPromptEvidenceCapture] = []
        for capture in evidence:
            if not isinstance(capture, LocalPromptEvidenceCapture):
                raise TypeError(
                    "evidence must contain LocalPromptEvidenceCapture values"
                )
            validated_evidence.append(
                LocalPromptEvidenceCapture.model_validate(
                    capture.model_dump(mode="python")
                )
            )
        candidate_ranks = [capture.candidate_rank for capture in validated_evidence]
        if len(set(candidate_ranks)) != len(candidate_ranks):
            raise ValueError("candidate_rank must be unique within a prompt set")

        snapshots: list[EvidenceSnapshotPayload] = []
        entries: list[PromptEvidenceEntry] = []
        for ordinal, capture in enumerate(validated_evidence, start=1):
            marker_match = re.match(
                r"\[S([1-9][0-9]*)\](?:\s|$)", capture.snapshot_text
            )
            if marker_match is None or int(marker_match.group(1)) != ordinal:
                raise ValueError(
                    "snapshot marker ordinal must match its prompt evidence ordinal"
                )
            candidate = candidates_by_rank.get(capture.candidate_rank)
            if candidate is None:
                raise ValueError("prompt evidence references unknown candidate_rank")
            snapshot_id = new_opaque_id("snapshot-payload")
            exact_bytes = capture.snapshot_text.encode("utf-8")
            comparison_bytes = self._comparison_snapshot_bytes(capture.snapshot_text)
            snapshot = EvidenceSnapshotPayload(
                payload_id=snapshot_id,
                storage_mode=EvidenceStorageMode.EMBEDDED,
                snapshot_text=capture.snapshot_text,
                title=candidate.title,
                source_identity=candidate.source_identity,
                locator=candidate.locator,
                lineage=candidate.lineage,
                transformations=capture.transformations,
                content_hash=self._fingerprint_codec.fingerprint(
                    CitationFingerprintDomain.EXACT_PAYLOAD,
                    "exact-snapshot-v1",
                    exact_bytes,
                ),
                comparison_hash=self._fingerprint_codec.fingerprint(
                    CitationFingerprintDomain.EXACT_PAYLOAD,
                    "comparison-nfc-lf-v1",
                    comparison_bytes,
                ),
            )
            entry = PromptEvidenceEntry(
                evidence_ordinal=ordinal,
                marker_ordinal=ordinal,
                run_id=run_id,
                snapshot_payload_ref=snapshot_id,
                storage_mode=EvidenceStorageMode.EMBEDDED,
            )
            snapshots.append(snapshot)
            entries.append(entry)

        prompt_set_id = new_opaque_id("prompt-set")
        prompt_set = PromptEvidenceSet(
            prompt_set_id=prompt_set_id,
            prompt_set_ordinal=len(self._prompt_evidence_sets) + 1,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            entries=tuple(entries),
            created_at=created_at,
        )

        self._evidence_snapshot_payloads.extend(snapshots)
        self._prompt_evidence_sets.append(prompt_set)
        return prompt_set_id

    @staticmethod
    def _comparison_snapshot_bytes(snapshot_text: str) -> bytes:
        normalized_newlines = snapshot_text.replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", normalized_newlines).encode("utf-8")

    @property
    def is_sealed(self) -> bool:
        """Report the intentionally unsealed state of this request builder."""

        return False

    def __repr__(self) -> str:
        return (
            "CitationTraceBuilder("
            f"request_id={self._request_id!r}, "
            f"generation_id={self._generation_id!r}, "
            f"evidence_runs={len(self._evidence_runs)}, "
            f"prompt_evidence_sets={len(self._prompt_evidence_sets)}, "
            "fingerprint_codec=<redacted>)"
        )


__all__ = [
    "CitationTraceBuilder",
    "LocalPromptEvidenceCapture",
    "LocalRetrievalCandidateCapture",
    "LocalRetrievalRunMetadata",
]
