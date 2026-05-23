"""Evidence and citation contracts for source-grounded Console answers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


EVIDENCE_SNIPPET_CHAR_LIMIT = 4_000
SECRET_METADATA_KEYS = frozenset({"credential", "credential_source", "token", "secret", "api_key", "password"})

EVIDENCE_STATUSES = frozenset({"available", "blocked", "missing", "stale", "unknown"})
CITATION_STATUSES = frozenset({"validated", "unknown", "stale", "missing", "uncited"})


@dataclass(frozen=True)
class EvidenceReference:
    """A durable reference to one source snippet that can ground an answer."""

    evidence_id: str
    source_id: str
    source_type: str
    title: str
    snippet: str
    authority_label: str
    workspace_id: str | None = None
    source_owner: str = "local"
    content_ref: str | None = None
    status: str = "available"
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    snippet_truncated: bool = False
    original_snippet_char_count: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("evidence_id", "source_id", "source_type", "title", "authority_label"):
            object.__setattr__(self, field_name, _required_text(field_name, getattr(self, field_name)))

        snippet = str(self.snippet or "")
        original_count = self.original_snippet_char_count if self.original_snippet_char_count is not None else len(snippet)
        truncated = bool(self.snippet_truncated) or len(snippet) > EVIDENCE_SNIPPET_CHAR_LIMIT
        if len(snippet) > EVIDENCE_SNIPPET_CHAR_LIMIT:
            snippet = snippet[:EVIDENCE_SNIPPET_CHAR_LIMIT]

        object.__setattr__(self, "snippet", snippet)
        object.__setattr__(self, "snippet_truncated", truncated)
        object.__setattr__(self, "original_snippet_char_count", original_count)
        object.__setattr__(self, "source_owner", _text_or_default(self.source_owner, "local"))
        object.__setattr__(self, "status", _validate_status("evidence", self.status, EVIDENCE_STATUSES))

        if self.score is not None and not 0 <= float(self.score) <= 1:
            raise ValueError("score must be between 0 and 1")

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload with sensitive metadata removed."""

        payload = {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "snippet": self.snippet,
            "authority_label": self.authority_label,
            "workspace_id": self.workspace_id,
            "source_owner": self.source_owner,
            "content_ref": self.content_ref,
            "status": self.status,
            "score": self.score,
            "metadata": _metadata_payload(self.metadata),
            "snippet_truncated": self.snippet_truncated,
            "original_snippet_char_count": self.original_snippet_char_count,
        }
        return _without_none(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvidenceReference":
        """Rebuild a reference from its serialized payload."""

        return cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            source_id=str(payload.get("source_id") or ""),
            source_type=str(payload.get("source_type") or ""),
            title=str(payload.get("title") or ""),
            snippet=str(payload.get("snippet") or ""),
            authority_label=str(payload.get("authority_label") or ""),
            workspace_id=_optional_text(payload.get("workspace_id")),
            source_owner=str(payload.get("source_owner") or "local"),
            content_ref=_optional_text(payload.get("content_ref")),
            status=str(payload.get("status") or "available"),
            score=payload.get("score"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
            snippet_truncated=bool(payload.get("snippet_truncated", False)),
            original_snippet_char_count=payload.get("original_snippet_char_count"),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    """A set of source snippets staged together for one grounded interaction."""

    bundle_id: str
    query: str
    references: tuple[EvidenceReference, ...] = field(default_factory=tuple)
    source: str = "Library Search/RAG"
    status: str = "available"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _required_text("bundle_id", self.bundle_id))
        object.__setattr__(self, "query", str(self.query or ""))
        object.__setattr__(self, "source", _text_or_default(self.source, "Library Search/RAG"))
        object.__setattr__(self, "references", tuple(self.references or ()))
        object.__setattr__(self, "status", _validate_status("evidence", self.status, EVIDENCE_STATUSES))

    def available_references(self) -> tuple[EvidenceReference, ...]:
        """Return references eligible to ground model output."""

        return tuple(reference for reference in self.references if reference.status == "available")

    def reference_by_id(self, evidence_id: str) -> EvidenceReference | None:
        """Return one reference by stable citation label."""

        for reference in self.references:
            if reference.evidence_id == evidence_id:
                return reference
        return None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload for handoffs, persistence, and export."""

        return {
            "bundle_id": self.bundle_id,
            "query": self.query,
            "source": self.source,
            "status": self.status,
            "metadata": _metadata_payload(self.metadata),
            "references": [reference.to_payload() for reference in self.references],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvidenceBundle":
        """Rebuild a bundle from its serialized payload."""

        references = payload.get("references") if isinstance(payload.get("references"), list) else []
        return cls(
            bundle_id=str(payload.get("bundle_id") or ""),
            query=str(payload.get("query") or ""),
            source=str(payload.get("source") or "Library Search/RAG"),
            status=str(payload.get("status") or "available"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
            references=tuple(
                EvidenceReference.from_payload(reference)
                for reference in references
                if isinstance(reference, Mapping)
            ),
        )


@dataclass(frozen=True)
class CitationRef:
    """A model-output citation marker validated against an evidence bundle."""

    evidence_id: str
    source_id: str
    quote: str = ""
    status: str = "validated"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_text("evidence_id", self.evidence_id))
        object.__setattr__(self, "source_id", _required_text("source_id", self.source_id))
        object.__setattr__(self, "quote", str(self.quote or ""))
        object.__setattr__(self, "status", _validate_status("citation", self.status, CITATION_STATUSES))

    def validate_against(self, bundle: EvidenceBundle) -> "CitationRef":
        """Return a trusted citation only when the evidence id and source id match."""

        reference = bundle.reference_by_id(self.evidence_id)
        if reference is None or reference.source_id != self.source_id:
            return replace(self, status="unknown")
        if reference.status != "available":
            return replace(self, status=reference.status if reference.status in CITATION_STATUSES else "unknown")
        return self

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe citation payload."""

        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "quote": self.quote,
            "status": self.status,
            "metadata": _metadata_payload(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CitationRef":
        """Rebuild a citation ref from its serialized payload."""

        return cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            source_id=str(payload.get("source_id") or ""),
            quote=str(payload.get("quote") or ""),
            status=str(payload.get("status") or "validated"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
        )


def _validate_status(kind: str, value: Any, allowed: frozenset[str]) -> str:
    status = str(value or "").strip().lower()
    if status not in allowed:
        raise ValueError(f"Unsupported {kind} status: {value!r}")
    return status


def _required_text(field_name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _text_or_default(value: Any, default: str) -> str:
    return _optional_text(value) or default


def _without_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _metadata_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_payload(item)
            for key, item in value.items()
            if item is not None and not _is_secret_metadata_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_payload(item) for item in value if item is not None]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _is_secret_metadata_key(key: str) -> bool:
    normalized = key.lower()
    return any(secret_key in normalized for secret_key in SECRET_METADATA_KEYS)
