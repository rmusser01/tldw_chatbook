"""Evidence and citation contracts for source-grounded Console answers."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


EVIDENCE_SNIPPET_CHAR_LIMIT = 4_000
EXACT_SECRET_METADATA_KEYS = frozenset({"token"})
TOKEN_SECRET_SUFFIXES = ("_token",)
SUBSTRING_SECRET_METADATA_KEYS = frozenset({"credential", "secret", "api_key", "password"})

EVIDENCE_STATUSES = frozenset({"available", "blocked", "missing", "stale", "unknown"})
CITATION_STATUSES = frozenset({"validated", "blocked", "unknown", "stale", "missing", "uncited"})


class EvidenceReference(BaseModel):
    """A durable reference to one source snippet that can ground an answer.

    Attributes:
        evidence_id: Stable citation label such as ``S1``.
        source_id: Source-system identifier for the referenced item.
        source_type: Source kind, for example ``note`` or ``media``.
        title: Human-readable source title.
        snippet: Text excerpt used as evidence.
        authority_label: User-visible local/server/workspace authority label.
        workspace_id: Optional workspace owner or eligibility scope.
        source_owner: Owner category for the source.
        content_ref: Optional durable source reference string.
        status: Evidence availability state.
        score: Optional retrieval score between 0 and 1.
        metadata: JSON-safe auxiliary metadata.
        snippet_truncated: Whether ``snippet`` was truncated for payload safety.
        original_snippet_char_count: Original snippet character count before truncation.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    evidence_id: str
    source_id: str
    source_type: str
    title: str
    snippet: str = Field(validation_alias=AliasChoices("snippet", "text"))
    authority_label: str
    workspace_id: str | None = None
    source_owner: str = "local"
    content_ref: str | None = None
    status: str = "available"
    score: float | None = Field(default=None, ge=0, le=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    snippet_truncated: bool = False
    original_snippet_char_count: int | None = None

    @field_validator("evidence_id", "source_id", "source_type", "title", "authority_label", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any, info: ValidationInfo) -> str:
        """Normalize and validate required text fields.

        Args:
            value: Raw value provided by callers or deserialized payloads.
            info: Pydantic field metadata for the field being validated.

        Returns:
            Non-empty stripped text.

        Raises:
            ValueError: If the normalized field value is empty.
        """

        return _required_text(info.field_name, value)

    @field_validator("snippet", mode="before")
    @classmethod
    def _normalize_snippet(cls, value: Any) -> str:
        """Normalize source snippets to text.

        Args:
            value: Raw snippet text.

        Returns:
            String snippet text, or an empty string for ``None``.
        """

        return _text_or_empty(value)

    @field_validator("workspace_id", "content_ref", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        """Normalize optional text fields.

        Args:
            value: Raw optional value.

        Returns:
            Stripped text, or ``None`` when no value is present.
        """

        return _optional_text(value)

    @field_validator("source_owner", mode="before")
    @classmethod
    def _normalize_source_owner(cls, value: Any) -> str:
        """Normalize the source-owner field.

        Args:
            value: Raw source owner value.

        Returns:
            Stripped source owner, defaulting to ``local``.
        """

        return _text_or_default(value, "local")

    @field_validator("status", mode="before")
    @classmethod
    def _validate_evidence_status(cls, value: Any) -> str:
        """Validate evidence status.

        Args:
            value: Raw status value.

        Returns:
            Normalized evidence status.

        Raises:
            ValueError: If the status is not supported.
        """

        return _validate_status("evidence", value, EVIDENCE_STATUSES)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> Mapping[str, Any]:
        """Normalize metadata to a mapping.

        Args:
            value: Raw metadata value.

        Returns:
            Metadata mapping, or an empty mapping for absent metadata.

        Raises:
            ValueError: If metadata is present but not mapping-like.
        """

        return _metadata_mapping(value)

    @model_validator(mode="after")
    def _apply_snippet_limits(self) -> "EvidenceReference":
        """Apply the snippet truncation contract.

        Returns:
            The validated evidence reference with snippet truncation metadata.
        """

        original_count = (
            self.original_snippet_char_count
            if self.original_snippet_char_count is not None
            else len(self.snippet)
        )
        truncated = self.snippet_truncated or len(self.snippet) > EVIDENCE_SNIPPET_CHAR_LIMIT
        snippet = self.snippet[:EVIDENCE_SNIPPET_CHAR_LIMIT]
        object.__setattr__(self, "snippet", snippet)
        object.__setattr__(self, "snippet_truncated", truncated)
        object.__setattr__(self, "original_snippet_char_count", original_count)
        return self

    def to_payload(self) -> dict[str, Any]:
        """Serialize the reference for handoffs, persistence, or export.

        Returns:
            JSON-safe dictionary with sensitive metadata removed.

        Raises:
            TypeError: If nested metadata contains unsupported values.
        """

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
        """Deserialize an evidence reference payload.

        Args:
            payload: Serialized evidence reference payload.

        Returns:
            Validated evidence reference.

        Raises:
            ValueError: If required fields or statuses are invalid.
        """

        return cls(
            evidence_id=payload.get("evidence_id"),
            source_id=payload.get("source_id"),
            source_type=payload.get("source_type"),
            title=payload.get("title"),
            snippet=_payload_value(payload, "snippet", fallback_key="text", default=""),
            authority_label=payload.get("authority_label"),
            workspace_id=payload.get("workspace_id"),
            source_owner=_payload_value(payload, "source_owner", default="local"),
            content_ref=payload.get("content_ref"),
            status=_payload_value(payload, "status", default="available"),
            score=payload.get("score"),
            metadata=payload.get("metadata"),
            snippet_truncated=bool(payload.get("snippet_truncated", False)),
            original_snippet_char_count=payload.get("original_snippet_char_count"),
        )


class EvidenceBundle(BaseModel):
    """A set of source snippets staged together for one grounded interaction.

    Attributes:
        bundle_id: Stable identifier for the evidence group.
        query: User query or retrieval query that produced the evidence.
        references: Evidence references in citation-label order.
        source: User-visible source of the evidence bundle.
        status: Bundle-level availability state.
        metadata: JSON-safe auxiliary metadata.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    bundle_id: str
    query: str
    references: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    source: str = "Library Search/RAG"
    status: str = "available"
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("bundle_id", mode="before")
    @classmethod
    def _normalize_bundle_id(cls, value: Any) -> str:
        """Normalize and validate the bundle identifier.

        Args:
            value: Raw bundle id.

        Returns:
            Non-empty stripped bundle id.

        Raises:
            ValueError: If the bundle id is empty.
        """

        return _required_text("bundle_id", value)

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: Any) -> str:
        """Normalize the bundle query.

        Args:
            value: Raw query value.

        Returns:
            Query text, or an empty string for ``None``.
        """

        return _text_or_empty(value)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> str:
        """Normalize the bundle source label.

        Args:
            value: Raw source label.

        Returns:
            Source label, defaulting to ``Library Search/RAG``.
        """

        return _text_or_default(value, "Library Search/RAG")

    @field_validator("references", mode="before")
    @classmethod
    def _normalize_references(cls, value: Any) -> tuple[Any, ...]:
        """Normalize references into a tuple.

        Args:
            value: Raw iterable of references.

        Returns:
            Tuple of reference-like values for Pydantic validation.
        """

        if value is None:
            return ()
        return tuple(value)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_evidence_status(cls, value: Any) -> str:
        """Validate bundle status.

        Args:
            value: Raw status value.

        Returns:
            Normalized evidence status.

        Raises:
            ValueError: If the status is not supported.
        """

        return _validate_status("evidence", value, EVIDENCE_STATUSES)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> Mapping[str, Any]:
        """Normalize metadata to a mapping.

        Args:
            value: Raw metadata value.

        Returns:
            Metadata mapping, or an empty mapping for absent metadata.

        Raises:
            ValueError: If metadata is present but not mapping-like.
        """

        return _metadata_mapping(value)

    def available_references(self) -> tuple[EvidenceReference, ...]:
        """Return references eligible to ground model output.

        Returns:
            Tuple of references whose status is ``available``.
        """

        return tuple(reference for reference in self.references if reference.status == "available")

    def reference_by_id(self, evidence_id: str) -> EvidenceReference | None:
        """Return one reference by stable citation label.

        Args:
            evidence_id: Citation label to look up.

        Returns:
            Matching evidence reference, or ``None`` when absent.
        """

        for reference in self.references:
            if reference.evidence_id == evidence_id:
                return reference
        return None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the bundle for handoffs, persistence, or export.

        Returns:
            JSON-safe dictionary with sensitive metadata removed.

        Raises:
            TypeError: If nested metadata contains unsupported values.
        """

        payload = {
            "bundle_id": self.bundle_id,
            "query": self.query,
            "source": self.source,
            "status": self.status,
            "metadata": _metadata_payload(self.metadata),
            "references": [reference.to_payload() for reference in self.references],
        }
        return _without_none(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvidenceBundle":
        """Deserialize an evidence bundle payload.

        Args:
            payload: Serialized evidence bundle payload.

        Returns:
            Validated evidence bundle.

        Raises:
            ValueError: If required fields or statuses are invalid.
        """

        references = payload.get("references") if isinstance(payload.get("references"), list) else []
        return cls(
            bundle_id=payload.get("bundle_id"),
            query=_payload_value(payload, "query", default=""),
            source=_payload_value(payload, "source", default="Library Search/RAG"),
            status=_payload_value(payload, "status", default="available"),
            metadata=payload.get("metadata"),
            references=tuple(
                EvidenceReference.from_payload(reference)
                for reference in references
                if isinstance(reference, Mapping)
            ),
        )


class CitationRef(BaseModel):
    """A model-output citation marker validated against an evidence bundle.

    Attributes:
        evidence_id: Stable citation label referenced by the answer.
        source_id: Source-system identifier expected for the evidence label.
        quote: Optional answer quote associated with the citation marker.
        status: Validation status for the citation marker.
        metadata: JSON-safe auxiliary metadata.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    evidence_id: str
    source_id: str
    quote: str = Field(default="", validation_alias=AliasChoices("quote", "text"))
    status: str = "validated"
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "source_id", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any, info: ValidationInfo) -> str:
        """Normalize and validate required text fields.

        Args:
            value: Raw value provided by callers or deserialized payloads.
            info: Pydantic field metadata for the field being validated.

        Returns:
            Non-empty stripped text.

        Raises:
            ValueError: If the normalized field value is empty.
        """

        return _required_text(info.field_name, value)

    @field_validator("quote", mode="before")
    @classmethod
    def _normalize_quote(cls, value: Any) -> str:
        """Normalize citation quote text.

        Args:
            value: Raw quote value.

        Returns:
            Quote text, or an empty string for ``None``.
        """

        return _text_or_empty(value)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_citation_status(cls, value: Any) -> str:
        """Validate citation status.

        Args:
            value: Raw status value.

        Returns:
            Normalized citation status.

        Raises:
            ValueError: If the status is not supported.
        """

        return _validate_status("citation", value, CITATION_STATUSES)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> Mapping[str, Any]:
        """Normalize metadata to a mapping.

        Args:
            value: Raw metadata value.

        Returns:
            Metadata mapping, or an empty mapping for absent metadata.

        Raises:
            ValueError: If metadata is present but not mapping-like.
        """

        return _metadata_mapping(value)

    def validate_against(self, bundle: EvidenceBundle) -> "CitationRef":
        """Validate the citation against an evidence bundle.

        Args:
            bundle: Evidence bundle that should contain the cited evidence.

        Returns:
            This citation when valid, or a copied citation with a downgraded
            status when the evidence is absent, blocked, stale, or mismatched.
        """

        reference = bundle.reference_by_id(self.evidence_id)
        if reference is None or reference.source_id != self.source_id:
            return self.model_copy(update={"status": "unknown"})
        if reference.status != "available":
            return self.model_copy(update={"status": reference.status})
        return self

    def to_payload(self) -> dict[str, Any]:
        """Serialize the citation reference.

        Returns:
            JSON-safe dictionary with sensitive metadata removed.

        Raises:
            TypeError: If nested metadata contains unsupported values.
        """

        payload = {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "quote": self.quote,
            "status": self.status,
            "metadata": _metadata_payload(self.metadata),
        }
        return _without_none(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CitationRef":
        """Deserialize a citation reference payload.

        Args:
            payload: Serialized citation reference payload.

        Returns:
            Validated citation reference.

        Raises:
            ValueError: If required fields or statuses are invalid.
        """

        return cls(
            evidence_id=payload.get("evidence_id"),
            source_id=payload.get("source_id"),
            quote=_payload_value(payload, "quote", fallback_key="text", default=""),
            status=_payload_value(payload, "status", default="validated"),
            metadata=payload.get("metadata"),
        )


def _payload_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    default: Any = None,
) -> Any:
    if key in payload:
        return payload[key]
    if fallback_key is not None and fallback_key in payload:
        return payload[fallback_key]
    return default


def _validate_status(kind: str, value: Any, allowed: frozenset[str]) -> str:
    status = _text_or_empty(value).strip().lower()
    if status not in allowed:
        raise ValueError(f"Unsupported {kind} status: {value!r}")
    return status


def _required_text(field_name: str, value: Any) -> str:
    text = _text_or_empty(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = _text_or_empty(value).strip()
    return text or None


def _text_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _text_or_default(value: Any, default: str) -> str:
    return _optional_text(value) or default


def _without_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("metadata must be a mapping")


def _metadata_payload(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_payload(item)
            for key, item in value.items()
            if item is not None and not _is_secret_metadata_key(str(key))
        }
    if isinstance(value, set):
        return [_metadata_payload(item) for item in sorted(value, key=str) if item is not None]
    if isinstance(value, (list, tuple)):
        return [_metadata_payload(item) for item in value if item is not None]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _is_secret_metadata_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in EXACT_SECRET_METADATA_KEYS
        or normalized.endswith(TOKEN_SECRET_SUFFIXES)
        or any(secret_key in normalized for secret_key in SUBSTRING_SECRET_METADATA_KEYS)
    )
