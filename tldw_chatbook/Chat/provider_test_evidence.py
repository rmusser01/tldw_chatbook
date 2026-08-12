"""Pure, process-local provider readiness and test evidence contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Literal, Protocol
from unicodedata import category as unicode_category

from .provider_endpoint_contract import canonical_connection_identity

ConfigurationFacet = Literal["incomplete", "configured"]
EndpointFacet = Literal[
    "not_tested",
    "testing",
    "reachable",
    "unreachable",
    "model_listing_unavailable",
]
ModelFacet = Literal["missing", "confirmed", "unconfirmed"]
CredentialSource = Literal["none", "stored", "environment", "draft"]
EndpointFailureCategory = Literal[
    "timeout",
    "connection_refused",
    "unauthorized",
    "forbidden",
    "http_status",
    "invalid_payload",
    "connection_error",
]
ReadinessVerdictCode = Literal[
    "incomplete",
    "model_missing",
    "not_tested",
    "testing",
    "connection_failed",
    "model_listing_unavailable",
    "model_unconfirmed",
    "verified",
]

_CONFIGURATION_FACETS = frozenset({"incomplete", "configured"})
_ENDPOINT_FACETS = frozenset(
    {
        "not_tested",
        "testing",
        "reachable",
        "unreachable",
        "model_listing_unavailable",
    }
)
_MODEL_FACETS = frozenset({"missing", "confirmed", "unconfirmed"})
_CREDENTIAL_SOURCES = frozenset({"none", "stored", "environment", "draft"})
_VERDICT_CODES = frozenset(
    {
        "incomplete",
        "model_missing",
        "not_tested",
        "testing",
        "connection_failed",
        "model_listing_unavailable",
        "model_unconfirmed",
        "verified",
    }
)
_FAILURE_CATEGORIES = frozenset(
    {
        "timeout",
        "connection_refused",
        "unauthorized",
        "forbidden",
        "http_status",
        "invalid_payload",
        "connection_error",
    }
)
_FAILURE_DETAILS = {
    "timeout": "The model listing request timed out.",
    "connection_refused": "The model listing connection was refused.",
    "unauthorized": "The model listing request was unauthorized.",
    "forbidden": "The model listing request was forbidden.",
    "http_status": "The model listing endpoint returned an HTTP status error.",
    "invalid_payload": "The model listing endpoint returned an invalid response.",
    "connection_error": "The model listing request had a connection error.",
}
_PROVIDER_KEY = re.compile(r"[a-z0-9_]+")
_MAX_PROVIDER_KEY_CHARS = 128
_MAX_CONNECTION_ENDPOINT_CHARS = 4096
_MAX_MODEL_ID_CHARS = 120
_MAX_MODEL_IDS = 100
_MAX_VERDICT_DETAIL_CHARS = 256
_UNSAFE_MODEL_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


@dataclass(frozen=True, slots=True)
class ProviderReadinessSnapshot:
    """Independent configuration, endpoint-test, and model facets."""

    configuration: ConfigurationFacet
    endpoint: EndpointFacet
    model: ModelFacet
    category: EndpointFailureCategory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, str) or (
            self.configuration not in _CONFIGURATION_FACETS
        ):
            raise ValueError("Configuration facet is invalid.")
        if not isinstance(self.endpoint, str) or self.endpoint not in _ENDPOINT_FACETS:
            raise ValueError("Endpoint facet is invalid.")
        if not isinstance(self.model, str) or self.model not in _MODEL_FACETS:
            raise ValueError("Model facet is invalid.")
        if self.category is not None and (
            not isinstance(self.category, str)
            or self.category not in _FAILURE_CATEGORIES
        ):
            raise ValueError("Endpoint failure category is invalid.")
        if self.endpoint == "unreachable":
            return
        if self.endpoint == "model_listing_unavailable" and self.category in {
            None,
            "http_status",
        }:
            return
        if self.category is not None:
            raise ValueError("Endpoint failure category conflicts with its facet.")


@dataclass(frozen=True, slots=True)
class ProviderReadinessVerdict:
    """One bounded user-facing interpretation of a readiness snapshot."""

    code: ReadinessVerdictCode
    detail: str
    verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or self.code not in _VERDICT_CODES:
            raise ValueError("Readiness verdict code is invalid.")
        if (
            not isinstance(self.detail, str)
            or not self.detail
            or len(self.detail) > _MAX_VERDICT_DETAIL_CHARS
            or not self.detail.isprintable()
            or any(
                unicode_category(character) in _UNSAFE_MODEL_CATEGORIES
                for character in self.detail
            )
        ):
            raise ValueError("Readiness verdict detail is invalid.")
        if type(self.verified) is not bool or self.verified != (
            self.code == "verified"
        ):
            raise ValueError("Readiness verdict verification state is inconsistent.")


def provider_readiness_verdict(
    snapshot: ProviderReadinessSnapshot,
) -> ProviderReadinessVerdict:
    """Return one honest verdict without inferring chat status from models 404."""

    if not isinstance(snapshot, ProviderReadinessSnapshot):
        raise ValueError("Readiness snapshot is invalid.")  # noqa: TRY004
    if snapshot.configuration == "incomplete":
        return ProviderReadinessVerdict(
            "incomplete", "Provider configuration is incomplete."
        )
    if snapshot.endpoint == "testing":
        return ProviderReadinessVerdict("testing", "Testing the model listing endpoint.")
    if snapshot.endpoint == "unreachable":
        return ProviderReadinessVerdict(
            "connection_failed",
            _FAILURE_DETAILS.get(
                snapshot.category,
                "The model listing endpoint could not be reached.",
            ),
        )
    if snapshot.endpoint == "model_listing_unavailable":
        return ProviderReadinessVerdict(
            "model_listing_unavailable",
            "Model listing unavailable; chat endpoint not tested.",
        )
    if snapshot.model == "missing":
        return ProviderReadinessVerdict("model_missing", "Select a model.")
    if snapshot.endpoint == "not_tested":
        return ProviderReadinessVerdict(
            "not_tested", "Provider configured; connection not tested."
        )
    if snapshot.model == "confirmed":
        return ProviderReadinessVerdict(
            "verified", "Model listing reached and selected model confirmed.", True
        )
    return ProviderReadinessVerdict(
        "model_unconfirmed",
        "Model listing reached; selected model was not confirmed.",
    )


@dataclass(frozen=True, slots=True)
class ProviderDraftIdentity:
    """Secret-free semantic identity for one provider settings draft."""

    provider_key: str
    connection_identity: tuple[str, str]
    credential_source: CredentialSource
    credential_revision: int
    draft_generation: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_key, str)
            or not self.provider_key
            or len(self.provider_key) > _MAX_PROVIDER_KEY_CHARS
            or _PROVIDER_KEY.fullmatch(self.provider_key) is None
        ):
            raise ValueError("Provider key is invalid.")
        if (
            not isinstance(self.connection_identity, tuple)
            or len(self.connection_identity) != 2
            or not all(isinstance(value, str) for value in self.connection_identity)
        ):
            raise ValueError("Connection identity is invalid.")
        identity_provider, identity_endpoint = self.connection_identity
        if (
            identity_provider != self.provider_key
            or not identity_endpoint
            or len(identity_endpoint) > _MAX_CONNECTION_ENDPOINT_CHARS
            or canonical_connection_identity(self.provider_key, identity_endpoint)
            != self.connection_identity
        ):
            raise ValueError("Connection identity is not canonical.")
        if not isinstance(self.credential_source, str) or (
            self.credential_source not in _CREDENTIAL_SOURCES
        ):
            raise ValueError("Credential source is invalid.")
        if type(self.credential_revision) is not int or self.credential_revision < 0:
            raise ValueError("Credential revision must be a non-negative integer.")
        if type(self.draft_generation) is not int or self.draft_generation < 0:
            raise ValueError("Draft generation must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ProviderTestEvidence:
    """Bounded probe evidence for one exact semantic draft identity."""

    identity: ProviderDraftIdentity
    endpoint: EndpointFacet
    model_ids: tuple[str, ...]
    category: EndpointFailureCategory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ProviderDraftIdentity):
            raise ValueError("Provider evidence identity is invalid.")  # noqa: TRY004
        if not isinstance(self.endpoint, str) or self.endpoint not in _ENDPOINT_FACETS:
            raise ValueError("Provider evidence endpoint facet is invalid.")
        _validate_model_ids(self.model_ids)
        if self.category is not None and (
            not isinstance(self.category, str)
            or self.category not in _FAILURE_CATEGORIES
        ):
            raise ValueError("Provider evidence category is invalid.")

        if self.endpoint == "reachable":
            if self.category is not None:
                raise ValueError("Reachable evidence cannot include a failure category.")
        elif self.endpoint == "unreachable":
            if self.model_ids:
                raise ValueError("Unreachable evidence cannot include model IDs.")
        elif self.endpoint == "model_listing_unavailable":
            if self.model_ids or self.category not in {None, "http_status"}:
                raise ValueError("Model-listing evidence is inconsistent.")
        elif self.model_ids or self.category is not None:
            raise ValueError("Untested evidence cannot include probe results.")


class _MutationResult(Protocol):
    @property
    def fully_applied(self) -> bool: ...

    file_replaced: bool
    conflict: bool


class _ProviderTestToken:
    """Opaque, single-use capability for settling one current probe."""

    __slots__ = ("_identity", "_sequence")

    def __init__(self, identity: ProviderDraftIdentity, sequence: int) -> None:
        self._identity = identity
        self._sequence = sequence

    def __repr__(self) -> str:
        return "<ProviderTestToken>"


class ProviderTestEvidenceStore:
    """Thread-safe process-local owner of latest-generation probe evidence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sequence = 0
        self._latest_generation = -1
        self._current_token: _ProviderTestToken | None = None
        self._evidence: ProviderTestEvidence | None = None

    def begin(self, identity: ProviderDraftIdentity) -> object:
        """Start a probe for ``identity`` and return its opaque settlement token."""

        if not isinstance(identity, ProviderDraftIdentity):
            raise ValueError("Provider draft identity is invalid.")  # noqa: TRY004
        with self._lock:
            if identity.draft_generation < self._latest_generation:
                raise ValueError("Draft generation is older than current evidence.")
            self._latest_generation = identity.draft_generation
            self._sequence += 1
            token = _ProviderTestToken(identity, self._sequence)
            self._current_token = token
            self._evidence = ProviderTestEvidence(identity, "testing", ())
            return token

    def settle(self, token: object, outcome: object) -> bool:
        """Attach a bounded result only when ``token`` still owns the draft."""

        with self._lock:
            if token is not self._current_token or not isinstance(
                token, _ProviderTestToken
            ):
                return False
            try:
                evidence = _coerce_probe_evidence(token._identity, outcome)
            except (AttributeError, TypeError, ValueError):
                return False
            if evidence.endpoint in {"not_tested", "testing"}:
                return False
            self._evidence = evidence
            self._current_token = None
            return True

    def evidence_for(
        self, identity: ProviderDraftIdentity
    ) -> ProviderTestEvidence | None:
        """Return evidence only for the exact semantic identity supplied."""

        if not isinstance(identity, ProviderDraftIdentity):
            return None
        with self._lock:
            if self._evidence is None or self._evidence.identity != identity:
                return None
            return self._evidence

    def invalidate(self, identity: ProviderDraftIdentity | None = None) -> bool:
        """Remove exact or current evidence and cancel its active operation."""

        if identity is not None and not isinstance(identity, ProviderDraftIdentity):
            return False
        with self._lock:
            if self._evidence is None:
                return False
            if identity is not None and self._evidence.identity != identity:
                return False
            self._evidence = None
            self._current_token = None
            return True

    def rebase_after_save(
        self,
        tested_identity: ProviderDraftIdentity,
        saved_identity: ProviderDraftIdentity,
        mutation_result: _MutationResult,
    ) -> bool:
        """Rebind settled evidence after an equivalent, fully applied save."""

        if not isinstance(tested_identity, ProviderDraftIdentity) or not isinstance(
            saved_identity, ProviderDraftIdentity
        ):
            return False
        with self._lock:
            evidence = self._evidence
            if evidence is None or evidence.identity != tested_identity:
                return False
            if not bool(getattr(mutation_result, "fully_applied", False)):
                if bool(getattr(mutation_result, "file_replaced", False)) or bool(
                    getattr(mutation_result, "conflict", False)
                ):
                    self._evidence = None
                    self._current_token = None
                return False
            if evidence.endpoint == "testing" or not _same_saved_semantics(
                tested_identity, saved_identity
            ) or saved_identity.draft_generation < tested_identity.draft_generation:
                self._evidence = None
                self._current_token = None
                return False
            self._evidence = ProviderTestEvidence(
                saved_identity,
                evidence.endpoint,
                evidence.model_ids,
                evidence.category,
            )
            self._current_token = None
            self._latest_generation = max(
                self._latest_generation, saved_identity.draft_generation
            )
            return True


def _validate_model_ids(model_ids: object) -> None:
    if not isinstance(model_ids, tuple):
        raise ValueError("Model IDs must be a tuple.")  # noqa: TRY004
    if len(model_ids) > _MAX_MODEL_IDS:
        raise ValueError("Too many model IDs.")
    seen: set[str] = set()
    for model_id in model_ids:
        if not isinstance(model_id, str):
            raise ValueError("Model ID is invalid.")  # noqa: TRY004
        if not model_id or len(model_id) > _MAX_MODEL_ID_CHARS:
            raise ValueError("Model ID length is invalid.")
        if (
            model_id != " ".join(model_id.split())
            or any(
                unicode_category(character) in _UNSAFE_MODEL_CATEGORIES
                for character in model_id
            )
            or not model_id.isprintable()
        ):
            raise ValueError("Model ID contains invalid characters.")
        if model_id in seen:
            raise ValueError("Model IDs must be unique.")
        seen.add(model_id)


def _coerce_probe_evidence(
    identity: ProviderDraftIdentity, outcome: object
) -> ProviderTestEvidence:
    if isinstance(outcome, ProviderTestEvidence):
        if outcome.identity != identity:
            raise ValueError("Probe evidence identity does not match the token.")
        return outcome
    endpoint = outcome.state  # type: ignore[attr-defined]
    model_ids = getattr(outcome, "model_ids", ())
    category = getattr(outcome, "category", None)
    return ProviderTestEvidence(identity, endpoint, model_ids, category)


def _same_saved_semantics(
    first: ProviderDraftIdentity, second: ProviderDraftIdentity
) -> bool:
    return (
        first.provider_key == second.provider_key
        and first.connection_identity == second.connection_identity
        and first.credential_source == second.credential_source
        and first.credential_revision == second.credential_revision
    )
