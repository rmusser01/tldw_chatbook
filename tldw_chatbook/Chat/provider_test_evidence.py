"""Pure, process-local provider readiness and test evidence contracts."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from itertools import count
from threading import Lock, RLock
from typing import TYPE_CHECKING, Literal, Protocol
from unicodedata import category as unicode_category

from .provider_endpoint_contract import canonical_connection_identity

if TYPE_CHECKING:
    from .console_session_settings import ConsoleSessionSettings

ConfigurationFacet = Literal["incomplete", "configured"]
EndpointFacet = Literal[
    "not_tested",
    "testing",
    "reachable",
    "unreachable",
    "model_listing_unavailable",
    "changed_since_test",
]
ModelFacet = Literal["missing", "confirmed", "unconfirmed"]
CredentialFacet = Literal[
    "not_required", "missing", "present_unverified", "authenticated"
]
GenerationFacet = Literal[
    "not_tested", "testing", "succeeded", "failed", "changed_since_test"
]
GenerationFailureCategory = Literal[
    "authentication",
    "rate_limit",
    "bad_request",
    "timeout",
    "connection_error",
    "provider_error",
]
CredentialSource = Literal["none", "stored", "environment", "draft"]
ConfigurationIssueCode = Literal[
    "provider_missing",
    "credential_missing",
    "endpoint_missing",
    "invalid_settings",
]
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
    "changed_since_test",
]

_CONFIGURATION_FACETS = frozenset({"incomplete", "configured"})
_ENDPOINT_FACETS = frozenset(
    {
        "not_tested",
        "testing",
        "reachable",
        "unreachable",
        "model_listing_unavailable",
        "changed_since_test",
    }
)
_EVIDENCE_ENDPOINT_FACETS = _ENDPOINT_FACETS - {"changed_since_test"}
_PROBE_ENDPOINT_FACETS = frozenset(
    {"reachable", "unreachable", "model_listing_unavailable"}
)
_MODEL_FACETS = frozenset({"missing", "confirmed", "unconfirmed"})
_CREDENTIAL_FACETS = frozenset(
    {"not_required", "missing", "present_unverified", "authenticated"}
)
_GENERATION_FACETS = frozenset(
    {"not_tested", "testing", "succeeded", "failed", "changed_since_test"}
)
_GENERATION_FAILURE_CATEGORIES = frozenset(
    {
        "authentication",
        "rate_limit",
        "bad_request",
        "timeout",
        "connection_error",
        "provider_error",
    }
)
_CREDENTIAL_SOURCES = frozenset({"none", "stored", "environment", "draft"})
_CONFIGURATION_ISSUES = frozenset(
    {
        "provider_missing",
        "credential_missing",
        "endpoint_missing",
        "invalid_settings",
    }
)
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
        "changed_since_test",
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
# TASK-25833: these are read by a first-time user deciding what to do next,
# not by the implementer. Each names what happened in plain language and the
# one action that addresses it -- "The model listing request had a connection
# error" named a subsystem and no next step.
_FAILURE_DETAILS = {
    "timeout": (
        "The request timed out - the server did not answer in time. "
        "Check it is running and not overloaded."
    ),
    "connection_refused": (
        "The connection was refused - nothing is listening at that address. "
        "Start the server, or check the endpoint and port."
    ),
    # Qodo review (PR #2256): "Go Back" was wrong -- this verdict renders on
    # the Provider step, which already contains the key input, and Back goes to
    # Welcome. Source-agnostic too: the credential may be stored or from the
    # environment, where "re-enter it" is not the remedy either.
    "unauthorized": (
        "Unauthorized - the server rejected this API key. "
        "Check the Authentication section on this step."
    ),
    "forbidden": (
        "Forbidden - this API key is not allowed to list models. Check its permissions."
    ),
    "http_status": (
        "The server returned an HTTP status error. Check the endpoint, then try again."
    ),
    "invalid_payload": (
        "The server returned an invalid response - not an OpenAI-compatible "
        "model list. Check the endpoint."
    ),
    "connection_error": (
        "Connection error - could not reach the server. Check the endpoint, "
        "and that the server is running."
    ),
}
_CONFIGURATION_ISSUE_DETAILS = {
    "provider_missing": "Select a provider.",
    "credential_missing": "Provider credentials are missing.",
    "endpoint_missing": "Provider endpoint is missing.",
    "invalid_settings": "Provider settings are invalid.",
}
_PROVIDER_KEY = re.compile(r"[a-z0-9_]+")
_MAX_PROVIDER_KEY_CHARS = 128
_MAX_CONNECTION_ENDPOINT_CHARS = 4096
_MAX_MODEL_ID_CHARS = 120
_MAX_MODEL_IDS = 100
_MAX_VERDICT_DETAIL_CHARS = 256
_UNSAFE_MODEL_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_BOUNDED_GENERATION_TEST_EXECUTION_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "moonshot", "zai"}
)


class ConsoleGenerationTestAvailability(StrEnum):
    """Whether Console can dispatch a bounded generation test."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


def console_generation_test_availability(
    provider: str,
    *,
    handler_keys: Collection[str] | None = None,
) -> ConsoleGenerationTestAvailability:
    """Project generation-test support from Console's real dispatch catalog.

    Args:
        provider: Provider selected by the Console draft.
        handler_keys: Optional dispatch catalog keys used for the projection.

    Returns:
        Whether the selected provider supports a bounded generation test.
    """

    from .console_provider_support import resolve_console_provider_identity

    identity = resolve_console_provider_identity(provider, handler_keys=handler_keys)
    if (
        identity.is_supported
        and identity.execution_key in _BOUNDED_GENERATION_TEST_EXECUTION_KEYS
    ):
        return ConsoleGenerationTestAvailability.SUPPORTED
    return ConsoleGenerationTestAvailability.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class ProviderReadinessSnapshot:
    """Independent configuration, credential, endpoint, model, and generation facets."""

    configuration: ConfigurationFacet
    endpoint: EndpointFacet
    model: ModelFacet
    category: EndpointFailureCategory | None = None
    configuration_issue: ConfigurationIssueCode | None = None
    credential: CredentialFacet = "not_required"
    generation: GenerationFacet = "not_tested"

    def __post_init__(self) -> None:
        if type(self.configuration) is not str or (
            self.configuration not in _CONFIGURATION_FACETS
        ):
            raise ValueError("Configuration facet is invalid.")
        if type(self.endpoint) is not str or self.endpoint not in _ENDPOINT_FACETS:
            raise ValueError("Endpoint facet is invalid.")
        if type(self.model) is not str or self.model not in _MODEL_FACETS:
            raise ValueError("Model facet is invalid.")
        if (
            type(self.credential) is not str
            or self.credential not in _CREDENTIAL_FACETS
        ):
            raise ValueError("Credential facet is invalid.")
        if (
            type(self.generation) is not str
            or self.generation not in _GENERATION_FACETS
        ):
            raise ValueError("Generation facet is invalid.")
        normalized_credential = _normalize_generation_credential(
            self.credential,
            self.generation,
            None,
            credential_required=self.credential != "not_required",
        )
        object.__setattr__(self, "credential", normalized_credential)
        if self.category is not None and (
            type(self.category) is not str or self.category not in _FAILURE_CATEGORIES
        ):
            raise ValueError("Endpoint failure category is invalid.")
        if self.configuration_issue is not None and (
            type(self.configuration_issue) is not str
            or self.configuration_issue not in _CONFIGURATION_ISSUES
        ):
            raise ValueError("Configuration issue is invalid.")
        if self.configuration == "configured" and self.configuration_issue is not None:
            raise ValueError("Configured readiness cannot include an issue.")
        if self.configuration == "incomplete" and self.credential == "authenticated":
            raise ValueError("Incomplete readiness cannot authenticate credentials.")
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
        if type(self.code) is not str or self.code not in _VERDICT_CODES:
            raise ValueError("Readiness verdict code is invalid.")
        if (
            type(self.detail) is not str
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
    if snapshot.endpoint == "changed_since_test":
        return ProviderReadinessVerdict(
            "changed_since_test",
            "Provider settings changed since test; test again.",
        )
    if snapshot.configuration == "incomplete":
        return ProviderReadinessVerdict(
            "incomplete",
            _CONFIGURATION_ISSUE_DETAILS.get(
                snapshot.configuration_issue,
                "Provider configuration is incomplete.",
            ),
        )
    if snapshot.endpoint == "testing":
        return ProviderReadinessVerdict(
            "testing", "Testing the model listing endpoint."
        )
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
    # TASK-25833: "reached; not confirmed" mixes a tick with a failure and
    # leaves the user unsure whether it is safe to continue. Say which half
    # worked and what to do about the other half.
    return ProviderReadinessVerdict(
        "model_unconfirmed",
        "Reached the server, but your chosen model was not in its list. "
        "Pick one on the next step.",
    )


@dataclass(frozen=True, slots=True)
class ProviderDraftIdentity:
    """Secret-free semantic identity for one provider settings draft."""

    provider_key: str
    connection_identity: tuple[str, str] = field(repr=False)
    credential_source: CredentialSource
    credential_revision: int
    draft_generation: int
    custom_endpoint_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.provider_key) is not str
            or not self.provider_key
            or len(self.provider_key) > _MAX_PROVIDER_KEY_CHARS
            or _PROVIDER_KEY.fullmatch(self.provider_key) is None
        ):
            raise ValueError("Provider key is invalid.")
        if self.custom_endpoint_id is not None and (
            type(self.custom_endpoint_id) is not str
            or re.fullmatch(r"custom-ep:[a-z0-9-]{1,64}", self.custom_endpoint_id)
            is None
            or self.provider_key not in {"custom", "llama_cpp", "ollama"}
        ):
            raise ValueError("Custom endpoint identity is invalid.")
        if (
            type(self.connection_identity) is not tuple
            or len(self.connection_identity) != 2
            or not all(type(value) is str for value in self.connection_identity)
        ):
            raise ValueError("Connection identity is invalid.")
        identity_provider, identity_endpoint = self.connection_identity
        if (
            not identity_provider
            or len(identity_provider) > _MAX_PROVIDER_KEY_CHARS
            or _PROVIDER_KEY.fullmatch(identity_provider) is None
            or not identity_endpoint
            or len(identity_endpoint) > _MAX_CONNECTION_ENDPOINT_CHARS
            or canonical_connection_identity(self.provider_key, identity_endpoint)
            != self.connection_identity
        ):
            raise ValueError("Connection identity is not canonical.")
        if type(self.credential_source) is not str or (
            self.credential_source not in _CREDENTIAL_SOURCES
        ):
            raise ValueError("Credential source is invalid.")
        if type(self.credential_revision) is not int or self.credential_revision < 0:
            raise ValueError("Credential revision must be a non-negative integer.")
        if type(self.draft_generation) is not int or self.draft_generation < 0:
            raise ValueError("Draft generation must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ProviderProbeResult:
    """Exact bounded probe result accepted by the evidence store."""

    endpoint: EndpointFacet
    model_ids: tuple[str, ...]
    category: EndpointFailureCategory | None = None

    def __post_init__(self) -> None:
        _validate_probe_result(self.endpoint, self.model_ids, self.category)


@dataclass(frozen=True, slots=True)
class ProviderGenerationProbeResult:
    """Exact bounded generation result accepted by the evidence store."""

    generation: Literal["succeeded", "failed"]
    category: GenerationFailureCategory | None = None

    def __post_init__(self) -> None:
        _validate_generation_result(self.generation, self.category)


@dataclass(frozen=True, slots=True)
class ConsoleGenerationTestRequest:
    """Validated modal draft and exact evidence identity for one paid test."""

    settings: ConsoleSessionSettings = field(repr=False)
    identity: ProviderDraftIdentity = field(repr=False)

    def __post_init__(self) -> None:
        from .console_session_settings import ConsoleSessionSettings

        if type(self.settings) is not ConsoleSessionSettings:
            raise TypeError("settings must be ConsoleSessionSettings")
        if type(self.identity) is not ProviderDraftIdentity:
            raise TypeError("identity must be ProviderDraftIdentity")


@dataclass(frozen=True, slots=True)
class ProviderTestEvidence:
    """Bounded independent evidence for one exact semantic draft identity."""

    identity: ProviderDraftIdentity
    endpoint: EndpointFacet
    model_ids: tuple[str, ...]
    category: EndpointFailureCategory | None = None
    credential: CredentialFacet = "not_required"
    generation: GenerationFacet = "not_tested"
    generation_category: GenerationFailureCategory | None = None
    #: TASK-33005.1: local time of the latest settled observation, for the
    #: "verified HH:MM" / "reachable HH:MM" words. Not part of equality.
    observed_at: datetime | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if type(self.identity) is not ProviderDraftIdentity:
            raise ValueError("Provider evidence identity is invalid.")
        if self.observed_at is not None and (
            type(self.observed_at) is not datetime
            or self.observed_at.utcoffset() is None
        ):
            raise ValueError("Provider evidence observation time is invalid.")
        if (
            type(self.endpoint) is not str
            or self.endpoint not in _EVIDENCE_ENDPOINT_FACETS
        ):
            raise ValueError("Provider evidence endpoint facet is invalid.")
        _validate_model_ids(self.model_ids)
        if self.category is not None and (
            type(self.category) is not str or self.category not in _FAILURE_CATEGORIES
        ):
            raise ValueError("Provider evidence category is invalid.")
        if (
            type(self.credential) is not str
            or self.credential not in _CREDENTIAL_FACETS
        ):
            raise ValueError("Provider evidence credential facet is invalid.")
        if (
            type(self.generation) is not str
            or self.generation not in _GENERATION_FACETS
        ):
            raise ValueError("Provider evidence generation facet is invalid.")
        if self.generation_category is not None and (
            type(self.generation_category) is not str
            or self.generation_category not in _GENERATION_FAILURE_CATEGORIES
        ):
            raise ValueError("Provider evidence generation category is invalid.")

        if self.endpoint == "reachable":
            if self.category is not None:
                raise ValueError(
                    "Reachable evidence cannot include a failure category."
                )
        elif self.endpoint == "unreachable":
            if self.model_ids:
                raise ValueError("Unreachable evidence cannot include model IDs.")
        elif self.endpoint == "model_listing_unavailable":
            if self.model_ids or self.category not in {None, "http_status"}:
                raise ValueError("Model-listing evidence is inconsistent.")
        elif self.model_ids or self.category is not None:
            raise ValueError("Untested evidence cannot include probe results.")

        if self.generation == "failed":
            if self.generation_category is None:
                raise ValueError("Failed generation evidence requires a category.")
        elif self.generation_category is not None:
            raise ValueError("Generation category conflicts with its facet.")
        normalized_credential = _normalize_generation_credential(
            self.credential,
            self.generation,
            self.generation_category,
            credential_required=self.identity.credential_source != "none",
        )
        object.__setattr__(self, "credential", normalized_credential)


class _MutationResult(Protocol):
    @property
    def fully_applied(self) -> bool: ...

    file_replaced: bool
    conflict: bool


#: TASK-33005.1: process-wide begin order. The shared owner keeps the result
#: of the probe that BEGAN last, so a slow probe settling late on one surface
#: never overwrites a newer one from another (and a clock step cannot either).
_BEGIN_ORDER = count(1)


class _ProviderTestToken:
    """Opaque, single-use capability for settling one current probe."""

    __slots__ = ("order",)

    def __init__(self) -> None:
        self.order = next(_BEGIN_ORDER)

    def __repr__(self) -> str:
        return "<ProviderTestToken>"


class _ProviderGenerationTestToken:
    """Opaque, single-use capability for settling one generation probe."""

    __slots__ = ("order",)

    def __init__(self) -> None:
        self.order = next(_BEGIN_ORDER)

    def __repr__(self) -> str:
        return "<ProviderGenerationTestToken>"


class _ProviderEvidenceSaveLease:
    """Opaque capability binding one save callback to a store revision."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<ProviderEvidenceSaveLease>"


class ProviderTestEvidenceStore:
    """Thread-safe draft-scoped owner of one surface's latest probe evidence.

    TASK-33005.1: a store built with ``app`` (a callable returning the running
    app) publishes every observed fact to that app's
    :class:`ProviderConnectionEvidence` and reads the other surfaces' facts
    back for an exact connection. A rebound earlier result (a
    ``ProviderTestEvidence`` outcome) was published when it was observed, so
    it is not published again. Without ``app`` the store is draft-only.
    """

    def __init__(self, app: Callable[[], object] | None = None) -> None:
        self._app = app
        self._shared_owner: ProviderConnectionEvidence | None = None
        self._lock = RLock()
        self._operation_epoch = 0
        self._generation_operation_epoch = 0
        self._latest_generation = -1
        self._current_token: _ProviderTestToken | None = None
        self._current_token_epoch: int | None = None
        self._current_identity: ProviderDraftIdentity | None = None
        self._settling: tuple[ProviderDraftIdentity, int] | None = None
        self._current_generation_token: _ProviderGenerationTestToken | None = None
        self._current_generation_token_epoch: int | None = None
        self._current_generation_identity: ProviderDraftIdentity | None = None
        self._generation_settling: tuple[ProviderDraftIdentity, int] | None = None
        self._generation_cancel_restore: (
            tuple[
                ProviderDraftIdentity,
                Literal["succeeded", "failed"],
                GenerationFailureCategory | None,
            ]
            | None
        ) = None
        self._save_lease: (
            tuple[_ProviderEvidenceSaveLease, ProviderDraftIdentity, int] | None
        ) = None
        self._evidence: ProviderTestEvidence | None = None

    def begin(self, identity: ProviderDraftIdentity) -> object:
        """Start a probe for ``identity`` and return its opaque settlement token."""

        if type(identity) is not ProviderDraftIdentity:
            raise ValueError("Provider draft identity is invalid.")
        with self._lock:
            if identity.draft_generation < self._latest_generation:
                raise ValueError("Draft generation is older than current evidence.")
            self._latest_generation = identity.draft_generation
            if self._owned_identity() not in {None, identity}:
                self._clear_all_test_state()
                self._advance_generation_operation()
            self._advance_operation()
            token = _ProviderTestToken()
            self._current_token = token
            self._current_token_epoch = self._operation_epoch
            self._current_identity = identity
            self._evidence = _replace_endpoint_evidence(
                self._evidence,
                identity=identity,
                endpoint="testing",
                model_ids=(),
                category=None,
            )
            return token

    def settle(self, token: object, outcome: object) -> bool:
        """Attach a bounded result only when ``token`` still owns the draft."""

        supported_outcome = type(outcome) in {
            ProviderProbeResult,
            ProviderTestEvidence,
        }
        with self._lock:
            if (
                type(token) is not _ProviderTestToken
                or token is not self._current_token
                or self._current_token_epoch != self._operation_epoch
            ):
                return False
            identity = self._current_identity
            claim_epoch = self._current_token_epoch
            self._current_token = None
            self._current_token_epoch = None
            self._current_identity = None
            self._save_lease = None
            if identity is None or claim_epoch is None:
                self._clear_endpoint_evidence(identity)
                self._advance_operation()
                return False
            settlement_claim = (identity, claim_epoch)
            self._settling = settlement_claim

        if not supported_outcome:
            self._reject_settlement(settlement_claim)
            return False
        try:
            evidence = _evidence_from_exact_outcome(identity, outcome)
        except ValueError:
            self._reject_settlement(settlement_claim)
            return False

        with self._lock:
            if (
                self._operation_epoch != claim_epoch
                or self._settling != settlement_claim
                or self._current_token is not None
                or self._owned_identity() != identity
            ):
                return False
            self._settling = None
            settled = replace(
                _replace_endpoint_evidence(
                    self._evidence,
                    identity=identity,
                    endpoint=evidence.endpoint,
                    model_ids=evidence.model_ids,
                    category=evidence.category,
                ),
                # A rebound earlier observation keeps its own time.
                observed_at=evidence.observed_at or _local_now(),
            )
            self._evidence = settled
            self._advance_operation()
        if type(outcome) is not ProviderTestEvidence:
            self._publish(settled, token.order, generation=False)
        return True

    def begin_generation(self, identity: ProviderDraftIdentity) -> object:
        """Start an exact-identity generation probe and return its token.

        Args:
            identity: Exact provider draft identity that owns the probe.

        Returns:
            Opaque token required to settle or cancel this probe.

        Raises:
            ValueError: If the identity is invalid or older than current evidence.
        """

        if type(identity) is not ProviderDraftIdentity:
            raise ValueError("Provider draft identity is invalid.")
        with self._lock:
            if identity.draft_generation < self._latest_generation:
                raise ValueError("Draft generation is older than current evidence.")
            self._latest_generation = identity.draft_generation
            if self._owned_identity() not in {None, identity}:
                self._clear_all_test_state()
                self._advance_operation()
            if self._current_generation_token is None:
                evidence = self._evidence
                self._generation_cancel_restore = (
                    (
                        identity,
                        evidence.generation,
                        evidence.generation_category,
                    )
                    if evidence is not None
                    and evidence.identity == identity
                    and evidence.generation in {"succeeded", "failed"}
                    else None
                )
            self._advance_generation_operation()
            token = _ProviderGenerationTestToken()
            self._current_generation_token = token
            self._current_generation_token_epoch = self._generation_operation_epoch
            self._current_generation_identity = identity
            self._evidence = _replace_generation_evidence(
                self._evidence,
                identity=identity,
                generation="testing",
                category=None,
            )
            return token

    def settle_generation(self, token: object, outcome: object) -> bool:
        """Attach a bounded generation result only to its exact current draft."""

        supported_outcome = type(outcome) in {
            ProviderGenerationProbeResult,
            ProviderTestEvidence,
        }
        with self._lock:
            if (
                type(token) is not _ProviderGenerationTestToken
                or token is not self._current_generation_token
                or self._current_generation_token_epoch
                != self._generation_operation_epoch
            ):
                return False
            identity = self._current_generation_identity
            claim_epoch = self._current_generation_token_epoch
            self._current_generation_token = None
            self._current_generation_token_epoch = None
            self._current_generation_identity = None
            self._save_lease = None
            if identity is None or claim_epoch is None:
                self._clear_generation_evidence(identity)
                self._advance_generation_operation()
                return False
            settlement_claim = (identity, claim_epoch)
            self._generation_settling = settlement_claim

        if not supported_outcome:
            self._reject_generation_settlement(settlement_claim)
            return False
        try:
            generation, category, observed_at = _generation_evidence_from_exact_outcome(
                identity, outcome
            )
        except ValueError:
            self._reject_generation_settlement(settlement_claim)
            return False

        with self._lock:
            if (
                self._generation_operation_epoch != claim_epoch
                or self._generation_settling != settlement_claim
                or self._current_generation_token is not None
                or self._owned_identity() != identity
            ):
                return False
            self._generation_settling = None
            settled = replace(
                _replace_generation_evidence(
                    self._evidence,
                    identity=identity,
                    generation=generation,
                    category=category,
                ),
                observed_at=observed_at or _local_now(),
            )
            self._evidence = settled
            self._generation_cancel_restore = None
            self._advance_generation_operation()
        if type(outcome) is not ProviderTestEvidence:
            self._publish(settled, token.order, endpoint=False)
        return True

    def evidence_for(
        self, identity: ProviderDraftIdentity
    ) -> ProviderTestEvidence | None:
        """Return evidence only for the exact semantic identity supplied.

        A linked store falls back to another surface's settled result for the
        same connection, re-stamped with ``identity`` -- but never for an
        identity ``begin`` would refuse (TASK-33001.3's invariant).
        """

        if type(identity) is not ProviderDraftIdentity:
            return None
        with self._lock:
            if self._evidence is not None and self._evidence.identity == identity:
                return self._evidence
            if identity.draft_generation < self._latest_generation:
                return None
        shared = self._shared()
        return shared.evidence_for(identity) if shared is not None else None

    def latest_evidence(self) -> ProviderTestEvidence | None:
        """Return the latest bounded snapshot for changed-since-test projection."""

        with self._lock:
            return self._evidence

    def mark_generation_changed(self, identity: ProviderDraftIdentity) -> bool:
        """Mark settled generation evidence stale without touching endpoint work."""

        if type(identity) is not ProviderDraftIdentity:
            return False
        with self._lock:
            evidence = self._evidence
            if (
                evidence is None
                or evidence.identity != identity
                or evidence.generation not in {"succeeded", "failed"}
            ):
                return False
            self._evidence = _replace_generation_evidence(
                evidence,
                identity=identity,
                generation="changed_since_test",
                category=None,
            )
            self._save_lease = None
            self._generation_cancel_restore = None
            self._advance_generation_operation()
            return True

    def invalidate(self, identity: ProviderDraftIdentity | None = None) -> bool:
        """Remove exact or current evidence and cancel its active operation."""

        if identity is not None and type(identity) is not ProviderDraftIdentity:
            return False
        with self._lock:
            owned_identities = self._owned_identities()
            if not owned_identities:
                return False
            if identity is not None and identity not in owned_identities:
                return False
            self._clear_all_test_state()
            self._advance_operation()
            self._advance_generation_operation()
            return True

    def cancel_probe(self, token: object) -> bool:
        """Cancel only the active probe owned by ``token``."""

        with self._lock:
            if (
                type(token) is not _ProviderTestToken
                or token is not self._current_token
                or self._current_token_epoch != self._operation_epoch
            ):
                return False
            identity = self._current_identity
            self._current_token = None
            self._current_token_epoch = None
            self._current_identity = None
            self._settling = None
            self._clear_endpoint_evidence(identity)
            self._advance_operation()
            return True

    def cancel_generation_probe(self, token: object) -> bool:
        """Cancel only the active generation test owned by ``token``."""

        with self._lock:
            if (
                type(token) is not _ProviderGenerationTestToken
                or token is not self._current_generation_token
                or self._current_generation_token_epoch
                != self._generation_operation_epoch
            ):
                return False
            identity = self._current_generation_identity
            self._current_generation_token = None
            self._current_generation_token_epoch = None
            self._current_generation_identity = None
            self._generation_settling = None
            self._clear_generation_evidence(identity)
            restore = self._generation_cancel_restore
            self._generation_cancel_restore = None
            if restore is not None and restore[0] == identity:
                self._evidence = _replace_generation_evidence(
                    self._evidence,
                    identity=restore[0],
                    generation=restore[1],
                    category=restore[2],
                )
            self._advance_generation_operation()
            return True

    def begin_save(self, identity: ProviderDraftIdentity) -> object | None:
        """Capture a value-free lease for the store's current operation revision."""

        if type(identity) is not ProviderDraftIdentity:
            raise ValueError("Provider draft identity is invalid.")
        with self._lock:
            if self._owned_identity() != identity:
                return None
            lease = _ProviderEvidenceSaveLease()
            self._save_lease = (lease, identity, self._operation_epoch)
            return lease

    def cancel_save(self, lease: object) -> bool:
        """Consume the current exact save lease without changing test evidence."""

        if type(lease) is not _ProviderEvidenceSaveLease:
            return False
        with self._lock:
            if self._save_lease is None or lease is not self._save_lease[0]:
                return False
            self._save_lease = None
            return True

    def rebase_after_save(
        self,
        tested_identity: ProviderDraftIdentity,
        saved_identity: ProviderDraftIdentity,
        mutation_result: _MutationResult,
        *,
        lease: object,
    ) -> bool:
        """Rebind settled evidence after an equivalent, fully applied save."""

        tested_identity_is_valid = type(tested_identity) is ProviderDraftIdentity
        saved_identity_is_valid = type(saved_identity) is ProviderDraftIdentity
        mutation_flags = _mutation_flags(mutation_result)
        if type(lease) is not _ProviderEvidenceSaveLease:
            return False

        with self._lock:
            leased_operation = self._save_lease
            if leased_operation is None or lease is not leased_operation[0]:
                return False
            if not tested_identity_is_valid:
                return False
            _, leased_identity, leased_epoch = leased_operation
            if (
                leased_identity != tested_identity
                or leased_epoch != self._operation_epoch
                or self._owned_identity() != tested_identity
            ):
                return False
            self._save_lease = None
            if not saved_identity_is_valid or mutation_flags is None:
                return False
            conflict, fully_applied = mutation_flags

            evidence = self._evidence
            can_preserve = bool(
                not conflict
                and fully_applied
                and evidence is not None
                and evidence.identity == tested_identity
                and evidence.endpoint != "testing"
                and evidence.generation != "testing"
                and _same_saved_semantics(tested_identity, saved_identity)
                and saved_identity.draft_generation >= tested_identity.draft_generation
            )
            if fully_applied and not conflict:
                self._latest_generation = max(
                    self._latest_generation,
                    saved_identity.draft_generation,
                )

            self._clear_all_test_state()
            self._advance_generation_operation()
            if can_preserve and evidence is not None:
                self._evidence = replace(evidence, identity=saved_identity)
            self._advance_operation()
        shared = self._shared() if can_preserve else None
        if shared is not None:
            shared.carry(tested_identity, saved_identity)
        return can_preserve

    def _shared(self) -> ProviderConnectionEvidence | None:
        """Resolve (once) the app's shared owner; ``None`` keeps this draft-only."""

        if self._shared_owner is None and self._app is not None:
            try:
                self._shared_owner = provider_connection_evidence(self._app())
            except (AttributeError, RuntimeError):  # No app: bare screens, units.
                return None
        return self._shared_owner

    def _publish(
        self,
        evidence: ProviderTestEvidence,
        order: int,
        *,
        endpoint: bool = True,
        generation: bool = True,
    ) -> None:
        shared = self._shared()
        if shared is not None:
            shared.publish(
                evidence, order=order, endpoint=endpoint, generation=generation
            )

    def _reject_settlement(
        self,
        settlement_claim: tuple[ProviderDraftIdentity, int],
    ) -> None:
        with self._lock:
            if (
                self._settling == settlement_claim
                and self._operation_epoch == settlement_claim[1]
            ):
                self._settling = None
                self._clear_endpoint_evidence(settlement_claim[0])
                self._advance_operation()

    def _reject_generation_settlement(
        self,
        settlement_claim: tuple[ProviderDraftIdentity, int],
    ) -> None:
        with self._lock:
            if (
                self._generation_settling == settlement_claim
                and self._generation_operation_epoch == settlement_claim[1]
            ):
                self._generation_settling = None
                self._clear_generation_evidence(settlement_claim[0])
                self._advance_generation_operation()

    def _owned_identity(self) -> ProviderDraftIdentity | None:
        if self._settling is not None:
            return self._settling[0]
        if self._generation_settling is not None:
            return self._generation_settling[0]
        if self._evidence is not None:
            return self._evidence.identity
        if self._current_generation_identity is not None:
            return self._current_generation_identity
        return self._current_identity

    def _owned_identities(self) -> set[ProviderDraftIdentity]:
        identities: set[ProviderDraftIdentity] = set()
        if self._settling is not None:
            identities.add(self._settling[0])
        if self._generation_settling is not None:
            identities.add(self._generation_settling[0])
        if self._evidence is not None:
            identities.add(self._evidence.identity)
        if self._current_identity is not None:
            identities.add(self._current_identity)
        if self._current_generation_identity is not None:
            identities.add(self._current_generation_identity)
        return identities

    def _clear_endpoint_evidence(self, identity: ProviderDraftIdentity | None) -> None:
        evidence = self._evidence
        if evidence is None or identity is None or evidence.identity != identity:
            return
        if evidence.generation == "not_tested":
            self._evidence = None
            return
        self._evidence = _replace_endpoint_evidence(
            evidence,
            identity=identity,
            endpoint="not_tested",
            model_ids=(),
            category=None,
        )

    def _clear_generation_evidence(
        self, identity: ProviderDraftIdentity | None
    ) -> None:
        evidence = self._evidence
        if evidence is None or identity is None or evidence.identity != identity:
            return
        if evidence.endpoint == "not_tested":
            self._evidence = None
            return
        self._evidence = _replace_generation_evidence(
            evidence,
            identity=identity,
            generation="not_tested",
            category=None,
        )

    def _clear_all_test_state(self) -> None:
        self._evidence = None
        self._current_token = None
        self._current_token_epoch = None
        self._current_identity = None
        self._settling = None
        self._current_generation_token = None
        self._current_generation_token_epoch = None
        self._current_generation_identity = None
        self._generation_settling = None
        self._generation_cancel_restore = None

    def _advance_operation(self) -> None:
        self._operation_epoch += 1
        self._settling = None
        self._save_lease = None

    def _advance_generation_operation(self) -> None:
        self._generation_operation_epoch += 1
        self._generation_settling = None
        self._save_lease = None


#: Per-process salt: a revision means nothing outside this process.
_CREDENTIAL_REVISION_SALT = secrets.token_bytes(32)
_OWNER_ATTRIBUTE = "_provider_connection_evidence"
_OWNER_CREATION_LOCK = Lock()


def connection_credential_revision(credential: object) -> int:
    """Return the secret-free revision of one resolved provider credential.

    TASK-33005.1: every surface stamps this into
    ``ProviderDraftIdentity.credential_revision`` -- computed from the key a
    send would use (``ProviderReadiness.api_key`` for saved state, the typed
    value for a draft) -- so one saved credential compares equal in Chat
    settings, Settings and the Console, and changing it changes the
    connection. A salted HMAC, so the number never reveals the key.

    Args:
        credential: The resolved credential, or ``None`` when none is sent.

    Returns:
        ``0`` when there is no credential, else a positive 56-bit integer.
    """

    if type(credential) is not str or not credential:
        return 0
    digest = hmac.new(
        _CREDENTIAL_REVISION_SALT, credential.encode("utf-8"), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:7], "big") + 1


class ProviderConnectionEvidence:
    """Process-memory settled evidence, one record per provider connection.

    TASK-33005.1 / ADR-033: a narrow owner attached to the app (see
    :func:`provider_connection_evidence`), never persisted. A connection is a
    ``ProviderDraftIdentity`` without its surface-local ``draft_generation``:
    provider, custom endpoint id, canonical endpoint, credential source and
    :func:`connection_credential_revision`. Surfaces publish observed facts
    through their draft stores; the Console reads here directly. A reader
    only ever gets its own connection's record: an unsaved draft's result
    stays with that draft, and a changed connection reads as untested here
    (the surface still holding the older result marks it changed since test).
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # ponytail: one record per connection ever tested this process; it
        # only grows with explicit tests, so no eviction until that matters.
        self._records: dict[tuple[object, ...], ProviderTestEvidence] = {}
        #: Per connection: begin order of the endpoint and generation facts held.
        self._orders: dict[tuple[object, ...], tuple[int, int]] = {}
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic count of record changes, for cheap change polling."""

        return self._version

    def publish(
        self,
        evidence: ProviderTestEvidence,
        *,
        order: int,
        endpoint: bool = True,
        generation: bool = True,
    ) -> bool:
        """Merge the observed facets of ``evidence`` into its connection's record.

        Only terminal facts move (a ``testing`` or changed-since-test marker is
        a surface's draft state). Each facet keeps the observation whose probe
        began last, whenever it settled.

        Args:
            evidence: The settled evidence.
            order: Begin order of the probe that observed it (its token's).
            endpoint: Whether to take the endpoint facet.
            generation: Whether to take the generation facet.

        Returns:
            Whether the record changed.
        """

        if type(evidence) is not ProviderTestEvidence or type(order) is not int:
            return False
        take_endpoint = endpoint and evidence.endpoint in _PROBE_ENDPOINT_FACETS
        take_generation = generation and evidence.generation in {"succeeded", "failed"}
        with self._lock:
            return self._merge(
                evidence.identity,
                evidence,
                order if take_endpoint else 0,
                order if take_generation else 0,
            )

    def carry(
        self, tested: ProviderDraftIdentity, saved: ProviderDraftIdentity
    ) -> bool:
        """Give the connection a Save produced the tested connection's facts.

        Called only after a save that wrote exactly the tested values (the
        draft store's rebase), e.g. a typed key that is now the stored key.
        Each fact keeps its begin order, so a newer result is never replaced.

        Returns:
            Whether the saved connection's record changed.
        """

        if {type(tested), type(saved)} != {ProviderDraftIdentity}:
            return False
        with self._lock:
            key = _connection_key(tested)
            record = self._records.get(key)
            if record is None:
                return False
            return self._merge(saved, record, *self._orders[key])

    def evidence_for(
        self, identity: ProviderDraftIdentity
    ) -> ProviderTestEvidence | None:
        """Return ``identity``'s connection record re-stamped with it, else ``None``."""

        if type(identity) is not ProviderDraftIdentity:
            return None
        with self._lock:
            record = self._records.get(_connection_key(identity))
        return None if record is None else replace(record, identity=identity)

    def _merge(
        self,
        identity: ProviderDraftIdentity,
        evidence: ProviderTestEvidence,
        endpoint_order: int,
        generation_order: int,
    ) -> bool:
        """Take each offered facet (order > 0) that began after the held one."""

        key = _connection_key(identity)
        record = self._records.get(key)
        # The record keeps its first identity, so both facets stay one record.
        identity = (
            record.identity if record else replace(identity, draft_generation=0)
        )
        held_endpoint, held_generation = self._orders.get(key, (0, 0))
        merged = record
        if endpoint_order > held_endpoint:
            held_endpoint = endpoint_order
            merged = _replace_endpoint_evidence(
                merged,
                identity=identity,
                endpoint=evidence.endpoint,
                model_ids=evidence.model_ids,
                category=evidence.category,
            )
        if generation_order > held_generation:
            held_generation = generation_order
            merged = _replace_generation_evidence(
                merged,
                identity=identity,
                generation=evidence.generation,
                category=evidence.generation_category,
            )
        if merged is record:
            return False
        self._orders[key] = (held_endpoint, held_generation)
        merged = replace(merged, observed_at=evidence.observed_at or _local_now())
        if merged == record and merged.observed_at == record.observed_at:
            return False  # Same facts at the same time: nothing for readers.
        self._records[key] = merged
        self._version += 1
        return True


def provider_connection_evidence(owner: object) -> ProviderConnectionEvidence:
    """Return the one connection evidence owner attached to an app process.

    Mirrors ``process_provider_test_evidence_store`` (TTS): a private
    attribute on the running app, so a restart starts with nothing tested.
    """

    existing = getattr(owner, _OWNER_ATTRIBUTE, None)
    if type(existing) is ProviderConnectionEvidence:
        return existing
    with _OWNER_CREATION_LOCK:
        existing = getattr(owner, _OWNER_ATTRIBUTE, None)
        if type(existing) is not ProviderConnectionEvidence:
            existing = ProviderConnectionEvidence()
            setattr(owner, _OWNER_ATTRIBUTE, existing)
        return existing


def _connection_key(identity: ProviderDraftIdentity) -> tuple[object, ...]:
    return (
        identity.provider_key,
        identity.custom_endpoint_id,
        identity.connection_identity,
        # Revision 0 sends no key, whichever source names it: Chat settings
        # reads an explicit credential_source, Settings what resolves, and
        # they differ only when nothing does (e.g. "stored" with no key).
        identity.credential_source if identity.credential_revision else "none",
        identity.credential_revision,
    )


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _validate_model_ids(model_ids: object) -> None:
    if type(model_ids) is not tuple:
        raise ValueError("Model IDs must be a tuple.")
    if len(model_ids) > _MAX_MODEL_IDS:
        raise ValueError("Too many model IDs.")
    seen: set[str] = set()
    for model_id in model_ids:
        if type(model_id) is not str:
            raise ValueError("Model ID is invalid.")
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


def _validate_probe_result(
    endpoint: object,
    model_ids: object,
    category: object,
) -> None:
    if type(endpoint) is not str or endpoint not in _PROBE_ENDPOINT_FACETS:
        raise ValueError("Provider probe endpoint facet is invalid.")
    _validate_model_ids(model_ids)
    if category is not None and (
        type(category) is not str or category not in _FAILURE_CATEGORIES
    ):
        raise ValueError("Provider probe category is invalid.")
    if endpoint == "reachable":
        if category is not None:
            raise ValueError("Reachable probe result cannot include a category.")
    elif endpoint == "unreachable":
        if model_ids:
            raise ValueError("Unreachable probe result cannot include model IDs.")
    elif model_ids or category not in {None, "http_status"}:
        raise ValueError("Model-listing probe result is inconsistent.")


def _validate_generation_result(generation: object, category: object) -> None:
    if type(generation) is not str or generation not in {"succeeded", "failed"}:
        raise ValueError("Provider generation result facet is invalid.")
    if category is not None and (
        type(category) is not str or category not in _GENERATION_FAILURE_CATEGORIES
    ):
        raise ValueError("Provider generation result category is invalid.")
    if generation == "failed" and category is None:
        raise ValueError("Failed provider generation requires a category.")
    if generation == "succeeded" and category is not None:
        raise ValueError("Successful provider generation cannot include a category.")


def _normalize_generation_credential(
    credential: CredentialFacet,
    generation: GenerationFacet,
    category: GenerationFailureCategory | None,
    *,
    credential_required: bool,
) -> CredentialFacet:
    """Return credential evidence consistent with the observed generation."""

    if generation == "succeeded" and credential == "missing":
        raise ValueError("Successful generation cannot have a missing credential.")
    if not credential_required:
        return "not_required"
    if generation == "succeeded":
        return "authenticated"
    if generation == "failed" and category == "authentication":
        return "present_unverified"
    if credential == "not_required":
        return "present_unverified"
    return credential


def _evidence_from_exact_outcome(
    identity: ProviderDraftIdentity,
    outcome: object,
) -> ProviderTestEvidence:
    if type(outcome) is ProviderTestEvidence:
        if outcome.identity != identity:
            raise ValueError("Probe evidence identity does not match the token.")
        if outcome.endpoint not in _PROBE_ENDPOINT_FACETS:
            raise ValueError("Probe evidence is not a terminal result.")
        return outcome
    if type(outcome) is not ProviderProbeResult:
        raise ValueError("Provider probe result type is invalid.")
    return ProviderTestEvidence(
        identity,
        outcome.endpoint,
        outcome.model_ids,
        outcome.category,
    )


def _generation_evidence_from_exact_outcome(
    identity: ProviderDraftIdentity,
    outcome: object,
) -> tuple[
    Literal["succeeded", "failed"],
    GenerationFailureCategory | None,
    datetime | None,
]:
    if type(outcome) is ProviderTestEvidence:
        # A rebound earlier result: same draft, and it keeps its own time.
        if outcome.identity != identity:
            raise ValueError("Generation evidence identity does not match the token.")
        _validate_generation_result(outcome.generation, outcome.generation_category)
        return outcome.generation, outcome.generation_category, outcome.observed_at
    if type(outcome) is not ProviderGenerationProbeResult:
        raise ValueError("Provider generation result type is invalid.")
    _validate_generation_result(outcome.generation, outcome.category)
    return outcome.generation, outcome.category, None


def _replace_endpoint_evidence(
    evidence: ProviderTestEvidence | None,
    *,
    identity: ProviderDraftIdentity,
    endpoint: EndpointFacet,
    model_ids: tuple[str, ...],
    category: EndpointFailureCategory | None,
) -> ProviderTestEvidence:
    if evidence is None or evidence.identity != identity:
        return ProviderTestEvidence(identity, endpoint, model_ids, category)
    return replace(evidence, endpoint=endpoint, model_ids=model_ids, category=category)


def _replace_generation_evidence(
    evidence: ProviderTestEvidence | None,
    *,
    identity: ProviderDraftIdentity,
    generation: GenerationFacet,
    category: GenerationFailureCategory | None,
) -> ProviderTestEvidence:
    if evidence is None or evidence.identity != identity:
        return ProviderTestEvidence(
            identity,
            "not_tested",
            (),
            credential=(
                "not_required"
                if identity.credential_source == "none"
                else "present_unverified"
            ),
            generation=generation,
            generation_category=category,
        )
    return replace(evidence, generation=generation, generation_category=category)


def _mutation_flags(mutation_result: object) -> tuple[bool, bool] | None:
    try:
        conflict = mutation_result.conflict  # type: ignore[attr-defined]
        fully_applied = mutation_result.fully_applied  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - callback inputs fail closed before locking.
        return None
    if type(conflict) is not bool or type(fully_applied) is not bool:
        return None
    return conflict, fully_applied


def _same_saved_semantics(
    first: ProviderDraftIdentity, second: ProviderDraftIdentity
) -> bool:
    credential_source_matches = first.credential_source == second.credential_source or (
        first.credential_source == "draft" and second.credential_source == "stored"
    )
    return (
        first.provider_key == second.provider_key
        and first.custom_endpoint_id == second.custom_endpoint_id
        and first.connection_identity == second.connection_identity
        and credential_source_matches
        and first.credential_revision == second.credential_revision
    )
