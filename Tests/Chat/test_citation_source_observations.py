from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json

import pytest
from pydantic import ValidationError

from Tests.Chat.test_citation_trace_repository import (
    _identity,
    _persist,
    _repository,
    _sealed_write,
)
from tldw_chatbook.Chat import citation_source_locators as locators
from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CanonicalSourceKind,
    CitationReadAuthorization,
    SourceCapability,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    LocalCitationIdentityContext,
    local_trace_namespace,
)
from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttempt,
    AnswerAttemptKind,
    AnswerAttemptPayload,
    CitationOccurrence,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    MarkerNamespace,
    PolicyCapability,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    SealedCitationWrite,
    StructuralValidationState,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 15, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        tmp_path / "citation-source-observations.sqlite",
        client_id="citation-source-observation-test",
    )
    yield database
    database.close_connection()


def _observation(
    *,
    observed_at: datetime = NOW,
    request_generation: int = 1,
    request_nonce: str = "nonce-1",
    availability: str = "available",
    permission: str = "allowed",
    content_state: str = "unchanged",
    location_state: str = "unchanged",
    capabilities: tuple[SourceCapability, ...] = (
        SourceCapability.RESOLVE_CURRENT,
        SourceCapability.OPEN_NATIVE,
        SourceCapability.COMPARE,
        SourceCapability.REFRESH_OBSERVATION,
    ),
    error_code: str | None = None,
    resolver_kind: CanonicalSourceKind = CanonicalSourceKind.MEDIA_DB,
    resolver_version: str = "1",
):
    return locators.CitationSourceObservation(
        resolver_kind=resolver_kind,
        resolver_version=resolver_version,
        availability=locators.CitationSourceAvailability(availability),
        permission=locators.CitationSourcePermission(permission),
        content_state=locators.CitationContentState(content_state),
        location_state=locators.CitationLocationState(location_state),
        capabilities=capabilities,
        observed_at=observed_at,
        request_generation=request_generation,
        request_nonce=request_nonce,
        error_code=error_code,
    )


def _observation_authorization(
    identity: LocalCitationIdentityContext,
    *,
    refresh_observation: bool = True,
    resolve_current: bool = True,
    open_native: bool = True,
    open_external: bool = False,
    compare: bool = True,
) -> CitationReadAuthorization:
    return CitationReadAuthorization(
        authority_scope=AuthorityScope.LOCAL_PROFILE,
        profile_id=identity.profile_id,
        governance_scope_id=identity.profile_id,
        allowlisted_authority_ids=(identity.local_authority_id,),
        refresh_observation=refresh_observation,
        resolve_current=resolve_current,
        open_native=open_native,
        open_external=open_external,
        compare=compare,
    )


def _observation_write() -> SealedCitationWrite:
    base = _sealed_write()
    return base.model_copy(
        update={
            "trace": base.trace.model_copy(
                update={
                    "policy_capabilities": (
                        *base.trace.policy_capabilities,
                        PolicyCapability.RESOLVE_CURRENT_SOURCE,
                        PolicyCapability.OPEN_NATIVE,
                        PolicyCapability.COMPARE_CURRENT_SOURCE,
                    )
                }
            )
        }
    )


def _rerun_write() -> SealedCitationWrite:
    base = _observation_write()
    run = base.trace.evidence_runs[0]
    second_prompt = PromptEvidenceSet(
        prompt_set_id="prompt-2",
        prompt_set_ordinal=2,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        entries=(
            PromptEvidenceEntry(
                evidence_ordinal=1,
                marker_ordinal=1,
                run_id=run.run_id,
                snapshot_payload_ref="snapshot-2",
                storage_mode=EvidenceStorageMode.EMBEDDED,
            ),
        ),
        created_at=NOW,
    )
    second_attempt = AnswerAttempt(
        attempt_id="attempt-2",
        attempt_ordinal=2,
        kind=AnswerAttemptKind.PIPELINE_RERUN,
        prompt_evidence_set_id=second_prompt.prompt_set_id,
        answer_payload_ref="answer-payload-2",
        occurrences=(
            CitationOccurrence(
                occurrence_id="occurrence-2",
                occurrence_ordinal=1,
                raw_marker="[S1]",
                marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
                evidence_ordinal=1,
                marker_start=7,
                marker_end=11,
                claim_start=0,
                claim_end=6,
                structural_state=StructuralValidationState.VALID,
            ),
        ),
        created_at=NOW,
    )
    second_snapshot = EvidenceSnapshotPayload(
        payload_id="snapshot-2",
        storage_mode=EvidenceStorageMode.EMBEDDED,
        snapshot_text="private exact rerun evidence",
        title="private rerun source title",
        source_identity={"document_id": "private-document-2"},
        locator={"source_kind": "media_db", "item_id": "private-item-2"},
        lineage={"chunk_id": "private-chunk-2"},
        transformations=("truncate",),
        content_hash="content-hmac-2",
        comparison_hash="comparison-hmac-2",
    )
    second_answer = AnswerAttemptPayload(
        payload_id="answer-payload-2",
        attempt_id=second_attempt.attempt_id,
        answer_body="Answer [S1].",
        body_integrity_hmac=base.answer_attempt_payloads[0].body_integrity_hmac,
    )
    return SealedCitationWrite(
        trace=base.trace.model_copy(
            update={
                "prompt_evidence_sets": (
                    *base.trace.prompt_evidence_sets,
                    second_prompt,
                ),
                "answer_attempts": (
                    *base.trace.answer_attempts,
                    second_attempt,
                ),
                "selected_attempt_id": second_attempt.attempt_id,
            }
        ),
        evidence_run_payloads=base.evidence_run_payloads,
        evidence_snapshot_payloads=(
            *base.evidence_snapshot_payloads,
            second_snapshot,
        ),
        answer_attempt_payloads=(
            *base.answer_attempt_payloads,
            second_answer,
        ),
    )


def _upsert(
    db: CharactersRAGDB,
    repository: CitationTraceRepository,
    observation,
    *,
    prompt_set_id: str = "prompt-1",
    evidence_ordinal: int = 1,
    snapshot_payload_id: str = "snapshot-1",
    authorization: CitationReadAuthorization | None = None,
):
    identity = _identity(db)
    with db.transaction() as cursor:
        return repository.upsert_source_observation(
            cursor,
            local_trace_namespace(identity, trace_id="trace-1"),
            prompt_set_id=prompt_set_id,
            evidence_ordinal=evidence_ordinal,
            snapshot_payload_id=snapshot_payload_id,
            observation=observation,
            authorization=authorization or _observation_authorization(identity),
        )


def _read(
    db: CharactersRAGDB,
    repository: CitationTraceRepository,
    *,
    prompt_set_id: str = "prompt-1",
    evidence_ordinal: int = 1,
    snapshot_payload_id: str = "snapshot-1",
    resolver_kind: CanonicalSourceKind = CanonicalSourceKind.MEDIA_DB,
    resolver_version: str = "1",
    authorization: CitationReadAuthorization | None = None,
):
    identity = _identity(db)
    return repository.get_source_observation(
        local_trace_namespace(identity, trace_id="trace-1"),
        prompt_set_id=prompt_set_id,
        evidence_ordinal=evidence_ordinal,
        snapshot_payload_id=snapshot_payload_id,
        resolver_kind=resolver_kind,
        resolver_version=resolver_version,
        authorization=authorization or _observation_authorization(identity),
    )


def _sealed_state(db: CharactersRAGDB) -> dict[str, tuple[tuple[object, ...], ...]]:
    connection = db.get_connection()
    queries = {
        "trace": """
            SELECT aggregate_json, completeness_at_seal, selected_attempt_id
            FROM rag_citation_traces ORDER BY trace_id
        """,
        "refs": """
            SELECT prompt_set_id, evidence_ordinal, run_id, snapshot_payload_id,
                   marker_ordinal, storage_mode
            FROM rag_trace_evidence_refs
            ORDER BY prompt_set_id, evidence_ordinal
        """,
        "snapshots": """
            SELECT payload_id, redaction_state, snapshot_text, title,
                   source_identity_json, locator_json, lineage_json,
                   transformations_json, content_hash, comparison_fingerprint,
                   purged_at
            FROM rag_evidence_snapshots ORDER BY payload_id
        """,
        "tombstones": """
            SELECT profile_id, origin_namespace, origin_payload_id,
                   revocation_scope_id, reason_code, policy_version,
                   revoked_at, retain_until
            FROM rag_payload_tombstones ORDER BY origin_payload_id
        """,
    }
    return {
        name: tuple(tuple(row) for row in connection.execute(sql).fetchall())
        for name, sql in queries.items()
    }


def test_observation_contract_exists_and_is_strict_frozen() -> None:
    assert hasattr(locators, "CitationSourceObservation")
    observation = _observation()
    with pytest.raises(ValidationError, match="frozen"):
        observation.availability = "missing"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        locators.CitationSourceObservation.model_validate(
            {**observation.model_dump(), "unexpected": True},
            strict=True,
        )


@pytest.mark.parametrize(
    ("availability", "permission", "content_state", "location_state", "error_code"),
    [
        ("available", "allowed", "unchanged", "unchanged", None),
        ("available", "allowed", "changed", "relocated", None),
        ("missing", "unknown", "unknown", "missing", None),
        ("offline", "unknown", "unknown", "unknown", "network_timeout"),
        ("error", "unknown", "unknown", "unknown", None),
        ("unknown", "revoked", "unknown", "unknown", None),
        (
            "unknown",
            "authentication_required",
            "unknown",
            "unknown",
            "authentication_expired",
        ),
        ("available", "allowed", "unknown", "ambiguous", None),
        ("unknown", "unknown", "unknown", "unknown", None),
    ],
)
def test_independent_observation_states_round_trip(
    availability: str,
    permission: str,
    content_state: str,
    location_state: str,
    error_code: str | None,
) -> None:
    capabilities = (
        (
            SourceCapability.RESOLVE_CURRENT,
            SourceCapability.REFRESH_OBSERVATION,
        )
        if availability == "available" and permission == "allowed"
        else ()
    )
    observation = _observation(
        availability=availability,
        permission=permission,
        content_state=content_state,
        location_state=location_state,
        capabilities=capabilities,
        error_code=error_code,
    )

    assert (
        locators.CitationSourceObservation.model_validate_json(
            observation.model_dump_json()
        )
        == observation
    )


@pytest.mark.parametrize(
    ("permission", "capabilities", "location_state"),
    [
        ("allowed", (), "relocated"),
        ("denied", (SourceCapability.RESOLVE_CURRENT,), "unknown"),
        ("revoked", (SourceCapability.REFRESH_OBSERVATION,), "unknown"),
    ],
)
def test_non_definitive_status_capabilities_remain_independent(
    permission: str,
    capabilities: tuple[SourceCapability, ...],
    location_state: str,
) -> None:
    observation = _observation(
        availability="available",
        permission=permission,
        content_state="unknown",
        location_state=location_state,
        capabilities=capabilities,
        error_code="safe_status",
    )

    assert observation.capabilities == capabilities
    assert observation.error_code == "safe_status"


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": datetime(2026, 7, 24, 15, 0)},
        {
            "permission": "revoked",
            "capabilities": (SourceCapability.OPEN_NATIVE,),
        },
        {
            "permission": "denied",
            "capabilities": (SourceCapability.COMPARE,),
        },
        {
            "availability": "available",
            "permission": "allowed",
            "location_state": "ambiguous",
            "capabilities": (SourceCapability.OPEN_NATIVE,),
        },
        {
            "availability": "offline",
            "permission": "unknown",
            "content_state": "unknown",
            "location_state": "unknown",
            "capabilities": (SourceCapability.OPEN_NATIVE,),
        },
        {
            "availability": "unknown",
            "permission": "authentication_required",
            "content_state": "unknown",
            "location_state": "unknown",
            "capabilities": (SourceCapability.COMPARE,),
        },
        {"capabilities": (SourceCapability.RESOLVE_CURRENT,) * 2},
        {"request_generation": -1},
    ],
)
def test_contradictory_or_malformed_observations_are_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _observation(**changes)


def test_observation_exact_json_error_and_identifier_bounds() -> None:
    exact_error = _observation(
        availability="error",
        permission="unknown",
        content_state="unknown",
        location_state="unknown",
        capabilities=(),
        error_code="e" * 256,
        resolver_version="v" * 256,
        request_nonce="n" * 256,
    )
    assert exact_error.error_code == "e" * 256
    with pytest.raises(ValidationError):
        _observation(
            availability="error",
            permission="unknown",
            content_state="unknown",
            location_state="unknown",
            capabilities=(),
            error_code="e" * 257,
        )
    with pytest.raises(ValidationError):
        _observation(error_code="unsafe\nstatus")
    with pytest.raises(ValidationError):
        _observation(resolver_version="é" * 129)
    with pytest.raises(ValidationError):
        _observation(request_nonce="é" * 129)

    exact_json = json.dumps("x" * (8 * 1024 - 2))
    assert len(exact_json.encode("utf-8")) == 8 * 1024
    assert locators.validate_source_observation_json_size(exact_json) == exact_json
    with pytest.raises(ValueError, match="8192"):
        locators.validate_source_observation_json_size(exact_json + "x")


def test_repository_replaces_only_newer_and_keeps_one_row(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    assert _upsert(db, repository, _observation()).value == "inserted"
    newer = _observation(
        observed_at=NOW + timedelta(seconds=1),
        request_generation=2,
        request_nonce="nonce-2",
        content_state="changed",
    )
    assert _upsert(db, repository, newer).value == "replaced"
    assert _read(db, repository) == newer
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_source_observations")
        .fetchone()[0]
        == 1
    )


def test_repository_rejects_stale_and_handles_equal_time_tokens_deterministically(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    current = _observation(request_generation=2, request_nonce="nonce-b")
    assert _upsert(db, repository, current).value == "inserted"

    older = _observation(
        observed_at=NOW - timedelta(seconds=1),
        request_generation=99,
        request_nonce="nonce-z",
        content_state="changed",
    )
    assert _upsert(db, repository, older).value == "stale"
    lower_generation = _observation(
        request_generation=1,
        request_nonce="nonce-z",
        content_state="changed",
    )
    assert _upsert(db, repository, lower_generation).value == "stale"
    lower_nonce = _observation(
        request_generation=2,
        request_nonce="nonce-a",
        content_state="changed",
    )
    assert _upsert(db, repository, lower_nonce).value == "stale"
    higher_generation = _observation(
        request_generation=3,
        request_nonce="nonce-a",
        content_state="changed",
    )
    assert _upsert(db, repository, higher_generation).value == "replaced"
    higher_nonce = _observation(
        request_generation=3,
        request_nonce="nonce-z",
        content_state="unchanged",
    )
    assert _upsert(db, repository, higher_nonce).value == "replaced"
    assert _read(db, repository) == higher_nonce


def test_repository_equal_token_is_idempotent_or_conflicting(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    observation = _observation()
    assert _upsert(db, repository, observation).value == "inserted"
    assert _upsert(db, repository, observation).value == "idempotent"

    conflicting = _observation(content_state="changed")
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_nonce_conflict",
    ):
        _upsert(db, repository, conflicting)
    assert _read(db, repository) == observation


def test_prompt_reruns_with_same_ordinal_and_different_snapshots_do_not_collide(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_rerun_write())
    first = _observation(request_nonce="nonce-first", content_state="unchanged")
    second = _observation(request_nonce="nonce-second", content_state="changed")

    assert _upsert(db, repository, first).value == "inserted"
    assert (
        _upsert(
            db,
            repository,
            second,
            prompt_set_id="prompt-2",
            snapshot_payload_id="snapshot-2",
        ).value
        == "inserted"
    )
    assert _read(db, repository) == first
    assert (
        _read(
            db,
            repository,
            prompt_set_id="prompt-2",
            snapshot_payload_id="snapshot-2",
        )
        == second
    )


@pytest.mark.parametrize(
    ("prompt_set_id", "evidence_ordinal", "snapshot_payload_id"),
    [
        ("missing-prompt", 1, "snapshot-1"),
        ("prompt-1", 99, "snapshot-1"),
        ("prompt-1", 1, "missing-snapshot"),
        ("prompt-1", 1, "snapshot-2"),
    ],
)
def test_observation_key_must_match_one_exact_trace_reference(
    db: CharactersRAGDB,
    prompt_set_id: str,
    evidence_ordinal: int,
    snapshot_payload_id: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_rerun_write())

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_reference_mismatch",
    ):
        _upsert(
            db,
            repository,
            _observation(),
            prompt_set_id=prompt_set_id,
            evidence_ordinal=evidence_ordinal,
            snapshot_payload_id=snapshot_payload_id,
        )


def test_observation_validates_namespace_inventory_trace_policy_and_authorization(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    identity = _identity(db)
    wrong_namespace = local_trace_namespace(
        LocalCitationIdentityContext(
            profile_id="other-profile",
            local_authority_id="other-authority",
            fingerprint_key_id=identity.fingerprint_key_id,
        ),
        trace_id="trace-1",
    )
    authorization = _observation_authorization(identity)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_namespace_mismatch",
    ):
        with db.transaction() as cursor:
            repository.upsert_source_observation(
                cursor,
                wrong_namespace,
                prompt_set_id="prompt-1",
                evidence_ordinal=1,
                snapshot_payload_id="snapshot-1",
                observation=_observation(),
                authorization=authorization,
            )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_resolver_unsupported",
    ):
        _upsert(
            db,
            repository,
            _observation(resolver_kind=CanonicalSourceKind.WEB_CONTENT),
        )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_resolver_unsupported",
    ):
        _upsert(db, repository, _observation(resolver_version="2"))
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_authorization_denied",
    ):
        _upsert(
            db,
            repository,
            _observation(),
            authorization=_observation_authorization(
                identity,
                refresh_observation=False,
            ),
        )

    other_db = CharactersRAGDB(
        db.db_path_str + ".other",
        client_id="citation-observation-other",
    )
    try:
        _persist(other_db, _repository(other_db), sealed_write=_sealed_write())
        with pytest.raises(
            CitationPersistenceUnavailable,
            match="source_observation_trace_policy_denied",
        ):
            _upsert(other_db, _repository(other_db), _observation())
        with other_db.transaction() as foreign_cursor:
            with pytest.raises(RuntimeError, match="repository database"):
                repository.upsert_source_observation(
                    foreign_cursor,
                    local_trace_namespace(identity, trace_id="trace-1"),
                    prompt_set_id="prompt-1",
                    evidence_ordinal=1,
                    snapshot_payload_id="snapshot-1",
                    observation=_observation(),
                    authorization=authorization,
                )
    finally:
        other_db.close_connection()


def test_observation_capabilities_cannot_exceed_inventory_or_authorization(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    identity = _identity(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_capability_denied",
    ):
        _upsert(
            db,
            repository,
            _observation(
                capabilities=(
                    SourceCapability.RESOLVE_CURRENT,
                    SourceCapability.OPEN_EXTERNAL,
                )
            ),
            authorization=_observation_authorization(
                identity,
                open_external=True,
            ),
        )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_capability_denied",
    ):
        _upsert(
            db,
            repository,
            _observation(),
            authorization=_observation_authorization(identity, compare=False),
        )


def test_observation_write_does_not_mutate_sealed_or_governed_state(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    before = _sealed_state(db)

    assert _upsert(db, repository, _observation()).value == "inserted"
    assert (
        _upsert(
            db,
            repository,
            _observation(
                observed_at=NOW + timedelta(seconds=1),
                request_generation=2,
                request_nonce="nonce-2",
                content_state="changed",
            ),
        ).value
        == "replaced"
    )
    assert _sealed_state(db) == before


def test_revoked_evidence_accepts_safe_observation_without_hydrating_content(
    db: CharactersRAGDB,
) -> None:
    from Tests.Chat.test_citation_payload_lifecycle import _lifecycle, _tombstone

    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    identity = _identity(db)
    _lifecycle(db).revoke(
        local_trace_namespace(identity, trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )
    revoked = _observation(
        availability="unknown",
        permission="revoked",
        content_state="unknown",
        location_state="unknown",
        capabilities=(),
        request_nonce="revoked-result",
    )

    assert _upsert(db, repository, revoked).value == "inserted"
    assert _read(db, repository) == revoked
    snapshot = (
        db.get_connection()
        .execute(
            """
        SELECT snapshot_text, title, source_identity_json, locator_json,
               content_hash, comparison_fingerprint
        FROM rag_evidence_snapshots WHERE payload_id = 'snapshot-1'
        """
        )
        .fetchone()
    )
    assert tuple(snapshot) == (None, None, None, None, None, None)


def test_safe_observation_read_is_keyless_but_still_authorized(
    db: CharactersRAGDB,
) -> None:
    writer = _repository(db)
    _persist(db, writer, sealed_write=_observation_write())
    observation = _observation()
    _upsert(db, writer, observation)
    identity = _identity(db)
    reader = CitationTraceRepository(
        db,
        policy=writer.policy,
        identity_context=identity,
        fingerprint_codec=None,
    )

    assert _read(db, reader) == observation
    wrong_authorization = CitationReadAuthorization(
        authority_scope=AuthorityScope.LOCAL_PROFILE,
        profile_id="other-profile",
        governance_scope_id="other-profile",
        allowlisted_authority_ids=(identity.local_authority_id,),
    )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="source_observation_authorization_denied",
    ):
        _read(db, reader, authorization=wrong_authorization)


def test_concurrent_equal_time_upserts_converge_to_deterministic_latest(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository, sealed_write=_observation_write())
    candidates = tuple(
        _observation(
            request_generation=4,
            request_nonce=f"nonce-{letter}",
            content_state="changed" if letter > "m" else "unchanged",
        )
        for letter in ("a", "z", "m", "q")
    )

    def write(candidate):
        try:
            return _upsert(db, repository, candidate)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        outcomes = tuple(executor.map(write, candidates))

    assert {outcome.value for outcome in outcomes} <= {
        "inserted",
        "replaced",
        "stale",
    }
    assert _read(db, repository) == max(
        candidates,
        key=lambda item: (
            item.observed_at,
            item.request_generation,
            item.request_nonce,
        ),
    )
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_source_observations")
        .fetchone()[0]
        == 1
    )
