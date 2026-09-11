"""Fake-software coverage for frozen speculative trace privacy."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_redaction import (
    BUILTIN_PII_RULESET_REVISION_ID,
    CREDENTIAL_FILTER_VERSION,
)
from tldw_chatbook.Chat.console_trace_custom_pii import CUSTOM_PII_RULESET_UNAVAILABLE
from tldw_chatbook.Chat.console_voice_trace_gateway import _artifact
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_trace_message,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_voice_trace_repository import _post_dispatch_import
from Tests.Chat.test_console_voice_trace_repository import (
    _completed_pair,
    _seal_post_dispatch_import,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    PostDispatchTraceHeaderComponent,
    PostDispatchTraceArtifact,
)
from tldw_chatbook.Chat.console_trace_repository import TraceIdentityConflict
from Tests.Chat.test_console_voice_capture import _call, _eligible, _context
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceRegistry,
    ProvisionalTraceUnavailable,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import PostDispatchTraceResponse
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    PostDispatchTraceSurfaceComponent,
    derive_post_dispatch_trace_node_id,
)


def policy(*, pii=True, revision=BUILTIN_PII_RULESET_REVISION_ID):
    return FrozenTracePolicy(
        new_opaque_id(), CREDENTIAL_FILTER_VERSION, pii, revision if pii else None
    )


def test_artifact_projects_before_seal_and_retains_only_immutable_span_metadata():
    original = {"content": "Write to alice@example.test"}
    artifact = _artifact(
        promotion_id=new_opaque_id(),
        call_sequence=0,
        label="system:0",
        value=original,
        policy=policy(),
    )
    assert original == {"content": "Write to alice@example.test"}
    assert json.loads(artifact.sanitized_bytes) == {"content": "Write to [PII omitted]"}
    assert (
        artifact.identity_digest == hashlib.sha256(artifact.sanitized_bytes).hexdigest()
    )
    assert artifact.field_redactions
    assert "alice" not in repr(artifact.field_redactions)
    assert artifact.retained_bytes > len(artifact.sanitized_bytes)
    with pytest.raises(FrozenInstanceError):
        artifact.field_redactions[0].field_path = "changed"
    with pytest.raises((TypeError, ValueError)):
        replace(artifact, field_redactions=list(artifact.field_redactions))


def test_unavailable_frozen_ruleset_seals_only_a_content_free_omission():
    artifact = _artifact(
        promotion_id=new_opaque_id(),
        call_sequence=0,
        label="system:0",
        value="alice@example.test",
        policy=policy(revision=new_opaque_id()),
    )
    assert json.loads(artifact.sanitized_bytes) == {
        "omitted": CUSTOM_PII_RULESET_UNAVAILABLE
    }
    assert artifact.field_redactions == ()


def test_credential_filter_precedes_pii_and_final_bytes_are_canonical_utf8():
    artifact = _artifact(
        promotion_id=new_opaque_id(),
        call_sequence=0,
        label="tool:0",
        value={"api_key": "fixture-secret", "text": "Élise"},
        policy=policy(pii=False),
    )
    assert b"fixture-secret" not in artifact.sanitized_bytes
    assert "Élise".encode() in artifact.sanitized_bytes


@pytest.mark.parametrize("ruleset_available", [True, False])
def test_winning_import_masks_canonical_pair_without_changing_messages(
    ruleset_available,
):
    db = CharactersRAGDB(":memory:", "voice-privacy")
    try:
        conversation_id = db.add_conversation({"title": "voice"})
        user_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "alice@example.test",
            }
        )
        assistant_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "parent_message_id": user_id,
                "content": "bob@example.test",
                "assistant_generation_state": "complete",
            }
        )
        request = _post_dispatch_import(
            db,
            conversation_id=conversation_id,
            user_message_id=user_id,
            assistant_message_id=assistant_id,
            call_count=1,
        )
        request = replace(
            request,
            policy=policy(
                revision=BUILTIN_PII_RULESET_REVISION_ID
                if ruleset_available
                else new_opaque_id()
            ),
        )
        repository = ConsoleTraceRepository()
        repository.import_post_dispatch_trace(db, request)
        with db.transaction() as cursor:
            for revision_id in (
                request.user_revision_id,
                request.assistant_revision_id,
            ):
                if ruleset_available:
                    projected = project_semantic_revision_trace_message(
                        cursor,
                        revision_id=revision_id,
                        expected_conversation_id=conversation_id,
                        policy_id=request.policy.policy_id,
                    )
                    assert "example.test" not in repr(projected)
                else:
                    binding = repository.get_revision_policy_binding(
                        cursor,
                        revision_id=revision_id,
                        policy_id=request.policy.policy_id,
                    )
                    assert (
                        binding.omission_reason_code == CUSTOM_PII_RULESET_UNAVAILABLE
                    )
            content = [
                row[0]
                for row in cursor.execute("SELECT content FROM messages ORDER BY rowid")
            ]
            assert content == ["alice@example.test", "bob@example.test"]
    finally:
        db.close_connection()


@pytest.mark.parametrize("normalization_version", ["canonical-json-v1", "json-v1"])
def test_artifact_import_retains_exact_sealed_masks_and_rejects_mask_tampering(
    normalization_version,
):
    db = CharactersRAGDB(":memory:", "voice-artifact-privacy")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
            call_count=1,
        )
        request = replace(request, policy=policy())
        artifact = _artifact(
            promotion_id=request.import_id,
            call_sequence=0,
            label="system:0",
            value={"content": "alice@example.test"},
            policy=request.policy,
        )
        artifact = replace(artifact, normalization_version=normalization_version)
        call = replace(
            request.calls[0],
            header_components=(
                PostDispatchTraceHeaderComponent("tool_schema", 0, artifact),
            ),
        )
        request = _seal_post_dispatch_import(replace(request, calls=(call,)))
        repository = ConsoleTraceRepository()
        repository.import_post_dispatch_trace(db, request)
        with db.transaction() as cursor:
            rows = repository.read_redaction_spans(
                cursor,
                policy_id=request.policy.policy_id,
                semantic_revision_id=None,
                artifact_id=artifact.artifact_id,
                field_path=artifact.field_redactions[0].field_path,
            )
            assert len(rows) == len(artifact.field_redactions)
            stored = repository.get_artifact(cursor, artifact.artifact_id)
            assert stored.sanitized_bytes == artifact.sanitized_bytes
            assert stored.artifact_id == artifact.artifact_id
            assert stored.identity_digest == artifact.identity_digest
            assert stored.normalization_version == normalization_version
        changed = replace(artifact, field_redactions=())
        forged = _seal_post_dispatch_import(
            replace(
                request,
                calls=(
                    replace(
                        call,
                        header_components=(
                            PostDispatchTraceHeaderComponent("tool_schema", 0, changed),
                        ),
                    ),
                ),
            )
        )
        with pytest.raises(TraceIdentityConflict):
            repository.import_post_dispatch_trace(db, forged)
        assert repository.import_post_dispatch_trace(db, request).already_imported
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "unsafe_field", ["artifact", "generation_parameters_json", "provider_name"]
)
def test_import_rejects_sealed_values_that_mandatory_credentials_would_change(
    unsafe_field,
):
    db = CharactersRAGDB(":memory:", "voice-sealed-credentials")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
            call_count=1,
        )
        if unsafe_field == "artifact":
            artifact = PostDispatchTraceArtifact(
                new_opaque_id(),
                "application/json",
                "canonical-json-v1",
                b'{"api_key":"fixture-secret"}',
            )
            call = replace(
                request.calls[0],
                header_components=(
                    PostDispatchTraceHeaderComponent("tool_schema", 0, artifact),
                ),
            )
        else:
            value = (
                '{"api_key":"fixture-secret"}'
                if unsafe_field.endswith("json")
                else "Bearer fixture-secret-credential-12345"
            )
            call = replace(request.calls[0], **{unsafe_field: value})
        request = _seal_post_dispatch_import(replace(request, calls=(call,)))
        with pytest.raises(TraceIdentityConflict):
            ConsoleTraceRepository().import_post_dispatch_trace(db, request)
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM console_trace_calls")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("media_type", "normalization_version", "payload"),
    [
        ("application/json", "json-v1", b'{"api_key":"fixture-secret"}'),
        ("application/json; charset=utf-8", "json-v1", b'{"api_key":"fixture-secret"}'),
        ("text/plain", "canonical-json-v1", b'{"api_key":"fixture-secret"}'),
        ("application/octet-stream", "bytes-v1", b'{"api_key":"fixture-secret"}'),
        ("application/json", "unknown-v1", b"{}"),
        ("text/plain", "canonical-json-v1", b"{}"),
    ],
)
def test_import_rejects_artifact_tag_sanitizer_bypass(
    media_type, normalization_version, payload
):
    db = CharactersRAGDB(":memory:", "voice-artifact-envelope")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
            call_count=1,
        )
        artifact = PostDispatchTraceArtifact(
            new_opaque_id(), media_type, normalization_version, payload
        )
        call = replace(
            request.calls[0],
            header_components=(
                PostDispatchTraceHeaderComponent("tool_schema", 0, artifact),
            ),
        )
        request = _seal_post_dispatch_import(replace(request, calls=(call,)))
        connection = db.get_connection()
        before = connection.execute(
            "SELECT count(*) FROM console_trace_artifacts"
        ).fetchone()[0]
        with pytest.raises(TraceIdentityConflict, match="post_dispatch_artifact"):
            ConsoleTraceRepository().import_post_dispatch_trace(db, request)
        assert (
            connection.execute("SELECT count(*) FROM console_trace_calls").fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM console_trace_artifacts"
            ).fetchone()[0]
            == before
        )
        assert db.get_message_by_id(user)["content"] == "protected user transcript"
        assert db.get_message_by_id(assistant)["content"] == "protected assistant reply"
    finally:
        db.close_connection()


@pytest.mark.parametrize("release", ["overflow", "abandon", "expiry"])
def test_production_counter_includes_masks_and_releases_them(release):
    frozen = policy()
    promotion = new_opaque_id()
    artifact = _artifact(
        promotion_id=promotion,
        call_sequence=0,
        label="tool:0",
        value="alice@example.test",
        policy=frozen,
    )
    call = replace(
        _call(promotion, 0, artifact_bytes=b"x"),
        response=PostDispatchTraceResponse.artifact(artifact),
        sealed_payload_bytes=8 + artifact.retained_bytes,
    )
    now = [0.0]
    limit = (
        call.sealed_payload_bytes - 1
        if release == "overflow"
        else call.sealed_payload_bytes
    )
    registry = ProvisionalTraceRegistry(
        attempt_byte_limit=limit, app_byte_limit=limit, clock=lambda: now[0]
    )
    attempt = registry.begin_attempt(
        promotion_id=promotion,
        attempt_id=new_opaque_id(),
        policy=frozen,
        eligibility=_eligible(),
    )
    if release == "overflow":
        assert limit > 8 + len(artifact.sanitized_bytes)
        with pytest.raises(ProvisionalTraceUnavailable):
            registry._retain_gateway_call(attempt, call)
    else:
        registry._retain_gateway_call(attempt, call)
        assert registry.retained_bytes == call.sealed_payload_bytes
        if release == "abandon":
            registry.abandon_attempt(attempt)
        else:
            now[0] = 1000.0
            assert registry.reap_expired() == 1
    assert registry.retained_bytes == 0


def test_import_claim_rejects_a_different_frozen_policy():
    registry = ProvisionalTraceRegistry()
    promotion = new_opaque_id()
    frozen = policy()
    attempt = registry.begin_attempt(
        promotion_id=promotion,
        attempt_id=new_opaque_id(),
        policy=frozen,
        eligibility=_eligible(),
    )
    envelope = registry._retain_gateway_call(attempt, _call(promotion, 0))
    manifest = registry.seal_attempt(attempt, expected_call_count=1)
    context = replace(
        _context(promotion), policy=replace(frozen, policy_id=new_opaque_id())
    )
    with pytest.raises(ProvisionalTraceUnavailable):
        claim = registry.claim(manifest, (envelope,))
        registry._import_claim(
            claim, context, lambda _request: pytest.fail("foreign policy imported")
        )
    assert registry.retained_bytes == 0


def test_custom_artifact_and_history_are_projected_once_and_retry_needs_no_rules(
    monkeypatch,
):
    from collections import OrderedDict
    from tldw_chatbook.Chat import console_trace_custom_pii as custom
    from tldw_chatbook.Chat import console_trace_regex_worker as regex_worker

    revision = new_opaque_id()
    rules = custom.validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": revision,
            "rules": [
                {
                    "id": "customer-id",
                    "label": "Customer ID",
                    "category": "customer_id",
                    "pattern": r"customer-[A-Z]{8}",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert custom.register_custom_pii_ruleset(rules)
    calls = []
    original_run = regex_worker.run_custom_pii_batch

    def counted(*args, **kwargs):
        calls.append(1)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(regex_worker, "run_custom_pii_batch", counted)
    db = CharactersRAGDB(":memory:", "voice-custom-retry")
    try:
        conversation, user, assistant = _completed_pair(db)
        history_id = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "role": "user",
                "content": "customer-ABCDWXYZ",
            }
        )
        history_revision = (
            db.get_connection()
            .execute(
                "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ?",
                (history_id,),
            )
            .fetchone()[0]
        )
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
            call_count=1,
        )
        frozen = policy(revision=revision)
        artifact = _artifact(
            promotion_id=request.import_id,
            call_sequence=0,
            label="system:0",
            value={"content": "customer-ABCDWXYZ"},
            policy=frozen,
        )
        assert len(calls) == 1
        assert b"customer-ABCDWXYZ" not in artifact.sanitized_bytes
        original_call = request.calls[0]
        history = PostDispatchTraceSurfaceComponent.revision(
            node_id=original_call.request_surface[0].node_id,
            component_kind="provider_message",
            revision_id=history_revision,
        )
        latest = replace(
            original_call.request_surface[0],
            node_id=derive_post_dispatch_trace_node_id(request.import_id, 0, 1),
        )
        call = replace(
            original_call,
            request_surface=(history, latest),
            header_components=(
                PostDispatchTraceHeaderComponent("tool_schema", 0, artifact),
            ),
        )
        request = _seal_post_dispatch_import(
            replace(request, policy=frozen, calls=(call,))
        )
        repository = ConsoleTraceRepository()
        first = repository.import_post_dispatch_trace(db, request)
        assert (
            len(calls) == 4
        )  # one sealed artifact, three distinct canonical revisions
        with db.transaction() as cursor:
            projected = project_semantic_revision_trace_message(
                cursor,
                revision_id=history_revision,
                expected_conversation_id=conversation,
                policy_id=frozen.policy_id,
            )
            assert projected == {"role": "user", "content": "[PII omitted]"}
        before = (
            db.get_connection()
            .execute("SELECT count(*) FROM console_trace_redaction_spans")
            .fetchone()[0]
        )
        remaining = OrderedDict(custom._RULESET_REGISTRY)
        remaining.pop(revision)
        monkeypatch.setattr(custom, "_RULESET_REGISTRY", remaining)

        def forbidden(*args, **kwargs):
            pytest.fail("committed retry reran custom detection")

        monkeypatch.setattr(custom, "redact_pii_value_for_ruleset_revision", forbidden)
        retried = repository.import_post_dispatch_trace(db, request)
        assert retried.already_imported and retried.call_ids == first.call_ids
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM console_trace_redaction_spans")
            .fetchone()[0]
            == before
        )
        assert (
            db.get_connection()
            .execute("SELECT content FROM messages WHERE id = ?", (history_id,))
            .fetchone()[0]
            == "customer-ABCDWXYZ"
        )
    finally:
        db.close_connection()


def test_worker_timeout_omits_artifact_and_canonical_values(monkeypatch):
    from tldw_chatbook.Chat import console_trace_custom_pii as custom
    from tldw_chatbook.Chat import console_trace_regex_worker as worker

    revision = new_opaque_id()
    rules = custom.validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": revision,
            "rules": [
                {
                    "id": "customer",
                    "label": "Customer",
                    "category": "customer_id",
                    "pattern": "customer",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert custom.register_custom_pii_ruleset(rules)
    monkeypatch.setattr(
        worker,
        "run_custom_pii_batch",
        lambda *_args, **_kwargs: worker._unavailable(
            worker.CUSTOM_PII_WORKER_TIMEOUT, terminated=True
        ),
    )
    frozen = policy(revision=revision)
    artifact = _artifact(
        promotion_id=new_opaque_id(),
        call_sequence=0,
        label="tool:0",
        value="customer alice@example.test",
        policy=frozen,
    )
    assert json.loads(artifact.sanitized_bytes) == {
        "omitted": worker.CUSTOM_PII_WORKER_TIMEOUT
    }
    assert artifact.field_redactions == ()
    db = CharactersRAGDB(":memory:", "voice-timeout")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = replace(
            _post_dispatch_import(
                db,
                conversation_id=conversation,
                user_message_id=user,
                assistant_message_id=assistant,
                call_count=1,
            ),
            policy=frozen,
        )
        repository = ConsoleTraceRepository()
        repository.import_post_dispatch_trace(db, request)
        with db.transaction() as cursor:
            for revision_id in (
                request.user_revision_id,
                request.assistant_revision_id,
            ):
                binding = repository.get_revision_policy_binding(
                    cursor, revision_id=revision_id, policy_id=frozen.policy_id
                )
                assert binding.omission_reason_code == worker.CUSTOM_PII_WORKER_TIMEOUT
    finally:
        db.close_connection()


def test_retired_history_binds_omission_without_substituting_current_body():
    db = CharactersRAGDB(":memory:", "voice-retired")
    try:
        conversation, user, assistant = _completed_pair(db)
        history_id = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "role": "user",
                "content": "alice@example.test",
            }
        )
        connection = db.get_connection()
        history_revision = connection.execute(
            "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ?",
            (history_id,),
        ).fetchone()[0]
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
            call_count=1,
        )
        call = request.calls[0]
        history = PostDispatchTraceSurfaceComponent.revision(
            node_id=call.request_surface[0].node_id,
            component_kind="provider_message",
            revision_id=history_revision,
        )
        current = replace(
            call.request_surface[0],
            node_id=derive_post_dispatch_trace_node_id(request.import_id, 0, 1),
        )
        request = _seal_post_dispatch_import(
            replace(
                request,
                policy=policy(),
                calls=(replace(call, request_surface=(history, current)),),
            )
        )
        version = connection.execute(
            "SELECT version FROM messages WHERE id = ?", (history_id,)
        ).fetchone()[0]
        assert db.update_message(
            history_id, {"content": "replacement bob@example.test"}, version
        )
        repository = ConsoleTraceRepository()
        repository.import_post_dispatch_trace(db, request)
        with db.transaction() as cursor:
            binding = repository.get_revision_policy_binding(
                cursor, revision_id=history_revision, policy_id=request.policy.policy_id
            )
            assert binding.omission_reason_code == "trace_source_unavailable"
        assert (
            connection.execute(
                "SELECT content FROM messages WHERE id = ?", (history_id,)
            ).fetchone()[0]
            == "replacement bob@example.test"
        )
    finally:
        db.close_connection()


def test_mask_admission_rolls_back_atomically_with_call_insertion():
    from tldw_chatbook.Chat.console_voice_trace_promotion import (
        ConfirmedPreCommitTraceImportError,
    )

    db = CharactersRAGDB(":memory:", "voice-mask-rollback")
    try:
        conversation = db.add_conversation({"title": "voice"})
        user = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "role": "user",
                "content": "alice@example.test",
            }
        )
        assistant = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "role": "assistant",
                "content": "bob@example.test",
                "parent_message_id": user,
                "assistant_generation_state": "complete",
            }
        )
        request = replace(
            _post_dispatch_import(
                db,
                conversation_id=conversation,
                user_message_id=user,
                assistant_message_id=assistant,
                call_count=1,
            ),
            policy=policy(),
        )
        connection = db.get_connection()
        connection.execute("""CREATE TEMP TRIGGER fail_after_masks BEFORE INSERT ON console_trace_calls
            WHEN EXISTS (SELECT 1 FROM console_trace_redaction_spans)
            BEGIN SELECT RAISE(ABORT, 'fixture after mask admission'); END""")
        with pytest.raises(ConfirmedPreCommitTraceImportError):
            ConsoleTraceRepository().import_post_dispatch_trace(db, request)
        for table in (
            "console_trace_redaction_spans",
            "console_trace_calls",
            "console_trace_owners",
            "console_trace_policies",
        ):
            assert (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            )
        assert [
            row[0]
            for row in connection.execute("SELECT content FROM messages ORDER BY rowid")
        ] == ["alice@example.test", "bob@example.test"]
    finally:
        db.close_connection()
