"""Actual captured normalized trace history remains inert and byte-exact."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def test_captured_trace_plan_preserves_native_artifact_graph(tmp_path, monkeypatch):
    with content_reads(tmp_path, monkeypatch, _seed_core) as reads:
        build = getattr(reads, "trace_plan", None)
        assert callable(build), "captured reader needs the fixed T trace plan"
        plan = build()
        assert not plan.blockers
        assert len(plan.field_ownership) == 34
        assert sum(len(columns) for _, columns in plan.field_ownership) == 331
        originals = {
            r.original: dict(r.values)
            for r in plan.source_rows
            if r.table == "console_trace_artifacts"
        }
        artifacts = [r for r in plan.rows if r.table == "console_trace_artifacts"]
        assert artifacts
        for row in artifacts:
            value, source = dict(row.values), originals[row.original]
            assert value["sanitized_bytes"] == source["sanitized_bytes"]
            assert value["identity_digest"] == source["identity_digest"]
            assert value["byte_length"] == len(value["sanitized_bytes"])
            assert value["artifact_id"] == plan.identities.mapped(
                "trace_artifact", source["artifact_id"]
            )
        nodes = [
            dict(r.values)
            for r in plan.rows
            if r.table == "console_trace_surface_nodes"
        ]
        assert any(
            n["artifact_id"] == dict(artifacts[0].values)["artifact_id"] for n in nodes
        )
        assert plan.pending_handlers == ("S", "A-runs", "integrated", "owner-write")
        for value in (plan, replace(plan, rows=(), blockers=(), pending_handlers=())):
            with pytest.raises(ValueError, match="pending"):
                value.require_complete()


@pytest.mark.parametrize("partial", [False, True, "unbound"])
def test_captured_trace_preserves_original_native_reconstruction(
    tmp_path, monkeypatch, partial
):
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        TraceCallState,
        TraceContentRef,
        TraceOmission,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    originals = {}

    def native_seed(db, purpose):
        conversation = db.add_conversation(
            {"id": "trace-conversation", "title": "Trace"}
        )
        user = db.add_message(
            {
                "id": "trace-user",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "question",
            }
        )
        answer = db.add_message(
            {
                "id": "trace-answer",
                "conversation_id": conversation,
                "sender": "Assistant",
                "role": "assistant",
                "content": "answer",
                "parent_message_id": user,
            }
        )
        repo = ConsoleTraceRepository()
        with db.transaction(immediate=True) as cursor:
            policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
            repo.ensure_policy(cursor, policy)
            segment = repo.create_segment(cursor)
            owner = repo.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=segment.segment_id
            )
            artifact = repo.store_sanitized_artifact(
                cursor,
                sanitized_bytes=b'{"role":"user","content":"question"}',
                media_type="application/json",
                normalization_version="test-v1",
            )
            reference = (
                TraceOmission("message", "source_unavailable")
                if partial
                else TraceContentRef(artifact.artifact_id, "message")
            )
            surface = repo.append_surface_node(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                predecessor_node_id=None,
                component_kind="message",
                reference=reference,
            )
            header = repo.create_or_reuse_request_header(
                cursor,
                provider_name="openai",
                model_name="trace-test",
                route_identity="fresh",
                endpoint_identity="https://example.invalid/v1",
                generation_parameters={"streaming": False},
                adapter_defaults={},
                response_format={},
                reasoning_controls={},
                components=(),
            )
            call = repo.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=user,
                run_id="74e2ca8f-8a81-4f0a-bffc-42785924f492",
                call_sequence=0,
                idempotency_key="native-captured-call-" + purpose,
                policy_id=policy.policy_id,
            )
            if partial == "unbound":
                repo.advance_call_state(
                    cursor,
                    call_id=call.call_id,
                    target=TraceCallState.NOT_DISPATCHED,
                    occurred_at="2026-10-26T00:00:00Z",
                    integrity_state="incomplete",
                )
            else:
                repo.bind_call(
                    cursor,
                    call_id=call.call_id,
                    surface_node_id=surface.node_id,
                    request_header_id=header.header_id,
                    provider_name="openai",
                    model_name="trace-test",
                    route_identity="fresh",
                )
                for state in (
                    TraceCallState.DISPATCH_STARTED,
                    TraceCallState.RESPONSE_STARTED,
                    TraceCallState.COMPLETE,
                ):
                    repo.advance_call_state(
                        cursor,
                        call_id=call.call_id,
                        target=state,
                        occurred_at="2026-10-26T00:00:00Z",
                        integrity_state="incomplete" if partial else "complete",
                    )
            repo.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                event_type="call_boundary",
                call_id=call.call_id,
            )
        native = ConsoleTraceNativeReader(db).read_calls(answer)
        assert len(native) == 1
        assert native[0].verification_status == (
            "unverified" if partial else "verified"
        )
        assert native[0].capture.response == {"omitted": "response_unavailable"}
        if partial == "unbound":
            assert native[0].capture.request == {
                "omitted": "native_request_reconstruction_unavailable"
            }
            assert native[0].uncertainty_codes == (
                "native_request_reconstruction_unavailable",
            )
            assert native[0].capture.omitted_keys == ("request",)
        elif partial:
            assert native[0].capture.request["messages_payload"] == [
                {"omitted": "source_unavailable"}
            ]
            assert native[0].capture.omitted_keys == ("messages_payload",)
        else:
            assert native[0].capture.request["messages_payload"] == [
                {"role": "user", "content": "question"}
            ]
        originals[purpose] = native

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=native_seed
    ) as reads:
        assert set(originals) == {"legacy", "canonical"}
        build = getattr(reads, "trace_plan", None)
        assert callable(build), (
            "captured reader must preserve actual native partial reconstruction"
        )
        plan = build()
        assert not any(b.table == "console_trace_calls" for b in plan.blockers)
        original_call = originals["legacy"][0]
        call = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "console_trace_calls"
            and row.original == (original_call.call_id,)
        )
        assert call["integrity_state"] == ("incomplete" if partial else "complete")
        assert call["outcome"] == (None if partial == "unbound" else "complete")
        assert call["state"] == (
            "not_dispatched" if partial == "unbound" else "complete"
        )


def test_captured_fixed_trace_input_seam_binds_native_rows_and_context(
    tmp_path, monkeypatch
):
    import hashlib

    with content_reads(tmp_path, monkeypatch, _seed_core) as reads:
        read = getattr(reads, "trace_inputs", None)
        assert callable(read), (
            "fixed T seam must run native consumers over retained connections"
        )
        inputs = read()
        assert (
            inputs.source.manifest_digest
            == hashlib.sha256(reads.captured.source.manifest_bytes).hexdigest()
        )
        assert (
            inputs.canonical.manifest_digest
            == hashlib.sha256(reads.captured.canonical.manifest_bytes).hexdigest()
        )
        assert {r.table for r in inputs.source.rows} >= {
            "messages",
            "console_trace_artifacts",
            "rag_identity_context",
        }
        assert inputs.source.citation.no_fingerprint_history is True
        assert inputs.source.citation.context.context_name == "default"
        with pytest.raises((AttributeError, TypeError)):
            inputs.source.rows += ()
        with pytest.raises((AttributeError, TypeError)):
            inputs.source.citation.no_fingerprint_history = False


def test_historical_locators_and_opaque_groups_share_native_reservations(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_semantic_revision_coordinator import (
        _reference_under_policies,
    )
    from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    def seed(db, purpose):
        conversation = db.add_conversation(
            {"id": "historical-conversation", "title": "History"}
        )
        message = db.add_message(
            {
                "id": "retired-message",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "retired",
            }
        )
        _revision, _node, policies = _reference_under_policies(
            db, conversation_id=conversation, message_id=message, policy_count=1
        )
        repository = ConsoleTraceRepository()
        with db.transaction(immediate=True) as cursor:
            owner_id, segment_id = cursor.execute(
                "SELECT owner_id,root_segment_id FROM console_trace_owners WHERE conversation_id=?",
                (conversation,),
            ).fetchone()
            repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id=message,
                run_id="legacy run:opaque tag / 7",
                call_sequence=0,
                idempotency_key="history-" + purpose,
                policy_id=policies[0],
            )
            SemanticRevisionCoordinator(db).mutate_message(
                cursor,
                message_id=message,
                creation_reason="hard_delete",
                hard_delete=True,
            )
        assert (
            not db.get_connection()
            .execute("SELECT 1 FROM messages WHERE id=?", (message,))
            .fetchone()
        )

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        identities = reads.identity_plan()
        retired = identities.mapped("message", "retired-message")
        assert retired != "retired-message"
        group = identities.mapped("trace_group", "legacy run:opaque tag / 7")
        assert group != "legacy run:opaque tag / 7"
        assert (
            len(
                [
                    r
                    for r in identities.reservations
                    if r.domain == "message" and r.original == "retired-message"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    r
                    for r in identities.reservations
                    if r.domain == "trace_group"
                    and r.original == "legacy run:opaque tag / 7"
                ]
            )
            == 1
        )


def test_captured_local_character_association_has_native_joined_proof(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        LocalCharacterConversationTarget,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.DB.character_conversation_search import (
        CharacterConversationSearchRepository,
    )

    authorities = {}

    def seed(db, purpose):
        authority = db.get_local_authority_id()
        authorities[purpose] = authority
        conversation = db.add_conversation(
            {
                "id": "character-history",
                "title": "Character history",
                "assistant_kind": "character",
                "character_id": 11,
                "assistant_id": "11",
            }
        )
        message = db.add_message(
            {
                "id": "character-message",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "native preview",
            }
        )
        repository = CharacterConversationSearchRepository(db)
        target = LocalCharacterConversationTarget(
            ResolvedLocalCharacterKey(authority, 11), conversation
        )
        preview = repository.validated_preview_messages(
            target, data_revision=db.get_character_conversation_search_revision()
        )
        assert preview is not None and [row["id"] for row in preview] == [message]

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        assert authorities["legacy"] != authorities["canonical"]
        inputs = reads.trace_inputs()
        associations = getattr(inputs.source, "local_characters", None)
        assert associations is not None, (
            "fixed retained seam needs actual local-character joined proof"
        )
        association = next(
            item for item in associations if item.conversation_id == "character-history"
        )
        assert association.source_authority_id == authorities["legacy"]
        assert association.character_id == 11
        assert association.resolved is True
        assert association.preview_eligible is True


def test_captured_native_repeated_artifact_refuses_before_materialization(
    tmp_path, monkeypatch
):
    import json

    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        TraceCallState,
        TraceContentRef,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    payload = json.dumps({"role": "user", "content": "x" * (128 * 1024)}).encode()

    def native_seed(db, purpose):
        conversation = db.add_conversation({"id": "budget-chat", "title": "Budget"})
        message = db.add_message(
            {
                "id": "budget-message",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "budget",
            }
        )
        repo = ConsoleTraceRepository()
        with db.transaction(immediate=True) as cursor:
            policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
            repo.ensure_policy(cursor, policy)
            segment = repo.create_segment(cursor)
            owner = repo.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=segment.segment_id
            )
            artifact = repo.store_sanitized_artifact(
                cursor,
                sanitized_bytes=payload,
                media_type="application/json",
                normalization_version="test-v1",
            )
            tail = None
            for sequence in range(128):
                tail = repo.append_surface_node(
                    cursor,
                    segment_id=segment.segment_id,
                    sequence=sequence,
                    predecessor_node_id=None if tail is None else tail.node_id,
                    component_kind="message",
                    reference=TraceContentRef(artifact.artifact_id, "message"),
                )
            header = repo.create_or_reuse_request_header(
                cursor,
                provider_name="openai",
                model_name="budget",
                route_identity="fresh",
                endpoint_identity="https://example.invalid/v1",
                generation_parameters={},
                adapter_defaults={},
                response_format={},
                reasoning_controls={},
                components=(),
            )
            call = repo.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=message,
                run_id="budget-run",
                call_sequence=0,
                idempotency_key="budget-call-" + purpose,
                policy_id=policy.policy_id,
            )
            repo.bind_call(
                cursor,
                call_id=call.call_id,
                surface_node_id=tail.node_id,
                request_header_id=header.header_id,
                provider_name="openai",
                model_name="budget",
                route_identity="fresh",
            )
            for state in (
                TraceCallState.DISPATCH_STARTED,
                TraceCallState.RESPONSE_STARTED,
                TraceCallState.COMPLETE,
            ):
                repo.advance_call_state(
                    cursor,
                    call_id=call.call_id,
                    target=state,
                    occurred_at="2026-10-27T00:00:00Z",
                    integrity_state="complete",
                )
            repo.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                event_type="call_boundary",
                call_id=call.call_id,
            )

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=native_seed
    ) as reads:
        decoded = []
        original = json.loads

        def observed(value, *args, **kwargs):
            if value == payload:
                decoded.append(len(value))
            return original(value, *args, **kwargs)

        monkeypatch.setattr(json, "loads", observed)
        with pytest.raises(Exception) as failure:
            reads.trace_inputs()
        assert str(failure.value) == "mobile_trace_input_limit"
        assert type(failure.value).__name__ == "_TraceReadLimit", (
            str(failure.value),
            len(decoded),
        )
        assert 0 < len(decoded) < 128


def test_captured_character_slot_maps_to_native_canonical_preview(
    tmp_path, monkeypatch
):
    import shutil
    from contextlib import closing

    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        LocalCharacterConversationTarget,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.character_conversation_search import (
        CharacterConversationSearchRepository,
    )

    authorities = {}

    def seed(db, purpose):
        authorities[purpose] = db.get_local_authority_id()
        conversation = db.add_conversation(
            {
                "id": "character-history",
                "title": "Character history",
                "assistant_kind": "character",
                "character_id": 11,
                "assistant_id": "11",
            }
        )
        message = db.add_message(
            {
                "id": "character-message",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "native preview",
            }
        )
        target = LocalCharacterConversationTarget(
            ResolvedLocalCharacterKey(authorities[purpose], 11),
            conversation,
        )
        preview = CharacterConversationSearchRepository(db).validated_preview_messages(
            target,
            data_revision=db.get_character_conversation_search_revision(),
        )
        assert preview is not None and [r["id"] for r in preview] == [message]

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

    ordinary_selector = effective_config_path()
    ordinary_bootstrap = bootstrap.default_bootstrap_root()
    assert ordinary_selector.is_relative_to(tmp_path)
    assert ordinary_bootstrap.is_relative_to(tmp_path)
    with monkeypatch.context() as fixture_patches:
        with content_reads(
            tmp_path, fixture_patches, _seed_core, native_seed=seed
        ) as reads:
            plan = reads.trace_plan()
            discharges = getattr(plan, "authority_discharges", None)
            assert discharges is not None, (
                "T needs the proved exact C authority-slot discharge"
            )
            discharge = next(
                d for d in discharges if d.original == ("character-history",)
            )
            row = dict(discharge.values)
            assert authorities["legacy"] != authorities["canonical"]
            assert row["assistant_authority_id"] == authorities["canonical"]
            assert row["character_id"] == plan.identities.mapped("character", 11) != 11
            assert not any(b.original == ("character-history",) for b in plan.blockers)
            card = next(
                r
                for r in plan.content.rows
                if r.table == "character_cards" and r.original == (11,)
            )
            message = next(
                r
                for r in plan.content.rows
                if r.table == "messages" and r.original == ("character-message",)
            )
            # Independent disposable oracle: preserve the ACTUAL canonical context.
            # Its captured source stays immutable; no constructor-generated identity
            # is substituted as proof and no table/trigger/authorizer is disabled.
            candidate = reads._connections[("canonical", "chat")][3]
            path = tmp_path / "mapped-character-oracle.sqlite"
            shutil.copy2(candidate.parent / candidate.name, path)
        with pytest.raises(ValueError, match="inactive"):
            reads.check()
    assert effective_config_path() == ordinary_selector
    assert bootstrap.default_bootstrap_root() == ordinary_bootstrap
    assert (tmp_path / "bootstrap").exists()
    with closing(CharactersRAGDB(path, "mapped-character-oracle")) as db:
        assert db.get_local_authority_id() == authorities["canonical"]
        with db.transaction(immediate=True) as cursor:
            for mapped in (card, discharge, message):
                values = dict(mapped.values)
                columns = ",".join('"' + key + '"' for key in values)
                placeholders = ",".join("?" for _ in values)
                cursor.execute(
                    f'INSERT INTO "{mapped.table}" ({columns}) VALUES ({placeholders})',
                    tuple(values.values()),
                )
        repository = CharacterConversationSearchRepository(db)
        target = LocalCharacterConversationTarget(
            ResolvedLocalCharacterKey(authorities["canonical"], row["character_id"]),
            row["id"],
        )
        revision = db.get_character_conversation_search_revision()
        preview = repository.validated_preview_messages(target, data_revision=revision)
        assert preview is not None
        assert [r["id"] for r in preview] == [
            plan.identities.mapped("message", "character-message")
        ]
        foreign = LocalCharacterConversationTarget(
            ResolvedLocalCharacterKey(authorities["legacy"], row["character_id"]),
            row["id"],
        )
        assert (
            repository.validated_preview_messages(foreign, data_revision=revision)
            is None
        )
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET deleted=1 WHERE id=?", (row["id"],)
            )
        assert (
            repository.validated_preview_messages(target, data_revision=revision)
            is None
        )
        assert (
            repository.validated_preview_messages(
                target,
                data_revision=db.get_character_conversation_search_revision(),
            )
            is None
        )

        def restore_eligible():
            with db.transaction(immediate=True) as cursor:
                cursor.execute(
                    "UPDATE conversations SET deleted=0,character_id=?,assistant_id=?,assistant_authority_id=? WHERE id=?",
                    (
                        row["character_id"],
                        str(row["character_id"]),
                        authorities["canonical"],
                        row["id"],
                    ),
                )
                cursor.execute(
                    "UPDATE character_cards SET deleted=0 WHERE id=?",
                    (row["character_id"],),
                )
            preview = repository.validated_preview_messages(
                target, data_revision=db.get_character_conversation_search_revision()
            )
            assert preview is not None and [item["id"] for item in preview] == [
                plan.identities.mapped("message", "character-message")
            ]

        # Each unavailable state starts from an independently observed positive.
        restore_eligible()
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE character_cards SET deleted=1 WHERE id=?",
                (row["character_id"],),
            )
        assert (
            repository.validated_preview_messages(
                target, data_revision=db.get_character_conversation_search_revision()
            )
            is None
        )
        restore_eligible()
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET assistant_authority_id=NULL WHERE id=?",
                (row["id"],),
            )
        assert (
            repository.validated_preview_messages(
                target, data_revision=db.get_character_conversation_search_revision()
            )
            is None
        )
        restore_eligible()
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET character_id=NULL,assistant_id='historical-missing' WHERE id=?",
                (row["id"],),
            )
        assert (
            repository.validated_preview_messages(
                target, data_revision=db.get_character_conversation_search_revision()
            )
            is None
        )
        restore_eligible()
        missing = LocalCharacterConversationTarget(
            target.character, "missing-conversation"
        )
        assert (
            repository.validated_preview_messages(
                missing, data_revision=db.get_character_conversation_search_revision()
            )
            is None
        )


def test_captured_normalized_legacy_read_budget_preserves_native_output(
    tmp_path, monkeypatch
):
    import json
    from dataclasses import asdict

    from Tests.Chat.test_console_trace_legacy_normalizer import _capture, _row
    from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer

    original = {}

    def seed(db, purpose):
        conversation = db.add_conversation({"id": "legacy-history", "title": "Legacy"})
        db.add_message(
            {
                "id": "legacy-user",
                "conversation_id": conversation,
                "sender": "user",
                "role": "user",
                "content": "saved question",
            }
        )
        message = db.add_message(
            {
                "id": "legacy-answer",
                "conversation_id": conversation,
                "sender": "assistant",
                "role": "assistant",
                "content": "answer",
            }
        )
        capture = _capture(messages=[{"role": "user", "content": "saved question"}])
        normalizer = LegacyTraceNormalizer(db)
        with db.transaction(immediate=True) as cursor:
            normalizer.normalize_exchange(
                cursor,
                _row(
                    capture,
                    exchange_id=17 if purpose == "legacy" else 18,
                    message_id=message,
                ),
            )
        calls = normalizer.read_calls(message)
        assert len(calls) == 1 and calls[0].provenance == "legacy_snapshot"
        original[purpose] = json.loads(json.dumps(asdict(calls[0])))

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        plan = reads.trace_plan()
        proof = next(
            r for r in plan.reconstructions if r.original_message_id == "legacy-answer"
        )
        assert [json.loads(p) for p in proof.original_legacy] == [original["legacy"]]
        mapped = json.loads(proof.mapped_legacy[0])
        expected = original["legacy"] | {
            "call_id": plan.identities.mapped(
                "trace_call", original["legacy"]["call_id"]
            ),
        }
        assert mapped == expected
        assert mapped["capture"]["run_tag"] == "legacy-run"
        assert mapped["chronology"] == "recorded_call_only"


def test_native_read_budget_refusal_is_not_sanitizer_uncertainty():
    from tldw_chatbook.Chat.console_exchange_capture import (
        _sanitize_semantic_json_string,
        sanitize_capture_value_with_omission,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer

    assert CredentialSanitizer().sanitize({"content": "ordinary"}).available
    for consume in (
        lambda budget: CredentialSanitizer().sanitize(
            {"content": "ordinary"}, _read_budget=budget
        ),
        lambda budget: sanitize_capture_value_with_omission(
            {"arguments": '{"key":"value"}'}, _read_budget=budget
        ),
        lambda budget: _sanitize_semantic_json_string(
            '{"key":"value"}', _read_budget=budget
        ),
    ):
        budget = _TraceReadBudget()
        budget.reserve(16 * 1024 * 1024 - 64)
        with pytest.raises(_TraceReadLimit, match="^mobile_trace_input_limit$"):
            consume(budget)


def _seed_response_trace(db, purpose, label, *, run_id=None, idempotency_key=None):
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        SemanticRevisionRef,
        TraceCallState,
        TraceContentRef,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    conversation = db.add_conversation({"id": label, "title": label})
    user = db.add_message(
        {
            "id": label + "-user",
            "conversation_id": conversation,
            "sender": "User",
            "role": "user",
            "content": "question",
        }
    )
    answer = db.add_message(
        {
            "id": label + "-answer",
            "conversation_id": conversation,
            "sender": "Assistant",
            "role": "assistant",
            "content": "answer",
            "parent_message_id": user,
        }
    )
    repo = ConsoleTraceRepository()
    with db.transaction(immediate=True) as cursor:
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        repo.ensure_policy(cursor, policy)
        segment = repo.create_segment(cursor)
        owner = repo.attach_owner(
            cursor, conversation_id=conversation, root_segment_id=segment.segment_id
        )
        artifact = repo.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b'{"role":"user","content":"'
            + purpose.encode()
            + b' question"}',
            media_type="application/json",
            normalization_version="test-v1",
        )
        node = repo.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=TraceContentRef(artifact.artifact_id, "message"),
        )
        header = repo.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="trace-test-" + purpose,
            route_identity="fresh",
            endpoint_identity="https://example.invalid/v1",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        call = repo.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id=user,
            run_id=label if run_id is None else run_id,
            call_sequence=0,
            idempotency_key=purpose + "-" + label
            if idempotency_key is None
            else idempotency_key,
            policy_id=policy.policy_id,
        )
        repo.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node.node_id,
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="trace-test-" + purpose,
            route_identity="fresh",
        )
        for state in (
            TraceCallState.DISPATCH_STARTED,
            TraceCallState.RESPONSE_STARTED,
            TraceCallState.COMPLETE,
        ):
            repo.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=state,
                occurred_at="2026-10-27T00:00:00Z",
                integrity_state="complete",
            )
        revision_id = cursor.execute(
            "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=?",
            (answer,),
        ).fetchone()[0]
        revision = repo.get_semantic_revision(cursor, revision_id)
        repo.store_response_link(
            cursor,
            call_id=call.call_id,
            response=SemanticRevisionRef(revision.revision_id),
        )
        repo.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="call_boundary",
            call_id=call.call_id,
        )
    return call.call_id, user, answer, revision.revision_id


def test_captured_trace_refuses_newly_lost_reverse_response_input(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

    seeded = {}

    def seed(db, purpose):
        bad = _seed_response_trace(db, purpose, "blocked-response")
        good = _seed_response_trace(db, purpose, "unrelated-response")
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE messages SET metadata_json=? WHERE id=?",
                ('{"future_unknown_metadata":true}', bad[2]),
            )
        native = ConsoleTraceNativeReader(db).read_calls(bad[1])
        assert len(native) == 1 and native[0].capture.response["content"] == "answer"
        seeded[purpose] = bad, good

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        plan = reads.trace_plan()
        bad, good = seeded["legacy"]
        assert any(
            b.table == "messages" and b.original == (bad[2],)
            for b in plan.content.blockers
        )
        assert not any(
            r.table == "console_trace_semantic_revisions" and r.original == (bad[3],)
            for r in plan.rows
        )
        assert any(
            b.table == "console_trace_calls"
            and b.original == (bad[0],)
            and b.reason == "native_trace_input_closure_unproved"
            for b in plan.blockers
        )
        assert not any(
            r.table == "console_trace_calls" and r.original == (bad[0],)
            for r in plan.rows
        )
        assert any(
            r.table == "console_trace_calls" and r.original == (good[0],)
            for r in plan.rows
        )
        assert any(r.original_message_id == good[1] for r in plan.reconstructions)
        assert not any(r.original_message_id == bad[1] for r in plan.reconstructions)


def test_captured_mapped_trace_runs_actual_native_reader(tmp_path, monkeypatch):
    import json
    import shutil
    from contextlib import closing
    from dataclasses import asdict

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
    from tldw_chatbook.Chat import console_trace_repository as native_repository
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        SemanticRevisionRef,
        TraceCallState,
        TraceContentRef,
    )
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    seeded = {}

    def seed(db, purpose):
        seeded[purpose] = _seed_response_trace(db, purpose, "mapped-native")

    selector, root = effective_config_path(), bootstrap.default_bootstrap_root()
    assert selector.is_relative_to(tmp_path) and root.is_relative_to(tmp_path)
    with monkeypatch.context() as fixture:
        with content_reads(tmp_path, fixture, _seed_core, native_seed=seed) as reads:
            plan = reads.trace_plan()
            assert not plan.blockers
            original_call_id, user, answer, revision_id = seeded["legacy"]
            original = {(r.table, r.original): dict(r.values) for r in plan.source_rows}
            mapped = {(r.table, r.original): dict(r.values) for r in plan.rows}
            call = mapped["console_trace_calls", (original_call_id,)]
            source_call = original["console_trace_calls", (original_call_id,)]

            def selected(table, identity):
                return mapped["console_trace_" + table, (identity,)]

            policy = selected("policies", source_call["policy_id"])
            segment = selected("segments", source_call["segment_id"])
            owner = selected("owners", source_call["owner_id"])
            node = selected("surface_nodes", source_call["surface_node_id"])
            original_node = original[
                "console_trace_surface_nodes", (source_call["surface_node_id"],)
            ]
            artifact = selected("artifacts", original_node["artifact_id"])
            header = selected("request_headers", source_call["request_header_id"])
            revision = selected("semantic_revisions", revision_id)
            response = next(
                dict(r.values)
                for r in plan.rows
                if r.table == "console_trace_response_links"
                and dict(r.values)["call_id"] == call["call_id"]
            )
            event = next(
                dict(r.values)
                for r in plan.rows
                if r.table == "console_trace_events"
                and dict(r.values)["call_id"] == call["call_id"]
            )
            proof = next(
                r for r in plan.reconstructions if r.original_message_id == user
            )
            candidate = reads._connections[("canonical", "chat")][3]
            path = tmp_path / "mapped-native-oracle.sqlite"
            shutil.copy2(candidate.parent / candidate.name, path)
        with pytest.raises(ValueError, match="inactive"):
            reads.check()
    assert (
        effective_config_path() == selector
        and bootstrap.default_bootstrap_root() == root
    )
    allocations = [
        segment["segment_id"],
        owner["owner_id"],
        artifact["artifact_id"],
        node["node_id"],
        header["header_id"],
        call["call_id"],
        revision["revision_id"],
        response["response_link_id"],
        event["event_id"],
    ]
    expected_allocations = tuple(allocations)
    used = []

    def allocate():
        assert allocations, "unexpected native identity allocation"
        identity = allocations.pop(0)
        used.append(identity)
        return identity

    repo = native_repository.ConsoleTraceRepository()
    with closing(CharactersRAGDB(path, "mapped-native-oracle")) as db:
        with db.transaction(immediate=True) as cursor:
            for table, identity in (
                ("conversations", "mapped-native"),
                ("messages", user),
                ("messages", answer),
            ):
                values = dict(
                    next(
                        r.values
                        for r in plan.content.rows
                        if r.table == table and r.original == (identity,)
                    )
                )
                columns = ",".join('"' + key + '"' for key in values)
                cursor.execute(
                    f'INSERT INTO "{table}" ({columns}) VALUES ({",".join("?" for _ in values)})',
                    tuple(values.values()),
                )
            with monkeypatch.context() as allocation:
                allocation.setattr(native_repository, "new_opaque_id", allocate)
                repo.ensure_policy(
                    cursor,
                    FrozenTracePolicy(
                        policy["policy_id"],
                        policy["credential_filter_version"],
                        bool(policy["pii_redaction_enabled"]),
                        policy["pii_ruleset_revision_id"],
                    ),
                )
                assert repo.create_segment(cursor).segment_id == segment["segment_id"]
                assert (
                    repo.attach_owner(
                        cursor,
                        conversation_id=owner["conversation_id"],
                        root_segment_id=owner["root_segment_id"],
                    ).owner_id
                    == owner["owner_id"]
                )
                assert (
                    repo.store_sanitized_artifact(
                        cursor,
                        sanitized_bytes=artifact["sanitized_bytes"],
                        media_type=artifact["media_type"],
                        normalization_version=artifact["normalization_version"],
                    ).artifact_id
                    == artifact["artifact_id"]
                )
                assert (
                    repo.append_surface_node(
                        cursor,
                        segment_id=node["segment_id"],
                        sequence=node["sequence"],
                        predecessor_node_id=node["predecessor_node_id"],
                        component_kind=node["component_kind"],
                        reference=TraceContentRef(
                            node["artifact_id"], node["component_kind"]
                        ),
                    ).node_id
                    == node["node_id"]
                )
                assert (
                    repo.create_or_reuse_request_header(
                        cursor,
                        provider_name=header["provider_name"],
                        model_name=header["model_name"],
                        route_identity=header["route_identity"],
                        endpoint_identity=header["endpoint_identity"],
                        generation_parameters=json.loads(
                            header["generation_parameters_json"]
                        ),
                        adapter_defaults=json.loads(header["adapter_defaults_json"]),
                        response_format=json.loads(header["response_format_json"]),
                        reasoning_controls=json.loads(
                            header["reasoning_controls_json"]
                        ),
                        components=(),
                    ).header_id
                    == header["header_id"]
                )
                assert (
                    repo.reserve_call(
                        cursor,
                        **{
                            key: call[key]
                            for key in (
                                "owner_id",
                                "segment_id",
                                "turn_id",
                                "run_id",
                                "call_sequence",
                                "idempotency_key",
                                "policy_id",
                            )
                        },
                    ).call_id
                    == call["call_id"]
                )
                repo.bind_call(
                    cursor,
                    call_id=call["call_id"],
                    **{
                        key: call[key]
                        for key in (
                            "surface_node_id",
                            "request_header_id",
                            "provider_name",
                            "model_name",
                            "route_identity",
                        )
                    },
                )
                for state, stamp in (
                    (TraceCallState.DISPATCH_STARTED, "dispatch_started_at"),
                    (TraceCallState.RESPONSE_STARTED, "response_started_at"),
                    (TraceCallState.COMPLETE, "settled_at"),
                ):
                    repo.advance_call_state(
                        cursor,
                        call_id=call["call_id"],
                        target=state,
                        occurred_at=call[stamp],
                        integrity_state=call["integrity_state"],
                    )
                assert (
                    repo.ensure_semantic_revision(
                        cursor,
                        **{
                            key: value
                            for key, value in revision.items()
                            if key not in {"revision_id", "created_at"}
                        },
                    ).revision_id
                    == revision["revision_id"]
                )
                assert (
                    repo.store_response_link(
                        cursor,
                        call_id=call["call_id"],
                        response=SemanticRevisionRef(revision["revision_id"]),
                    ).response_link_id
                    == response["response_link_id"]
                )
                assert (
                    repo.append_event(
                        cursor,
                        **{
                            key: value
                            for key, value in event.items()
                            if key not in {"event_id", "created_at"}
                        },
                    ).event_id
                    == event["event_id"]
                )
            assert not allocations and tuple(used) == expected_allocations
            for table, row, column in (
                ("segments", segment, "segment_id"),
                ("owners", owner, "owner_id"),
                ("surface_nodes", node, "node_id"),
                ("request_headers", header, "header_id"),
                ("calls", call, "call_id"),
                ("semantic_revisions", revision, "revision_id"),
                ("response_links", response, "response_link_id"),
                ("events", event, "event_id"),
            ):
                stored = cursor.execute(
                    f'SELECT * FROM "console_trace_{table}" WHERE "{column}"=?',
                    (row[column],),
                ).fetchone()
                # Ordinary constructors generate their own audit creation time;
                # every native reader input and lifecycle timestamp is exact.
                assert {k: v for k, v in dict(stored).items() if k != "created_at"} == {
                    k: v for k, v in row.items() if k != "created_at"
                }
        actual = ConsoleTraceNativeReader(db).read_calls(
            plan.identities.mapped("message", user)
        )
        assert [json.loads(json.dumps(asdict(item))) for item in actual] == [
            json.loads(value) for value in proof.mapped_native
        ]


def test_captured_trace_group_tie_and_idempotency_conflicts_are_exact(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_trace_models import TraceCallState
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_projection import ConsoleTraceProjection
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    seeded = {}

    def seed(db, purpose):
        first = _seed_response_trace(db, purpose, "tie", run_id="!first")
        collision = _seed_response_trace(
            db, purpose, "occupied", idempotency_key="shared-native-key"
        )
        repo = ConsoleTraceRepository()
        with db.transaction(immediate=True) as cursor:
            call = repo.get_call(cursor, first[0])
            second = repo.reserve_call(
                cursor,
                owner_id=call.owner_id,
                segment_id=call.segment_id,
                turn_id=call.turn_id,
                run_id="!second" if purpose == "legacy" else "#canonical-second",
                call_sequence=0,
                idempotency_key=purpose + "-second",
                policy_id=call.policy_id,
            )
            repo.bind_call(
                cursor,
                call_id=second.call_id,
                surface_node_id=call.surface_node_id,
                request_header_id=call.request_header_id,
                provider_name=call.provider_name,
                model_name=call.model_name,
                route_identity=call.route_identity,
            )
            for state in (
                TraceCallState.DISPATCH_STARTED,
                TraceCallState.RESPONSE_STARTED,
                TraceCallState.COMPLETE,
            ):
                repo.advance_call_state(
                    cursor,
                    call_id=second.call_id,
                    target=state,
                    occurred_at="2026-10-27T00:00:00Z",
                    integrity_state="complete",
                )
            repo.append_event(
                cursor,
                segment_id=call.segment_id,
                sequence=1,
                event_type="call_boundary",
                call_id=second.call_id,
            )
        raw = ConsoleTraceNativeReader(db).read_calls(first[1])
        selected = ConsoleTraceProjection(
            legacy_reader=lambda _: (),
            normalized_reader=lambda _: raw,
            normalized_reads_enabled=True,
        ).read_calls(first[1])
        if purpose == "legacy":
            assert [c.call_id for c in raw] == [first[0], second.call_id]
            assert [c.call_id for c in selected] == [first[0], second.call_id]
        seeded[purpose] = first, second.call_id, collision

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        plan = reads.trace_plan()
        first, _second, collision = seeded["legacy"]
        assert any(
            b.table == "console_trace_calls"
            and b.original == (collision[0],)
            and b.reason == "native_call_idempotency_conflict"
            for b in plan.blockers
        )
        assert not any(
            r.table == "console_trace_calls" and r.original == (collision[0],)
            for r in plan.rows
        )
        original = next(
            dict(r.values)
            for r in plan.source_rows
            if r.table == "console_trace_calls" and r.original == (collision[0],)
        )
        assert original["idempotency_key"] == "shared-native-key"
        assert plan.identities.mapped("trace_group", "!first") != "!first"
        assert plan.identities.mapped("trace_group", "!second") == "!second"
        assert any(
            b.table == "messages"
            and b.original == (first[1],)
            and b.reason == "native_trace_projection_selection_changed"
            for b in plan.blockers
        )
        assert not any(r.original_message_id == first[1] for r in plan.reconstructions)
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_captured_character_slots_preserve_inactive_and_unproved_states(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        LocalCharacterConversationTarget,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.DB.character_conversation_search import (
        CharacterConversationSearchRepository,
    )

    states = (
        "foreign",
        "none",
        "unresolved",
        "deleted-conversation",
        "deleted-card",
        "missing-mapped-card",
    )
    originals = {}

    def seed(db, purpose):
        authority = db.get_local_authority_id()
        repository = CharacterConversationSearchRepository(db)
        for state in states:
            name = (
                "occupied-card"
                if state == "missing-mapped-card"
                else purpose + "-" + state
            )
            card = db.add_character_card({"name": name})
            conversation = db.add_conversation(
                {
                    "id": state,
                    "title": state,
                    "assistant_kind": "character",
                    "assistant_id": str(card),
                    "character_id": card,
                }
            )
            with db.transaction(immediate=True) as cursor:
                if state == "foreign":
                    cursor.execute(
                        "UPDATE conversations SET assistant_authority_id=? WHERE id=?",
                        ("c3a7d11c-e8b1-4b02-8aef-ab3042487665", conversation),
                    )
                elif state == "none":
                    cursor.execute(
                        "UPDATE conversations SET assistant_authority_id=NULL WHERE id=?",
                        (conversation,),
                    )
                elif state == "unresolved":
                    cursor.execute(
                        "UPDATE conversations SET character_id=NULL,assistant_id='historical-missing' WHERE id=?",
                        (conversation,),
                    )
                elif state == "deleted-conversation":
                    cursor.execute(
                        "UPDATE conversations SET deleted=1 WHERE id=?", (conversation,)
                    )
                elif state == "deleted-card":
                    cursor.execute(
                        "UPDATE character_cards SET deleted=1 WHERE id=?", (card,)
                    )
            target = LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(authority, card), conversation
            )
            preview = repository.validated_preview_messages(
                target, data_revision=db.get_character_conversation_search_revision()
            )
            assert (preview is not None) == (state == "missing-mapped-card")
            originals[purpose, state] = card

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=seed
    ) as reads:
        plan = reads.trace_plan()
        discharges = {r.original[0]: dict(r.values) for r in plan.authority_discharges}
        assert set(discharges) == {"deleted-conversation", "deleted-card"}
        assert discharges["deleted-conversation"]["deleted"] == 1
        mapped_card = next(
            dict(r.values)
            for r in plan.content.rows
            if r.table == "character_cards"
            and r.original == (originals["legacy", "deleted-card"],)
        )
        assert mapped_card["deleted"] == 1
        assert all(
            not item.preview_eligible
            for item in plan.inputs.source.local_characters
            if item.conversation_id in states
            and item.conversation_id != "missing-mapped-card"
        )
        assert any(
            b.table == "conversations" and b.original == ("foreign",)
            for b in plan.blockers
        )
        # C closes the rejected card dependency before exposing any T slot.
        assert any(
            b.table == "conversations"
            and b.original == ("missing-mapped-card",)
            and b.reason == "dependency_blocked"
            for b in plan.content.blockers
        )
        assert not any(
            slot.conversation_id == "missing-mapped-card"
            for slot in plan.content.authority_slots
        )
        assert any(
            b.table == "conversations" and b.original == ("unresolved",)
            for b in plan.content.blockers
        )
        none = next(
            dict(r.values)
            for r in plan.content.rows
            if r.table == "conversations" and r.original == ("none",)
        )
        assert none["assistant_authority_id"] is None
        assert any(
            r.original == ("foreign",)
            and dict(r.values)["assistant_authority_id"]
            == "c3a7d11c-e8b1-4b02-8aef-ab3042487665"
            for r in plan.content.source_rows
            if r.table == "conversations"
        )


@pytest.mark.parametrize(
    "fault", ["pre_nonchat_tamper", "post_nonchat_tamper", "cancel"]
)
def test_fixed_native_trace_seam_retains_all_member_fences(
    tmp_path, monkeypatch, fault
):
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

    with content_reads(tmp_path, monkeypatch, _seed_core) as reads:
        candidate = reads._connections[("canonical", "workspaces")][3]
        path = candidate.parent / candidate.name
        reached = []
        native = ConsoleTraceNativeReader.read_calls

        def tamper():
            with path.open("ab") as stream:
                stream.write(b"fixed-trace-nonchat-tamper")

        def read_and_fault(reader, *args, **kwargs):
            result = native(reader, *args, **kwargs)
            if not reached:
                reached.append(True)
                if fault == "cancel":
                    reads.cancel.set()
                else:
                    tamper()
            return result

        if fault == "pre_nonchat_tamper":
            tamper()
        else:
            monkeypatch.setattr(ConsoleTraceNativeReader, "read_calls", read_and_fault)
        error = InterruptedError if fault == "cancel" else ValueError
        reason = "cancelled" if fault == "cancel" else "candidate_changed"
        with pytest.raises(error, match=reason):
            reads.trace_inputs()
        assert bool(reached) == (fault != "pre_nonchat_tamper")
    with pytest.raises(ValueError, match="inactive"):
        reads.check()


@pytest.mark.parametrize("inherited", [False, True])
def test_captured_surface_iteration_reserves_before_materialization(
    tmp_path, monkeypatch, inherited
):
    import json
    import linecache
    import sys

    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        TraceCallState,
        TraceContentRef,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_service import (
        ConsoleTraceService,
        _ProjectionRoot,
    )

    payload = json.dumps({"role": "user", "content": "x" * 4096}).encode()
    children = set()
    ordinary = []

    def seed(db, purpose):
        if purpose != "legacy":
            return
        conversation = db.add_conversation({"id": "surface-budget", "title": "Budget"})
        message = db.add_message(
            {
                "id": "surface-user",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "surface",
            }
        )
        parent_conversation = (
            db.add_conversation({"id": "surface-parent", "title": "Parent"})
            if inherited
            else conversation
        )
        repository = ConsoleTraceRepository()
        with db.transaction(immediate=True) as cursor:
            policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
            repository.ensure_policy(cursor, policy)
            parent = repository.create_segment(cursor)
            owner = repository.attach_owner(
                cursor,
                conversation_id=parent_conversation,
                root_segment_id=parent.segment_id,
            )
            artifact = repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=payload,
                media_type="application/json",
                normalization_version="surface-budget-v1",
            )
            tail = None
            for sequence in range(16):
                tail = repository.append_surface_node(
                    cursor,
                    segment_id=parent.segment_id,
                    sequence=sequence,
                    predecessor_node_id=None if tail is None else tail.node_id,
                    component_kind="message",
                    reference=TraceContentRef(artifact.artifact_id, "message"),
                )
            segment = parent
            if inherited:
                repository.append_event(
                    cursor,
                    segment_id=parent.segment_id,
                    sequence=0,
                    event_type="surface_append",
                    surface_node_id=tail.node_id,
                )
                segment = repository.create_segment(
                    cursor,
                    parent_segment_id=parent.segment_id,
                    inherited_through_sequence=0,
                    inherited_surface_head_id=tail.node_id,
                )
                children.add(segment.segment_id)
                owner = repository.attach_owner(
                    cursor,
                    conversation_id=conversation,
                    root_segment_id=segment.segment_id,
                )
            header = repository.create_or_reuse_request_header(
                cursor,
                provider_name="openai",
                model_name="surface-budget",
                route_identity="fresh",
                endpoint_identity="https://example.invalid/v1",
                generation_parameters={},
                adapter_defaults={},
                response_format={},
                reasoning_controls={},
                components=(),
            )
            for sequence in range(32):
                call = repository.reserve_call(
                    cursor,
                    owner_id=owner.owner_id,
                    segment_id=segment.segment_id,
                    turn_id=message,
                    run_id="surface-run",
                    call_sequence=sequence,
                    idempotency_key=f"surface-{sequence}",
                    policy_id=policy.policy_id,
                )
                repository.bind_call(
                    cursor,
                    call_id=call.call_id,
                    surface_node_id=tail.node_id,
                    request_header_id=header.header_id,
                    provider_name="openai",
                    model_name="surface-budget",
                    route_identity="fresh",
                )
                for state in (
                    TraceCallState.DISPATCH_STARTED,
                    TraceCallState.RESPONSE_STARTED,
                    TraceCallState.COMPLETE,
                ):
                    repository.advance_call_state(
                        cursor,
                        call_id=call.call_id,
                        target=state,
                        occurred_at="2026-10-27T00:00:00Z",
                        integrity_state="complete",
                    )
                repository.append_event(
                    cursor,
                    segment_id=segment.segment_id,
                    sequence=sequence,
                    event_type="call_boundary",
                    call_id=call.call_id,
                )
        # Ordinary None callers must still reconstruct the same ordered surface.
        calls = ConsoleTraceNativeReader(db).read_calls(message)
        assert len(calls) == 32
        for call in calls:
            assert (
                call.capture.request["messages_payload"] == [json.loads(payload)] * 16
            )
        ordinary.append(len(calls))

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=seed
    ) as reads:
        materializations = []
        cache_hits = []
        inherited_entries = []
        streamed = []
        original_projection = ConsoleTraceService._surface_projection
        original_materialize = _ProjectionRoot.materialize
        original_iter = _ProjectionRoot.iter_entries

        def observed_projection(self, cursor, segment_id, tail, **kwargs):
            assert kwargs.get("_read_budget") is not None
            cached = self._surface_ref_cache.get(segment_id)
            if cached is not None and cached.head_id == (
                None if tail is None else tail.node_id
            ):
                cache_hits.append(cached.root.size)
            if segment_id in children:
                inherited_entries.append(segment_id)
            return original_projection(self, cursor, segment_id, tail, **kwargs)

        def observed_materialize(self):
            result = original_materialize(self)
            caller = sys._getframe(2)
            materializations.append(
                (
                    caller.f_code.co_name,
                    linecache.getline(
                        caller.f_code.co_filename, caller.f_lineno
                    ).strip(),
                    len(result),
                )
            )
            return result

        def observed_iter(self):
            caller = sys._getframe(1)
            streamed.append(caller.f_code.co_name)
            yield from original_iter(self)

        monkeypatch.setattr(
            ConsoleTraceService, "_surface_projection", observed_projection
        )
        monkeypatch.setattr(_ProjectionRoot, "materialize", observed_materialize)
        monkeypatch.setattr(_ProjectionRoot, "iter_entries", observed_iter)
        with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
            reads.trace_inputs()
        assert ordinary == [32]
        assert cache_hits and all(size == 16 for size in cache_hits)
        assert bool(inherited_entries) is inherited
        inherited_boundary = (
            any("inherited_projection.entries" in item[1] for item in materializations)
            or "_surface_projection" in streamed
        )
        assert inherited_boundary is inherited
        assert materializations == [], (
            "budgeted iteration materialized complete native surfaces",
            materializations,
        )
    with pytest.raises(ValueError, match="inactive"):
        reads.check()
