from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from tldw_profile_core import (
    AgentVisibility,
    InterviewAudience,
    InterviewPack,
    InterviewQuestion,
    InterviewTurn,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    ProposalOperation,
    RecordState,
    SemanticKey,
    SyncMode,
    WorkingContextPayload,
)

from tldw_chatbook.Personal_Context.interview_coordinator import (
    InterviewCommitOutcomeUnknownError,
    ProfileInterviewCoordinator,
)
from tldw_chatbook.Personal_Context.interview_draft_repository import (
    InterviewDraftConflictError,
    InterviewDraftRepository,
)
from tldw_chatbook.Personal_Context.interview_provider import (
    ConfiguredModelQuestionProvider,
    FixedQuestionProvider,
    InterviewProviderError,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import PersonalContextService


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def _pack(audience: InterviewAudience = InterviewAudience.PERSONAL) -> InterviewPack:
    topics = (
        ("preferences", "identity")
        if audience is InterviewAudience.PERSONAL
        else ("goal", "working_context", "convention")
    )
    return InterviewPack(
        pack_id=f"{audience.value}-pack",
        pack_version=1,
        audience=audience,
        coverage_version=1,
        coverage_topics=topics,
        questions=tuple(
            InterviewQuestion(
                question_id=f"fixed-{index}",
                topic=topic,
                text=f"Tell me about {topic}?",
            )
            for index, topic in enumerate(topics)
        ),
    )


class _ServiceSpy:
    def __init__(self, records=()) -> None:
        self.records = tuple(records)
        self.commit_calls = []
        self.runtime_values = []
        self.list_calls = []
        self.validation_calls = []

    def list_records(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.records

    def validate_interview_target(self, *, scope_id, audience):
        self.validation_calls.append((scope_id, InterviewAudience(audience)))

    def commit_interview_changes(self, **kwargs):
        self.commit_calls.append(kwargs)
        return tuple(f"record-{index}" for index, _ in enumerate(kwargs["changes"]))

    def set_runtime_enabled(self, value):
        self.runtime_values.append(value)


class _FailingService(_ServiceSpy):
    def commit_interview_changes(self, **kwargs):
        self.commit_calls.append(kwargs)
        raise RuntimeError("safe commit failure")


class _RuntimeFailingService(_ServiceSpy):
    def set_runtime_enabled(self, value):
        self.runtime_values.append(value)
        raise RuntimeError("runtime policy unavailable")


class _BlockingCommitService(_ServiceSpy):
    def __init__(self) -> None:
        super().__init__()
        self.commit_entered = threading.Event()
        self.release_commit = threading.Event()

    def commit_interview_changes(self, **kwargs):
        self.commit_calls.append(kwargs)
        self.commit_entered.set()
        assert self.release_commit.wait(5)
        return tuple(f"record-{index}" for index, _ in enumerate(kwargs["changes"]))


class _RejectingTargetService(_ServiceSpy):
    def validate_interview_target(self, *, scope_id, audience):
        self.validation_calls.append((scope_id, InterviewAudience(audience)))
        raise ValueError("Interview scope does not match its audience.")


class _ProviderSpy:
    provider_id = "configured-provider"
    provider_label = "Configured Provider"
    model_id = "model-profile"

    def __init__(self, *, invalid=False, fail_first=False, topic="preferences") -> None:
        self.invalid = invalid
        self.fail_first = fail_first
        self.topic = topic
        self.calls = []

    def next_question(self, request):
        self.calls.append(request)
        if self.fail_first and len(self.calls) == 1:
            raise InterviewProviderError("provider_unavailable")
        text = "What do you prefer and why?" if self.invalid else "What do you prefer?"
        return InterviewQuestion(
            question_id=f"adaptive-{len(self.calls)}",
            topic=self.topic,
            text=text,
        )


class _RepeatedQuestionIdProvider(_ProviderSpy):
    def next_question(self, request):
        self.calls.append(request)
        return InterviewQuestion(
            question_id="reused-question-id",
            topic="preferences" if len(self.calls) == 1 else "identity",
            text="What should agents know?",
        )


class _FutureFixedQuestionIdProvider(_ProviderSpy):
    def next_question(self, request):
        self.calls.append(request)
        if len(self.calls) > 1:
            raise InterviewProviderError("provider_unavailable")
        return InterviewQuestion(
            question_id="fixed-1",
            topic="preferences",
            text="What response style do you prefer?",
        )


def _coordinator(*, audience=InterviewAudience.PERSONAL, adaptive=None, service=None):
    pack = _pack(audience)
    return ProfileInterviewCoordinator(
        service=service or _ServiceSpy(),
        drafts=InterviewDraftRepository.memory_only(clock=lambda: NOW),
        fixed_provider=FixedQuestionProvider(pack),
        adaptive_provider=adaptive,
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )


def test_fixed_mode_makes_zero_configured_provider_calls_and_finish_does_not_write() -> (
    None
):
    configured = _ProviderSpy()
    service = _ServiceSpy()
    coordinator = _coordinator(adaptive=configured, service=service)

    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "Prefer concise replies")
    diff = coordinator.finish(session.session_id)

    assert configured.calls == []
    assert diff.additions
    assert service.commit_calls == []


def test_session_discloses_memory_only_and_durable_draft_storage(
    tmp_path, memory_protector
) -> None:
    memory = _coordinator()
    durable = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=InterviewDraftRepository(
            tmp_path / "interview-drafts.db",
            key_protector=memory_protector,
            clock=lambda: NOW,
        ),
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-durable",
    )

    assert (
        memory.start(
            kind="personal", scope_id="scope-global", mode="fixed"
        ).draft_is_memory_only
        is True
    )
    assert (
        durable.start(
            kind="personal", scope_id="scope-global", mode="fixed"
        ).draft_is_memory_only
        is False
    )


def test_review_rewrite_is_validated_persisted_and_returns_refreshed_selection_id() -> (
    None
):
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    initial = coordinator.finish_early(session.session_id)
    initial_revision = drafts.require_active(session.session_id).revision

    rewritten = coordinator.rewrite_review_change(
        session.session_id,
        change_id=initial.changes[0].change_id,
        proposed_payload={
            "kind": "preference",
            "subject": "response.detail",
            "polarity": "dislike",
            "value": "verbose replies",
        },
        controls={
            "sync_mode": "device_only",
            "agent_visibility": "user_only",
        },
    )

    assert rewritten.change_id != initial.changes[0].change_id
    assert rewritten.diff == coordinator.review(session.session_id)
    assert drafts.require_active(session.session_id).revision == initial_revision + 1
    change = rewritten.diff.changes[0].change
    assert change.proposed_payload == PreferencePayload(
        subject="response.detail", polarity="dislike", value="verbose replies"
    )
    assert change.semantic_key == SemanticKey(
        namespace="preference", subject="response.detail"
    )
    assert change.controls == ProfileControls(
        sync_mode=SyncMode.DEVICE_ONLY,
        agent_visibility=AgentVisibility.USER_ONLY,
    )
    with pytest.raises(ValueError, match="invalid or stale"):
        coordinator.commit(
            session.session_id,
            selections=(initial.changes[0].change_id,),
            enable_runtime=False,
        )

    receipt = coordinator.commit(
        session.session_id,
        selections=(rewritten.change_id,),
        enable_runtime=False,
    )
    assert receipt.committed_record_ids == ("record-0",)


def test_review_rewrite_rejects_secret_without_changing_draft() -> None:
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    review = coordinator.finish_early(session.session_id)
    before = drafts.require_active(session.session_id)
    secret = "api_key: RAW_SECRET_CANARY_abcdefghijklmnop"

    with pytest.raises(ValueError, match="secret material") as failure:
        coordinator.rewrite_review_change(
            session.session_id,
            change_id=review.changes[0].change_id,
            proposed_payload={
                "kind": "preference",
                "subject": "preferences",
                "polarity": "like",
                "value": secret,
            },
            controls=review.changes[0].change.controls.model_dump(mode="json"),
        )

    after = drafts.require_active(session.session_id)
    assert secret not in str(failure.value)
    assert after.revision == before.revision
    assert after.payload == before.payload


def test_review_resume_and_rewrite_preserve_private_duplicate_warning() -> None:
    private = ProfileRecord(
        profile_id="profile-1",
        record_id="private-record",
        scope_id="scope-global",
        kind="preference",
        payload=PreferencePayload(
            subject="preferences", polarity="like", value="private value"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="preferences"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.USER_ONLY,
        ),
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="private-version",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )
    coordinator = _coordinator(service=_ServiceSpy((private,)))
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")

    initial = coordinator.finish_early(session.session_id)
    resumed = coordinator.review(session.session_id)
    rewritten = coordinator.rewrite_review_change(
        session.session_id,
        change_id=resumed.changes[0].change_id,
        proposed_payload=resumed.changes[0].change.proposed_payload.model_dump(
            mode="json"
        ),
        controls=resumed.changes[0].change.controls.model_dump(mode="json"),
    )

    assert initial.changes[0].possible_private_duplicate is True
    assert resumed.changes[0].possible_private_duplicate is True
    assert rewritten.diff.changes[0].possible_private_duplicate is True


def test_review_does_not_renormalize_persisted_create_after_visible_record_appears() -> (
    None
):
    service = _ServiceSpy()
    coordinator = _coordinator(service=service)
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    initial = coordinator.finish_early(session.session_id)
    initial_item = initial.changes[0]
    assert initial_item.change.operation is ProposalOperation.CREATE

    service.records = (
        ProfileRecord(
            profile_id="profile-1",
            record_id="visible-record",
            scope_id="scope-global",
            kind="preference",
            payload=PreferencePayload(
                subject="preferences", polarity="like", value="new concurrent value"
            ),
            semantic_key=SemanticKey(namespace="preference", subject="preferences"),
            state=RecordState.ACTIVE,
            controls=ProfileControls(
                sync_mode=SyncMode.SYNCABLE,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
            provenance=ProfileProvenance(
                source="manual", actor="user", reason_code="settings_edit"
            ),
            version_id="visible-version",
            parent_version_id=None,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    resumed = coordinator.review(session.session_id)
    assert resumed.changes[0].change_id == initial_item.change_id
    assert resumed.changes[0].change.operation is ProposalOperation.CREATE

    rewritten = coordinator.rewrite_review_change(
        session.session_id,
        change_id=resumed.changes[0].change_id,
        proposed_payload=resumed.changes[0].change.proposed_payload.model_dump(
            mode="json"
        ),
        controls=resumed.changes[0].change.controls.model_dump(mode="json"),
    )
    receipt = coordinator.commit(
        session.session_id,
        selections=(rewritten.change_id,),
        enable_runtime=False,
    )
    assert receipt.committed_record_ids == ("record-0",)


def test_adaptive_mode_pins_disclosed_provider_model_and_counts_twenty_invalid_attempts() -> (
    None
):
    provider = _ProviderSpy(invalid=True)
    coordinator = _coordinator(adaptive=provider)

    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    assert session.turns == ()
    assert session.provider_label == "Configured Provider"
    assert session.model_id == "model-profile"
    assert session.external_retention_notice == (
        "External retention is controlled by the selected provider."
    )
    for _ in range(19):
        progress = coordinator.retry(session.session_id)

    assert session.provider_label == "Configured Provider"
    assert session.model_id == "model-profile"
    assert progress.question_attempts == 20
    assert progress.can_ask_another is False
    assert len(provider.calls) == 20
    assert progress.question is None


def test_adaptive_fallback_preserves_attempt_budget_and_never_exceeds_twenty() -> None:
    provider = _ProviderSpy(invalid=True)
    coordinator = _coordinator(adaptive=provider)
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    for _ in range(18):
        session = coordinator.retry(session.session_id)

    fallback = coordinator.use_fixed_fallback(session.session_id)
    final = coordinator.skip(session.session_id)

    assert len(provider.calls) == 19
    assert fallback.question_attempts == 20
    assert final.question_attempts == 20
    assert final.can_ask_another is False


def test_repeated_adaptive_topic_retains_only_the_last_exact_answer() -> None:
    coordinator = _coordinator(adaptive=_ProviderSpy())
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    session = coordinator.answer(session.session_id, "detailed")
    coordinator.answer(session.session_id, "concise")

    diff = coordinator.finish_early(session.session_id)

    assert len(diff.changes) == 1
    assert diff.changes[0].change.proposed_payload.value == "concise"


def test_adaptive_session_fails_closed_if_pinned_provider_or_model_changes() -> None:
    provider = _ProviderSpy(fail_first=True)
    coordinator = _coordinator(adaptive=provider)
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    provider.provider_label = "Different Provider"
    provider.model_id = "different-model"

    retried = coordinator.retry(session.session_id)

    assert retried.provider_label == "Configured Provider"
    assert retried.model_id == "model-profile"
    assert retried.provider_error == "provider_selection_changed"
    assert len(provider.calls) == 1


def test_skip_advances_without_answer_and_finish_early_stops_questioning() -> None:
    coordinator = _coordinator()
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")

    skipped = coordinator.skip(session.session_id)
    finished = coordinator.finish_early(session.session_id)

    assert skipped.turns == ()
    assert skipped.question.question_id == "fixed-1"
    assert finished.changes == ()


def test_protected_session_resumes_with_question_turns_and_disclosure(tmp_path) -> None:
    from tldw_chatbook.Personal_Context.key_protector import (
        InMemoryProfileKeyProtector,
    )

    protector = InMemoryProfileKeyProtector()
    drafts = InterviewDraftRepository(
        tmp_path / "interviews.db", key_protector=protector, clock=lambda: NOW
    )
    first = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = first.start(kind="personal", scope_id="scope-global", mode="fixed")
    first.answer(session.session_id, "concise")

    resumed = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=InterviewDraftRepository(
            tmp_path / "interviews.db",
            key_protector=protector,
            clock=lambda: NOW,
        ),
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
    ).resume("session-1")

    assert resumed.turns == (InterviewTurn(question_id="fixed-0", answer="concise"),)
    assert resumed.question.question_id == "fixed-1"
    assert resumed.provider_label == "Fixed local questionnaire"


def test_changed_question_pack_makes_saved_draft_stale_without_calls_or_writes() -> (
    None
):
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    service = _ServiceSpy()
    first = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    first.start(kind="personal", scope_id="scope-global", mode="fixed")
    persisted = drafts.load("session-1").payload
    assert persisted["pack_id"] == "personal-pack"
    assert persisted["pack_version"] == 1
    assert persisted["coverage_version"] == 1
    changed_pack = _pack().model_copy(update={"pack_id": "replacement-pack"})
    configured = _ProviderSpy()
    reopened = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(changed_pack),
        adaptive_provider=configured,
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="question pack is stale"):
        reopened.resume("session-1")

    assert configured.calls == []
    assert service.commit_calls == []


def test_target_validation_precedes_adaptive_provider_and_record_access() -> None:
    service = _RejectingTargetService()
    provider = _ProviderSpy()
    coordinator = _coordinator(adaptive=provider, service=service)

    with pytest.raises(ValueError, match="scope does not match"):
        coordinator.start(kind="personal", scope_id="scope-workspace", mode="adaptive")

    assert provider.calls == []
    assert service.list_calls == []


def test_resume_revalidates_target_before_exposing_records_or_calling_provider() -> (
    None
):
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    first = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    first.start(kind="personal", scope_id="scope-global", mode="fixed")
    service = _RejectingTargetService()
    provider = _ProviderSpy()
    reopened = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        adaptive_provider=provider,
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="scope does not match"):
        reopened.resume("session-1")

    assert provider.calls == []
    assert service.list_calls == []


def test_configured_provider_uses_pinned_model_structured_output_and_no_tools() -> None:
    calls = []

    def call(
        *, api_endpoint, messages_payload, model, tools, streaming, response_format
    ):
        calls.append(
            {
                "api_endpoint": api_endpoint,
                "messages_payload": messages_payload,
                "model": model,
                "tools": tools,
                "streaming": streaming,
                "response_format": response_format,
            }
        )
        return {
            "question_id": "q-1",
            "topic": "preferences",
            "text": "What response style do you prefer?",
        }

    provider = ConfiguredModelQuestionProvider(
        provider_id="openai",
        provider_label="OpenAI",
        model_id="gpt-profile",
        call=call,
    )
    question = provider.next_question(
        request=type(
            "Request",
            (),
            {
                "pack": _pack(),
                "turns": (),
                "existing_records": (),
                "scope_id": "scope-global",
                "question_attempt": 1,
            },
        )()
    )

    assert question.question_id == "q-1"
    assert calls[0]["api_endpoint"] == "openai"
    assert calls[0]["model"] == "gpt-profile"
    assert calls[0]["tools"] is None
    assert calls[0]["streaming"] is False
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert (
        json.dumps(
            InterviewQuestion.model_json_schema(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        in calls[0]["messages_payload"][0]["content"]
    )


def test_configured_provider_rejects_malformed_structured_output() -> None:
    provider = ConfiguredModelQuestionProvider(
        provider_id="openai",
        provider_label="OpenAI",
        model_id="gpt-profile",
        call=lambda **_kwargs: '{"question_id":"missing-fields"}',
    )

    with pytest.raises(InterviewProviderError) as failure:
        provider.next_question(
            type(
                "Request",
                (),
                {
                    "pack": _pack(),
                    "turns": (),
                    "existing_records": (),
                    "scope_id": "scope-global",
                    "question_attempt": 1,
                },
            )()
        )

    assert failure.value.reason_code == "invalid_question"


def test_configured_provider_extracts_openai_shaped_chat_api_response() -> None:
    provider = ConfiguredModelQuestionProvider(
        provider_id="openai",
        provider_label="OpenAI",
        model_id="gpt-profile",
        call=lambda **_kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "question_id": "q-openai",
                                "topic": "preferences",
                                "text": "What response style do you prefer?",
                            }
                        )
                    }
                }
            ]
        },
    )

    question = provider.next_question(
        type(
            "Request",
            (),
            {
                "pack": _pack(),
                "turns": (),
                "existing_records": (),
                "scope_id": "scope-global",
                "question_attempt": 1,
            },
        )()
    )

    assert question.question_id == "q-openai"


def test_secret_answer_is_refused_and_provider_failure_can_retry_or_fallback() -> None:
    provider = _ProviderSpy(fail_first=True)
    coordinator = _coordinator(adaptive=provider)
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )

    assert session.provider_error == "provider_unavailable"
    progress = coordinator.retry(session.session_id)
    assert progress.question is not None
    secret = "api_key: RAW_SECRET_CANARY_abcdefghijklmnop"
    with pytest.raises(ValueError, match="secret material") as failure:
        coordinator.answer(session.session_id, secret)
    assert secret not in str(failure.value)

    fixed = coordinator.use_fixed_fallback(session.session_id)
    assert fixed.provider_label == "Fixed local questionnaire"


def test_adaptive_request_excludes_user_only_wrong_scope_archived_and_expired_records() -> (
    None
):
    def record(
        record_id,
        *,
        scope="scope-global",
        visibility="agent_visible",
        state="active",
        expires_at=None,
        sync_mode="syncable",
    ):
        is_expiring = expires_at is not None
        payload = (
            WorkingContextPayload(subject=record_id, value=f"VALUE-{record_id}")
            if is_expiring
            else PreferencePayload(
                subject=record_id, polarity="like", value=f"VALUE-{record_id}"
            )
        )
        timestamp = NOW.replace(year=2024) if is_expiring else NOW
        return ProfileRecord(
            profile_id="profile-1",
            record_id=record_id,
            scope_id=scope,
            kind=payload.kind,
            payload=payload,
            semantic_key=SemanticKey(namespace=payload.kind, subject=record_id),
            state=RecordState(state),
            controls=ProfileControls(
                sync_mode=SyncMode(sync_mode),
                agent_visibility=AgentVisibility(visibility),
            ),
            provenance=ProfileProvenance(
                source="manual", actor="user", reason_code="settings_edit"
            ),
            version_id=f"version-{record_id}",
            parent_version_id=None,
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=expires_at,
            no_expiry=False,
        )

    eligible = record("eligible")
    provider = _ProviderSpy()
    coordinator = _coordinator(
        adaptive=provider,
        service=_ServiceSpy(
            (
                eligible,
                record("private", visibility="user_only"),
                record("wrong-scope", scope="scope-other"),
                record("archived", state="archived"),
                record("expired", expires_at=NOW.replace(year=2025)),
                record("device-only", sync_mode="device_only"),
            )
        ),
    )

    coordinator.start(kind="personal", scope_id="scope-global", mode="adaptive")

    assert provider.calls[0].existing_records == (eligible,)


def test_adaptive_out_of_pack_topic_consumes_attempt_and_never_becomes_proposal() -> (
    None
):
    provider = _ProviderSpy(topic="outside-coverage")
    coordinator = _coordinator(adaptive=provider)

    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    diff = coordinator.finish_early(session.session_id)

    assert session.question_attempts == 1
    assert session.question is None
    assert session.provider_error == "invalid_question"
    assert diff.changes == ()


def test_adaptive_reused_question_id_is_invalid_and_cannot_reclassify_answer() -> None:
    provider = _RepeatedQuestionIdProvider()
    coordinator = _coordinator(adaptive=provider)
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )

    progress = coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish_early(session.session_id)

    assert progress.question_attempts == 2
    assert progress.question is None
    assert progress.provider_error == "invalid_question"
    assert len(diff.changes) == 1
    assert diff.changes[0].change.proposed_payload.kind == "preference"


def test_fixed_fallback_cannot_reuse_an_adaptive_question_id() -> None:
    coordinator = _coordinator(adaptive=_FutureFixedQuestionIdProvider())
    session = coordinator.start(
        kind="personal", scope_id="scope-global", mode="adaptive"
    )
    coordinator.answer(session.session_id, "concise")
    fallback = coordinator.use_fixed_fallback(session.session_id)

    progress = coordinator.skip(fallback.session_id)
    diff = coordinator.finish_early(session.session_id)

    assert progress.question is None
    assert progress.provider_error == "invalid_question"
    assert len(diff.changes) == 1
    assert diff.changes[0].change.proposed_payload.kind == "preference"
    assert diff.changes[0].change.proposed_payload.value == "concise"


def test_concurrent_retries_reserve_attempt_with_cas_before_provider_call(
    monkeypatch,
) -> None:
    provider = _ProviderSpy(invalid=True)
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    first = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        adaptive_provider=provider,
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    second = ProfileInterviewCoordinator(
        service=_ServiceSpy(),
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        adaptive_provider=provider,
        clock=lambda: NOW,
    )
    session = first.start(kind="personal", scope_id="scope-global", mode="adaptive")
    for _ in range(18):
        first.retry(session.session_id)
    real_require = drafts.require_active
    barrier = threading.Barrier(2)

    def synchronized_require(session_id):
        stored = real_require(session_id)
        barrier.wait(timeout=5)
        return stored

    monkeypatch.setattr(drafts, "require_active", synchronized_require)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(coordinator.retry, session.session_id)
            for coordinator in (first, second)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except InterviewDraftConflictError:
                outcomes.append("conflict")

    assert len(provider.calls) == 20
    assert sum(item == "conflict" for item in outcomes) == 1


def test_commit_applies_only_selected_diff_once_then_destroys_draft() -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)

    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )

    assert len(service.commit_calls) == 1
    assert len(service.commit_calls[0]["changes"]) == 1
    assert receipt.committed_record_ids == ("record-0",)
    assert service.runtime_values == [True]
    assert drafts.load(session.session_id) is None


def test_commit_reserves_draft_before_canonical_mutation() -> None:
    service = _BlockingCommitService()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            coordinator.commit,
            session.session_id,
            selections=(diff.changes[0].change_id,),
            enable_runtime=True,
        )
        assert service.commit_entered.wait(1)
        reserved = drafts.require_active(session.session_id)
        assert reserved.payload["status"] == "committing"
        assert reserved.payload["runtime_requested"] is True
        service.release_commit.set()
        future.result(timeout=5)


def test_concurrent_commits_allow_only_one_canonical_mutation(monkeypatch) -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    first = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    second = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
    )
    session = first.start(kind="personal", scope_id="scope-global", mode="fixed")
    first.answer(session.session_id, "concise")
    diff = first.finish(session.session_id)
    real_require = drafts.require_active
    barrier = threading.Barrier(2)

    def synchronized_require(session_id):
        stored = real_require(session_id)
        barrier.wait(timeout=5)
        return stored

    monkeypatch.setattr(drafts, "require_active", synchronized_require)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                coordinator.commit,
                session.session_id,
                selections=(diff.changes[0].change_id,),
                enable_runtime=True,
            )
            for coordinator in (first, second)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except InterviewDraftConflictError:
                outcomes.append("conflict")

    assert len(service.commit_calls) == 1
    assert sum(item == "conflict" for item in outcomes) == 1


def test_newer_rewrite_prevents_stale_commit_before_canonical_mutation(
    monkeypatch,
) -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    committing = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    rewriting = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
    )
    session = committing.start(kind="personal", scope_id="scope-global", mode="fixed")
    committing.answer(session.session_id, "concise")
    review = committing.finish(session.session_id)
    old_change = review.changes[0]
    reservation_entered = threading.Event()
    release_reservation = threading.Event()
    real_save = committing._save

    def delayed_reservation(state):
        if state["status"] == "committing":
            reservation_entered.set()
            assert release_reservation.wait(5)
        real_save(state)

    monkeypatch.setattr(committing, "_save", delayed_reservation)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            committing.commit,
            session.session_id,
            selections=(old_change.change_id,),
            enable_runtime=False,
        )
        assert reservation_entered.wait(1)
        payload = old_change.change.proposed_payload.model_dump(mode="json")
        payload["value"] = "rewritten before commit"
        rewritten = rewriting.rewrite_review_change(
            session.session_id,
            change_id=old_change.change_id,
            proposed_payload=payload,
            controls=old_change.change.controls.model_dump(mode="json"),
        )
        release_reservation.set()
        with pytest.raises(InterviewDraftConflictError):
            future.result(timeout=5)

    assert service.commit_calls == []
    assert rewriting.review(session.session_id) == rewritten.diff


def test_failed_commit_preserves_draft_and_does_not_toggle_runtime() -> None:
    service = _FailingService()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)

    with pytest.raises(RuntimeError, match="safe commit failure"):
        coordinator.commit(
            session.session_id,
            selections=(diff.changes[0].change_id,),
            enable_runtime=True,
        )

    assert coordinator.resume(session.session_id).status == "review"
    assert coordinator.review(session.session_id) == diff
    assert service.runtime_values == []


def test_failed_commit_with_failed_review_recovery_is_terminal_unknown(
    monkeypatch,
) -> None:
    service = _FailingService()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)
    real_save = coordinator._save

    def fail_review_recovery(state):
        if state["status"] == "review":
            raise InterviewDraftConflictError("PRIVATE_RECOVERY_CANARY")
        real_save(state)

    monkeypatch.setattr(coordinator, "_save", fail_review_recovery)

    with pytest.raises(InterviewCommitOutcomeUnknownError) as failure:
        coordinator.commit(
            session.session_id,
            selections=(diff.changes[0].change_id,),
            enable_runtime=True,
        )

    assert str(failure.value) == "Interview commit outcome is unknown."
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert "PRIVATE_RECOVERY_CANARY" not in repr(failure.value)
    assert drafts.require_active(session.session_id).payload["status"] == "committing"
    assert service.runtime_values == []


def test_runtime_failure_after_durable_records_is_reported_separately() -> None:
    service = _RuntimeFailingService()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)

    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )

    assert receipt.committed_record_ids == ("record-0",)
    assert receipt.runtime_update_succeeded is False
    assert receipt.draft_cleanup_succeeded is True
    assert drafts.load(session.session_id) is None


def test_post_commit_marker_failure_does_not_interrupt_finalization(
    monkeypatch,
) -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)
    real_save = coordinator._save
    save_calls = 0

    def fail_first_post_commit_marker(state):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise InterviewDraftConflictError("concurrent marker update")
        real_save(state)

    monkeypatch.setattr(coordinator, "_save", fail_first_post_commit_marker)

    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )

    assert receipt.committed_record_ids == ("record-0",)
    assert receipt.runtime_update_succeeded is True
    assert receipt.draft_cleanup_succeeded is True
    assert receipt.draft_cleanup_retry_required is False
    assert service.runtime_values == [True]
    assert drafts.load(session.session_id) is None


@pytest.mark.parametrize("runtime_fails", [False, True])
def test_post_commit_cleanup_failure_is_independent_and_explicitly_retriable(
    monkeypatch, runtime_fails
) -> None:
    service = _RuntimeFailingService() if runtime_fails else _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)
    real_delete = drafts.delete
    monkeypatch.setattr(
        drafts,
        "delete",
        lambda _session_id: (_ for _ in ()).throw(
            RuntimeError("protector unavailable")
        ),
    )

    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )

    assert receipt.committed_record_ids == ("record-0",)
    assert receipt.runtime_update_succeeded is (not runtime_fails)
    assert receipt.draft_cleanup_succeeded is False
    assert receipt.draft_cleanup_retry_required is True
    monkeypatch.setattr(drafts, "delete", real_delete)
    assert coordinator.retry_draft_cleanup(session.session_id) is True
    assert drafts.load(session.session_id) is None


def test_cleanup_failure_leaves_only_terminal_committed_resume_and_cleanup(
    monkeypatch,
) -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)
    real_delete = drafts.delete
    monkeypatch.setattr(
        drafts,
        "delete",
        lambda _session_id: (_ for _ in ()).throw(
            RuntimeError("protector unavailable")
        ),
    )
    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )

    resumed = coordinator.resume(session.session_id)
    before = drafts.load(session.session_id)
    assert resumed.status == "committed"
    assert resumed.committed_record_ids == receipt.committed_record_ids
    assert resumed.runtime_requested is True
    assert resumed.runtime_update_succeeded is True
    actions = (
        lambda: coordinator.finish(session.session_id),
        lambda: coordinator.answer(session.session_id, "changed"),
        lambda: coordinator.skip(session.session_id),
        lambda: coordinator.retry(session.session_id),
        lambda: coordinator.use_fixed_fallback(session.session_id),
        lambda: coordinator.commit(
            session.session_id,
            selections=(diff.changes[0].change_id,),
            enable_runtime=False,
        ),
    )
    for action in actions:
        with pytest.raises(ValueError, match="already committed"):
            action()
        assert drafts.load(session.session_id) == before

    monkeypatch.setattr(drafts, "delete", real_delete)
    assert coordinator.retry_draft_cleanup(session.session_id) is True
    assert drafts.load(session.session_id) is None


def test_failed_runtime_outcome_marker_resumes_as_unknown(monkeypatch) -> None:
    service = _ServiceSpy()
    drafts = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=drafts,
        fixed_provider=FixedQuestionProvider(_pack()),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )
    session = coordinator.start(kind="personal", scope_id="scope-global", mode="fixed")
    coordinator.answer(session.session_id, "concise")
    diff = coordinator.finish(session.session_id)
    real_save = coordinator._save
    save_calls = 0

    def fail_runtime_outcome_marker(state):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 3:
            raise InterviewDraftConflictError("outcome marker unavailable")
        real_save(state)

    monkeypatch.setattr(coordinator, "_save", fail_runtime_outcome_marker)
    monkeypatch.setattr(
        drafts,
        "delete",
        lambda _session_id: (_ for _ in ()).throw(
            RuntimeError("protector unavailable")
        ),
    )

    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=True,
    )
    resumed = coordinator.resume(session.session_id)

    assert receipt.runtime_update_succeeded is True
    assert receipt.draft_cleanup_succeeded is False
    assert resumed.runtime_requested is True
    assert resumed.runtime_update_succeeded is None


def test_workspace_interview_outputs_only_workspace_safe_kinds_and_scope() -> None:
    service = _ServiceSpy()
    coordinator = _coordinator(
        audience=InterviewAudience.WORKSPACE,
        service=service,
    )
    session = coordinator.start(
        kind="workspace", scope_id="scope-workspace", mode="fixed"
    )
    while session.question is not None:
        progress = coordinator.answer(session.session_id, "workspace answer")
        session = progress
    diff = coordinator.finish("session-1")
    coordinator.commit(
        "session-1",
        selections=tuple(item.change_id for item in diff.changes),
        enable_runtime=False,
    )

    committed = service.commit_calls[0]
    assert committed["scope_id"] == "scope-workspace"
    assert {item.proposed_payload.kind for item in committed["changes"]} <= {
        "goal",
        "working_context",
        "convention",
    }
    assert all(item.target_record_id is None for item in committed["changes"])


def test_expired_working_context_reinterview_creates_fresh_bounded_record(
    tmp_path, memory_protector
) -> None:
    next_id = iter(range(1, 100)).__next__
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "personal-context.db", key_protector=memory_protector
        ),
        clock=lambda: NOW - timedelta(days=2),
        id_factory=lambda label: f"{label}-{next_id()}",
    )
    service.create_profile()
    scope = service.create_workspace_scope("workspace-1", "Project")
    expired = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=WorkingContextPayload(
            subject="working_context", value="obsolete context"
        ),
        semantic_key={
            "namespace": "working_context",
            "subject": "working_context",
        },
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        expires_at=NOW - timedelta(days=1),
    )
    service.clock = lambda: NOW
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=InterviewDraftRepository.memory_only(clock=lambda: NOW),
        fixed_provider=FixedQuestionProvider(_pack(InterviewAudience.WORKSPACE)),
        clock=lambda: NOW,
        id_factory=lambda _label: "session-1",
    )

    session = coordinator.start(kind="workspace", scope_id=scope.scope_id, mode="fixed")
    session = coordinator.skip(session.session_id)
    assert session.question is not None
    assert session.question.topic == "working_context"
    coordinator.answer(session.session_id, "fresh context")
    diff = coordinator.finish_early(session.session_id)

    assert len(diff.changes) == 1
    assert diff.changes[0].change.operation is ProposalOperation.CREATE
    receipt = coordinator.commit(
        session.session_id,
        selections=(diff.changes[0].change_id,),
        enable_runtime=False,
    )
    fresh = service.get_record(receipt.committed_record_ids[0])
    assert fresh is not None
    assert fresh.record_id != expired.record_id
    assert fresh.expires_at == NOW + timedelta(days=30)
    assert fresh in service.list_records(scope_ids=(scope.scope_id,))
