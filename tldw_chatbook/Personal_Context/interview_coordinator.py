"""Bounded, resumable Personal Context interview state machine."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError
from tldw_profile_core import (
    AgentVisibility,
    ConventionPayload,
    GoalPayload,
    IdentityPayload,
    InterviewAudience,
    InterviewProposalBatch,
    InterviewProposedChange,
    InterviewQuestion,
    InterviewTurn,
    PreferencePayload,
    ProfileControls,
    ProfileRecord,
    ProposalOperation,
    RecordState,
    SemanticKey,
    SyncMode,
    WorkingContextPayload,
)

from .interview_diff import InterviewDiff, build_interview_diff
from .interview_draft_repository import InterviewDraftRepository
from .interview_provider import (
    FixedQuestionProvider,
    InterviewProviderError,
    InterviewProviderRequest,
    InterviewQuestionProvider,
)


_MAX_QUESTION_ATTEMPTS = 20


def _clock() -> datetime:
    return datetime.now(UTC)


def _id(label: str) -> str:
    return f"{label}-{uuid.uuid4()}"


@dataclass(frozen=True, slots=True)
class InterviewSession:
    session_id: str
    kind: str
    scope_id: str
    mode: str
    provider_label: str
    model_id: str | None
    external_retention_notice: str
    question_attempts: int
    question: InterviewQuestion | None
    status: str
    committed_record_ids: tuple[str, ...]
    turns: tuple[InterviewTurn, ...] = field(default=(), repr=False)
    can_ask_another: bool = True
    provider_error: str | None = None


@dataclass(frozen=True, slots=True)
class InterviewCommitReceipt:
    committed_record_ids: tuple[str, ...]
    runtime_update_succeeded: bool
    draft_cleanup_succeeded: bool
    draft_cleanup_retry_required: bool


class ProfileInterviewCoordinator:
    """Coordinate questions, encrypted drafts, review, and one atomic commit."""

    def __init__(
        self,
        *,
        service: Any,
        drafts: InterviewDraftRepository,
        fixed_provider: FixedQuestionProvider,
        adaptive_provider: InterviewQuestionProvider | None = None,
        clock: Callable[[], datetime] = _clock,
        id_factory: Callable[[str], str] = _id,
    ) -> None:
        self._service = service
        self._drafts = drafts
        self._fixed = fixed_provider
        self._adaptive = adaptive_provider
        self._clock = clock
        self._ids = id_factory

    def start(self, *, kind: str, scope_id: str, mode: str) -> InterviewSession:
        audience = InterviewAudience(kind)
        if audience is not self._fixed.pack.audience:
            raise ValueError("Interview pack audience mismatch.")
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("Interview mode must be fixed or adaptive.")
        self._service.validate_interview_target(
            scope_id=scope_id,
            audience=audience,
        )
        provider = self._provider(mode)
        state = {
            "version": 1,
            "session_id": self._ids("interview-session"),
            "pack_id": self._fixed.pack.pack_id,
            "pack_version": self._fixed.pack.pack_version,
            "coverage_version": self._fixed.pack.coverage_version,
            "kind": audience.value,
            "scope_id": scope_id,
            "mode": mode,
            "provider_id": provider.provider_id,
            "provider_label": provider.provider_label,
            "model_id": provider.model_id,
            "external_retention_notice": (
                "External retention is controlled by the selected provider."
                if mode == "adaptive"
                else "No external provider is used by the fixed questionnaire."
            ),
            "question_attempts": 0,
            "fixed_question_index": 0,
            "question": None,
            "turns": [],
            "asked_questions": [],
            "provider_error": None,
            "status": "active",
            "batch": None,
            "expires_at": (self._clock() + timedelta(days=30)).isoformat(),
        }
        # Persist disclosure before any configured provider is invoked.
        self._save(state)
        self._ask_next(state)
        return self._view(state)

    def resume(self, session_id: str) -> InterviewSession:
        return self._view(self._load(session_id))

    def answer(self, session_id: str, answer: str) -> InterviewSession:
        state = self._load(session_id)
        self._require_not_committed(state)
        question = self._question(state)
        if question is None:
            raise ValueError("No interview question is awaiting an answer.")
        try:
            turn = InterviewTurn(question_id=question.question_id, answer=answer)
        except ValidationError as exc:
            if any(error.get("type") == "value_error" for error in exc.errors()):
                raise ValueError(
                    "Interview answers cannot contain secret material."
                ) from None
            raise ValueError("Interview answer is invalid.") from None
        state["turns"].append(turn.model_dump(mode="json"))
        state["question"] = None
        self._ask_next(state)
        return self._view(state)

    def skip(self, session_id: str) -> InterviewSession:
        state = self._load(session_id)
        self._require_not_committed(state)
        if self._question(state) is None:
            raise ValueError("No interview question is available to skip.")
        state["question"] = None
        self._ask_next(state)
        return self._view(state)

    def retry(self, session_id: str) -> InterviewSession:
        state = self._load(session_id)
        self._require_not_committed(state)
        if self._question(state) is not None:
            raise ValueError("Answer or skip the current question first.")
        self._ask_next(state)
        return self._view(state)

    def use_fixed_fallback(self, session_id: str) -> InterviewSession:
        state = self._load(session_id)
        self._require_not_committed(state)
        state.update(
            {
                "mode": "fixed",
                "provider_id": self._fixed.provider_id,
                "provider_label": self._fixed.provider_label,
                "model_id": self._fixed.model_id,
                "external_retention_notice": (
                    "No external provider is used by the fixed questionnaire."
                ),
                "fixed_question_index": 0,
                "question": None,
                "provider_error": None,
            }
        )
        self._ask_next(state)
        return self._view(state)

    def finish(self, session_id: str) -> InterviewDiff:
        state = self._load(session_id)
        self._require_not_committed(state)
        batch = self._proposal_batch(state)
        existing = tuple(
            record
            for record in self._service.list_records(
                scope_ids=(state["scope_id"],), include_archived=True
            )
            if record.scope_id == state["scope_id"]
        )
        diff = build_interview_diff(batch, existing, now=self._clock())
        normalized = InterviewProposalBatch(
            pack_id=batch.pack_id,
            pack_version=batch.pack_version,
            audience=batch.audience,
            changes=tuple(item.change for item in diff.changes),
        )
        state.update(
            {
                "status": "review",
                "question": None,
                "provider_error": None,
                "batch": normalized.model_dump(mode="json"),
            }
        )
        self._save(state)
        return diff

    def finish_early(self, session_id: str) -> InterviewDiff:
        return self.finish(session_id)

    def commit(
        self,
        session_id: str,
        *,
        selections: tuple[str, ...],
        enable_runtime: bool,
    ) -> InterviewCommitReceipt:
        state = self._load(session_id)
        self._require_not_committed(state)
        if state["status"] != "review" or state["batch"] is None:
            raise ValueError("Interview must be reviewed before commit.")
        batch = InterviewProposalBatch.model_validate(state["batch"])
        diff = build_interview_diff(batch, (), now=self._clock())
        selected_ids = set(selections)
        known_ids = {item.change_id for item in diff.changes}
        if len(selected_ids) != len(selections) or not selected_ids <= known_ids:
            raise ValueError("Interview selections are invalid or stale.")
        selected = tuple(
            item.change for item in diff.changes if item.change_id in selected_ids
        )
        committed = self._service.commit_interview_changes(
            scope_id=state["scope_id"],
            audience=InterviewAudience(state["kind"]),
            changes=selected,
        )
        committed_ids = tuple(
            item.record_id if isinstance(item, ProfileRecord) else str(item)
            for item in committed
        )
        state.update(
            {
                "status": "committed",
                "committed_record_ids": committed_ids,
            }
        )
        try:
            self._save(state)
        except Exception:
            # Records are already durable. The marker is only a best-effort guard
            # for a draft whose subsequent key cleanup also fails.
            pass
        runtime_update_succeeded = True
        try:
            self._service.set_runtime_enabled(enable_runtime)
        except Exception:
            runtime_update_succeeded = False
        draft_cleanup_succeeded = True
        try:
            self._drafts.delete(session_id)
        except Exception:
            draft_cleanup_succeeded = False
        return InterviewCommitReceipt(
            committed_record_ids=committed_ids,
            runtime_update_succeeded=runtime_update_succeeded,
            draft_cleanup_succeeded=draft_cleanup_succeeded,
            draft_cleanup_retry_required=not draft_cleanup_succeeded,
        )

    def retry_draft_cleanup(self, session_id: str) -> bool:
        """Retry destruction after a separately reported post-commit failure."""

        self._drafts.delete(session_id)
        return True

    def discard(self, session_id: str) -> None:
        self._drafts.delete(session_id)

    def _provider(self, mode: str) -> InterviewQuestionProvider:
        if mode == "fixed":
            return self._fixed
        if self._adaptive is None:
            raise ValueError("Adaptive interview provider is unavailable.")
        return self._adaptive

    def _ask_next(self, state: dict[str, Any]) -> None:
        if state["status"] != "active":
            return
        provider = self._provider(state["mode"])
        if state["mode"] == "adaptive" and (
            provider.provider_id != state["provider_id"]
            or provider.provider_label != state["provider_label"]
            or provider.model_id != state["model_id"]
        ):
            state.update(
                {"question": None, "provider_error": "provider_selection_changed"}
            )
            self._save(state)
            return
        if state["question_attempts"] >= _MAX_QUESTION_ATTEMPTS or (
            state["mode"] == "fixed"
            and state["fixed_question_index"] >= len(self._fixed.pack.questions)
        ):
            state.update(
                {"question": None, "provider_error": None, "status": "complete"}
            )
            self._save(state)
            return
        state["question_attempts"] += 1
        self._save(state)
        request = InterviewProviderRequest(
            pack=self._fixed.pack,
            scope_id=state["scope_id"],
            question_attempt=(
                state["fixed_question_index"] + 1
                if state["mode"] == "fixed"
                else state["question_attempts"]
            ),
            turns=self._turns(state),
            existing_records=self._eligible_existing(state["scope_id"]),
        )
        try:
            question = provider.next_question(request)
        except InterviewProviderError as exc:
            if exc.reason_code == "question_pack_complete":
                state.update(
                    {"question": None, "provider_error": None, "status": "complete"}
                )
            else:
                state.update({"question": None, "provider_error": exc.reason_code})
        except (ValidationError, TypeError, ValueError):
            state.update({"question": None, "provider_error": "invalid_question"})
        except Exception:
            state.update({"question": None, "provider_error": "provider_unavailable"})
        else:
            known_question_ids = {
                item["question_id"] for item in state["asked_questions"]
            }
            invalid_topic = (
                state["mode"] == "adaptive"
                and question.topic not in self._fixed.pack.coverage_topics
            )
            if invalid_topic or question.question_id in known_question_ids:
                state.update({"question": None, "provider_error": "invalid_question"})
                if state["mode"] == "fixed":
                    state["fixed_question_index"] += 1
                    if state["fixed_question_index"] >= len(self._fixed.pack.questions):
                        state["status"] = "complete"
            else:
                state["question"] = question.model_dump(mode="json")
                state["asked_questions"].append(question.model_dump(mode="json"))
                state["provider_error"] = None
                if state["mode"] == "fixed":
                    state["fixed_question_index"] += 1
        self._save(state)

    def _eligible_existing(self, scope_id: str) -> tuple[ProfileRecord, ...]:
        now = self._clock()
        return tuple(
            record
            for record in self._service.list_records(
                scope_ids=(scope_id,), include_archived=False
            )
            if record.scope_id == scope_id
            and record.state is RecordState.ACTIVE
            and record.controls.agent_visibility is AgentVisibility.AGENT_VISIBLE
            and record.controls.sync_mode is SyncMode.SYNCABLE
            and (record.expires_at is None or record.expires_at > now)
        )

    def _proposal_batch(self, state: dict[str, Any]) -> InterviewProposalBatch:
        questions = {
            question.question_id: question
            for question in (
                InterviewQuestion.model_validate(item)
                for item in state["asked_questions"]
            )
        }
        changes = tuple(
            self._change_for_answer(
                InterviewAudience(state["kind"]),
                questions[turn.question_id],
                turn.answer,
            )
            for turn in self._turns(state)
            if turn.question_id in questions
        )
        return InterviewProposalBatch(
            pack_id=state["pack_id"],
            pack_version=state["pack_version"],
            audience=InterviewAudience(state["kind"]),
            changes=changes,
        )

    @staticmethod
    def _change_for_answer(
        audience: InterviewAudience, question: InterviewQuestion, answer: str
    ) -> InterviewProposedChange:
        topic = question.topic
        if audience is InterviewAudience.WORKSPACE:
            payload = {
                "goal": GoalPayload(subject=topic, outcome=answer),
                "working_context": WorkingContextPayload(subject=topic, value=answer),
                "convention": ConventionPayload(subject=topic, value=answer),
            }[topic]
        elif topic == "identity":
            payload = IdentityPayload(subject=topic, value=answer)
        else:
            payload = PreferencePayload(subject=topic, polarity="like", value=answer)
        return InterviewProposedChange(
            operation=ProposalOperation.CREATE,
            proposed_payload=payload,
            controls=ProfileControls(
                sync_mode=SyncMode.SYNCABLE,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
            semantic_key=SemanticKey(namespace=payload.kind, subject=topic),
        )

    def _save(self, state: dict[str, Any]) -> None:
        payload = {
            key: value for key, value in state.items() if key != "_draft_revision"
        }
        stored = self._drafts.save(
            state["session_id"],
            payload,
            expires_at=datetime.fromisoformat(state["expires_at"]),
            expected_revision=state.get("_draft_revision"),
        )
        state["_draft_revision"] = stored.revision

    def _load(self, session_id: str) -> dict[str, Any]:
        stored = self._drafts.require_active(session_id)
        state = dict(stored.payload)
        pinned_pack = (
            state.get("pack_id"),
            state.get("pack_version"),
            state.get("coverage_version"),
        )
        active_pack = (
            self._fixed.pack.pack_id,
            self._fixed.pack.pack_version,
            self._fixed.pack.coverage_version,
        )
        if pinned_pack != active_pack:
            raise ValueError("Saved interview question pack is stale.")
        self._service.validate_interview_target(
            scope_id=state["scope_id"],
            audience=InterviewAudience(state["kind"]),
        )
        state["_draft_revision"] = stored.revision
        return state

    @staticmethod
    def _turns(state: dict[str, Any]) -> tuple[InterviewTurn, ...]:
        return tuple(InterviewTurn.model_validate(item) for item in state["turns"])

    @staticmethod
    def _question(state: dict[str, Any]) -> InterviewQuestion | None:
        return (
            None
            if state["question"] is None
            else InterviewQuestion.model_validate(state["question"])
        )

    @staticmethod
    def _require_not_committed(state: dict[str, Any]) -> None:
        if state["status"] == "committed":
            raise ValueError(
                "Interview is already committed; only draft cleanup may be retried."
            )

    def _view(self, state: dict[str, Any]) -> InterviewSession:
        return InterviewSession(
            session_id=state["session_id"],
            kind=state["kind"],
            scope_id=state["scope_id"],
            mode=state["mode"],
            provider_label=state["provider_label"],
            model_id=state["model_id"],
            external_retention_notice=state["external_retention_notice"],
            question_attempts=state["question_attempts"],
            question=self._question(state),
            status=state["status"],
            committed_record_ids=tuple(state.get("committed_record_ids", ())),
            turns=self._turns(state),
            can_ask_another=(
                state["status"] == "active"
                and state["question_attempts"] < _MAX_QUESTION_ATTEMPTS
            ),
            provider_error=state["provider_error"],
        )
