"""Session-local native review authority; model parameters only narrow intent."""

from __future__ import annotations

import hashlib
import re
import threading
from copy import copy
from dataclasses import asdict, dataclass
from typing import Literal
from uuid import uuid4

from tldw_chatbook.Agents.code_review_models import (
    ReviewBoundaryReceipt,
    ReviewRefused,
    ReviewRunChoice,
    canonical_record_bytes,
)
from tldw_chatbook.Agents.code_review_service import (
    CodeReviewService,
    PreparedReview,
    ReviewStageIssuer,
    request_fingerprints,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


@dataclass(frozen=True)
class _AcceptedRequest:
    session_id: str
    message_id: str
    source_epoch: int
    content_digest: str
    stages: frozenset[str]
    review_id: str | None = None
    slot_id: str | None = None


_QUALIFICATION = re.compile(
    r"\b(if|unless|when|would|could|after|before|until|once|only|pending|provided|"
    r"subject|approval|approve|later|maybe|perhaps|example|says)\b"
)
_VERIFY_REQUEST = re.compile(
    r"(?:verify|consolidate)(?:\s+(?:them|(?:(?:this|the|my|captured|current|"
    r"retained|all|every)\s+)*(?:review|changes|patch|reports?|findings?)))?[.!]?"
)
_NEGATION = re.compile(r"\b(not|never|don.t|without)\b")
_REVIEWER_RETRY = re.compile(
    r"(?:please\s+)?retry\s+(?:the\s+)?reviewer\s+(r[1-9]\d*)[.!]?", re.I
)
_RESUME_REVIEW = re.compile(r"resume\s+(?:(?:the|this)\s+)?review[.!]?", re.I)


def _explicit_stages(text: str, trusted_skill_bindings=()) -> frozenset[str]:
    """Keep a qualified later stage as intent requiring a new native acceptance."""
    text = text.strip().lower()
    mention = re.match(r"^\$multi-model-review(?:\s+|$)", text)
    if mention:
        if "multi-model-review" not in trusted_skill_bindings:
            return frozenset()
        text = text[mention.end() :].strip()
    if (
        any(mark in text for mark in ('"', "“", "”", "`", "\n>", "\n >"))
        or text.startswith((">", "'"))
        or re.search(r"(?<!\w)'[^']*'", text)
    ):
        return frozenset()
    text = re.sub(r"^(?:can you\s+)?(?:please\s+)?", "", text)
    text = re.sub(r"^do (?:a )?multi-model code review\b", "review", text)
    text = text.removesuffix("?")
    if re.search(r"\b(?:not|never|don.t)\s+(?:review|retry|resume)\b", text):
        return frozenset()
    if re.match(r"review\b", text):
        clause = re.search(
            r"\b(?:and|then)\s+(?:please\s+)?(?P<verify>verify|consolidate)\b", text
        )
        review_text = text[: clause.start()] if clause else text
        if _QUALIFICATION.search(review_text):
            return frozenset()
        verify_text = text[clause.start("verify") :] if clause else ""
        # Accept the complete unqualified stage clause, not a positive keyword
        # buried in arbitrary qualifications. Unknown suffixes need new native
        # acceptance instead of expanding a deny-word list indefinitely.
        verify = bool(_VERIFY_REQUEST.fullmatch(verify_text)) and not _NEGATION.search(
            review_text
        )
        return frozenset(("review", "verify") if verify else ("review",))
    if _QUALIFICATION.search(text) or _NEGATION.search(text):
        return frozenset()
    if _VERIFY_REQUEST.fullmatch(text):
        return frozenset(("verify",))
    if re.match(r"retry\s+(?:the\s+)?reviewer\b", text):
        return frozenset(("retry_reviewer",))
    if re.match(r"retry\s+(?:the\s+)?verification\b", text):
        return frozenset(("retry_verification",))
    if _RESUME_REVIEW.fullmatch(text):
        return frozenset(("resume_review",))
    return frozenset()


_MAX_ACCEPTED_REQUESTS = 64


class ConsoleCodeReviewCoordinator:
    """Host issuer for one Console store; no model-facing acceptance methods."""

    def __init__(
        self,
        store,
        *,
        runtime_owner_id=None,
        on_boundary=None,
        on_boundary_retired=None,
    ):
        self.store = store
        self._lock = threading.RLock()
        self._boundary_admission_lock = threading.RLock()
        self._presentation_lock = threading.Lock()
        self._requests: dict[str, _AcceptedRequest] = {}
        self._narrowed: dict[str, tuple] = {}
        self._fenced: set[str] = set()
        self.issuer = ReviewStageIssuer(self)
        self.runtime_owner_id = runtime_owner_id
        self._on_boundary = on_boundary
        self._on_boundary_retired = on_boundary_retired
        self._boundaries = {}
        self._reviews = {}
        self._review_requests = {}
        self._stage_admissions = {}
        self._accepted_stages = {}
        self.present_report = None
        self.present_offer = None
        self.present_status = None
        self._offers = {}

    def _message(self, message_id: str, session_id: str):
        try:
            message = self.store.get_message(message_id)
            valid = (
                message.role is ConsoleMessageRole.USER
                and not getattr(message.metadata, "origin", None)
                and self.store.session_id_for_message(message_id) == session_id
                and message_id in self.store.active_path_message_ids(session_id)
            )
        except KeyError:
            valid = False
        if not valid:
            raise ReviewRefused("accepted_user_message_required")
        return message

    def prune_retired(self):
        with self._lock:
            reviews = tuple(self._reviews.items())
        retired = []
        for review_id, service in reviews:
            try:
                with service.store._tab.lock, service.store.lease(review_id):
                    pass
            except ReviewRefused:
                retired.append(review_id)
        with self._lock:
            for review_id in retired:
                self._reviews.pop(review_id, None)
                self._boundaries = {
                    key: value
                    for key, value in self._boundaries.items()
                    if value[0].review_id != review_id
                }
                self._accepted_stages = {
                    key: value
                    for key, value in self._accepted_stages.items()
                    if value[1].review_id != review_id
                }
                self._offers = {
                    key: value
                    for key, value in self._offers.items()
                    if value[0] != review_id
                }
            retired_messages = set()
            for review_id in retired:
                retired_messages.update(self._review_requests.pop(review_id, ()))
            for key in retired_messages:
                self._narrowed.pop(key, None)
                self._requests.pop(key, None)
                self._stage_admissions.pop(key, None)
            if retired_messages:
                with self.issuer._lock:
                    self.issuer._grants = {
                        key: value
                        for key, value in self.issuer._grants.items()
                        if value.acceptance_id not in retired_messages
                    }
                    self.issuer._intents = {
                        key: value
                        for key, value in self.issuer._intents.items()
                        if value.message_id not in retired_messages
                    }

    def close(self):
        with self._lock:
            self.issuer.close(self)
            self._requests.clear()
            self._narrowed.clear()
            self._reviews.clear()
            self._review_requests.clear()
            self._boundaries.clear()
            self._stage_admissions.clear()
            self._accepted_stages.clear()
            self._offers.clear()
            self._fenced.clear()

    def reset_runtime(self):
        """Retire all old executable authority before a new runtime attachment."""
        self.close()
        self.issuer = ReviewStageIssuer(self)

    def validate_acceptance(self, authority):
        """An edit, branch change or native fence revokes an issued capability."""
        message_id = getattr(authority, "acceptance_id", None) or authority.message_id
        with self._lock:
            accepted = self._requests.get(message_id)
            if accepted is None or accepted.session_id in self._fenced:
                raise ReviewRefused("stage_grant_unavailable")
        message = self._message(message_id, accepted.session_id)
        if (
            hashlib.sha256(message.content.encode()).hexdigest()
            != accepted.content_digest
        ):
            raise ReviewRefused("accepted_user_message_changed")

    def bind_accepted_request(
        self,
        message_id: str,
        *,
        session_id: str,
        source_epoch: int,
        trusted_skill_bindings=(),
    ) -> None:
        """Bind only the USER identity supplied by native turn acceptance."""
        self.prune_retired()
        message = self._message(message_id, session_id)
        if type(source_epoch) is not int or source_epoch < 0:
            raise ReviewRefused("source_epoch_mismatch")
        accepted = _AcceptedRequest(
            session_id,
            message_id,
            source_epoch,
            hashlib.sha256(message.content.encode()).hexdigest(),
            _explicit_stages(message.content, trusted_skill_bindings),
            slot_id=(
                match.group(1).upper()
                if (match := _REVIEWER_RETRY.fullmatch(message.content.strip()))
                else None
            ),
        )
        with self._lock:
            if session_id in self._fenced:
                if message_id in self._requests:
                    raise ReviewRefused("review_session_fenced")
                # A newly accepted manual request can start fresh authority;
                # previously revoked grants and receipts remain revoked.
                self._fenced.discard(session_id)
            offer = self._offers.get(session_id)
            if (
                offer is not None
                and message.content.strip().lower() in {"yes", "yes please", "proceed"}
                and offer[2] == source_epoch
            ):
                accepted = _AcceptedRequest(
                    session_id,
                    message_id,
                    source_epoch,
                    accepted.content_digest,
                    frozenset((offer[1],)),
                    offer[0],
                )
            if offer is not None and message.content.strip().lower() in {
                "no",
                "no thanks",
                "decline",
            }:
                self._offers.pop(session_id, None)
            previous = self._requests.get(message_id)
            if previous is not None and previous != accepted:
                raise ReviewRefused("source_epoch_mismatch")
            if previous is None:
                if len(self._requests) >= _MAX_ACCEPTED_REQUESTS:
                    raise ReviewRefused("review_authority_quota_exceeded")
                self._requests[message_id] = accepted
                self.issuer.register_intent(self, accepted)

    def narrow_request(
        self,
        message_id: str,
        *,
        session_id: str,
        source_epoch: int,
        target,
        team,
        source_authority,
        reviewers=None,
        verifier=None,
    ):
        """Freeze exact parameters under already accepted native stage intent."""
        message = self._message(message_id, session_id)
        fingerprints = request_fingerprints(target, team, reviewers, verifier)
        with source_authority.scope():
            with self._lock:
                if session_id in self._fenced:
                    raise ReviewRefused("review_session_fenced")
                accepted = self._requests.get(message_id)
                if (
                    accepted is None
                    or accepted.session_id != session_id
                    or "review" not in accepted.stages
                    or accepted.content_digest
                    != hashlib.sha256(message.content.encode()).hexdigest()
                ):
                    raise ReviewRefused("accepted_review_intent_required")
                if (
                    accepted.source_epoch != source_epoch
                    or source_authority.source_epoch != source_epoch
                ):
                    raise ReviewRefused("source_epoch_mismatch")
                previous = self._narrowed.get(message_id)
                if previous is not None:
                    if any(
                        getattr(previous[0], key) != value
                        for key, value in fingerprints.items()
                    ):
                        raise ReviewRefused("accepted_request_already_narrowed")
                    return previous
                fields = dict(
                    session_id=session_id,
                    acceptance_id=message_id,
                    source_epoch=source_epoch,
                    **fingerprints,
                )
                review = self.issuer.issue(self, stage="review", **fields)
                verification = (
                    self.issuer.issue(self, stage="verify", **fields)
                    if "verify" in accepted.stages
                    else None
                )
                self._narrowed[message_id] = (review, verification)
                return review, verification

    def service_for_request(self, message_id: str, **kwargs):
        """Compose the authenticated deferred provider from accepted native intent."""
        with self._lock:
            intent = self._requests.get(message_id)
            if intent is None or "review" not in intent.stages:
                raise ReviewRefused("accepted_review_intent_required")
            self.issuer.validate_intent(intent)
        return CodeReviewService(
            issuer=self.issuer,
            grant=None,
            native_intent=intent,
            native_coordinator=self,
            **kwargs,
        )

    def _accepted_boundary_service(self, receipt):
        """Revalidate retained accepted authority without extending live admission."""
        with self._lock:
            entry = self._boundaries.get(getattr(receipt, "boundary_id", None))
            if (
                entry is None
                or entry[0] is not receipt
                or entry[2] != "accepted"
                or receipt.session_id in self._fenced
                or receipt.owner_id != self.runtime_owner_id
            ):
                raise ReviewRefused("review_boundary_unavailable")
            previous = entry[1]
        previous.validate_authority()
        with (
            previous.store.lease(receipt.review_id) as context,
            previous.source_authority.scope(),
            previous.store._tab.lock,
        ):
            previous.store._live(context)
            grant = (
                previous.verification_grant
                if receipt.stage == "verify"
                else previous.grant
            )
            previous.issuer.validate(grant)
            pending = next(
                (
                    slot
                    for slot in previous.slots(context)
                    if slot.slot_id not in context.launch_claims
                    and (
                        slot.role == "reviewer"
                        or previous.verification_grant is not None
                    )
                ),
                None,
            )
            if (
                context.revision != receipt.revision
                or context.source_epoch != receipt.source_epoch
                or context.session_id != receipt.session_id
                or (context.conversation_id or context.session_id)
                != receipt.conversation_id
                or previous.native_chain_id != receipt.chain_id
                or pending is None
                or pending.slot_id != receipt.slot_id
                or grant.grant_id != receipt.grant_id
                or grant.stage != receipt.stage
                or grant.target_fingerprint != receipt.target_fingerprint
                or grant.team_fingerprint != receipt.team_fingerprint
                or grant.override_fingerprint != receipt.override_fingerprint
                or context.scheduled_attempts.get(
                    receipt.slot_id, receipt.slot_attempt_id
                )
                != receipt.slot_attempt_id
            ):
                raise ReviewRefused("review_boundary_unavailable")
        with self._lock:
            if (
                receipt.session_id in self._fenced
                or self._boundaries.get(receipt.boundary_id) != entry
            ):
                raise ReviewRefused("review_boundary_unavailable")
        return previous

    def service_for_boundary(self, receipt, *, capacity, destination_check):
        """Reuse only an accepted exact native wake, never serialized identities."""
        service = copy(self._accepted_boundary_service(receipt))
        service.actor = None
        service.capacity = capacity
        service.destination_check = destination_check
        service.native_resume = (receipt.review_id, receipt)
        return service

    def service_for_stage(self, message_id):
        with self._lock:
            entry = self._accepted_stages.get(message_id)
        if entry is None:
            raise ReviewRefused("accepted_stage_intent_required")
        _, prepared, previous, grant = entry
        service = copy(previous)
        service.actor = None
        service.native_resume = (prepared.review_id, None)
        service.native_allowed_slots = frozenset(prepared.slots)
        if grant.stage == "verify":
            service.grant = grant
            service.verification_grant = grant
        else:
            service.grant = grant
            service.verification_grant = None
        return service

    def bind_run(self, service, *, chain_id):
        service.native_chain_id = chain_id
        resume = getattr(service, "native_resume", None)
        if resume is None:
            return
        review_id, receipt = resume
        if receipt is not None and chain_id != receipt.chain_id:
            raise ReviewRefused("review_scope_mismatch")
        from tldw_chatbook.Agents.code_review_models import ReviewRunBinding

        with service.store._tab.lock, service.store.lease(review_id) as context:
            if receipt is not None:
                if (
                    context.revision != receipt.revision
                    or receipt.slot_id in context.launch_claims
                ):
                    raise ReviewRefused("review_boundary_unavailable")
                context.scheduled_attempts[receipt.slot_id] = receipt.slot_attempt_id
            payload = canonical_record_bytes(
                dict(
                    primary_run_id=service.actor.run_id,
                    grant_id=service.grant.grant_id,
                    verification_grant_id=getattr(
                        service.verification_grant, "grant_id", None
                    ),
                )
            )
            with service.store.reserve(
                review_id, source_bytes=0, report_bytes=len(payload), files=1
            ) as reservation:
                reservation.write(payload, category="report")
                reservation.commit(0, len(payload), 1)
            context.primary_authorities[service.actor.run_id] = (
                service.actor,
                service.grant,
                service.verification_grant,
            )
            service.store.bind_attempt(
                service.actor.run_id,
                ReviewRunBinding(
                    review_id=review_id,
                    slot_id="primary",
                    attempt_id=service.actor.run_id,
                    role="primary",
                    target_digest=context.target_digest,
                    source_epoch=context.source_epoch,
                ),
            )
        self.track_prepared(service, review_id)

    def on_settled(self, review_id):
        """Serialize native presentation without holding store or ledger locks."""
        with self._presentation_lock:
            self._present_settled(review_id)

    def _present_settled(self, review_id):
        """Present actual originals, then notify outside every store lock."""
        with self._lock:
            service = self._reviews.get(review_id)
        if service is None:
            return
        pending = []
        with service.store._tab.lock, service.store.lease(review_id) as context:
            for slot in service.slots(context):
                if slot.role != "reviewer":
                    continue
                binding = context.active_attempts.get(slot.slot_id)
                if (
                    binding is None
                    or (slot.slot_id, binding.attempt_id) in context.presented_attempts
                ):
                    continue
                report = service._report_envelope(
                    context, slot.slot_id, binding.attempt_id
                )
                if not report.get("terminal_status"):
                    continue
                pending.append((slot.slot_id, binding.attempt_id, report))
        for slot_id, attempt_id, report in pending:
            if self.present_report is None:
                continue
            report = dict(report)
            report["original_page"] = service.native_report_page(review_id, slot_id)
            presented = self.present_report(
                service.store._session_id, review_id, report
            )
            if presented:
                with service.store._tab.lock, service.store.lease(review_id) as context:
                    binding = context.active_attempts.get(slot_id)
                    if binding is not None and binding.attempt_id == attempt_id:
                        context.presented_attempts.add((slot_id, attempt_id))
        if service.verification_grant is None and self.present_offer is not None:
            with service.store._tab.lock, service.store.lease(review_id) as context:
                reviewers = [
                    slot for slot in service.slots(context) if slot.role == "reviewer"
                ]
                all_presented = all(
                    (slot.slot_id, context.active_attempts[slot.slot_id].attempt_id)
                    in context.presented_attempts
                    for slot in reviewers
                    if slot.slot_id in context.active_attempts
                )
                all_presented = all_presented and all(
                    slot.slot_id in context.active_attempts for slot in reviewers
                )
            with self._lock:
                offered = any(offer[0] == review_id for offer in self._offers.values())
            if (
                all_presented
                and not offered
                and self.present_offer(service.store._session_id, review_id)
            ):
                self.offer_stage(review_id)
        receipt = self.seal_next_boundary(review_id)
        if receipt is not None and self._on_boundary is not None:
            self._on_boundary(receipt)

    def report_service(self, session_id: str, review_id: str | None = None):
        """Native screen lookup constrained to the owning active Console chat."""
        with self._lock:
            candidates = [
                (key, value)
                for key, value in self._reviews.items()
                if value.store._session_id == session_id
                and (review_id is None or review_id == key)
            ]
        if not candidates:
            raise ReviewRefused("review_expired")
        key, service = candidates[-1]
        service.validate_authority()
        return key, service

    def offer_stage(self, review_id, stage="verify"):
        """Only native UI may offer a stage; model prose cannot install an offer."""
        with self._lock:
            service = self._reviews[review_id]
            self._offers[service.store._session_id] = (
                review_id,
                stage,
                service.source_authority.source_epoch,
            )

    def track_prepared(self, service, review_id):
        with self._lock:
            self._reviews[review_id] = service
            self._review_requests.setdefault(review_id, set()).update(
                grant.acceptance_id
                for grant in (service.grant, service.verification_grant)
                if grant is not None
            )

    def bind_stage_admission(
        self, message_id, *, capacity, routing, app_config, destination_check
    ):
        """Native composition supplies the later accepted turn's current admission."""
        with self._lock:
            if message_id not in self._requests:
                raise ReviewRefused("accepted_user_message_required")
            self._stage_admissions[message_id] = dict(
                capacity=capacity,
                routing=routing,
                app_config=app_config,
                destination_check=destination_check,
            )

    def accept_stage(
        self,
        review_id: str,
        stage: Literal[
            "verify", "retry_reviewer", "retry_verification", "resume_review"
        ],
        *,
        accepted_message_id: str,
        choice: ReviewRunChoice | None = None,
        slot_id: str | None = None,
    ) -> PreparedReview:
        with self._lock:
            service = self._reviews.get(review_id)
            accepted = self._requests.get(accepted_message_id)
            if accepted is None:
                raise ReviewRefused("accepted_user_message_required")
            self._message(accepted_message_id, accepted.session_id)
            self.issuer.validate_intent(accepted)
            if accepted.review_id is not None and accepted.review_id != review_id:
                raise ReviewRefused("review_scope_mismatch")
            if stage not in accepted.stages:
                raise ReviewRefused("accepted_stage_intent_required")
            if service is None or service.store._session_id != accepted.session_id:
                raise ReviewRefused("review_scope_mismatch")
            admission = self._stage_admissions.get(accepted_message_id)
            if admission is None:
                raise ReviewRefused("stage_admission_unavailable")
            previous = self._accepted_stages.get(accepted_message_id)
        if stage == "retry_reviewer":
            if accepted.slot_id is None:
                raise ReviewRefused("reviewer_selection_required")
            if slot_id is not None and slot_id != accepted.slot_id:
                raise ReviewRefused("accepted_reviewer_selection_mismatch")
            slot_id = accepted.slot_id
        if stage == "resume_review" and (choice is not None or slot_id is not None):
            raise ReviewRefused("accepted_stage_selection_mismatch")
        identity = (review_id, stage, choice, slot_id)
        if previous is not None:
            if previous[0] != identity:
                raise ReviewRefused("accepted_request_already_narrowed")
            return previous[1]
        current = service.for_run(service.actor, capacity=admission["capacity"])
        current.routing = admission["routing"]
        current.app_config = admission["app_config"]
        current.destination_check = admission["destination_check"]
        capacity = current.capacity()
        if capacity.remaining_children == 0:
            raise ReviewRefused("child_budget_unavailable")
        with (
            service.store.lease(review_id) as context,
            service.source_authority.scope(),
            service.store._tab.lock,
        ):
            service.store._live(context)
            if accepted.source_epoch != context.source_epoch:
                raise ReviewRefused("source_epoch_mismatch")
            role = (
                "reviewer"
                if stage in {"retry_reviewer", "resume_review"}
                else "verifier"
            )
            if stage == "resume_review":
                selected_slots = tuple(
                    slot
                    for slot in service.slots(context)
                    if slot.role == "reviewer"
                    and slot.slot_id not in context.launch_claims
                )
                if not selected_slots:
                    raise ReviewRefused("review_no_unstarted_reviewers")
                if (
                    len(selected_slots)
                    > capacity.remaining_children
                    + capacity.children_per_turn * capacity.remaining_continuations
                ):
                    raise ReviewRefused("child_budget_unavailable")
                with self._lock:
                    if any(
                        receipt.review_id == review_id
                        and state == "accepted"
                        and any(
                            slot.slot_id == receipt.slot_id for slot in selected_slots
                        )
                        for receipt, _, state in self._boundaries.values()
                    ):
                        raise ReviewRefused("review_attempt_outcome_uncertain")
            else:
                selected_id = slot_id if stage == "retry_reviewer" else "V1"
                slot = next(
                    (
                        slot
                        for slot in service.slots(context)
                        if slot.slot_id == selected_id
                    ),
                    None,
                )
                if slot is None and choice is None:
                    raise ReviewRefused(
                        "verifier_missing"
                        if role == "verifier"
                        else "review_slot_unknown"
                    )
                if slot is not None and slot.role != role:
                    raise ReviewRefused("review_slot_unknown")
                existing = context.launch_claims.get(selected_id)
                if stage.startswith("retry_"):
                    if existing is None or existing.run_id is None:
                        raise ReviewRefused("review_attempt_not_settled")
                    stream = context.streams[(selected_id, existing.binding.attempt_id)]
                    if not stream.terminal_status:
                        raise ReviewRefused("review_attempt_not_settled")
                elif existing is not None:
                    raise ReviewRefused("review_stage_already_started")
                selected_slots = (slot,)
            selected_claims = {
                slot.slot_id
                if slot is not None
                else selected_id: context.launch_claims.get(
                    slot.slot_id if slot is not None else selected_id
                )
                for slot in selected_slots
            }
            revision = context.revision
        # Routing/readiness and native destination confirmation may block on a
        # person. They never run under the review-store publication lock.
        if choice is not None:
            choice = ReviewRunChoice.model_validate(choice)
            if role == "verifier" and choice.focus:
                raise ReviewRefused("invalid_verifier_focus")
            selected_slots = (current.admit_slot(selected_id, role, choice),)
        else:
            for slot in selected_slots:
                refusal = current.destination_check(slot.target)
                if refusal:
                    raise ReviewRefused(refusal)
        # Shared only with ledger acceptance, before either side acquires its
        # store/DB locks. Native consent above never holds this scope.
        with (
            service.store.lease(review_id) as context,
            service.source_authority.scope(),
            self._boundary_admission_lock,
            service.store._tab.lock,
        ):
            service.store._live(context)
            if context.revision != revision:
                raise ReviewRefused("review_revision_conflict")
            with self._lock:
                if any(
                    receipt.review_id == review_id
                    and receipt.slot_id in selected_claims
                    and state == "accepted"
                    for receipt, _, state in self._boundaries.values()
                ):
                    raise ReviewRefused("review_attempt_outcome_uncertain")
                if any(
                    context.launch_claims.get(slot_id) is not claim
                    for slot_id, claim in selected_claims.items()
                ):
                    raise ReviewRefused("review_attempt_state_changed")
                if accepted_message_id in self._accepted_stages:
                    previous = self._accepted_stages[accepted_message_id]
                    if previous[0] == identity:
                        return previous[1]
                    raise ReviewRefused("accepted_request_already_narrowed")
            self.issuer.validate_intent(accepted)
            replacements = {slot.slot_id: slot for slot in selected_slots}
            reviewers = tuple(
                replacements.get(item.slot_id, item).choice
                for item in service.slots(context)
                if item.role == "reviewer"
            )
            verifier = (
                selected_slots[0].choice
                if role == "verifier"
                else next(
                    (
                        item.choice
                        for item in service.slots(context)
                        if item.role == "verifier"
                    ),
                    None,
                )
            )
            fingerprints = request_fingerprints(
                context.preparation.target, service.team, reviewers, verifier
            )
            grant = self.issuer.issue(
                self,
                session_id=accepted.session_id,
                acceptance_id=accepted_message_id,
                stage="verify" if role == "verifier" else "review",
                source_epoch=context.source_epoch,
                **fingerprints,
            )
            attempts = {slot.slot_id: uuid4().hex for slot in selected_slots}
            payload = canonical_record_bytes(
                dict(
                    acceptance_id=accepted_message_id,
                    stage=stage,
                    slots=[
                        dict(
                            slot_id=slot.slot_id,
                            role=slot.role,
                            choice=slot.choice.model_dump(mode="json"),
                            definition=asdict(slot.definition),
                            target=asdict(slot.target),
                            fallbacks=[
                                asdict(candidate) for candidate in slot.fallbacks
                            ],
                            attempt_id=attempts[slot.slot_id],
                        )
                        for slot in selected_slots
                    ],
                    grant_id=grant.grant_id,
                )
            )
            with service.store.reserve(
                review_id, source_bytes=0, report_bytes=len(payload), files=1
            ) as reservation:
                reservation.write(payload, category="report")
                reservation.commit(0, len(payload), 1)
            for slot in selected_slots:
                context.stage_slots[slot.slot_id] = slot
                context.scheduled_attempts[slot.slot_id] = attempts[slot.slot_id]
                context.launch_claims.pop(slot.slot_id, None)
            context.continuation_status = None
            context.revision += 1
            result = PreparedReview(
                review_id=review_id,
                revision=context.revision,
                slots=tuple(slot.slot_id for slot in selected_slots),
                summary="stage_admitted",
            )
            retired = []
            with self._lock:
                # Supersede only live eligibility. Accepted/completed durable
                # attempts and charges remain owned by the existing ledger.
                for key, entry in tuple(self._boundaries.items()):
                    if entry[0].review_id == review_id and entry[2] == "live":
                        self._boundaries[key] = (entry[0], entry[1], "superseded")
                        retired.append(entry[0])
                self._accepted_stages[accepted_message_id] = (
                    identity,
                    result,
                    current,
                    grant,
                )
                self._review_requests.setdefault(review_id, set()).add(
                    accepted_message_id
                )
        if self._on_boundary_retired is not None:
            for receipt in retired:
                self._on_boundary_retired(receipt)
        return result

    def seal_next_boundary(self, review_id: str) -> ReviewBoundaryReceipt | None:
        with self._lock:
            service = self._reviews.get(review_id)
        if service is None or not self.runtime_owner_id:
            return None
        chain_id = getattr(service, "native_chain_id", None)
        if not chain_id or service.capacity().remaining_children:
            return None
        with (
            service.store.lease(review_id) as context,
            service.source_authority.scope(),
            service.store._tab.lock,
        ):
            service.store._live(context)
            slots = tuple(
                slot
                for slot in service.slots(context)
                if slot.role == "reviewer" or service.verification_grant is not None
            )
            pending = next(
                (slot for slot in slots if slot.slot_id not in context.launch_claims),
                None,
            )
            if pending is None:
                return None
            predecessors = []
            for slot in slots:
                if slot is pending:
                    break
                claim = context.launch_claims.get(slot.slot_id)
                if claim is None or claim.run_id is None:
                    return None
                stream = context.streams.get((slot.slot_id, claim.binding.attempt_id))
                if (
                    stream is None
                    or not stream.terminal_status
                    or not stream.terminal_receipt
                ):
                    return None
                predecessors.append((claim.run_id, stream.terminal_receipt))
            if not predecessors:
                return None
            grant = (
                service.verification_grant
                if pending.role == "verifier"
                else service.grant
            )
            service.issuer.validate(grant)
            with self._lock:
                if context.session_id in self._fenced:
                    return None
                for receipt, issued_service, state in self._boundaries.values():
                    if (
                        receipt.review_id == review_id
                        and receipt.slot_id == pending.slot_id
                        and receipt.revision == context.revision
                        and receipt.chain_id == chain_id
                        and state != "superseded"
                    ):
                        return receipt if state == "live" else None
                receipt = ReviewBoundaryReceipt(
                    boundary_id=uuid4().hex,
                    review_id=review_id,
                    revision=context.revision,
                    slot_id=pending.slot_id,
                    slot_attempt_id=uuid4().hex,
                    stage=grant.stage,
                    predecessor_run_ids=tuple(run for run, _ in predecessors),
                    predecessor_settlement_ids=tuple(
                        settlement for _, settlement in predecessors
                    ),
                    grant_id=grant.grant_id,
                    source_epoch=context.source_epoch,
                    chain_id=chain_id,
                    owner_id=self.runtime_owner_id,
                    session_id=context.session_id,
                    conversation_id=context.conversation_id or context.session_id,
                    target_fingerprint=grant.target_fingerprint,
                    team_fingerprint=grant.team_fingerprint,
                    override_fingerprint=grant.override_fingerprint,
                )
                payload = canonical_record_bytes(asdict(receipt))
                with service.store.reserve(
                    review_id, source_bytes=0, report_bytes=len(payload), files=1
                ) as reservation:
                    reservation.write(payload, category="report")
                    reservation.commit(0, len(payload), 1)
                self._boundaries[receipt.boundary_id] = (receipt, service, "live")
                return receipt

    def boundary_acceptance_scope(self, receipt):
        """Serialize durable acceptance with final manual-stage publication.

        This independent lock is acquired before DB/store locks. Ordinary
        boundary validation and claim never acquire it.
        """
        return self._boundary_admission_lock

    def boundary_transition(self, receipt, state):
        """Receive durable lifecycle facts only after the ledger transaction commits."""
        with self._lock:
            entry = self._boundaries.get(receipt.boundary_id)
            if entry is None or entry[0] is not receipt:
                return
            if state == "released":
                return
            if state in {"accepted", "completed"}:
                self._boundaries[receipt.boundary_id] = (receipt, entry[1], state)

    def validate_boundary(self, receipt: ReviewBoundaryReceipt) -> bool:
        with self._lock:
            entry = self._boundaries.get(getattr(receipt, "boundary_id", None))
            if entry is None or entry[0] is not receipt or entry[2] != "live":
                return False
            if (
                receipt.session_id in self._fenced
                or receipt.owner_id != self.runtime_owner_id
            ):
                return False
            service = entry[1]
        try:
            with (
                service.store.lease(receipt.review_id) as context,
                service.source_authority.scope(),
                service.store._tab.lock,
            ):
                service.store._live(context)
                if (
                    context.revision != receipt.revision
                    or context.source_epoch != receipt.source_epoch
                ):
                    return False
                if receipt.slot_id in context.launch_claims:
                    return False
                grant = (
                    service.verification_grant
                    if receipt.stage == "verify"
                    else service.grant
                )
                service.issuer.validate(grant)
                if grant.grant_id != receipt.grant_id:
                    return False
            with self._lock:
                return (
                    receipt.session_id not in self._fenced
                    and self._boundaries.get(receipt.boundary_id) == entry
                )
        except (ReviewRefused, OSError):
            return False

    def mark_continuation_blocked(self, receipt, reason):
        """Project retained progress without restoring continuation authority."""
        with self._lock:
            entry = self._boundaries.get(receipt.boundary_id)
            if entry is None or entry[0] is not receipt or entry[2] == "superseded":
                return
            service = entry[1]
        status = "continuation_blocked:" + reason
        try:
            with (
                service.store._tab.lock,
                service.store.lease(receipt.review_id) as context,
            ):
                if context.revision != receipt.revision:
                    return
                if context.continuation_status == status:
                    return
                context.continuation_status = status
        except ReviewRefused:
            return
        if self.present_status is not None:
            self.present_status(receipt.session_id, receipt.review_id, status)

    def fence_session(self, session_id: str) -> None:
        """Revoke live eligibility before stop/close can cancel child workers."""
        with self._lock:
            self._fenced.add(session_id)
            self._offers.pop(session_id, None)
            self.issuer.revoke_session(self, session_id)
            receipts = [
                entry[0]
                for entry in self._boundaries.values()
                if entry[0].session_id == session_id
            ]
        for receipt in receipts:
            self.mark_continuation_blocked(receipt, "session_fenced")


def compose_native_review_provider(
    controller,
    *,
    session_id,
    message,
    roots,
    resolution,
    destination_check,
    conversation_id,
    wake_authorization,
    tool_configuration,
    trusted_skill_bindings=(),
):
    """Compose from accepted Console state, before the runtime mints its run ID."""
    from tldw_chatbook import config
    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Agents.agent_routing import load_agents_routing_config
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
    from tldw_chatbook.Agents.code_review_service import ReviewCapacity
    from tldw_chatbook.Agents.code_review_source import ReviewSourceAuthority
    from tldw_chatbook.Agents.code_review_store import ReviewStore
    from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    coordinator = controller.code_review
    budget = RunBudget()
    limits = AutomaticWorkLimits.from_settings()

    def capacity():
        return ReviewCapacity(
            budget.max_subagents,
            budget.max_subagents,
            limits.generations,
            budget.max_tool_result_chars // 4,
        )

    receipt = getattr(wake_authorization, "review_boundary", None)
    if receipt is not None:
        if not wake_authorization.accepted:
            raise ReviewRefused("review_boundary_unavailable")
        service = coordinator.service_for_boundary(
            receipt, capacity=capacity, destination_check=destination_check
        )
        return CodeReviewToolProvider(service)
    if message is None:
        return None
    with coordinator._lock:
        offered = session_id in coordinator._offers
    if not _explicit_stages(message.content, trusted_skill_bindings) and not offered:
        return None
    if len(roots) != 1 or isinstance(roots[0].root, RemoteRoot):
        raise ReviewRefused("source_selection_required")
    root = roots[0]
    # Source identity is retained with the exact admitted binding; its guard
    # checks current authority on every capture/read and continuation.
    epoch = 0
    coordinator.bind_accepted_request(
        message.id,
        session_id=session_id,
        source_epoch=epoch,
        trusted_skill_bindings=trusted_skill_bindings,
    )
    settings = config.load_settings()
    routing = load_agents_routing_config()
    stages = coordinator._requests[message.id].stages
    if not stages:
        return None
    if "review" not in stages:
        with coordinator._lock:
            reviews = [
                (rid, service)
                for rid, service in coordinator._reviews.items()
                if service.store._session_id == session_id
                and (
                    coordinator._requests[message.id].review_id is None
                    or rid == coordinator._requests[message.id].review_id
                )
            ]
        if len(reviews) != 1:
            raise ReviewRefused("review_selection_required")
        coordinator.bind_stage_admission(
            message.id,
            capacity=capacity,
            routing=routing,
            app_config=settings,
            destination_check=destination_check,
        )
        coordinator.accept_stage(
            reviews[0][0], next(iter(stages)), accepted_message_id=message.id
        )
        return CodeReviewToolProvider(coordinator.service_for_stage(message.id))
    review_store = ReviewStore(controller._scratch_spaces, session_id)
    service = coordinator.service_for_request(
        message.id,
        store=review_store,
        actor=None,
        source_authority=ReviewSourceAuthority(root, epoch),
        preset_db=controller._agent_bridge.runs_db,
        app_config=settings,
        routing=routing,
        team=config.get_code_review_team(),
        primary_provider=resolution.execution_key or resolution.provider,
        primary_model=resolution.model,
        capacity=capacity,
        destination_check=destination_check,
        max_tool_result_chars=budget.max_tool_result_chars,
        ephemeral=next(
            s.ephemeral for s in controller.store.sessions() if s.id == session_id
        ),
    )
    service.native_conversation_id = conversation_id
    return CodeReviewToolProvider(service)
