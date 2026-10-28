"""Accepted, run-bound review preparation and public operations (ADR-281).

The issuer is a process-local host port. Native accepted-message issuance belongs
at the Console acceptance boundary; wire values never construct this authority.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from .agent_models import AgentDefinition, ToolResult, definition_from_row
from .agent_routing import (
    AgentsRoutingConfig,
    RoutingError,
    SpawnTarget,
    freeze_spawn_target,
    resolve_preset_fallback_targets,
    resolve_spawn_target,
)
from .code_review_guide import fit_code_review_guide, load_code_review_guide
from .code_review_models import (
    CheckReceipt,
    CoverageGap,
    ReviewRefused,
    ReviewRunBinding,
    ReviewRunChoice,
    ReviewTarget,
    ReviewTeam,
    ReviewWireModel,
    SourceQuery,
    SourceRange,
    canonical_record_bytes,
    review_result_char_limit,
    validate_record_chunk,
)
from .code_review_source import (
    ReviewPage,
    ReviewSourceAuthority,
    _access,
    _base_page,
    _fits,
    _in_scope,
    _load_cursor,
    _persist_page,
    _replay,
    _safe,
    capture_review_source,
    original_source_path,
    query_review_source,
    read_review_source,
    source_cursor_selection,
    use_review_source_authority,
)
from .code_review_store import ReleaseReceipt, ReviewStore, SubmissionAck
from .run_context import CurrentRunActor, current_run_actor
from .tool_catalog import ToolPathTarget

if TYPE_CHECKING:
    from .code_review_checks import ReviewCheckClaim, ReviewCheckNotice
    from .code_review_history import ReviewHistoryCheckpoint

_GUIDES = {
    "review": "Prepare the accepted target and selected presets. Reviewers inspect captured source independently and submit structured findings and coverage. Present each original report before optional verification. If wait_agents returns check_requested, read the proposal using summary/document_ref, then submit one check_actions claim or decline. Read summary again for the host specification; invoke only that exact existing process tool through its ordinary permissions, then wait for the same verifier. Stock execution may be unavailable; never retry accepted or uncertain execution.",
    "verify": "Verification requires an accepted verification stage and a fresh configured verifier. Inspect blinded units and captured evidence. Submit confirmed, refuted or unresolved dispositions; suggested checks are not execution evidence. Non-reproduction alone is not refutation: without counter-evidence keep the claim unresolved. Read batch pages, submit an independent disposition for every unit, then continue the returned cursor. Propose at most one executable check at a time, then read verification_batch with document_ref=proposal_id to await its host result. A proposal names related units; it does not establish that execution is indispensable. Independently supported source-only decisions remain allowed. If execution is necessary to decide and is unavailable or unfinished, keep the unit unresolved with next_check. Only an executed host receipt fully read by this live verifier is check evidence.",
}


_REVIEWER_BRIEF = (
    "Review independently using captured source. Report source-grounded findings "
    "with their trigger, impact and uncertainty, and submit structured coverage "
    "including inspection gaps even when there are no findings."
)


@dataclass(frozen=True)
class ReviewStageGrant:
    session_id: str
    acceptance_id: str
    stage: Literal["review", "verify"]
    source_epoch: int
    target_fingerprint: str
    team_fingerprint: str
    override_fingerprint: str
    grant_id: str = field(default_factory=lambda: uuid.uuid4().hex)


_ISSUERS = weakref.WeakSet()


class ReviewStageIssuer:
    """Registered host issuer, retaining exact issued objects until revocation."""

    def __init__(self, owner: object) -> None:
        self._owner = owner
        self._lock = threading.RLock()
        self._grants: dict[int, ReviewStageGrant] = {}
        self._intents: dict[int, object] = {}
        _ISSUERS.add(self)

    def issue(self, owner: object, **fields) -> ReviewStageGrant:
        if owner is not self._owner or self not in _ISSUERS:
            raise ReviewRefused("stage_grant_unavailable")
        grant = ReviewStageGrant(**fields)
        if (
            not grant.session_id
            or not grant.acceptance_id
            or grant.stage not in ("review", "verify")
            or type(grant.source_epoch) is not int
            or grant.source_epoch < 0
            or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in (
                    grant.target_fingerprint,
                    grant.team_fingerprint,
                    grant.override_fingerprint,
                )
            )
        ):
            raise ReviewRefused("stage_grant_unavailable")
        with self._lock:
            self._grants[id(grant)] = grant
        return grant

    def validate(self, grant: ReviewStageGrant) -> None:
        with self._lock:
            if self not in _ISSUERS or self._grants.get(id(grant)) is not grant:
                raise ReviewRefused("stage_grant_unavailable")
        validate = getattr(self._owner, "validate_acceptance", None)
        if validate is not None:
            validate(grant)

    def register_intent(self, owner: object, intent: object) -> None:
        """Register native accepted intent without creating an executable grant."""
        if owner is not self._owner or self not in _ISSUERS:
            raise ReviewRefused("stage_grant_unavailable")
        with self._lock:
            self._intents[id(intent)] = intent

    def validate_intent(self, intent: object) -> None:
        with self._lock:
            if self not in _ISSUERS or self._intents.get(id(intent)) is not intent:
                raise ReviewRefused("accepted_review_intent_required")
        validate = getattr(self._owner, "validate_acceptance", None)
        if validate is not None:
            validate(intent)

    def close(self, owner: object) -> None:
        if owner is not self._owner:
            raise ReviewRefused("stage_grant_unavailable")
        with self._lock:
            self._grants.clear()
            self._intents.clear()
            _ISSUERS.discard(self)

    def revoke_session(self, owner: object, session_id: str) -> None:
        """Revoke one session without invalidating other native owners."""
        if owner is not self._owner:
            raise ReviewRefused("stage_grant_unavailable")
        with self._lock:
            self._grants = {
                key: grant
                for key, grant in self._grants.items()
                if grant.session_id != session_id
            }
            self._intents = {
                key: intent
                for key, intent in self._intents.items()
                if intent.session_id != session_id
            }


def _choices(team, reviewers, verifier):
    selected = (
        tuple(ReviewRunChoice.model_validate(item) for item in reviewers)
        if reviewers is not None
        else tuple(ReviewRunChoice(**item.model_dump()) for item in team.reviewers)
    )
    chosen_verifier = verifier
    if chosen_verifier is None and team.verifier_preset_id:
        chosen_verifier = ReviewRunChoice(preset_id=team.verifier_preset_id)
    if chosen_verifier is not None:
        chosen_verifier = ReviewRunChoice.model_validate(chosen_verifier)
        if chosen_verifier.focus:
            raise ReviewRefused("invalid_verifier_focus")
    return selected, chosen_verifier


def request_fingerprints(target, team, reviewers=None, verifier=None) -> dict[str, str]:
    """Fingerprint accepted target, team/focus and explicit run-only overrides."""
    target = TypeAdapter(ReviewTarget).validate_python(target)
    reviewers, verifier = _choices(team, reviewers, verifier)
    choices = (*reviewers, *((verifier,) if verifier else ()))

    def digest(value):
        return hashlib.sha256(canonical_record_bytes(value)).hexdigest()

    return dict(
        target_fingerprint=digest(target.model_dump(mode="json")),
        team_fingerprint=digest(
            {
                "reviewers": [
                    r.model_dump(mode="json", exclude={"destination"})
                    for r in reviewers
                ],
                "verifier": verifier.model_dump(mode="json", exclude={"destination"})
                if verifier
                else None,
            }
        ),
        override_fingerprint=digest(
            {
                "destinations": [
                    choice.destination.model_dump(mode="json")
                    if choice.destination
                    else None
                    for choice in choices
                ]
            }
        ),
    )


@dataclass(frozen=True)
class ReviewCapacity:
    remaining_children: int
    children_per_turn: int
    remaining_continuations: int
    remaining_input_tokens: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ReviewRefused("child_budget_unavailable")
        if self.remaining_children > self.children_per_turn:
            raise ReviewRefused("child_budget_unavailable")


@dataclass(frozen=True)
class ReviewSlot:
    slot_id: str
    role: str
    choice: ReviewRunChoice
    definition: AgentDefinition
    target: SpawnTarget
    fallbacks: tuple


@dataclass(frozen=True)
class ReviewPreparation:
    primary_actor: CurrentRunActor
    acceptance_id: str
    grant: ReviewStageGrant
    source_authority: ReviewSourceAuthority
    target: ReviewTarget
    slots: tuple[ReviewSlot, ...]
    guide_text: str


class PreparedReview(ReviewWireModel):
    review_id: str
    revision: int
    slots: tuple[str, ...]
    summary: str


@dataclass
class ReviewLaunchClaim:
    """One host-owned launch identity; an unresolved claim is never retried."""

    review_id: str
    slot: ReviewSlot
    binding: ReviewRunBinding
    grant: ReviewStageGrant
    run_id: str | None = None
    receipt: ToolResult | None = None
    launch_allowed: bool = True


class CodeReviewService:
    """One actual run's authority and response capacity, never tool arguments."""

    def __init__(
        self,
        *,
        store: ReviewStore,
        issuer: ReviewStageIssuer,
        grant: ReviewStageGrant | None,
        actor: CurrentRunActor | None,
        source_authority: ReviewSourceAuthority,
        preset_db,
        app_config: dict,
        routing: AgentsRoutingConfig,
        team: ReviewTeam,
        primary_provider: str,
        primary_model: str,
        capacity: Callable[[], ReviewCapacity],
        destination_check: Callable[[SpawnTarget], str | None],
        max_tool_result_chars: int,
        ephemeral: bool = False,
        verification_grant: ReviewStageGrant | None = None,
        native_intent: object | None = None,
        native_coordinator: object | None = None,
    ):
        if type(issuer) is not ReviewStageIssuer:
            raise ReviewRefused("stage_grant_unavailable")
        if grant is not None:
            issuer.validate(grant)
        else:
            issuer.validate_intent(native_intent)
        authority = grant or native_intent
        if (
            authority.session_id != store._session_id
            or authority.source_epoch != source_authority.source_epoch
        ):
            raise ReviewRefused("review_scope_mismatch")
        self.store, self.issuer, self.grant, self.actor = store, issuer, grant, actor
        self.source_authority, self.preset_db = source_authority, preset_db
        self.app_config, self.routing, self.team = deepcopy(app_config), routing, team
        self.primary_provider, self.primary_model = primary_provider, primary_model
        self.capacity, self.destination_check = capacity, destination_check
        self.result_limit = review_result_char_limit(max_tool_result_chars)
        self.ephemeral = ephemeral
        self.verification_grant = verification_grant
        self.native_intent = native_intent
        self.native_coordinator = native_coordinator
        self.diagnostic = ""
        self.source_permission = None
        self.check_process_provider = None
        self._released: dict[str, ReleaseReceipt] = {}

    @property
    def enabled(self):
        from tldw_chatbook.config import coerce_bool_setting

        return not self.ephemeral and coerce_bool_setting(
            self.app_config.get("tools", {}).get("code_review_enabled", True),
            default=True,
        )

    def validate_authority(self):
        if self.grant is not None:
            self.issuer.validate(self.grant)
        else:
            self.issuer.validate_intent(self.native_intent)

    def bind_native_run(self, actor, *, chain_id: str | None) -> None:
        """Called by the native runtime after it owns the exact primary run."""
        if self.native_coordinator is None:
            return
        if actor.kind != "primary" or (self.actor is not None and self.actor != actor):
            raise ReviewRefused("run_not_bound")
        self.validate_authority()
        self.actor = actor
        self.native_coordinator.bind_run(self, chain_id=chain_id)

    def _actor(self, *, primary=False):
        if not self.enabled:
            raise ReviewRefused("feature_disabled")
        self.validate_authority()
        if self.actor is None or current_run_actor() != self.actor:
            raise ReviewRefused("run_not_bound")
        if primary and self.actor.kind != "primary":
            raise ReviewRefused("role_denied")
        return self.actor

    def _bound(self, context):
        self._actor()
        binding = self.store._bound(context)
        preparation = context.preparation
        if not isinstance(preparation, ReviewPreparation):
            raise ReviewRefused("review_scope_mismatch")
        association = context.primary_authorities.get(
            self.actor.run_id if binding.role == "primary" else self.actor.parent_run_id
        )
        if association is not None:
            expected, review_grant, verification_grant = association
            expected_grant = (
                verification_grant if binding.role == "verifier" else review_grant
            )
            if (
                self.grant is not expected_grant
                or (binding.role == "primary" and self.actor != expected)
                or (
                    binding.role != "primary"
                    and (
                        self.actor.kind != "subagent"
                        or self.actor.parent_run_id != expected.run_id
                    )
                )
                or self.source_authority is not preparation.source_authority
                or self.grant.source_epoch != context.source_epoch
                or self.grant.target_fingerprint != preparation.grant.target_fingerprint
                or self.grant.session_id != context.session_id
            ):
                raise ReviewRefused("review_scope_mismatch")
            return binding
        expected = preparation.primary_actor
        if (
            (binding.role == "primary" and self.actor != expected)
            or (
                binding.role != "primary"
                and (
                    self.actor.kind != "subagent"
                    or self.actor.parent_run_id != expected.run_id
                )
            )
            or self.grant.acceptance_id != preparation.acceptance_id
            or self.source_authority is not preparation.source_authority
        ):
            raise ReviewRefused("review_scope_mismatch")
        if self.grant.stage != (
            "verify" if binding.role == "verifier" else "review"
        ) or any(
            getattr(self.grant, key) != getattr(preparation.grant, key)
            for key in (
                "session_id",
                "source_epoch",
                "target_fingerprint",
                "team_fingerprint",
                "override_fingerprint",
            )
        ):
            raise ReviewRefused("stage_grant_scope_mismatch")
        return binding

    def for_run(
        self,
        actor: CurrentRunActor,
        *,
        capacity: Callable[[], ReviewCapacity] | None = None,
        grant: ReviewStageGrant | None = None,
        max_tool_result_chars: int | None = None,
    ) -> CodeReviewService:
        """Make a private actor adapter while retaining shared store authority."""
        service = copy(self)
        service.actor = actor
        service.grant = grant or self.grant
        service.capacity = capacity or self.capacity
        if max_tool_result_chars is not None:
            service.result_limit = min(
                self.result_limit, review_result_char_limit(max_tool_result_chars)
            )
        service._released = {}
        service.diagnostic = ""
        service.verifier_input_tokens = None
        service.verifier_estimate = None
        service.verifier_required_continuation = False
        return service

    @contextmanager
    def _source_context(self, review_id):
        """Lease/snapshot, call external authority unlocked, then revalidate."""
        with self.store.lease(review_id) as context:
            with self.store._tab.lock:
                binding = self._bound(context)
            with use_review_source_authority(self.store, self.source_authority):
                with _access(review_id) as (_, current, _, checked, root, exclusions):
                    if (
                        current is not context
                        or checked != binding
                        or self._bound(context) != binding
                    ):
                        raise ReviewRefused("source_authority_changed")
                    yield context, root, exclusions

    @staticmethod
    def slots(context):
        original = {slot.slot_id: slot for slot in context.preparation.slots}
        original.update(context.stage_slots)
        return tuple(original.values())

    def claim_review_slot(self, review_id: str, slot_id: str) -> ReviewLaunchClaim:
        self._actor(primary=True)
        if self.native_coordinator is not None:
            with self.store._tab.lock, self.store.lease(review_id) as context:
                self._bound(context)
                candidate = next(
                    (slot for slot in self.slots(context) if slot.slot_id == slot_id),
                    None,
                )
                source_scope = context.preparation.target.scope
                admission_context, admission_revision = context, context.revision
            # A later accepted stage gets the actual new primary's permission
            # owner at runtime binding. Recheck before paying for a child; the
            # retained capture and stage intent do not confer filesystem access.
            if self.source_permission is None:
                raise ReviewRefused("source_permission_unavailable")
            if self.source_permission is not None:
                refusal = self.source_permission(
                    "fs_read",
                    {
                        "root_alias": self.source_authority.admitted.alias,
                        "path": source_scope,
                    },
                )
                if refusal:
                    raise ReviewRefused(refusal)
            if candidate is not None:
                refusal = self.destination_check(candidate.target)
                if refusal:
                    raise ReviewRefused(refusal)
        with self._source_context(review_id) as (context, _, _):
            self._bound(context)
            if self.native_coordinator is not None and (
                context is not admission_context
                or context.revision != admission_revision
                or next(
                    (item for item in self.slots(context) if item.slot_id == slot_id),
                    None,
                )
                is not candidate
            ):
                raise ReviewRefused("review_revision_conflict")
            preparation = context.preparation
            allowed = getattr(self, "native_allowed_slots", None)
            if allowed is not None and slot_id not in allowed:
                raise ReviewRefused("stage_grant_scope_mismatch")
            slot = next((s for s in self.slots(context) if s.slot_id == slot_id), None)
            if slot is None:
                raise ReviewRefused("review_slot_unknown")
            if self.native_coordinator is not None and slot.role == "verifier":
                for reviewer in self.slots(context):
                    if reviewer.role != "reviewer":
                        continue
                    previous = context.active_attempts.get(reviewer.slot_id)
                    if (
                        previous is None
                        or (reviewer.slot_id, previous.attempt_id)
                        not in context.presented_attempts
                    ):
                        raise ReviewRefused("original_reports_not_presented")
            grant = self.verification_grant if slot.role == "verifier" else self.grant
            if grant is None:
                raise ReviewRefused("stage_grant_unavailable")
            self.issuer.validate(grant)
            native_association = context.primary_authorities.get(self.actor.run_id)
            if grant.stage != ("verify" if slot.role == "verifier" else "review") or (
                native_association is None
                and any(
                    getattr(grant, key) != getattr(preparation.grant, key)
                    for key in (
                        "session_id",
                        "acceptance_id",
                        "source_epoch",
                        "target_fingerprint",
                        "team_fingerprint",
                        "override_fingerprint",
                    )
                )
            ):
                raise ReviewRefused("stage_grant_scope_mismatch")
            existing = context.launch_claims.get(slot_id)
            if existing is not None:
                return replace(existing, launch_allowed=False)
            if self.capacity().remaining_children == 0:
                raise ReviewRefused("child_budget_unavailable")
            binding = ReviewRunBinding(
                review_id=review_id,
                slot_id=slot_id,
                attempt_id=context.scheduled_attempts.get(slot_id) or uuid.uuid4().hex,
                role=slot.role,
                target_digest=context.target_digest,
                source_epoch=context.source_epoch,
                check_feedback_mode="unavailable_inline"
                if slot.role == "verifier"
                else None,
            )
            claim = ReviewLaunchClaim(review_id, slot, binding, grant)
            context.launch_claims[slot_id] = claim
            context.attempt_slots[(slot_id, binding.attempt_id)] = slot
            return claim

    def release_unaccepted_claim(self, review_id: str, slot_id: str) -> None:
        """A conclusive native admission refusal leaves the slot launchable."""
        self._actor(primary=True)
        with self.store._tab.lock, self.store.lease(review_id) as context:
            self._bound(context)
            claim = context.launch_claims.get(slot_id)
            if claim is not None and claim.run_id is None:
                del context.launch_claims[slot_id]

    def retire_unstarted_launch(self, claim: ReviewLaunchClaim, run_id: str) -> None:
        """Host-only cleanup after the native owner proves no thread started."""
        with self.store._tab.lock, self.store.lease(claim.review_id) as context:
            if (
                context.launch_claims.get(claim.slot.slot_id) is not claim
                or claim.run_id != run_id
            ):
                raise ReviewRefused("review_launch_unknown")
            self.store.retire_unstarted_attempt(run_id, claim.binding)
            del context.launch_claims[claim.slot.slot_id]

    def attach_review_run(self, claim: ReviewLaunchClaim, run_id: str) -> None:
        """Publish the accepted run before its first tool can observe the slot."""
        self.issuer.validate(claim.grant)
        with self.store._tab.lock, self.store.lease(claim.review_id) as context:
            if context.launch_claims.get(claim.slot.slot_id) is not claim:
                raise ReviewRefused("review_launch_unknown")
            if claim.run_id is not None and claim.run_id != run_id:
                raise ReviewRefused("review_launch_already_bound")
            self.store.bind_attempt(run_id, claim.binding)
            claim.run_id = run_id
            if claim.binding.role == "verifier":
                self.store.bind_check_actor(
                    self.actor.run_id, claim.review_id, claim.binding.attempt_id
                )

    def launch_task(self, claim: ReviewLaunchClaim) -> str:
        with self.store.lease(claim.review_id) as context:
            return (
                (
                    _GUIDES["verify"]
                    if claim.slot.role == "verifier"
                    else context.preparation.guide_text
                )
                + "\n"
                + f"Review {claim.review_id}; slot {claim.slot.slot_id}; target {claim.binding.target_digest}; "
                f"target_kind={context.preparation.target.kind}.\n"
                + "Use only code_review_read captured source and code_review_submit role-owned records. "
                "Source queries are list(relative_dir,side), search(text,relative_dir,side), "
                "resolve(relative_path,side); direct reads use document_ref and source_range(start_line,end_line). "
                "Read submission_revision from your source/batch page or last acknowledgment; "
                "use it as expected_revision with a stable submission_id, never the context revision. "
                "On submission_revision_conflict reread and retry the unchanged payload once. "
                "Chunks are at most eight records, each at most 8 KiB UTF-8 JSON.\n"
                + (
                    "For each opaque unit submit kind=dispositions records with schema_version=1, "
                    "verification_unit_id, verdict confirmed/refuted/unresolved, reason, evidence "
                    "and next_check for unresolved. Source evidence requires kind=source, source_ref, "
                    "start_line,end_line,digest; check evidence requires kind=check,receipt_id. "
                    "Confirm only with direct evidence establishing the trigger and impact; refute "
                    "only with concrete counter-evidence. Agreement, proposed commands and non-reproduction "
                    "alone are not proof. Checkpoint each batch before following its cursor. "
                    "For checks submit kind=check_proposals with schema_version=1,argv,intent,verification_unit_ids; "
                    "take proposal_id from accepted_ids and read view=verification_batch,document_ref=proposal_id. "
                    "Inline executable checks and stock Raw CLI are unavailable; do not wait for the "
                    "parent, repeat proposals or enable a fleet. Source-only decisions remain valid; "
                    "if indispensable execution is unavailable use unresolved with next_check. "
                    "Only executed host receipts fully read in this live attempt prove checks. "
                    "Finish with kind=verifier_complete,records=[] after independent dispositions.\n"
                    if claim.slot.role == "verifier"
                    else "Submit kind=findings records with schema_version=1,local_key,title,locations "
                    "(source_ref,start_line,end_line),trigger,impact,evidence,severity "
                    "(critical/high/medium/low),uncertainty. Source evidence includes kind=source, "
                    "source_ref,start_line,end_line,digest. Submit one kind=coverage record with "
                    "schema_version=1,inspected_refs,gaps(reason,optional source_ref),empty_findings. "
                    "Then kind=reviewer_complete,records=[one coverage record]; actual settlement seals completion.\n"
                )
                + (
                    f"Additional focus: {claim.slot.choice.focus}\n"
                    if claim.slot.choice.focus
                    else ""
                )
                + (
                    f"Executable check feedback: {claim.binding.check_feedback_mode}.\n"
                    if claim.slot.role == "verifier"
                    else ""
                )
            )

    def validate_run(self, binding: ReviewRunBinding) -> None:
        """Recheck the live association and source authority at runtime dispatch."""
        with self.store.lease(binding.review_id) as context:
            if self._bound(context) != binding:
                raise ReviewRefused("run_binding_conflict")
            with use_review_source_authority(self.store, self.source_authority):
                with _access(binding.review_id):
                    pass

    def encode(self, result) -> str:
        self._actor()
        text = (
            result.model_dump_json()
            if hasattr(result, "model_dump_json")
            else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        )
        if (
            len(text) > self.result_limit
            or (
                self.verifier_estimate(text) > self.capacity().remaining_input_tokens
                if getattr(self, "verifier_estimate", None) is not None
                else len(text.encode("utf-8")) > self.capacity().remaining_input_tokens * 4
            )
        ):
            self.diagnostic = "review_response_capacity_unavailable"
            raise ReviewRefused(self.diagnostic)
        return text

    def admit_slot(
        self, slot_id: str, role: str, choice: ReviewRunChoice
    ) -> ReviewSlot:
        """Apply the same routing/readiness/consent admission to a new stage."""
        row = self.preset_db.get_agent_definition(choice.preset_id)
        if row is None:
            raise ReviewRefused("preset_missing")
        definition = definition_from_row(row)
        if not definition.enabled:
            raise ReviewRefused("preset_disabled")
        override = choice.destination
        try:
            resolved = resolve_spawn_target(
                self.app_config,
                parent_provider=self.primary_provider,
                parent_model=self.primary_model,
                preset=definition,
                override_provider=override.provider or "" if override else "",
                override_model=override.model or "" if override else "",
                routing=self.routing,
            )
            # The general spawn resolver may inherit both parent fields.
            # Review choices need a configured or explicitly requested route.
            if (
                (
                    resolved.source == "inherit"
                    and not (
                        definition.model
                        or self.routing.subagent_default_model
                        or (override and override.model)
                    )
                )
                or not resolved.provider
                or not resolved.model
            ):
                raise ReviewRefused("no_model_resolved")
            resolved = freeze_spawn_target(self.app_config, resolved)
            fallbacks = resolve_preset_fallback_targets(
                self.app_config,
                definition,
                primary_provider=resolved.provider,
                primary_model=resolved.model,
            )
        except RoutingError as exc:
            raise ReviewRefused(exc.code) from exc
        refusal = self.destination_check(resolved)
        if refusal:
            raise ReviewRefused(refusal)
        return ReviewSlot(slot_id, role, choice, definition, resolved, fallbacks)

    def prepare(
        self,
        target: ReviewTarget,
        reviewers: tuple[ReviewRunChoice, ...] | None = None,
        verifier: ReviewRunChoice | None = None,
    ) -> PreparedReview:
        self._actor(primary=True)
        target = TypeAdapter(ReviewTarget).validate_python(target)
        fingerprints = request_fingerprints(target, self.team, reviewers, verifier)
        if target.binding_ref != self.source_authority.admitted.binding_id:
            raise ReviewRefused("source_binding_mismatch")
        if self.grant is not None and (
            self.grant.stage != "review"
            or any(
                getattr(self.grant, key) != value for key, value in fingerprints.items()
            )
        ):
            raise ReviewRefused("stage_grant_scope_mismatch")
        verification_requested = self.verification_grant is not None or (
            self.native_intent is not None and "verify" in self.native_intent.stages
        )
        selected, selected_verifier = _choices(self.team, reviewers, verifier)
        if not selected:
            raise ReviewRefused("review_team_empty")
        capacity = self.capacity()
        if self.verification_grant is not None:
            self.issuer.validate(self.verification_grant)
            if self.verification_grant.stage != "verify" or any(
                getattr(self.verification_grant, key) != getattr(self.grant, key)
                for key in (
                    "session_id",
                    "acceptance_id",
                    "source_epoch",
                    *fingerprints,
                )
            ):
                raise ReviewRefused("stage_grant_scope_mismatch")
        if verification_requested and selected_verifier is None:
            raise ReviewRefused("verifier_missing")
        required = len(selected) + int(verification_requested)
        if (
            capacity.remaining_children == 0
            or capacity.children_per_turn == 0
            or required
            > capacity.remaining_children
            + capacity.children_per_turn * capacity.remaining_continuations
        ):
            raise ReviewRefused("child_budget_unavailable")
        slots = []
        for index, (role, choice) in enumerate(
            [("reviewer", c) for c in selected]
            + ([("verifier", selected_verifier)] if selected_verifier else [])
        ):
            slots.append(
                self.admit_slot(
                    "V1" if role == "verifier" else f"R{index + 1}", role, choice
                )
            )
        routes = [(s.target.provider, s.target.model, s.target.base_url) for s in slots]
        warnings = []
        if len(set(routes)) < len(routes):
            warnings.append("duplicate_destinations")
        if selected_verifier is None:
            warnings.append("verifier_missing")
        summary = ",".join(warnings) or "prepared"

        def ack(review_id):
            return PreparedReview(
                review_id=review_id,
                revision=0,
                slots=tuple(s.slot_id for s in slots),
                summary=summary,
            )

        if self.source_permission is not None:
            arguments = {
                "root_alias": self.source_authority.admitted.alias,
                "path": target.scope,
            }
            names = ("git_diff", "fs_read") if target.kind != "patch" else ("fs_read",)
            for name in names:
                refusal = self.source_permission(name, arguments)
                if refusal:
                    raise ReviewRefused(refusal)

        # Parameters remain a proposal until current routing, capacity and
        # source permission admission succeed. Only the host then narrows intent.
        if self.grant is None:
            self.grant, self.verification_grant = (
                self.native_coordinator.narrow_request(
                    self.native_intent.message_id,
                    session_id=self.store._session_id,
                    source_epoch=self.source_authority.source_epoch,
                    target=target,
                    team=self.team,
                    reviewers=reviewers,
                    verifier=verifier,
                    source_authority=self.source_authority,
                )
            )

        # The store fits the exact minted ID before allocating any capture files.
        context = self.store.create(
            target_digest=fingerprints["target_fingerprint"],
            source_epoch=self.grant.source_epoch,
            conversation_id=getattr(self, "native_conversation_id", None),
            admit_id=lambda review_id: self.encode(ack(review_id)),
        )
        try:
            manifest = capture_review_source(
                target, self.source_authority, self.store, context.review_id
            )
            self._actor(primary=True)
            snapshot = canonical_record_bytes(
                {
                    "schema_version": 1,
                    "guide": {"schema_version": 1, "text": _REVIEWER_BRIEF},
                    "target": target.model_dump(mode="json"),
                    "slots": [
                        {
                            "slot_id": slot.slot_id,
                            "role": slot.role,
                            "choice": slot.choice.model_dump(mode="json"),
                            "definition": asdict(slot.definition),
                            "target": asdict(slot.target),
                            "fallbacks": [
                                asdict(candidate) for candidate in slot.fallbacks
                            ],
                        }
                        for slot in slots
                    ],
                }
            )
            with (
                self.store._tab.lock,
                self.store.reserve(
                    context.review_id,
                    source_bytes=0,
                    report_bytes=len(snapshot),
                    files=1,
                ) as reservation,
            ):
                self.encode(ack(context.review_id))
                reservation.write(snapshot, category="report")
                reservation.commit(0, len(snapshot), 1)
                context.preparation = ReviewPreparation(
                    self.actor,
                    self.grant.acceptance_id,
                    self.grant,
                    self.source_authority,
                    target,
                    tuple(slots),
                    _REVIEWER_BRIEF,
                )
            self.store.bind_attempt(
                self.actor.run_id,
                ReviewRunBinding(
                    review_id=context.review_id,
                    slot_id="primary",
                    attempt_id=self.grant.acceptance_id,
                    role="primary",
                    target_digest=manifest.target_digest,
                    source_epoch=manifest.source_epoch,
                ),
            )
            if self.native_coordinator is not None:
                self.native_coordinator.track_prepared(self, context.review_id)
            return ack(context.review_id)
        except Exception:
            self.store.release(context.review_id, expected_revision=context.revision)
            raise

    def guide(self, stage: str) -> dict:
        self._actor(primary=True)
        if stage not in ("review", "verify"):
            raise ReviewRefused("invalid_stage")
        result = load_code_review_guide(
            stage, self.team, max_result_chars=self.result_limit
        )
        if "diagnostic" in result:
            self.encode(result)
            return result
        destinations = []
        choices = [
            ("reviewer", choice.preset_id, choice.focus)
            for choice in self.team.reviewers
        ]
        if self.team.verifier_preset_id:
            choices.append(("verifier", self.team.verifier_preset_id, ""))
        result["participants"] = []
        for role, preset_id, focus in choices:
            row = self.preset_db.get_agent_definition(preset_id)
            participant = {
                "role": role,
                "preset_id": preset_id,
                "display_name": row["name"] if row else None,
                "focus": focus,
                "enabled": bool(row["enabled"]) if row else False,
                "destination": {"provider": row["provider"], "model": row["model"]}
                if row
                else None,
            }
            result["participants"].append(participant)
            if row is None or not row["enabled"]:
                result["diagnostics"].append(
                    {
                        "preset_id": preset_id,
                        "reason": "preset_missing"
                        if row is None
                        else "preset_disabled",
                    }
                )
            if row is not None:
                try:
                    target = resolve_spawn_target(
                        self.app_config,
                        parent_provider=self.primary_provider,
                        parent_model=self.primary_model,
                        preset=definition_from_row(row),
                        routing=self.routing,
                        readiness=lambda _config, _provider: None,
                    )
                    participant["destination"] = {
                        "provider": target.provider,
                        "model": target.model,
                    }
                    if target.provider and target.model:
                        destinations.append((target.provider, target.model))
                except RoutingError as exc:
                    result["diagnostics"].append(
                        {"preset_id": preset_id, "reason": exc.code}
                    )
        if len(destinations) != len(set(destinations)):
            result["diagnostics"].append("duplicate_destinations")
        if self.native_coordinator is not None and stage == "review":
            result["default_target"] = {
                "kind": "working_tree",
                "binding_ref": self.source_authority.admitted.binding_id,
                "scope": ".",
                "staged": True,
                "unstaged": True,
                "untracked_paths": [],
            }
        result = fit_code_review_guide(result, self.result_limit)
        self.encode(result)
        return result

    def _selection(
        self, review_id, document_ref, cursor, source_query, source_range=None
    ):
        if source_query is not None:
            source_query = TypeAdapter(SourceQuery).validate_python(source_query)
        if cursor:
            selection = source_cursor_selection(review_id, cursor)
            if selection["op"] == "content":
                if source_query is not None or (
                    document_ref is not None and document_ref != selection["source_ref"]
                ):
                    raise ReviewRefused("source_cursor_conflict")
                document_ref = selection["source_ref"]
                selected_range = SourceRange(
                    start_line=selection["start_line"], end_line=selection["end_line"]
                )
                if source_range is not None and source_range != selected_range:
                    raise ReviewRefused("source_cursor_conflict")
                source_range = selected_range
            else:
                restored = TypeAdapter(SourceQuery).validate_python(selection)
                if (
                    document_ref is not None
                    or source_range is not None
                    or (source_query is not None and source_query != restored)
                ):
                    raise ReviewRefused("source_cursor_conflict")
                source_query = restored
        return document_ref, source_query, source_range

    def source_instruction_targets(
        self,
        review_id: str,
        *,
        document_ref: str | None = None,
        cursor: str | None = None,
        source_query: SourceQuery | None = None,
    ) -> tuple[ToolPathTarget, ...]:
        with self.store.lease(review_id) as context:
            with self.store._tab.lock:
                self._bound(context)
            with use_review_source_authority(self.store, self.source_authority):
                document_ref, query, _ = self._selection(
                    review_id, document_ref, cursor, source_query
                )
                if document_ref:
                    return (
                        ToolPathTarget(
                            original_source_path(review_id, document_ref), "exact"
                        ),
                    )
                if query:
                    with _access(review_id) as (_, _, state, _, root, exclusions):
                        relative = (
                            query.relative_path
                            if query.op == "resolve"
                            else query.relative_dir
                        )
                        if not _in_scope(relative, state.manifest.scope):
                            raise ReviewRefused("source_outside_scope")
                        path = _safe(
                            root, exclusions, relative, directory=query.op != "resolve"
                        )
                        return (
                            ToolPathTarget(
                                path, "exact" if query.op == "resolve" else "directory"
                            ),
                        )
                raise ReviewRefused("invalid_source_selection")

    def read(
        self,
        review_id: str,
        view: str,
        document_ref: str | None = None,
        cursor: str | None = None,
        source_query: SourceQuery | None = None,
        source_range: SourceRange | None = None,
    ) -> ReviewPage:
        if (
            (
                view != "source"
                and (source_query is not None or source_range is not None)
            )
            or (
                source_query is not None
                and (document_ref is not None or source_range is not None)
            )
            or (source_range is not None and not document_ref and not cursor)
        ):
            raise ReviewRefused("invalid_arguments")
        if (
            view == "source"
            and self.native_coordinator is not None
            and self.source_permission is None
        ):
            raise ReviewRefused("source_permission_unavailable")
        if view == "source" and self.source_permission is not None:
            targets = self.source_instruction_targets(
                review_id,
                document_ref=document_ref,
                cursor=cursor,
                source_query=source_query,
            )
            root = self.source_authority.admitted.root
            for target in targets:
                relative = str(target.path.relative_to(root))
                name = "fs_read" if target.kind == "exact" else "fs_grep"
                refusal = self.source_permission(
                    name,
                    {
                        "root_alias": self.source_authority.admitted.alias,
                        "path": relative,
                    },
                )
                if refusal:
                    raise ReviewRefused(refusal)
        if document_ref and view in {"batch", "verification_batch", "summary"}:
            with self.store._tab.lock, self.store.lease(review_id) as context:
                is_check = document_ref in context.checks
            if is_check or view in {"batch", "verification_batch"}:
                return self._read_check_page(review_id, view, document_ref, cursor)
        if view in {"batch", "verification_batch"}:
            with self.store._tab.lock, self.store.lease(review_id) as context:
                binding = self._bound(context)
                if binding.role != "verifier":
                    raise ReviewRefused("role_denied")
                capacity = getattr(self, "verifier_input_tokens", None)
                estimate = getattr(self, "verifier_estimate", None)
                if capacity is None or estimate is None:
                    raise ReviewRefused("review_context_capacity_unknown")
            # Batch source approval may block; do not retain this caller's lock.
            return self.next_verification_batch(
                review_id,
                remaining_input_tokens=capacity,
                estimate=estimate,
                max_result_chars=self.result_limit,
                cursor=cursor,
            )
        with self.store.lease(review_id) as context:
            with self.store._tab.lock:
                binding = self._bound(context)
            if view != "source":
                if binding.role != "primary":
                    raise ReviewRefused(
                        "role_denied"
                        if view != "batch" or binding.role != "verifier"
                        else "batch_unavailable"
                    )
                # Reports can quote any captured source, including raw output.
                # Policy callbacks must run outside the publication lock.
                with self.store._tab.lock:
                    revision = context.revision
                    paths = tuple(
                        sorted(
                            {
                                context.source_state.documents[ref].relative_path
                                for ref in context.sources
                            }
                        )
                    )
                if (
                    self.native_coordinator is not None
                    and self.source_permission is None
                ):
                    raise ReviewRefused("source_permission_unavailable")
                if self.source_permission is not None:
                    for path in paths:
                        refusal = self.source_permission(
                            "fs_read",
                            {
                                "root_alias": self.source_authority.admitted.alias,
                                "path": path,
                            },
                        )
                        if refusal:
                            raise ReviewRefused(refusal)
                with self._source_context(review_id) as (current, root, exclusions):
                    if (
                        current is not context
                        or context.revision != revision
                        or self._bound(context) != binding
                        or paths
                        != tuple(
                            sorted(
                                {
                                    context.source_state.documents[ref].relative_path
                                    for ref in context.sources
                                }
                            )
                        )
                    ):
                        raise ReviewRefused("source_authority_changed")
                    for path in paths:
                        _safe(root, exclusions, path)
                    return self._read_report(
                        context, binding, view, document_ref, cursor
                    )
            if source_range is not None:
                source_range = SourceRange.model_validate(source_range)
            try:
                with use_review_source_authority(self.store, self.source_authority):
                    ref, query, selection = self._selection(
                        review_id, document_ref, cursor, source_query, source_range
                    )
                    if query is not None:
                        capacity = self.capacity().remaining_input_tokens
                        if capacity <= 0:
                            raise ReviewRefused("review_response_capacity_unavailable")
                        # Every Unicode character is at most four UTF-8 bytes.
                        page = query_review_source(
                            review_id,
                            query,
                            cursor=cursor,
                            max_result_chars=self.result_limit,
                            max_tokens=min(1024, capacity),
                            estimate=getattr(self, "verifier_estimate", None),
                        )
                    elif ref:
                        page = read_review_source(
                            context.source_state.manifest,
                            ref,
                            start_line=selection.start_line if selection else 1,
                            end_line=selection.end_line if selection else None,
                            max_tokens=min(
                                1024, self.capacity().remaining_input_tokens
                            ),
                            cursor=cursor,
                            max_result_chars=self.result_limit,
                            estimate=getattr(self, "verifier_estimate", None),
                        )
                    else:
                        raise ReviewRefused("invalid_source_selection")
                    self.encode(page)
                    return page
            except ReviewRefused as exc:
                if exc.reason == "review_control_too_large_for_tool_result":
                    self.diagnostic = "review_response_capacity_unavailable"
                    raise ReviewRefused(self.diagnostic) from exc
                raise

    def _submit(
        self,
        review_id: str,
        kind: str,
        records: list[dict],
        expected_revision: int,
        submission_id: str,
    ) -> SubmissionAck:
        with self._source_context(review_id) as (context, root, exclusions):
            binding = self._bound(context)
            if binding.role == "primary" and kind != "check_actions":
                raise ReviewRefused("role_denied")
            if (
                context.streams[(binding.slot_id, binding.attempt_id)].terminal_status
                is not None
            ):
                raise ReviewRefused(
                    "verification_attempt_closed"
                    if binding.role == "verifier"
                    else "review_run_settled"
                )
            parsed = validate_record_chunk(kind, records)
            refs = set()
            for record in parsed:
                refs.update(
                    location.source_ref for location in getattr(record, "locations", ())
                )
                refs.update(
                    evidence.source_ref
                    for evidence in getattr(record, "evidence", ())
                    if hasattr(evidence, "source_ref")
                )
                refs.update(getattr(record, "inspected_refs", ()))
                refs.update(
                    gap.source_ref
                    for gap in getattr(record, "gaps", ())
                    if gap.source_ref
                )
            for ref in refs:
                document = context.source_state.documents.get(ref)
                if document is None or ref not in context.sources:
                    raise ReviewRefused("invalid_source_reference")
                _safe(root, exclusions, document.relative_path)
            ack = self.store.publish(
                review_id,
                kind=kind,
                records=records,
                expected_revision=expected_revision,
                submission_id=submission_id,
                admit_ack=self.encode,
            )
            if kind == "check_proposals":
                context.checks[ack.accepted_ids[0]].deadline = getattr(
                    self, "check_deadline", None
                )
            elif kind == "check_actions":
                action = parsed[0]
                if action.action == "claim":
                    self._claim_check(context, action.proposal_id, root, exclusions)
                else:
                    self._release_unstarted_check(context.checks[action.proposal_id])
            return ack

    def note_invalid_submission(self, review_id: str, submission_id: str) -> None:
        """Preserve a malformed envelope gap only for its authenticated owner."""
        with self.store._tab.lock, self.store.lease(review_id) as context:
            self._bound(context)
            self.store.note_rejection(review_id, submission_id, "invalid_arguments")

    def submit(
        self,
        review_id: str,
        kind: str,
        records: list[dict],
        expected_revision: int,
        submission_id: str,
    ) -> SubmissionAck:
        with self.store.lease(review_id) as context:
            with self.store._tab.lock:
                self._bound(context)
            try:
                ack = self._submit(
                    review_id, kind, records, expected_revision, submission_id
                )
                return ack
            except ReviewRefused as exc:
                self.store.note_rejection(review_id, submission_id, exc.reason)
                raise

    def _read_check_page(self, review_id, view, proposal_id, cursor):
        if view != "summary":
            self.await_check_result(review_id, proposal_id)
        with self.store._tab.lock, self.store.lease(review_id) as context:
            binding = self._bound(context)
            check = context.checks.get(proposal_id)
            if check is None:
                raise ReviewRefused("invalid_check_proposal")
            primary = view == "summary"
            if primary and (
                binding.role != "primary" or check.primary_run_id != self.actor.run_id
            ):
                raise ReviewRefused("check_actor_mismatch")
            if not primary and (binding.role != "verifier" or check.binding != binding):
                raise ReviewRefused("check_attempt_mismatch")
            state = SimpleNamespace(
                cursors=context.report_cursors, pages=context.report_pages
            )
            selection = {"view": view, "proposal_id": proposal_id}
            offset = (
                _load_cursor(state, cursor, binding, selection)["position"]["offset"]
                if cursor
                else 0
            )
            capacity = min(1024, self.capacity().remaining_input_tokens)
            estimate = getattr(self, "verifier_estimate", None)
            if cursor:
                replay = _replay(state, cursor, self.result_limit, capacity, estimate)
                if replay is not None:
                    return replay
            if primary:
                record = {
                    "proposal_id": proposal_id,
                    "proposal": check.proposal.model_dump(mode="json"),
                    "state": check.state,
                }
                if check.claim is not None:
                    record["claim"] = {
                        "claim_id": check.claim.claim_id,
                        "tool_name": check.claim.tool_name,
                        "arguments": check.claim.arguments,
                    }
                if check.receipt is not None:
                    record["result"] = check.receipt.model_dump(mode="json")
                text = ""
            else:
                record = check.receipt.model_dump(mode="json")
                text = record["evidence"]
                record["evidence"] = ""
            token = uuid.uuid4().hex

            def page(length, more):
                result = _base_page(
                    context.source_state,
                    context,
                    binding,
                    records=(record,),
                    text=text[offset : offset + length],
                    cursor=token if more else None,
                )
                if primary:
                    result = result.model_copy(
                        update={
                            "submission_revision": context.check_streams[
                                (self.actor.run_id, check.binding.attempt_id)
                            ].revision
                        }
                    )
                return result

            whole = page(len(text) - offset, False)
            if _fits(whole, self.result_limit, capacity, estimate):
                if not primary and offset == 0:
                    record["evidence"], text = text, ""
                    whole = page(0, False)
                result = whole
            else:
                lo, hi = 0, len(text) - offset
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if _fits(page(mid, True), self.result_limit, capacity, estimate):
                        lo = mid
                    else:
                        hi = mid - 1
                result = page(lo, True)
                if not lo or not _fits(result, self.result_limit, capacity, estimate):
                    check.transport_reason = "check_result_transport_unavailable"
                    raise ReviewRefused(check.transport_reason)
            self.encode(result)
            result = _persist_page(
                self.store,
                context,
                state,
                binding,
                selection,
                result,
                {"offset": offset + len(result.text)},
                cursor,
            )
            if not primary and result.cursor is None:
                check.result_read = True
            return result

    def _resolve_check(self, context, check, outcome, reason):
        if check.receipt is not None:
            return check.receipt
        if check.native_execution_receipt_id is not None:
            outcome = "uncertain"
        if len(reason) > 1024:
            reason = reason[:1000] + " [reason truncated]"
        receipt = check.result(outcome, reason)
        payload = {"check_result": receipt.model_dump(mode="json")}
        if check.execution is not None:
            check.execution.reservation.write(
                canonical_record_bytes(payload), category="report"
            )
        elif check.result_reservation is not None:
            reservation = check.result_reservation
            body = canonical_record_bytes(payload)
            reservation.write(body, category="report")
            reservation.commit(0, len(body), 1)
            reservation.__exit__(None, None, None)
            check.result_reservation = None
        else:
            self.store.write_verification_state(context, payload)
        check.receipt, check.state = receipt, "resolved"
        context.check_receipt_ids.add(receipt.receipt_id)
        check.changed.set()
        self._release_unstarted_check(check)
        return receipt

    @staticmethod
    def _release_unstarted_check(check):
        if check.result_reservation is not None:
            check.result_reservation.__exit__(None, None, None)
            check.result_reservation = None
        if (
            check.execution is not None
            and not check.execution.started
            and check.native_execution_receipt_id is None
        ):
            check.execution.release()

    def claim_check(self, review_id: str, proposal_id: str) -> ReviewCheckClaim:
        """Claim once; absence of an enforcing executor never admits a process."""
        with self._source_context(review_id) as (context, root, exclusions):
            return self._claim_check(context, proposal_id, root, exclusions)

    def _claim_check(self, context, proposal_id, root, exclusions):
        """Apply/replay a claim inside an already-authorized publication scope."""
        from .code_review_checks import ReviewCheckClaim

        self._bound(context)
        actor = self._actor(primary=True)
        check = context.checks.get(proposal_id)
        if (
            check is None
            or check.primary_run_id != actor.run_id
            or context.check_actors.get(actor.run_id) != check.binding.attempt_id
        ):
            raise ReviewRefused("check_actor_mismatch")
        if (
            context.streams[
                (check.binding.slot_id, check.binding.attempt_id)
            ].terminal_status
            is not None
        ):
            raise ReviewRefused("verification_attempt_closed")
        if check.claim is not None:
            return check.claim
        if check.receipt is None:
            try:
                self._prepare_check_execution(context, check, root, exclusions)
            except ReviewRefused as exc:
                self._resolve_check(context, check, "unavailable", exc.reason)
        claim = ReviewCheckClaim(
            uuid.uuid4().hex,
            context.review_id,
            proposal_id,
            check.binding.attempt_id,
            actor.run_id,
            context.target_digest,
            tool_name="shell_exec" if check.execution else None,
            arguments=dict(check.execution.arguments) if check.execution else None,
            receipt=check.receipt,
        )
        check.claim = claim
        return claim

    def _prepare_check_execution(self, context, check, root, exclusions):
        """Reserve before an existing enforcing executor materializes a copy."""
        import os
        import shlex
        import subprocess

        from .code_review_checks import (
            CHECK_GENERATED_BYTES,
            CHECK_GENERATED_FILES,
            CHECK_OUTPUT_BYTES,
            ReviewCheckExecution,
        )
        from .raw_shell_tool_provider import RawShellToolProvider

        provider = self.check_process_provider
        factory = None
        if type(provider) is RawShellToolProvider:
            getter = getattr(provider._runtime, "review_check_boundary_factory", None)
            factory = getter() if getter else None
        if not callable(factory):
            self._resolve_check(
                context, check, "unavailable", "quota_enforcing_executor_unavailable"
            )
            return
        if not provider.catalog_enabled():
            self._resolve_check(
                context, check, "unavailable", "process_permission_unavailable"
            )
            return
        sources = {}
        priorities = {"base": 0, "head": 1, "staged": 2, "worktree": 3}
        for side in priorities:
            for doc in context.source_state.documents.values():
                if doc.side == side:
                    _safe(root, exclusions, doc.relative_path)
                    sources[doc.relative_path] = self.store.read_file(
                        context, context.sources[doc.source_ref].path
                    )
            for absent_side, path in context.source_state.manifest.absent_sources:
                if absent_side == side:
                    sources.pop(path, None)
        source_bytes = (
            sum(len(value) for value in sources.values()) + CHECK_GENERATED_BYTES
        )
        files = len(sources) + CHECK_GENERATED_FILES + 1
        reservation = self.store.reserve(
            context.review_id,
            source_bytes=source_bytes,
            report_bytes=CHECK_OUTPUT_BYTES,
            files=files,
            participant_id=check.binding.slot_id,
        )
        boundary = None
        try:
            boundary = factory(
                parent=context.bundle_path,
                sources=sources,
                max_bytes=source_bytes,
                max_files=files - 1,
            )
            root = boundary.root.resolve(strict=True)
            if (
                not root.is_relative_to(context.bundle_path)
                or root == context.bundle_path
                or not root.is_dir()
            ):
                raise ReviewRefused("invalid_disposable_check_root")
            deadline = min(
                getattr(self, "check_deadline", time.monotonic()),
                check.deadline or time.monotonic(),
            )
            seconds = min(300.0, deadline - time.monotonic())
            if seconds <= 0:
                raise ReviewRefused("check_deadline_expired")
            command = (
                subprocess.list2cmdline(check.proposal.argv)
                if os.name == "nt"
                else shlex.join(check.proposal.argv)
            )
            arguments = {
                "command": command,
                "shell": "auto",
                "initial_directory": str(root),
                "timeout_seconds": seconds,
            }
            check.execution = ReviewCheckExecution(
                self, provider, check, boundary, reservation, arguments
            )
            check.state = "claimed"
        except BaseException:
            if boundary is None or boundary.release():
                reservation.__exit__(None, None, None)
            raise

    def prepare_check_process(self, call):
        """Bind the exact claiming primary's specification before real dispatch."""
        from contextlib import nullcontext

        if call.name != "shell_exec":
            return nullcontext()
        actor = self._actor(primary=True)
        with self.store._tab.lock:
            candidates = [
                (context.review_id, check, check.execution)
                for context in self.store._tab.contexts.values()
                for check in context.checks.values()
                if check.primary_run_id == actor.run_id and check.execution is not None
            ]
        for review_id, check, execution in candidates:
            with self._source_context(review_id) as (context, _, _):
                if (
                    context.checks.get(check.proposal_id) is not check
                    or check.execution is not execution
                ):
                    raise ReviewRefused("check_execution_specification_mismatch")
                if execution.started:
                    if call.args == execution.arguments or check.receipt is None:
                        raise ReviewRefused("check_execution_already_accepted")
                    continue
                if check.receipt is not None or call.args != execution.arguments:
                    raise ReviewRefused("check_execution_specification_mismatch")
                return execution.activate()
        return nullcontext()

    def register_check_result(
        self, review_id: str, proposal_id: str, execution_receipt_id: str
    ) -> CheckReceipt:
        """Accept facts only from this claim's native raw process adapter."""
        with self.store._tab.lock, self.store.lease(review_id) as context:
            check = context.checks.get(proposal_id)
            execution = check.execution if check is not None else None
            native = execution.native_result if execution is not None else None
            if native is None or native.invocation_id != execution_receipt_id:
                raise ReviewRefused("invalid_native_check_receipt")
            if execution.registered_result is native:
                return check.receipt
            if execution.accepted_id is None:
                receipt = self._resolve_check(
                    context, check, "unavailable", "process_not_started"
                )
            else:
                if (
                    execution.accepted_id != execution_receipt_id
                    or check.binding.target_digest != context.target_digest
                ):
                    raise ReviewRefused("check_receipt_target_mismatch")
                evidence = f"terminal_state: {native.terminal_state}\nexit_code: {native.exit_code}\ncleanup_proven: {native.cleanup_proven}\nstdout:\n{native.stdout_preview}\nstderr:\n{native.stderr_preview}"
                encoded = evidence.encode("utf-8")
                evidence = encoded[:4096].decode("utf-8", errors="ignore")
                if len(encoded) > 4096 or native.truncated:
                    evidence += "\noutput_truncated: true"
                receipt = check.result(
                    "executed" if native.cleanup_proven else "uncertain", evidence
                )
                payload = canonical_record_bytes(
                    {"check_result": receipt.model_dump(mode="json")}
                )
                execution.reservation.write(payload, category="report")
                if check.result_reservation is not None:
                    check.result_reservation.__exit__(None, None, None)
                    check.result_reservation = None
                check.receipt, check.state = receipt, "resolved"
                context.check_receipt_ids.add(receipt.receipt_id)
                check.changed.set()
            execution.registered_result = native
            if native.cleanup_proven:
                execution.release()
            return receipt

    def pending_check_for_wait(
        self, primary_run_id: str, handle_ids: tuple[str, ...]
    ) -> ReviewCheckNotice | None:
        """Called by the actual fleet wait with its owned child run references."""
        from .code_review_checks import ReviewCheckNotice

        with self.store._tab.lock:
            for context in self.store._tab.contexts.values():
                self.store._live(context)
                if context.check_actors.get(primary_run_id) is None:
                    continue
                for check in context.checks.values():
                    if (
                        check.primary_run_id == primary_run_id
                        and check.verifier_run_id in handle_ids
                        and check.receipt is None
                        and check.state == "pending"
                    ):
                        check.notice_count += 1
                        return ReviewCheckNotice(
                            context.review_id, check.proposal_id, check.verifier_run_id
                        )
        return None

    def await_check_result(self, review_id: str, proposal_id: str) -> CheckReceipt:
        """Wait within the actual verifier deadline; no caller-selected duration."""
        with self.store.lease(review_id) as context:
            while True:
                with self._source_context(review_id) as (current, _, _):
                    if current is not context:
                        raise ReviewRefused("source_authority_changed")
                    binding = self._bound(context)
                    check = context.checks.get(proposal_id)
                    if (
                        binding.role != "verifier"
                        or check is None
                        or check.binding != binding
                        or check.verifier_run_id != self.actor.run_id
                    ):
                        raise ReviewRefused("check_attempt_mismatch")
                    if (
                        context.streams[
                            (binding.slot_id, binding.attempt_id)
                        ].terminal_status
                        is not None
                    ):
                        raise ReviewRefused("verification_attempt_closed")
                    if check.receipt is not None:
                        return check.receipt
                    deadline = getattr(self, "check_deadline", None)
                    cancelled = getattr(self, "check_cancelled", lambda: False)
                    if deadline is None:
                        return self._resolve_check(
                            context, check, "unavailable", "check_deadline_unavailable"
                        )
                    if getattr(self, "check_budget_expired", lambda: False)():
                        return self._resolve_check(
                            context, check, "expired", "verifier_deadline_expired"
                        )
                    if cancelled():
                        return self._resolve_check(
                            context, check, "cancelled", "verifier_cancelled"
                        )
                    if time.monotonic() >= deadline:
                        return self._resolve_check(
                            context, check, "expired", "verifier_deadline_expired"
                        )
                    # Acceptance is a notification, not a terminal result. Clear
                    # under the publication lock before the next bounded wait.
                    check.changed.clear()
                check.wait_count += 1
                check.changed.wait(min(0.05, max(0, deadline - time.monotonic())))

    def end_check_actor(
        self, run_id: str, reason: str, *, unstarted_only: bool = False
    ) -> None:
        """Signal unstarted work before native fleet draining or terminal settlement."""
        with self.store._tab.lock:
            for context in self.store._tab.contexts.values():
                for check in context.checks.values():
                    if self.store._tab.closed.is_set():
                        check.changed.set()
                        if (
                            run_id in {check.primary_run_id, check.verifier_run_id}
                            and check.execution is not None
                            and not check.execution.started
                            and not check.execution.released
                        ):
                            check.execution.release()
                        continue
                    if (
                        run_id in {check.primary_run_id, check.verifier_run_id}
                        and check.receipt is None
                        and not (
                            unstarted_only
                            and check.native_execution_receipt_id is not None
                        )
                    ):
                        self._resolve_check(
                            context,
                            check,
                            "unavailable"
                            if unstarted_only or "tool" in reason
                            else "cancelled",
                            reason,
                        )

    def capture_review_output(self, run_id: str, text: str, *, terminal: bool) -> None:
        """Host-only capture of assistant text, never tool input or thinking metadata."""
        from tldw_chatbook.Utils.log_sanitizer import redact_log_line

        from .agent_service import (
            _UNQUOTED_REASONING_FIELD_RE,
            _without_reasoning_fields,
            _without_think_blocks,
        )

        self.issuer.validate(self.grant)
        filtered = redact_log_line(text, max_length=0)
        filtered, think = _without_think_blocks(filtered)
        reasoning = False
        if _UNQUOTED_REASONING_FIELD_RE.search(filtered):
            filtered, reasoning = _without_reasoning_fields(filtered)
        self.store.capture_review_output(
            run_id,
            filtered,
            terminal=terminal,
            withheld=filtered != text or think or reasoning,
        )

    def settle_review_run(
        self, run_id: str, *, terminal_status: str, receipt_id: str
    ) -> None:
        """Host-only actual terminal receipt; neither model JSON nor a row grants it."""
        self.end_check_actor(run_id, "verification_attempt_closed")
        self.store.settle_review_run(
            run_id, terminal_status=terminal_status, receipt_id=receipt_id
        )

    def _authorize_verification_sources(self, review_id, units):
        """Recheck live permissions on original binding paths before claim delivery."""
        if self.native_coordinator is not None and self.source_permission is None:
            raise ReviewRefused("source_permission_unavailable")
        refs = {
            location["source_ref"]
            for unit in units
            for location in unit["claim"]["locations"]
        }
        refs.update(
            evidence["source_ref"]
            for unit in units
            for evidence in unit["claim"]["evidence"]
            if evidence["kind"] == "source"
        )
        with use_review_source_authority(self.store, self.source_authority):
            for ref in refs:
                path = original_source_path(review_id, ref)
                if self.source_permission is not None:
                    refusal = self.source_permission(
                        "fs_read",
                        {
                            "root_alias": self.source_authority.admitted.alias,
                            "path": str(
                                path.relative_to(self.source_authority.admitted.root)
                            ),
                        },
                    )
                    if refusal:
                        raise ReviewRefused(refusal)

    def record_verification_gap(self, review_id, reason):
        """Preserve a request-level refusal without labeling claims oversized."""
        with self.store._tab.lock, self.store.lease(review_id) as context:
            binding = self._bound(context)
            if binding.role != "verifier":
                return
            ledger = self.store.verification_ledger(context, binding)
            if reason not in ledger.request_gaps:
                self.store.write_verification_state(
                    context, {"attempt_id": binding.attempt_id, "request_gap": reason}
                )
                ledger.request_gaps.append(reason)

    def next_verification_batch(
        self,
        review_id: str,
        *,
        remaining_input_tokens: int,
        estimate: Callable[[str], int],
        max_result_chars: int,
        cursor: str | None,
    ) -> ReviewPage:
        """Deliver whole blinded units under independent input/transport caps."""
        from .code_review_verification import unresolved

        if type(remaining_input_tokens) is not int or remaining_input_tokens <= 0:
            raise ReviewRefused("review_context_reduction_blocked")
        claim_budget = min(4096, remaining_input_tokens // 2)
        limit = min(self.result_limit, review_result_char_limit(max_result_chars))
        with self.store._tab.lock, self.store.lease(review_id) as context:
            binding = self._bound(context)
            units = self.store.verification_ledger(context, binding).units.public_units
        # Permission callbacks may reach native approval/SQLite owners.
        self._authorize_verification_sources(review_id, units)
        with self.store._tab.lock, self.store.lease(review_id) as context:
            binding = self._bound(context)
            stream = context.streams[(binding.slot_id, binding.attempt_id)]
            if stream.terminal_status is not None:
                raise ReviewRefused("review_run_settled")
            ledger = self.store.verification_ledger(context, binding)
            if cursor in ledger.pages:
                page = ReviewPage.model_validate_json(
                    self.store.read_file(context, ledger.pages[cursor])
                )
                if (
                    len(page.model_dump_json()) > limit
                    or estimate(canonical_record_bytes(page.records).decode())
                    > claim_budget
                    or estimate(page.model_dump_json()) > remaining_input_tokens
                ):
                    raise ReviewRefused("review_page_capacity_changed")
                return page
            if cursor is not None and cursor not in ledger.positions:
                raise ReviewRefused("invalid_cursor")
            if ledger.delivered - ledger.dispositions.keys():
                raise ReviewRefused("verification_batch_unacknowledged")
            offset = ledger.positions.get(cursor, 0)
            records, gaps, skipped = [], [], {}
            token = uuid.uuid4().hex
            units = ledger.units.public_units

            def page(next_cursor):
                return _base_page(
                    context.source_state,
                    context,
                    binding,
                    records=tuple(records),
                    gaps=tuple(gaps),
                    text="Zero findings: no verification units." if not units else "",
                    cursor=next_cursor,
                )

            while offset < len(units) and len(records) < 8:
                unit = units[offset]
                candidate = records + [unit]
                reason = None
                claim_tokens = estimate(canonical_record_bytes(candidate).decode())
                if claim_tokens > claim_budget:
                    if records:
                        break
                    if (
                        claim_tokens <= 4096
                        and getattr(self, "verifier_required_continuation", False)
                    ):
                        # Retained private owners prevent proving owner-free fit.
                        self.record_verification_gap(
                            review_id, "review_context_reduction_blocked"
                        )
                        raise ReviewRefused("review_context_reduction_blocked")
                    reason = "claim_too_large"
                records.append(unit)
                prospective = page(token if offset + 1 < len(units) else None)
                if reason is None and len(prospective.model_dump_json()) > limit:
                    reason = "review_record_too_large_for_tool_result"
                if (
                    reason is None
                    and estimate(prospective.model_dump_json()) > remaining_input_tokens
                ):
                    records.pop()
                    if records:
                        break
                    raise ReviewRefused("review_context_reduction_blocked")
                if reason:
                    records.pop()
                    if records:
                        break
                    skipped[unit["verification_unit_id"]] = unresolved(
                        unit["verification_unit_id"], reason
                    )
                    if not any(gap.reason == reason for gap in gaps):
                        gaps.append(CoverageGap(reason=reason))
                offset += 1
            result = page(token if offset < len(units) else None)
            if (
                len(result.model_dump_json()) > limit
                or estimate(result.model_dump_json()) > remaining_input_tokens
            ):
                raise ReviewRefused("review_response_capacity_unavailable")
            # One charged immutable record commits gaps, delivered IDs and cursor
            # before any of them can be acknowledged or evicted from history.
            self.store.write_verification_state(
                context,
                {
                    "attempt_id": binding.attempt_id,
                    "page": result.model_dump(mode="json"),
                    "dispositions": skipped,
                    "offset": offset,
                },
            )
            payload = result.model_dump_json().encode()
            with self.store.reserve(
                review_id, source_bytes=0, report_bytes=len(payload), files=1
            ) as reservation:
                path = reservation.write(payload, category="report")
                reservation.commit(0, len(payload), 1)
            ledger.pages[cursor] = path
            if result.cursor:
                ledger.positions[result.cursor] = offset
            ledger.cursor = result.cursor
            ledger.delivered.update(row["verification_unit_id"] for row in records)
            ledger.dispositions.update(skipped)
            return result

    def verifier_checkpoint(self, review_id: str) -> ReviewHistoryCheckpoint:
        """Return this live attempt's bounded recoverable checkpoint only."""
        from .code_review_history import ReviewHistoryCheckpoint

        with self.store._tab.lock, self.store.lease(review_id) as context:
            binding = self._bound(context)
            stream = context.streams[(binding.slot_id, binding.attempt_id)]
            if stream.terminal_status is not None:
                raise ReviewRefused("review_run_settled")
            ledger = self.store.verification_ledger(context, binding)
            return ReviewHistoryCheckpoint(
                attempt_id=binding.attempt_id,
                target_digest=binding.target_digest,
                submission_revision=stream.revision,
                acknowledged_cursors=tuple(ledger.acknowledged_cursors),
                cursor=ledger.cursor,
                pending_unit_ids=tuple(
                    sorted(ledger.delivered - ledger.dispositions.keys())
                ),
                check_refs=tuple(
                    check.proposal_id
                    for check in context.checks.values()
                    if check.binding == binding and check.result_read
                ),
                pending_check_refs=tuple(
                    check.proposal_id
                    for check in context.checks.values()
                    if check.binding == binding and not check.result_read
                ),
            )

    def expand_dispositions(self, review_id: str) -> tuple[dict, ...]:
        """Expand each accepted original exactly once through the private map."""
        with self.store._tab.lock, self.store.lease(review_id) as context:
            self._bound(context)
            self._actor(primary=True)
            binding = next(
                (
                    value
                    for value in context.active_attempts.values()
                    if value.role == "verifier"
                ),
                None,
            )
            if binding is None:
                raise ReviewRefused("verification_unavailable")
            ledger = self.store.verification_ledger(context, binding)
            from .code_review_verification import unresolved

            return tuple(
                {
                    "original_id": original,
                    "original": self.store.read_record(
                        context, context.original_index[original]
                    ),
                    "disposition": ledger.dispositions.get(
                        unit, unresolved(unit, "verification_incomplete")
                    ),
                }
                for original, unit in ledger.units.original_to_unit.items()
            )

    def finding_index(self, review_id: str, cursor: str | None = None) -> ReviewPage:
        return self.read(review_id, "index", cursor=cursor)

    def _report_envelope(self, context, slot_id, attempt_id=None):
        slot = next(s for s in self.slots(context) if s.slot_id == slot_id)
        binding = context.active_attempts.get(slot_id)
        attempt_id = attempt_id or (binding.attempt_id if binding else None)
        stream = context.streams.get((slot_id, attempt_id))
        slot = context.attempt_slots.get((slot_id, attempt_id), slot)
        envelope = {
            "schema_version": 1,
            "slot_id": slot_id,
            "attempt_id": attempt_id,
            "target_digest": context.target_digest,
            "guide_fingerprint": hashlib.sha256(
                context.preparation.guide_text.encode()
            ).hexdigest(),
            "preset_fingerprint": hashlib.sha256(
                canonical_record_bytes(asdict(slot.definition))
            ).hexdigest(),
            "requested": {"provider": slot.target.provider, "model": slot.target.model},
            "report_status": "not_started",
            "continuation_status": context.continuation_status,
            "empty_findings": False,
        }
        if stream is None:
            return envelope
        coverage = (
            self.store.read_record(context, stream.coverage)
            if stream.coverage
            else None
        )
        valid = bool(
            coverage is not None
            and coverage["empty_findings"] == (not stream.local_keys)
            and (coverage["inspected_refs"] or coverage["gaps"])
        )
        if slot.role == "verifier":
            ledger = stream.verification
            units = set(ledger.units.original_to_unit.values()) if ledger else set()
            valid = bool(
                ledger is not None
                and units == ledger.delivered == ledger.dispositions.keys()
                and not ledger.request_gaps
            )
        sealed = (
            stream.terminal_status == "done"
            and bool(stream.terminal_receipt)
            and stream.completion_requested
            and valid
            and not stream.rejected
            and stream.capture_terminal
            and bool(stream.capture_receipt)
            and not stream.capture_problem
        )
        envelope.update(
            {
                "run_id": stream.run_id,
                "submission_revision": stream.revision,
                "host_sequence": stream.host_sequence,
                "capture_receipt": stream.capture_receipt,
                "terminal_status": stream.terminal_status,
                "terminal_receipt": stream.terminal_receipt,
                "report_status": "sealed"
                if sealed
                else (
                    "incomplete" if stream.terminal_status is not None else "pending"
                ),
                "seal_receipt": hashlib.sha256(
                    canonical_record_bytes(
                        {
                            "terminal": stream.terminal_receipt,
                            "revision": stream.revision,
                            "capture": stream.host_sequence,
                            "attempt": attempt_id,
                        }
                    )
                ).hexdigest()
                if sealed
                else None,
                "capture_status": stream.capture_problem
                or (
                    "withheld"
                    if stream.withheld
                    else "complete"
                    if stream.capture_terminal
                    else "pending"
                ),
                "raw_ref": stream.raw_ref,
                "coverage": coverage,
                "coverage_ref": "coverage:" + slot_id + ":" + attempt_id
                if coverage
                else None,
                "rejected_submissions": tuple(stream.rejected.values()),
                "empty_findings": bool(
                    sealed and coverage and coverage["empty_findings"]
                ),
                "actual_routes": tuple(stream.routes),
                "active_route": stream.routes[-1] if stream.routes else None,
            }
        )
        return envelope

    def _read_report(self, context, binding, view, document_ref, cursor):
        """Pack whole originals/index entries before the generic truncation seam."""
        if view not in {"summary", "report", "index"}:
            raise ReviewRefused("report_unavailable")
        state = SimpleNamespace(
            cursors=context.report_cursors, pages=context.report_pages
        )
        selection = {"view": view, "document_ref": document_ref}
        capacity = self.capacity().remaining_input_tokens
        if cursor:
            position = _load_cursor(state, cursor, binding, selection)["position"]
            replay = _replay(state, cursor, self.result_limit, capacity)
            if replay is not None:
                self.encode(replay)
                return replay
        else:
            if view == "summary":
                ids = [
                    [s.slot_id, attempt]
                    for s in self.slots(context)
                    for attempt in (
                        [a for slot, a in context.streams if slot == s.slot_id]
                        or [None]
                    )
                ]
            elif document_ref and document_ref.startswith("raw-"):
                if view != "report":
                    raise ReviewRefused("invalid_report_reference")
                ids = [document_ref]
            elif document_ref and document_ref.startswith("coverage:"):
                ids = [document_ref]
            else:
                ids = [document_ref] if document_ref else list(context.original_index)
            position = {"ids": ids, "offset": 0, "raw_offset": 0}
        ids, offset = position["ids"], position["offset"]
        records, text = [], ""
        next_token = uuid.uuid4().hex

        def page(*, next_cursor=None):
            return _base_page(
                context.source_state,
                context,
                binding,
                records=tuple(records),
                text=text,
                cursor=next_cursor,
            )

        while offset < len(ids):
            ref = ids[offset]
            if isinstance(ref, str) and ref.startswith("raw-"):
                stream = next(
                    (s for s in context.streams.values() if s.raw_ref == ref), None
                )
                if stream is None:
                    raise ReviewRefused("invalid_report_reference")
                raw = "".join(
                    self.store.read_file(context, path).decode()
                    for path in stream.raw_chunks
                )
                position.setdefault("raw_limit", len(raw))
                raw = raw[: position["raw_limit"]]
                records.append(
                    {
                        "raw_ref": ref,
                        "capture_status": stream.capture_problem
                        or (
                            "withheld"
                            if stream.withheld
                            else "complete"
                            if stream.capture_terminal
                            else "pending"
                        ),
                        "terminal_status": stream.terminal_status,
                    }
                )
                start = position["raw_offset"]
                lo, hi = 0, len(raw) - start
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    text = raw[start : start + mid]
                    if _fits(page(next_cursor=next_token), self.result_limit, capacity):
                        lo = mid
                    else:
                        hi = mid - 1
                text = raw[start : start + lo]
                if not lo and start < len(raw):
                    raise ReviewRefused("review_response_capacity_unavailable")
                position["raw_offset"] = start + lo
                if start + lo == len(raw):
                    offset += 1
                break
            if view == "summary":
                candidate = {"slots": [self._report_envelope(context, *ref)]}
            elif ref.startswith("coverage:"):
                _, slot, attempt = ref.split(":", 2)
                if (slot, attempt) not in context.streams:
                    raise ReviewRefused("invalid_report_reference")
                candidate = {
                    "record_ref": ref,
                    "record": self._report_envelope(context, slot, attempt)["coverage"],
                }
            else:
                location = context.original_index.get(ref)
                if location is None:
                    raise ReviewRefused("invalid_report_reference")
                chunk = json.loads(self.store.read_file(context, location[0]))
                envelope = self._report_envelope(
                    context, chunk["slot_id"], chunk["attempt_id"]
                )
                candidate = {
                    "original_id": ref,
                    "record_ref": ref,
                    "slot_id": chunk["slot_id"],
                    "attempt_id": chunk["attempt_id"],
                    "report_status": envelope["report_status"],
                    "coverage_ref": envelope["coverage_ref"],
                }
                if view == "report":
                    candidate["record"] = chunk["records"][location[1]]
            records.append(candidate)
            if not _fits(page(next_cursor=next_token), self.result_limit, capacity):
                records.pop()
                if records:
                    break
                records.append(
                    {"record_ref": ref, "gap": "model_read_record_too_large"}
                )
                self.encode(page(next_cursor=next_token))
                offset += 1
                break
            offset += 1
        position["offset"] = offset
        result = page(next_cursor=next_token if offset < len(ids) else None)
        self.encode(result)
        return _persist_page(
            self.store, context, state, binding, selection, result, position, cursor
        )

    def native_report_slots(self, review_id: str) -> tuple[str, ...]:
        """Return only this retained context's actual participant references."""
        from .run_context import use_run_actor

        with (
            use_run_actor(self.actor),
            self.store._tab.lock,
            self.store.lease(review_id) as context,
        ):
            self._bound(context)
            self._actor(primary=True)
            return tuple(
                slot.slot_id for slot in self.slots(context) if slot.role == "reviewer"
            )

    def native_source_options(self, review_id: str) -> tuple[tuple[str, str], ...]:
        """Return captured evidence identities without expanding document bodies."""
        from .run_context import use_run_actor

        with (
            use_run_actor(self.actor),
            self._source_context(review_id) as (context, root, exclusions),
        ):
            self._actor(primary=True)
            return tuple(
                (f"{doc.relative_path} · {doc.side} · captured source", "source:" + ref)
                for ref, doc in context.source_state.documents.items()
                if ref in context.sources and _safe(root, exclusions, doc.relative_path)
            )

    def native_report_page(
        self, review_id: str, selection: str, offset: int = 0
    ) -> dict:
        """Materialize one native page; evidence stays independently byte-paged.

        Native offsets are local UI state, never model authority. Every read
        revalidates the live context, source authority, and captured file identity.
        Accepted records remain intact in storage and in explicit exports.
        """
        import codecs

        from .run_context import use_run_actor

        page_chars = 8192
        if offset < 0:
            raise ReviewRefused("invalid_cursor")
        with (
            use_run_actor(self.actor),
            self._source_context(review_id) as (context, root, exclusions),
        ):
            self._actor(primary=True)
            if selection.startswith("source:"):
                ref = selection[7:]
                source = context.sources.get(ref)
                doc = context.source_state.documents.get(ref)
                if source is None or doc is None:
                    raise ReviewRefused("invalid_source_reference")
                _safe(root, exclusions, doc.relative_path)
                data = self.store.read_file(
                    context, source.path, offset=offset, limit=page_chars
                )
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                body = decoder.decode(data, final=offset + len(data) >= doc.byte_count)
                consumed = len(data) - len(decoder.getstate()[0])
                return {
                    "body": body,
                    "next_offset": offset + consumed
                    if offset + consumed < doc.byte_count
                    else None,
                }

            def sections():
                if selection.startswith("raw:"):
                    binding = context.active_attempts.get(selection[4:])
                    if binding is None:
                        yield "Raw output unavailable: participant has not started."
                        return
                    stream = context.streams[(selection[4:], binding.attempt_id)]
                    for path in stream.raw_chunks:
                        yield self.store.read_file(context, path).decode("utf-8")
                    return
                if selection == "unified":
                    # Sort only identities/keys; expand one original at a time.
                    binding = context.active_attempts.get("V1")
                    if binding is None:
                        raise ReviewRefused("verification_unavailable")
                    ledger = self.store.verification_ledger(context, binding)
                    from .code_review_verification import unresolved

                    def disposition(unit):
                        return ledger.dispositions.get(
                            unit, unresolved(unit, "verification_incomplete")
                        )

                    def rank(pair):
                        key, unit = pair
                        record = self.store.read_record(
                            context, context.original_index[key]
                        )
                        return (
                            {"confirmed": 0, "unresolved": 1, "refuted": 2}[
                                disposition(unit)["verdict"]
                            ],
                            {"critical": 0, "high": 1, "medium": 2, "low": 3}[
                                record["severity"]
                            ],
                            key,
                        )

                    yield "Unified review — session retention\n\n"
                    yield (
                        json.dumps(
                            {
                                "review_id": review_id,
                                "target_digest": context.target_digest,
                                "target": {
                                    "kind": context.source_state.manifest.kind,
                                    "scope": context.source_state.manifest.scope,
                                    "revisions": dict(
                                        context.source_state.manifest.revisions
                                    ),
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n\n"
                    )
                    for slot in self.slots(context):
                        yield (
                            json.dumps(
                                self._report_envelope(context, slot.slot_id),
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n\n"
                        )
                    for key, unit in sorted(
                        ledger.units.original_to_unit.items(), key=rank
                    ):
                        record = self.store.read_record(
                            context, context.original_index[key]
                        )
                        decision = disposition(unit)
                        yield (
                            json.dumps(
                                {
                                    "original_id": key,
                                    "original": record,
                                    "disposition": decision,
                                    "captured_evidence": self._native_source_evidence(
                                        context,
                                        record.get("evidence", [])
                                        + decision.get("evidence", []),
                                        root,
                                        exclusions,
                                    ),
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n\n"
                        )
                    for check in context.checks.values():
                        yield (
                            json.dumps(
                                check.receipt.model_dump(mode="json")
                                if check.receipt
                                else {
                                    "proposal_id": check.proposal_id,
                                    "status": "proposed",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    return
                yield f"Original report {selection} — {review_id}\nSession retention: export to keep a durable copy.\n\n"
                yield (
                    json.dumps(
                        self._report_envelope(context, selection),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n\n"
                )
                for key, location in context.original_index.items():
                    if key.startswith(selection + "-F"):
                        record = self.store.read_record(context, location)
                        yield (
                            key
                            + "\n"
                            + json.dumps(record, ensure_ascii=False, indent=2)
                            + "\n\n"
                        )
                        yield (
                            json.dumps(
                                {
                                    "captured_evidence": self._native_source_evidence(
                                        context,
                                        record.get("evidence", []),
                                        root,
                                        exclusions,
                                    )
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n\n"
                        )

            remaining, pieces, count = offset, [], 0
            for section in sections():
                if remaining >= len(section):
                    remaining -= len(section)
                    continue
                piece = section[remaining : remaining + page_chars + 1 - count]
                remaining = 0
                pieces.append(piece)
                count += len(piece)
                if count > page_chars:
                    break
            body = "".join(pieces)
            return {
                "body": body[:page_chars],
                "next_offset": offset + page_chars if len(body) > page_chars else None,
            }

    def render_all_reports(self, review_id: str) -> str:
        """Export every retained original/raw stream and available unified view."""
        sections = []
        for slot in self.native_report_slots(review_id):
            sections.append(self.render_original_report(review_id, slot))
            sections.append(
                "Raw output " + slot + "\n" + self.render_raw_report(review_id, slot)
            )
        try:
            sections.append(self.render_unified_report(review_id))
        except ReviewRefused as exc:
            if exc.reason != "verification_unavailable":
                raise
            sections.append("Verification has not run.")
        return "\n\n".join(sections)

    def release_native(self, review_id: str):
        """Only a trusted native discard action calls this method; not a wire flag."""
        from .run_context import use_run_actor

        with use_run_actor(self.actor), self.store._tab.lock:
            context = self.store._context(review_id)
            self._bound(context)
            self._actor(primary=True)
            if any(
                stream.terminal_status is None and stream.run_id
                for (slot, _), stream in context.streams.items()
                if slot != "primary"
            ):
                raise ReviewRefused("review_busy")
            return self.store.release(review_id, expected_revision=context.revision)

    def render_original_report(self, review_id: str, slot_id: str) -> str:
        """Render retained original records intact for the owning native reader."""
        from .run_context import use_run_actor

        with (
            use_run_actor(self.actor),
            self._source_context(review_id) as (context, root, exclusions),
        ):
            self._bound(context)
            self._actor(primary=True)
            report = self._report_envelope(context, slot_id)
            records = [
                (key, self.store.read_record(context, location))
                for key, location in context.original_index.items()
                if key.startswith(slot_id + "-F")
            ]
            sections = [
                f"Original report {slot_id} — {review_id}",
                "Session retention: export to keep a durable copy.",
                json.dumps(report, ensure_ascii=False, indent=2),
            ]
            for key, record in records:
                sections.append(
                    key + "\n" + json.dumps(record, ensure_ascii=False, indent=2)
                )
                sections.append(
                    json.dumps(
                        {
                            "captured_evidence": self._native_source_evidence(
                                context, record.get("evidence", []), root, exclusions
                            )
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return "\n\n".join(sections)

    def _native_source_evidence(self, context, evidence, root, exclusions):
        """Keep evidence reference-backed; native source pages materialize bytes."""
        result = []
        for ref in evidence:
            if ref.get("kind") != "source":
                continue
            source = context.sources.get(ref["source_ref"])
            document = context.source_state.documents.get(ref["source_ref"])
            if source is None or document is None:
                raise ReviewRefused("invalid_source_reference")
            _safe(root, exclusions, document.relative_path)
            result.append(
                {**ref, "relative_path": document.relative_path, "side": document.side}
            )
        return result

    def render_raw_report(self, review_id: str, slot_id: str) -> str:
        """Read policy-permitted raw chunks without the model transport ceiling."""
        from .run_context import use_run_actor

        with (
            use_run_actor(self.actor),
            self.store._tab.lock,
            self.store.lease(review_id) as context,
        ):
            self._bound(context)
            self._actor(primary=True)
            binding = context.active_attempts.get(slot_id)
            if binding is None:
                return "Raw output unavailable: participant has not started."
            stream = context.streams[(slot_id, binding.attempt_id)]
            return "".join(
                self.store.read_file(context, path).decode("utf-8")
                for path in stream.raw_chunks
            )

    def render_unified_report(self, review_id: str) -> str:
        """Expand the complete private host map without modifying a finding."""
        from .run_context import use_run_actor

        with (
            use_run_actor(self.actor),
            self._source_context(review_id) as (context, root, exclusions),
        ):
            self._bound(context)
            self._actor(primary=True)
            expanded = self.expand_dispositions(review_id)
            severity = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            rows = sorted(
                expanded,
                key=lambda row: (
                    {"confirmed": 0, "unresolved": 1, "refuted": 2}[
                        row["disposition"]["verdict"]
                    ],
                    severity[row["original"]["severity"]],
                    row["original_id"],
                ),
            )
            for row in rows:
                row["captured_evidence"] = self._native_source_evidence(
                    context,
                    row["original"].get("evidence", [])
                    + row["disposition"].get("evidence", []),
                    root,
                    exclusions,
                )
            body = {
                "review_id": review_id,
                "target_digest": context.target_digest,
                "target": {
                    "kind": context.source_state.manifest.kind,
                    "scope": context.source_state.manifest.scope,
                    "revisions": dict(context.source_state.manifest.revisions),
                },
                "participants": [
                    self._report_envelope(context, slot.slot_id)
                    for slot in self.slots(context)
                ],
                "original_dispositions": rows,
                "checks": [
                    check.receipt.model_dump(mode="json")
                    if check.receipt
                    else {"proposal_id": check.proposal_id, "status": "proposed"}
                    for check in context.checks.values()
                ],
            }
            return "Unified review — session retention\n\n" + json.dumps(
                body, ensure_ascii=False, indent=2
            )

    def release(self, review_id: str, expected_revision: int) -> ReleaseReceipt:
        self._actor(primary=True)
        with self.store._tab.lock:
            if review_id in self._released:
                self.store._live()
                return self.store.release(
                    review_id,
                    expected_revision=expected_revision,
                    admit_ack=self.encode,
                )
            context = self.store._context(review_id)
            if self._bound(context).role != "primary":
                raise ReviewRefused("role_denied")
            receipt = self.store.release(
                review_id, expected_revision=expected_revision, admit_ack=self.encode
            )
            self._released[review_id] = receipt
            while len(self._released) > 4:
                self._released.pop(next(iter(self._released)))
            return receipt
