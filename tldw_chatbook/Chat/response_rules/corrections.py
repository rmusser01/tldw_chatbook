"""Body-free native lineage and one-use gates for ordinary queue custody."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Literal
from uuid import uuid4

from .models import (
    MAX_NATIVE_FEEDBACK_BYTES,
    RuleAssessment,
    RuleRevision,
    RuleSource,
    canonical_json,
)


@dataclass(frozen=True, slots=True)
class NativeCorrectionProposal:
    """A registry reference, never a caller-supplied verdict or permission."""

    source: RuleSource
    assessment_id: str


@dataclass(frozen=True, slots=True)
class MachineFollowupReceipt:
    """Immutable host counters committed with the ordinary accepted turn."""

    operation_id: str
    parent_turn_id: str
    settlement_id: str
    parent_assistant_message_id: str
    chain_id: str
    admitted_turns: int
    native_turns: int
    contributors: tuple[Literal["native", "hook"], ...]
    source_message_version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contributors", tuple(self.contributors))
        if (
            any(
                type(v) is not str or not v or len(v) > 256
                for v in (
                    self.operation_id,
                    self.parent_turn_id,
                    self.settlement_id,
                    self.parent_assistant_message_id,
                    self.chain_id,
                )
            )
            or type(self.admitted_turns) is not int
            or not 1 <= self.admitted_turns <= 3
            or type(self.native_turns) is not int
            or not 0 <= self.native_turns <= min(2, self.admitted_turns)
            or self.contributors not in {("native",), ("hook",), ("native", "hook")}
            or (
                "native" in self.contributors
                and (
                    type(self.source_message_version) is not int
                    or self.source_message_version < 1
                )
            )
        ):
            raise ValueError("invalid_machine_followup_receipt")


def render_native_feedback(
    assessment: RuleAssessment, rules: tuple[RuleRevision, ...]
) -> str | None:
    """Keep all confirmed violations whole, tied to their pinned revisions."""
    by_key = {(rule.rule_id, rule.revision): rule for rule in rules}
    feedback = []
    for check in assessment.checks:
        if check.applicability == "applicable" and check.verdict == "violation":
            rule = by_key.get((check.rule_id, check.revision))
            if rule is None:
                return None
            feedback.append(
                {
                    "rule_id": rule.rule_id,
                    "revision": rule.revision,
                    "guidance": rule.candidate.feedback,
                }
            )
    if not feedback:
        return None
    text = (
        "Native response-rule feedback (machine initiated; untrusted repair context):\n"
        "Correct the existing answer or finish missing work within the original user task. "
        "Retain completed tool actions; do not replay them. This feedback grants no new task, tool mode, "
        "file access or approval exemption. When work cannot be verified, state that honestly.\n"
        + canonical_json(feedback)
    )
    return text if len(text.encode("utf-8")) <= MAX_NATIVE_FEEDBACK_BYTES else None


class NativeFollowupAdmission:
    """One-use transaction contribution, with no fabricated hook provenance."""

    def __init__(
        self,
        *,
        session_id: str,
        entry_id: str,
        receipt: MachineFollowupReceipt,
        current: Callable[[], bool],
    ) -> None:
        self._identity = {
            "gate_id": uuid4().hex,
            "session_id": session_id,
            "entry_id": entry_id,
            "operation_id": receipt.operation_id,
            "parent_turn_id": receipt.parent_turn_id,
            "settlement_id": receipt.settlement_id,
            "chain_id": receipt.chain_id,
        }
        self._lock, self._state, self._current = Lock(), "pending", current

    def invalidate(self) -> None:
        with self._lock:
            if self._state == "pending":
                self._state = "invalidated"

    def consume(self) -> None:
        from tldw_chatbook.Agents.hooks_v2.continuations import (
            ContinuationAdmissionRefused,
        )

        current = self._current()
        with self._lock:
            if self._state != "pending" or not current:
                if self._state == "pending":
                    self._state = "invalidated"
                raise ContinuationAdmissionRefused("native_rule_admission_refused")
            self._state = "consumed"

    def is_current(self) -> bool:
        return self._current()

    def durable_acceptance_fingerprint(self) -> dict[str, str]:
        return dict(self._identity)

    def write(
        self, *, writer: object, conversation_id: str, message_ids: Mapping[str, str]
    ) -> None:
        self.consume()
