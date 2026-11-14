"""TASK-34100.3 pass 3: the per-domain append rule the trace planner admits.

Turn N of a saved chat with a reasoning-replay model sends the recorded
messages, then the previous reply and the new user message, then every
recorded reasoning checkpoint, then the previous reply's checkpoint. Only an
exact append in each domain is admitted; any change to what was recorded
(ADR-337: "a request whose saved history really changed") returns None so
the planner's existing rules refuse it.

``matches`` stands in for the planner's durable comparison: recorded entry
``i`` equals incoming entry ``j`` exactly when their domains and labels are
equal, and, like the planner's, a match records the incoming value in the
retained values. ``prefix`` is computed the way the planner computes it.
"""

from __future__ import annotations

import uuid

import pytest

from tldw_chatbook.Chat.console_trace_domain_append import domain_append_indices
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    DerivedTraceProvenance,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
    TraceTransformKind,
)

M = "messages_payload"
C = "provider_continuations"
POLICY = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)


def _message(label: str) -> SavedRevisionTraceProvenance:
    # One stable UUIDv4-shaped revision id per label.
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"task-34100-3/{label}").hex
    return SavedRevisionTraceProvenance(
        str(uuid.UUID(digest[:12] + "4" + digest[13:16] + "8" + digest[17:]))
    )


def _attached(owner: str) -> DerivedTraceProvenance:
    return DerivedTraceProvenance(
        TraceTransformKind.CONTINUATION_ATTACHMENT,
        (_message(owner),),
        artifact=ProviderArtifactTraceProvenance(
            TraceProvenanceSource.CONTINUATION, POLICY
        ),
    )


def _plan(recorded, incoming, *, calls=None, retained=None):
    """Run the rule over (domain, label, descriptor) rows, as the planner does.

    Args:
        calls: Optional list that records every ``matches`` call the rule
            makes (the planner's flat prefix loop is not recorded).
        retained: Optional dict standing in for the planner's retained
            artifact values; ``matches`` writes each match into it.
    """

    recorded_domains = [domain for domain, _label in recorded]
    recorded_labels = [label for _domain, label in recorded]
    incoming_domains = [domain for domain, _label, _descriptor in incoming]
    incoming_labels = [label for _domain, label, _descriptor in incoming]
    retained = {} if retained is None else retained

    def same(i: int, j: int) -> bool:
        return (
            recorded_domains[i] == incoming_domains[j]
            and recorded_labels[i] == incoming_labels[j]
        )

    prefix = 0
    while prefix < min(len(recorded), len(incoming)) and same(prefix, prefix):
        prefix += 1

    def matches(i: int, j: int) -> bool:
        if calls is not None:
            calls.append((i, j))
        matched = same(i, j)
        if matched:
            retained[i] = incoming_labels[j]
        return matched

    return domain_append_indices(
        recorded_domains=recorded_domains,
        domains=incoming_domains,
        descriptors=[descriptor for _domain, _label, descriptor in incoming],
        prefix=prefix,
        matches=matches,
        retained=retained,
    )


#: Turn 2's request as the trace recorded it: [u1, a1, u2] + [c(a1)].
RECORDED = [(M, "u1"), (M, "a1"), (M, "u2"), (C, "c-a1")]


def _turn_three(**overrides):
    """Turn 3's request: [u1, a1, u2, a2, u3] + [c(a1), c(a2)]."""

    rows = {
        "u1": (M, "u1", _message("u1")),
        "a1": (M, "a1", _message("a1")),
        "u2": (M, "u2", _message("u2")),
        "a2": (M, "a2", _message("a2")),
        "u3": (M, "u3", _message("u3")),
        "c-a1": (C, "c-a1", _attached("a1")),
        "c-a2": (C, "c-a2", _attached("a2")),
    }
    order = overrides.pop("order", ["u1", "a1", "u2", "a2", "u3", "c-a1", "c-a2"])
    rows.update(overrides)
    return [rows[key] for key in order]


def test_an_exact_append_in_each_domain_is_admitted() -> None:
    # Incoming indices: a2=3, u3=4 (new messages), c(a2)=6 (new checkpoint).
    assert _plan(RECORDED, _turn_three()) == (3, 4, 6)


def test_later_turns_keep_appending() -> None:
    recorded = [
        (M, "u1"), (M, "a1"), (M, "u2"), (M, "a2"), (M, "u3"),
        (C, "c-a1"), (C, "c-a2"),
    ]
    incoming = _turn_three(
        a3=(M, "a3", _message("a3")),
        u4=(M, "u4", _message("u4")),
        **{"c-a3": (C, "c-a3", _attached("a3"))},
        order=["u1", "a1", "u2", "a2", "u3", "a3", "u4", "c-a1", "c-a2", "c-a3"],
    )
    assert _plan(recorded, incoming) == (5, 6, 9)


@pytest.mark.parametrize(
    ("case", "incoming"),
    [
        # A recorded message was edited.
        ("edited reply", _turn_three(a1=(M, "a1-edited", _message("a1")))),
        # A recorded checkpoint changed its bytes.
        ("changed checkpoint", _turn_three(**{"c-a1": (C, "c-a1-v2", _attached("a1"))})),
        # The recorded checkpoint is no longer replayed.
        ("dropped checkpoint", _turn_three(order=["u1", "a1", "u2", "a2", "u3", "c-a2"])),
        # Only the newest checkpoint is replayed (here with bytes equal to
        # the recorded one): the domain did not grow, so this is not an
        # append in each domain. The existing rules decide it.
        (
            "newest in the recorded slot",
            _turn_three(
                **{"c-a2": (C, "c-a1", _attached("a2"))},
                order=["u1", "a1", "u2", "a2", "u3", "c-a2"],
            ),
        ),
        # A new checkpoint for an EARLIER reply: history gained an attachment.
        ("new checkpoint on a recorded reply", _turn_three(**{"c-a2": (C, "c-a2", _attached("a1"))})),
        # The recorded checkpoint now claims the new reply as its owner.
        ("recorded slot owned by a new reply", _turn_three(**{"c-a1": (C, "c-a1", _attached("a2"))})),
        # Checkpoints reordered.
        ("reordered checkpoints", _turn_three(order=["u1", "a1", "u2", "a2", "u3", "c-a2", "c-a1"])),
        # A recorded message removed (history shortened, then grown).
        ("removed message", _turn_three(order=["u1", "u2", "a2", "u3", "c-a1", "c-a2"])),
        # An untyped continuation cannot prove its owner.
        (
            "untyped checkpoint",
            _turn_three(
                **{
                    "c-a2": (
                        C,
                        "c-a2",
                        ProviderArtifactTraceProvenance(
                            TraceProvenanceSource.CONTINUATION, POLICY
                        ),
                    )
                }
            ),
        ),
    ],
)
def test_a_changed_history_is_left_to_the_existing_rules(case, incoming) -> None:
    assert _plan(RECORDED, incoming) is None, case


def test_growth_in_one_domain_is_left_to_the_existing_rules() -> None:
    # Messages before an unchanged checkpoint: the planner already admits it.
    only_messages = _turn_three(order=["u1", "a1", "u2", "a2", "u3", "c-a1"])
    assert _plan(RECORDED, only_messages) is None
    # A new checkpoint with no new message would attach to a recorded reply.
    only_checkpoint = _turn_three(
        **{"c-a2": (C, "c-a2", _attached("a1"))},
        order=["u1", "a1", "u2", "c-a1", "c-a2"],
    )
    assert _plan(RECORDED, only_checkpoint) is None
    # Nothing recorded in the continuation domain: an ordinary flat append.
    assert _plan(RECORDED[:3], _turn_three()) is None


def test_domains_out_of_provider_order_are_refused() -> None:
    # Defensive invariants: the planner builds both sequences domain-sorted,
    # so its hook never hands the rule an interleaved one.
    interleaved = _turn_three(order=["u1", "a1", "u2", "c-a1", "a2", "u3", "c-a2"])
    assert _plan(RECORDED, interleaved) is None
    assert _plan([(M, "u1"), (C, "c-a1"), (M, "a1")], _turn_three()) is None


def _reply_artifact() -> ProviderArtifactTraceProvenance:
    """How the Console carries every assistant reply: one equal descriptor."""
    return ProviderArtifactTraceProvenance(TraceProvenanceSource.ACTIVE_REQUEST, POLICY)


def _attached_to_a_reply() -> DerivedTraceProvenance:
    return DerivedTraceProvenance(
        TraceTransformKind.CONTINUATION_ATTACHMENT,
        (_reply_artifact(),),
        artifact=ProviderArtifactTraceProvenance(
            TraceProvenanceSource.CONTINUATION, POLICY
        ),
    )


def _console_turn_three(order=None):
    """Turn 3 as the mounted Console sends it: replies are equal artifacts."""
    rows = {
        "u1": (M, "u1", _message("u1")),
        "a1": (M, "a1", _reply_artifact()),
        "u2": (M, "u2", _message("u2")),
        "a2": (M, "a2", _reply_artifact()),
        "u3": (M, "u3", _message("u3")),
        "c-a1": (C, "c-a1", _attached_to_a_reply()),
        "c-a2": (C, "c-a2", _attached_to_a_reply()),
        "c-a1-again": (C, "c-a1-again", _attached_to_a_reply()),
    }
    order = order or ["u1", "a1", "u2", "a2", "u3", "c-a1", "c-a2"]
    return [rows[key] for key in order]


def test_console_reply_artifacts_are_owners_by_side_not_by_identity() -> None:
    """The real Console shape: every reply is the same ACTIVE_REQUEST artifact.

    Pre-merge review M2 asked that a checkpoint's owner occur exactly once
    among the incoming messages, so that "attached to a recorded / new
    message" would be a proof. In the mounted Console (TASK-34100.3 pilots,
    2026-11-13) every assistant reply is ``ProviderArtifactTraceProvenance(
    ACTIVE_REQUEST, policy)``, equal for every reply, so a uniqueness rule
    refused every Kimi turn 3+ again. The owner check can only say which side
    a checkpoint belongs to; the no-swap guarantee is the recorded slots'
    byte equality.
    """
    assert _plan(RECORDED, _console_turn_three()) == (3, 4, 6)
    # The limitation, pinned: with equal reply artifacts, a new checkpoint
    # that really belongs to reply 1 cannot be told from reply 2's. It is
    # appended (and recorded exactly as sent); only a saved-revision owner
    # lets the rule refuse it (``new checkpoint on a recorded reply`` above).
    again = _console_turn_three(
        order=["u1", "a1", "u2", "a2", "u3", "c-a1", "c-a1-again"]
    )
    assert _plan(RECORDED, again) == (3, 4, 6)
    # A recorded slot whose bytes changed is still refused: content decides.
    changed = _console_turn_three(order=["u1", "a1", "u2", "a2", "u3", "c-a2", "c-a1"])
    assert _plan(RECORDED, changed) is None


#: Turn 3's recorded surface: [u1, a1, u2, a2, u3] + [c(a1), c(a2)].
RECORDED_TURN_THREE = [
    (M, "u1"), (M, "a1"), (M, "u2"), (M, "a2"), (M, "u3"),
    (C, "c-a1"), (C, "c-a2"),
]


def _turn_four(**overrides):
    """Turn 4's request: [u1 .. u3, a3, u4] + [c(a1), c(a2), c(a3)]."""

    return _turn_three(
        a3=(M, "a3", _message("a3")),
        u4=(M, "u4", _message("u4")),
        **{"c-a3": (C, "c-a3", _attached("a3"))},
        **overrides,
        order=["u1", "a1", "u2", "a2", "u3", "a3", "u4", "c-a1", "c-a2", "c-a3"],
    )


def test_the_rule_compares_only_the_recorded_continuations() -> None:
    """Pre-merge review M4: the flat prefix already compared every message.

    Only the recorded continuations are compared, each with the incoming
    continuation at the same index. None of them is incoming index 0, so the
    planner's (0, 0) rendered-system branch of ``matches`` is never taken.
    """
    calls: list[tuple[int, int]] = []

    assert _plan(RECORDED_TURN_THREE, _turn_four(), calls=calls) == (5, 6, 9)
    assert calls == [(5, 7), (6, 8)]


def test_an_inexact_message_prefix_compares_nothing() -> None:
    calls: list[tuple[int, int]] = []
    edited = _turn_four(a1=(M, "a1-edited", _message("a1")))

    assert _plan(RECORDED_TURN_THREE, edited, calls=calls) is None
    assert calls == []


def test_a_refused_probe_leaves_the_retained_values_as_it_found_them() -> None:
    """Pre-merge review M3: the probe is speculative, so a None undoes it.

    The first recorded checkpoint matches (and ``matches`` records it), the
    second has new bytes. The rule returns None, and the planner's retained
    values must be exactly what they were, or a later rule would inherit a
    value from a comparison it never made.
    """
    retained: dict[object, object] = {0: "u1", 1: "a1"}
    changed = _turn_four(**{"c-a2": (C, "c-a2-v2", _attached("a2"))})
    calls: list[tuple[int, int]] = []

    assert _plan(RECORDED_TURN_THREE, changed, calls=calls, retained=retained) is None
    assert calls == [(5, 7), (6, 8)], "the first checkpoint did match and record"
    assert retained == {0: "u1", 1: "a1"}


def test_an_admitted_probe_keeps_what_it_compared() -> None:
    """The planner still gets each recorded checkpoint's original value."""
    retained: dict[object, object] = {}

    assert _plan(RECORDED_TURN_THREE, _turn_four(), retained=retained) == (5, 6, 9)
    assert retained == {5: "c-a1", 6: "c-a2"}
