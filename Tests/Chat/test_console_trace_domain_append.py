"""TASK-34100.3 pass 3: the per-domain append rule the trace planner admits.

Turn N of a saved chat with a reasoning-replay model sends the recorded
messages, then the previous reply and the new user message, then every
recorded reasoning checkpoint, then the previous reply's checkpoint. Only an
exact append in each domain is admitted; any change to what was recorded
(ADR-337: "a request whose saved history really changed") returns None so
the planner's existing rules refuse it.

``matches`` stands in for the planner's durable comparison: recorded entry
``i`` equals incoming entry ``j`` exactly when their labels are equal.
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


def _plan(recorded, incoming):
    """Run the rule over (domain, label, descriptor) rows."""

    recorded_labels = [label for _domain, label in recorded]
    incoming_labels = [label for _domain, label, _descriptor in incoming]
    return domain_append_indices(
        recorded_domains=[domain for domain, _label in recorded],
        domains=[domain for domain, _label, _descriptor in incoming],
        descriptors=[descriptor for _domain, _label, descriptor in incoming],
        matches=lambda i, j: recorded_labels[i] == incoming_labels[j],
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
    interleaved = _turn_three(order=["u1", "a1", "u2", "c-a1", "a2", "u3", "c-a2"])
    assert _plan(RECORDED, interleaved) is None
    assert _plan([(M, "u1"), (C, "c-a1"), (M, "a1")], _turn_three()) is None
