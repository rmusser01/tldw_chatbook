"""Admit a surface that grows at the end of each provider domain (ADR-337).

A traced Console request carries two ordered domains: ``messages_payload``
and ``provider_continuations``. The planner compares the incoming request
with the recorded surface as one flat list (messages, then continuations),
so it can only admit growth at the end of that list, or new messages before
an unchanged continuation suffix.

A saved chat with a model that replays its reasoning (every Kimi model
Moonshot lists) grows in both domains on every turn after the second: the
new turn appends the previous reply and the new user message, and that
reply's reasoning checkpoint follows the checkpoints already recorded.
Nothing recorded changes; flattened, though, the new messages sit before the
old checkpoints and the new checkpoint after them, so the flat comparison
saw two change points and refused the send ("Trace capture blocked",
TASK-34100.3 / TASK-34100.3.1).

The surface store already keeps per-domain ordinals (a warm append places
each new descriptor at the end of its own domain) and the cold bootstrap
already reads a delta as "whatever follows the recorded prefix of each
domain". This module only decides when that per-domain append is the
request's exact meaning:

- every recorded message is matched, in order, by the incoming message at the
  same index (the planner's flat prefix has already compared them, so it
  passes ``prefix`` and only an exact message prefix qualifies), and every
  recorded continuation by the incoming continuation at the same index (the
  caller's ``matches`` performs the same durable, credential-filtered
  comparison the flat prefix uses);
- at least one message and at least one continuation are new;
- every continuation is a typed attachment whose owner equals one of the
  recorded messages (for a recorded slot) or one of the new messages (for a
  new slot).

The owner check is by descriptor equality and says only which side of the
history a checkpoint belongs to. It cannot name the reply: the Console carries
every assistant reply as the same ``ACTIVE_REQUEST`` artifact descriptor, so
every checkpoint names an owner that occurs once per reply (requiring it to be
unique refused every Kimi turn 3+, pre-merge review 2026-11-13). That no recorded
reply gains, loses or swaps a checkpoint comes from content: each recorded
slot must equal its recorded bytes, and a Kimi checkpoint carries its own
reply's text, so a swap would need byte-identical checkpoints and would send
identical bytes either way (the strength of the planner's existing suffix
rule).

Anything else (an edited, removed or reordered recorded entry, an untyped
continuation, or a new checkpoint whose owner is not a new message) returns
``None`` and the caller's existing rules decide, which refuse it as before.
Only an owner that carries an identity (a saved revision) can show that a new
checkpoint belongs to an earlier reply; with the Console's reply artifacts
such a checkpoint is appended like any new one, and the ledger still records
exactly the request sent. A ``None`` leaves no trace of the probe in the
planner's retained artifact values.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence

from tldw_chatbook.Chat.console_trace_provenance import (
    DerivedTraceProvenance,
    TraceProvenance,
    TraceTransformKind,
)

MESSAGES = "messages_payload"
CONTINUATIONS = "provider_continuations"


def _owner(descriptor: TraceProvenance) -> TraceProvenance | None:
    """Return the message a continuation attachment belongs to, if typed."""

    if (
        type(descriptor) is not DerivedTraceProvenance
        or descriptor.transform is not TraceTransformKind.CONTINUATION_ATTACHMENT
        or len(descriptor.inputs) != 1
    ):
        return None
    return descriptor.inputs[0]


def domain_append_indices(
    *,
    recorded_domains: Sequence[str],
    domains: Sequence[str],
    descriptors: Sequence[TraceProvenance],
    prefix: int,
    matches: Callable[[int, int], bool],
    retained: MutableMapping[int, object],
) -> tuple[int, ...] | None:
    """Return the incoming indices to append per domain, or None.

    Args:
        recorded_domains: Domain of each recorded surface entry, in provider
            domain order (all messages, then all continuations).
        domains: Domain of each incoming descriptor, in the same order.
        descriptors: Incoming descriptors aligned with ``domains``.
        prefix: The planner's flat prefix: how many leading entries matched
            at the same index. It stops at the first recorded continuation,
            so it equals the recorded message count exactly when every
            recorded message matched.
        matches: ``matches(recorded_index, incoming_index)`` from the planner;
            True only when the recorded entry durably equals that incoming
            entry. A match records the incoming artifact value in
            ``retained``.
        retained: The planner's retained artifact values, which ``matches``
            writes. Restored when this returns None, so a refused probe
            leaves nothing behind.

    Returns:
        The incoming indices of the new messages followed by the new
        continuations, when the request is an exact append in each domain;
        otherwise None.
    """

    recorded_messages = sum(domain == MESSAGES for domain in recorded_domains)
    recorded_continuations = len(recorded_domains) - recorded_messages
    incoming_messages = sum(domain == MESSAGES for domain in domains)
    incoming_continuations = len(domains) - incoming_messages
    if (
        # Defensive invariants, not gates: the planner builds both sequences
        # domain-sorted, so neither can interleave through its hook.
        tuple(recorded_domains)
        != (MESSAGES,) * recorded_messages + (CONTINUATIONS,) * recorded_continuations
        or tuple(domains)
        != (MESSAGES,) * incoming_messages + (CONTINUATIONS,) * incoming_continuations
        # Defensive too: the hook runs only when prefix < len(recorded), which
        # with no recorded continuation means a recorded message mismatched.
        or recorded_continuations == 0
        or prefix != recorded_messages
        or incoming_messages <= recorded_messages
        or incoming_continuations <= recorded_continuations
    ):
        return None
    recorded_owners = frozenset(descriptors[:recorded_messages])
    new_owners = frozenset(descriptors[recorded_messages:incoming_messages])
    for offset in range(incoming_continuations):
        owner = _owner(descriptors[incoming_messages + offset])
        allowed = recorded_owners if offset < recorded_continuations else new_owners
        # Equality, not identity: the Console carries every reply as the same
        # ACTIVE_REQUEST artifact descriptor, so an owner cannot be unique.
        if owner is None or owner not in allowed:
            return None
    # Only the recorded continuations are left to compare; their incoming
    # indices are all >= 1, so ``matches`` never takes its (0, 0)
    # rendered-system branch here and writes only ``retained``.
    before = dict(retained)
    if not all(
        matches(recorded_messages + offset, incoming_messages + offset)
        for offset in range(recorded_continuations)
    ):
        retained.clear()
        retained.update(before)
        return None
    return tuple(range(recorded_messages, incoming_messages)) + tuple(
        range(incoming_messages + recorded_continuations, len(domains))
    )


__all__ = ["domain_append_indices"]
