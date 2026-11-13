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
  same index, and every recorded continuation by the incoming continuation
  at the same index (the caller's ``matches`` performs the same durable,
  credential-filtered comparison the flat prefix uses);
- at least one message and at least one continuation are new;
- every continuation in the recorded positions attaches to a recorded
  message, and every new continuation attaches to a new message, so no
  recorded message gains, loses or swaps a reasoning checkpoint.

Anything else (an edited, removed or reordered recorded entry, a new
checkpoint on an earlier reply) returns ``None`` and the caller's existing
rules decide, which refuse it as before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
    matches: Callable[[int, int], bool],
) -> tuple[int, ...] | None:
    """Return the incoming indices to append per domain, or None.

    Args:
        recorded_domains: Domain of each recorded surface entry, in provider
            domain order (all messages, then all continuations).
        domains: Domain of each incoming descriptor, in the same order.
        descriptors: Incoming descriptors aligned with ``domains``.
        matches: ``matches(recorded_index, incoming_index)`` from the planner;
            True only when the recorded entry durably equals that incoming
            entry.

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
        tuple(recorded_domains)
        != (MESSAGES,) * recorded_messages + (CONTINUATIONS,) * recorded_continuations
        or tuple(domains)
        != (MESSAGES,) * incoming_messages + (CONTINUATIONS,) * incoming_continuations
        or recorded_continuations == 0
        or incoming_messages <= recorded_messages
        or incoming_continuations <= recorded_continuations
    ):
        return None
    recorded_owners = tuple(descriptors[:recorded_messages])
    new_owners = tuple(descriptors[recorded_messages:incoming_messages])
    for offset in range(incoming_continuations):
        owner = _owner(descriptors[incoming_messages + offset])
        allowed = recorded_owners if offset < recorded_continuations else new_owners
        if owner is None or owner not in allowed:
            return None
    if not all(matches(index, index) for index in range(recorded_messages)):
        return None
    if not all(
        matches(recorded_messages + offset, incoming_messages + offset)
        for offset in range(recorded_continuations)
    ):
        return None
    return tuple(range(recorded_messages, incoming_messages)) + tuple(
        range(incoming_messages + recorded_continuations, len(domains))
    )


__all__ = ["domain_append_indices"]
