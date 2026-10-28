"""Capture-only live native custody; never provider execution authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

    from .console_trace_service import (
        CompletedToolTurnWitness,
        SurfaceNodeRecord,
        TraceCallRecord,
        TraceProvenance,
        VerifiedSurfaceReplacement,
        VerifiedSurfaceReplacementRange,
        _SurfaceProjection,
    )

from collections.abc import Mapping
from dataclasses import dataclass, field

from tldw_chatbook.Chat.provider_continuation import ProviderContinuationCheckpoint


@dataclass(frozen=True, slots=True, repr=False)
class LiveNativeContinuationCustody:
    """One acknowledged invocation value, issued and rechecked by its host."""

    invocation: object
    owner_message_id: str
    run_id: str
    message_version: int
    checkpoint: ProviderContinuationCheckpoint
    model_results: Mapping[str, str]
    issuer: object = field(compare=False, repr=False)
    durable_checkpoint: ProviderContinuationCheckpoint = field(repr=False)

    def _host(self):
        from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter

        if (
            type(self.issuer) is not _StreamingModelAdapter
            or self.issuer._live_native_custody is not self
        ):
            raise ValueError("live_native_custody_expired")
        return self.issuer

    def scope(self):
        host = self._host()
        return host._store._generation_owner_scope(host._assistant_message_id)

    def validate(self, cursor, issued):
        host = self._host()
        if issued is not self:
            raise ValueError("live_native_custody_expired")
        host._store.validate_live_native_trace_owner(
            cursor,
            message_id=host._assistant_message_id,
            generation_token=host._generation_token,
            checkpoint=self.durable_checkpoint,
            message_version=self.message_version,
        )


@dataclass(frozen=True, slots=True)
class LiveNativeContinuationWitness:
    """Bound one existing continuation slot and one new message suffix."""

    custody: LiveNativeContinuationCustody = field(repr=False)
    previous_call_id: str
    continuation_ordinal: int
    message_start: int
    message_count: int

    @property
    def descriptor_count(self) -> int:
        return self.message_count + 1


CONTINUATION_ROUND_NORMALIZATION = "native-continuation-round-v1"
CONTINUATION_DELTA_COMPONENT = "continuation_delta"


def _projection_bytes(value):
    import json

    from tldw_chatbook.Chat.console_prepared_request import thaw_json

    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _bounded_projection(value):
    """Validate trace projection structure without creating execution state."""
    from tldw_chatbook.Chat.provider_continuation import (
        _MAX_ARGUMENT_BYTES,
        _MAX_CALLS,
        _MAX_JSON_DEPTH,
        _MAX_JSON_NODES,
        _MAX_PAYLOAD_BYTES,
        _MAX_REASONING_BYTES,
        _MAX_RESULT_CHARS,
        _MAX_ROUNDS,
    )
    from tldw_chatbook.Utils.input_validation import strict_json_loads

    raw = _projection_bytes(value)
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise ValueError("continuation_trace_size")
    value = strict_json_loads(
        raw.decode("utf-8"), max_depth=_MAX_JSON_DEPTH, max_nodes=_MAX_JSON_NODES
    )
    if (
        type(value) is not dict
        or type(value.get("rounds")) is not list
        or not 1 <= len(value["rounds"]) <= _MAX_ROUNDS
    ):
        raise ValueError("continuation_trace_shape")
    version = value.get("schema_version")
    keys = {
        "schema_version",
        "checkpoint_revision",
        "provider",
        "protocol",
        "model",
        "api_base_url",
        "state",
        "rounds",
    }
    if version in {2, 3}:
        keys.add("managed_resume")
    if version == 3:
        keys.add("selected_provider")
    if (
        type(version) is not int
        or version not in {1, 2, 3}
        or set(value) != keys
        or type(value["checkpoint_revision"]) is not int
        or value["checkpoint_revision"] <= 0
        or value["state"] not in {"active", "complete"}
        or any(
            type(value[key]) is not str
            for key in ("provider", "protocol", "model", "api_base_url")
        )
        or (version == 3 and type(value["selected_provider"]) is not str)
    ):
        raise ValueError("continuation_trace_shape")
    calls = 0
    reasoning_bytes = 0
    for round_ in value["rounds"]:
        if (
            type(round_) is not dict
            or set(round_) != {"assistant_content", "reasoning_blocks", "calls"}
            or type(round_["assistant_content"]) is not str
            or type(round_["reasoning_blocks"]) is not list
            or type(round_["calls"]) is not list
        ):
            raise ValueError("continuation_trace_shape")
        for block in round_["reasoning_blocks"]:
            if type(block) is not str:
                raise ValueError("continuation_trace_shape")
            reasoning_bytes += len(block.encode("utf-8"))
        for call in round_["calls"]:
            calls += 1
            if (
                type(call) is not dict
                or set(call) != {"call_id", "name", "arguments", "state", "result"}
                or any(type(call[key]) is not str for key in call)
                or call["state"] not in {"completed", "failed"}
                or len(call["arguments"].encode("utf-8")) > _MAX_ARGUMENT_BYTES
                or len(call["result"]) > _MAX_RESULT_CHARS
            ):
                raise ValueError("continuation_trace_shape")
    if calls > _MAX_CALLS or reasoning_bytes > _MAX_REASONING_BYTES:
        raise ValueError("continuation_trace_size")
    return value


def compact_continuation_projection(previous, current, policy_id):
    """Encode one new sanitized round only after checking the whole old prefix."""
    previous = _bounded_projection(previous)
    current = _bounded_projection(current)
    if (
        len(current["rounds"]) != len(previous["rounds"]) + 1
        or current["rounds"][:-1] != previous["rounds"]
    ):
        raise ValueError("continuation_trace_projection_changed")
    return {
        "checkpoint": {key: value for key, value in current.items() if key != "rounds"},
        "prior_round_count": len(previous["rounds"]),
        "round": current["rounds"][-1],
        "policy_id": policy_id,
    }


def resolve_continuation_node(repository, cursor, node, *, policy_id=None):
    """Materialize a trace-only round chain through exact replacement edges."""
    from tldw_chatbook.Chat.provider_continuation import (
        _MAX_JSON_DEPTH,
        _MAX_JSON_NODES,
        _MAX_PAYLOAD_BYTES,
        _MAX_ROUNDS,
    )
    from tldw_chatbook.Utils.input_validation import strict_json_loads

    pending = []
    seen = set()
    while True:
        if node is None or node.node_id in seen or len(seen) >= _MAX_ROUNDS:
            raise ValueError("continuation_trace_chain")
        seen.add(node.node_id)
        if node.reference_kind != "artifact" or node.component_kind not in {
            "continuation",
            CONTINUATION_DELTA_COMPONENT,
        }:
            raise ValueError("continuation_trace_domain")
        artifact = repository.get_artifact(cursor, node.artifact_id)
        if (
            artifact is None
            or artifact.media_type != "application/json"
            or len(artifact.sanitized_bytes) > _MAX_PAYLOAD_BYTES
        ):
            raise ValueError("continuation_trace_artifact")
        value = strict_json_loads(
            artifact.sanitized_bytes.decode("utf-8"),
            max_depth=_MAX_JSON_DEPTH,
            max_nodes=_MAX_JSON_NODES,
        )
        if node.component_kind != CONTINUATION_DELTA_COMPONENT:
            if artifact.normalization_version == CONTINUATION_ROUND_NORMALIZATION:
                raise ValueError("continuation_trace_envelope")
            value = _bounded_projection(value)
            break
        if (
            artifact.normalization_version != CONTINUATION_ROUND_NORMALIZATION
            or type(value) is not dict
            or set(value) != {"checkpoint", "prior_round_count", "round", "policy_id"}
            or type(value["checkpoint"]) is not dict
            or "rounds" in value["checkpoint"]
            or type(value["prior_round_count"]) is not int
            or not 1 <= value["prior_round_count"] < _MAX_ROUNDS
            or repository.get_policy(cursor, value["policy_id"]) is None
        ):
            raise ValueError("continuation_trace_envelope")
        if policy_id is None:
            policy_id = value["policy_id"]
        if value["policy_id"] != policy_id:
            raise ValueError("continuation_trace_policy")
        edges = cursor.execute(
            "SELECT segment_id, predecessor_head_id, start_node_id, start_sequence, end_node_id, end_sequence "
            "FROM console_trace_surface_replacements WHERE replacement_node_id = ?",
            (node.node_id,),
        ).fetchall()
        if len(edges) != 1:
            raise ValueError("continuation_trace_edge")
        segment, head, start, start_sequence, end, end_sequence = edges[0]
        if (
            segment != node.segment_id
            or head != node.predecessor_node_id
            or start != end
            or start_sequence != end_sequence
            or start_sequence >= node.sequence
        ):
            raise ValueError("continuation_trace_edge")
        predecessors = repository.read_lineage_surface_nodes(
            cursor,
            segment_id=node.segment_id,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
        )
        if len(predecessors) != 1 or predecessors[0].node_id != start:
            raise ValueError("continuation_trace_lineage")
        pending.append(value)
        node = predecessors[0]
    for delta in reversed(pending):
        if len(value["rounds"]) != delta["prior_round_count"]:
            raise ValueError("continuation_trace_count")
        value = _bounded_projection(
            {**delta["checkpoint"], "rounds": value["rounds"] + [delta["round"]]}
        )
    return value


def prepare_live_native_advance(
    self,
    cursor,
    *,
    owner_id,
    segment_id,
    route_identity,
    preparation_identity,
    provenance,
    values,
    custody,
    ordinal,
    previous_call_id,
    parent,
    active,
    matches,
    retained_artifact_values,
    rendered_system,
    current_policy_id,
    reserved_call,
    known_credentials,
):
    from tldw_chatbook.Chat import console_trace_service as trace

    previous_custody = parent.live_native_custody
    prior_values = parent.live_native_surface
    if (
        route_identity != "tool_loop"
        or previous_call_id is None
        or previous_custody.invocation is not custody.invocation
        or previous_custody.owner_message_id != custody.owner_message_id
        or previous_custody.run_id != custody.run_id
        or parent.live_native_ordinal != ordinal
        or prior_values is None
        or any(
            policy.policy_id != current_policy_id for policy in parent.surface_policies
        )
    ):
        raise ValueError("live_native_custody")
    old = previous_custody.checkpoint
    new = custody.checkpoint
    if (
        new.state != "active"
        or new.checkpoint_revision <= old.checkpoint_revision
        or len(new.rounds) != len(old.rounds) + 1
        or new.rounds[:-1] != old.rounds
        or trace.replace(
            new, rounds=old.rounds, checkpoint_revision=old.checkpoint_revision
        )
        != old
        or any(
            call.state not in {"completed", "failed"} or call.result is None
            for round_ in new.rounds
            for call in round_.calls
        )
    ):
        raise ValueError("live_native_checkpoint_prefix")
    old_messages = prior_values["messages_payload"]
    old_continuations = prior_values["provider_continuations"]
    message_count = len(provenance.messages_payload)
    start = len(old_messages)
    suffix = values[start:message_count]
    if (
        not 1 <= len(suffix) <= trace.MAX_SURFACE_REPLACEMENT_SPAN
        or len(provenance.continuations) != len(old_continuations)
        or type(ordinal) is not int
        or not 0 <= ordinal < len(old_continuations)
    ):
        raise ValueError("live_native_surface_shape")
    parent_messages = tuple(
        item for _, item in parent.descriptors.iter_domain("messages_payload")
    )
    parent_continuations = tuple(
        item for _, item in parent.descriptors.iter_domain("provider_continuations")
    )
    for index, prior in enumerate(old_messages):
        if not matches(index, index) or (
            not (
                index == 0
                and trace._is_rendered_system_row(
                    provenance.messages_payload[0], values[0]
                )
            )
            and (
                trace._artifact_bytes(prior) != trace._artifact_bytes(values[index])
                or parent_messages[index] != provenance.messages_payload[index]
            )
        ):
            raise ValueError("live_native_message_prefix")
    for index, prior in enumerate(old_continuations):
        if index == ordinal:
            continue
        if (
            not matches(start + index, message_count + index)
            or trace._checkpoint_semantic_value(prior) != values[message_count + index]
            or parent_continuations[index] != provenance.continuations[index]
        ):
            raise ValueError("live_native_other_owner")
    latest = new.rounds[-1]
    native = [row for row in suffix if row.get("role") in {"assistant", "tool"}]
    expected_calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in latest.calls
    ]
    if (
        len(native) != 1 + len(latest.calls)
        or not latest.calls
        or native[0].get("role") != "assistant"
        or native[0].get("content") not in (None, latest.assistant_content)
        or trace._artifact_bytes(native[0].get("tool_calls"))
        != trace._artifact_bytes(expected_calls)
        or any(
            row.get("role") != "tool"
            or row.get("tool_call_id") != call.call_id
            or row.get("content") != custody.model_results.get(call.call_id)
            for row, call in zip(native[1:], latest.calls, strict=True)
        )
        or any(row.get("role") not in {"assistant", "tool", "user"} for row in suffix)
        or any(
            start + index not in provenance.tool_loop for index in range(len(suffix))
        )
    ):
        raise ValueError("live_native_batch")
    physical, sequence, _key = active[start + ordinal]
    nodes = self.repository.read_lineage_surface_nodes(
        cursor,
        segment_id=segment_id,
        start_sequence=sequence,
        end_sequence=sequence,
    )
    if len(nodes) != 1:
        raise ValueError("live_native_node")
    node = nodes[0]
    plan = trace.VerifiedSurfaceReplacementRange(
        predecessor_head_id=parent.head_id,
        start_node_id=node.node_id,
        end_node_id=node.node_id,
        start_sequence=sequence,
        end_sequence=sequence,
        current_ordinal=physical,
        component_name="provider_continuations",
        component_ordinal=ordinal,
    )
    message_delta = tuple(provenance.messages_payload[start:])
    continuation_delta = (provenance.continuations[ordinal],)
    witness = trace.LiveNativeContinuationWitness(
        custody, previous_call_id, ordinal, start, len(suffix)
    )
    admission = trace.SurfaceDeltaAdmission(
        owner_id,
        segment_id,
        parent.head_id,
        route_identity,
        preparation_identity,
        message_delta + continuation_delta,
        parent.capability,
        plan,
        live_native=witness,
        live_native_custody=custody,
        live_native_ordinal=ordinal,
    )
    delta_provenance = trace.replace(
        provenance,
        messages=message_delta
        if provenance.messages == provenance.messages_payload
        else provenance.messages,
        messages_payload=message_delta,
        continuations=continuation_delta,
        tool_loop=tuple(
            index - start for index in provenance.tool_loop if index >= start
        ),
    )
    boundary = self.prepare_surface_provenance(
        cursor,
        parent.capability,
        provenance=delta_provenance,
        admission=admission,
        values=tuple(suffix) + (values[message_count + ordinal],),
        reserved_call=reserved_call,
        known_credentials=known_credentials,
        retained_artifact_values=retained_artifact_values,
        rendered_system=rendered_system(),
    )
    return admission, boundary


def validate_completed_tool_turn(
    self,
    cursor: sqlite3.Cursor,
    *,
    owner_id: str,
    segment_id: str,
    witness: CompletedToolTurnWitness,
    plan: VerifiedSurfaceReplacementRange | VerifiedSurfaceReplacement,
    descriptors: tuple[TraceProvenance, ...],
    values: tuple[object, ...],
    current_turn_id: str | None,
    current_policy_id: str | None,
    reserved_call: TraceCallRecord | None = None,
    known_credentials: tuple[str, ...] = (),
) -> None:
    """Recheck bounded run evidence; a response link alone grants nothing."""
    from tldw_chatbook.Chat import console_trace_service as trace

    restoring_source = witness.source_revision_id is not None
    source_after_failure = restoring_source and witness.assistant_revision_id is None
    discarded = witness.discarded_assistant_message_id is not None
    closed = witness.closed_assistant_message_id is not None
    unanswered = discarded or closed
    terminal_states = {trace.TraceCallState.COMPLETE}
    if restoring_source or unanswered:
        terminal_states.update(
            {
                trace.TraceCallState.ERROR,
                trace.TraceCallState.STOPPED,
                trace.TraceCallState.INTERRUPTED,
            }
        )
    if source_after_failure and not unanswered:
        terminal_states.remove(trace.TraceCallState.COMPLETE)
    # Read durable Discard again at final binding. A closed witness may
    # span completed uncaptured followups and still have a discarded owner.
    # The full ownership/closure proof below remains mandatory.
    explicit_discard = discarded or (
        closed
        and cursor.execute(
            "SELECT 1 FROM messages WHERE id = ? "
            "AND assistant_generation_state = 'discarded'",
            (witness.closed_assistant_message_id,),
        ).fetchone()
        is not None
    )
    if explicit_discard:
        # Explicit discard settles the assistant owner separately. A run
        # interrupted during trace construction can leave earlier calls
        # response-bearing but unsettled; do not invent their outcomes.
        terminal_states.add(trace.TraceCallState.RESPONSE_STARTED)
    project_transition = witness.project_context_count is not None
    self._validate_owner(cursor, owner_id=owner_id, segment_id=segment_id)
    owner = self.repository.get_owner(cursor, owner_id)
    origin = self.repository.get_call(cursor, witness.origin_call_id)
    terminal = self.repository.get_call(cursor, witness.terminal_call_id)
    tail = self._effective_surface_tail(cursor, segment_id)
    latest = self.repository.get_latest_call_boundary(cursor, segment_id)
    if reserved_call is not None:
        durable = self.repository.get_call(cursor, reserved_call.call_id)
        if (
            durable != reserved_call
            or latest is None
            or latest.call_id != reserved_call.call_id
            or (reserved_call.owner_id, reserved_call.segment_id)
            != (owner_id, segment_id)
            or reserved_call.state is not trace.TraceCallState.RESERVED
            or any(
                value is not None
                for value in (
                    reserved_call.surface_node_id,
                    reserved_call.request_header_id,
                    reserved_call.dispatch_started_at,
                    reserved_call.response_started_at,
                    reserved_call.settled_at,
                )
            )
        ):
            raise ValueError("completed_tool_turn_reservation")
        previous = cursor.execute(
            """SELECT call_id FROM console_trace_events
                   WHERE segment_id = ? AND event_type = 'call_boundary'
                     AND sequence < ? ORDER BY sequence DESC LIMIT 1""",
            (segment_id, latest.sequence),
        ).fetchone()
        latest_id = None if previous is None else previous[0]
    else:
        latest_id = None if latest is None else latest.call_id
    chain_terminal = terminal
    fallback = (
        project_transition
        and terminal is not None
        and terminal.route_identity == "llama_fallback"
    )
    if fallback:
        # The non-streaming retry owns a distinct run ID. Its exact
        # immediately preceding failed stream owns the unchanged surface;
        # neither an arbitrary older call nor the response link proves it.
        fallback_events = cursor.execute(
            """SELECT sequence FROM console_trace_events
                   WHERE segment_id = ? AND event_type = 'call_boundary'
                     AND call_id = ? ORDER BY sequence LIMIT 2""",
            (segment_id, terminal.call_id),
        ).fetchall()
        previous_event = (
            cursor.execute(
                """SELECT call_id FROM console_trace_events
                       WHERE segment_id = ? AND event_type = 'call_boundary'
                         AND sequence < ? ORDER BY sequence DESC LIMIT 1""",
                (segment_id, fallback_events[0][0]),
            ).fetchone()
            if len(fallback_events) == 1
            else None
        )
        chain_terminal = (
            self.repository.get_call(cursor, previous_event[0])
            if previous_event is not None
            else None
        )
        if (
            origin != terminal
            or terminal.call_sequence != 0
            or chain_terminal is None
            or chain_terminal.state is not trace.TraceCallState.ERROR
            or chain_terminal.route_identity
            not in {"agent_first", "fresh", "tool_loop"}
            or (
                chain_terminal.owner_id,
                chain_terminal.segment_id,
                chain_terminal.turn_id,
                chain_terminal.policy_id,
                chain_terminal.surface_node_id,
                chain_terminal.provider_name,
                chain_terminal.model_name,
            )
            != (
                terminal.owner_id,
                terminal.segment_id,
                terminal.turn_id,
                terminal.policy_id,
                terminal.surface_node_id,
                terminal.provider_name,
                terminal.model_name,
            )
        ):
            raise ValueError("completed_project_fallback_lineage")
        origin = self.repository.get_run_origin(cursor, chain_terminal.run_id)
    if (
        owner is None
        or origin is None
        or terminal is None
        or chain_terminal is None
        or tail is None
        or latest_id != terminal.call_id
        or terminal.state not in terminal_states
        or closed
        and not explicit_discard
        and terminal.response_started_at is None
        or (
            (restoring_source or unanswered)
            and terminal.state is not trace.TraceCallState.RESPONSE_STARTED
            and terminal.settled_at is None
        )
        or (
            chain_terminal.route_identity != "tool_loop"
            and not (
                (project_transition or restoring_source)
                and chain_terminal == origin
                and chain_terminal.route_identity in {"agent_first", "fresh"}
            )
        )
        or origin.route_identity
        not in (
            {"fresh", "agent_first"}
            if restoring_source or project_transition
            else {"agent_first"}
        )
        or origin.call_sequence != 0
        or self.repository.get_run_origin(cursor, chain_terminal.run_id) != origin
        or (
            origin.owner_id,
            origin.segment_id,
            origin.turn_id,
            origin.run_id,
            origin.policy_id,
        )
        != (
            chain_terminal.owner_id,
            chain_terminal.segment_id,
            chain_terminal.turn_id,
            chain_terminal.run_id,
            chain_terminal.policy_id,
        )
        or (terminal.owner_id, terminal.segment_id) != (owner_id, segment_id)
        or terminal.turn_id == current_turn_id
        or terminal.surface_node_id != tail.node_id
        or plan.predecessor_head_id != tail.node_id
    ):
        raise ValueError("completed_tool_turn_unavailable")
    if unanswered:
        next_turn_id = current_turn_id
        seen_turns = {current_turn_id, origin.turn_id}
        followups = (
            witness.closed_followups
            if closed
            else tuple(
                (revision_id, assistant_id, None)
                for revision_id, assistant_id in witness.discarded_followups
            )
        )
        for revision_id, assistant_id, response_id in reversed(followups):
            revision = self.repository.get_semantic_revision(cursor, revision_id)
            response = (
                self.repository.get_semantic_revision(cursor, response_id)
                if response_id is not None
                else None
            )
            if (
                revision is None
                or revision.normalized_role != "user"
                or revision.source_conversation_id != owner.conversation_id
                or revision.source_message_id in seen_turns
                or closed
                and revision.live_message_id != revision.source_message_id
                or response_id is not None
                and (
                    response is None
                    or response.normalized_role != "assistant"
                    or response.source_conversation_id != owner.conversation_id
                    or response.source_message_id != assistant_id
                    or response.live_message_id != assistant_id
                )
                or self._closed_turn_assistant(
                    cursor,
                    conversation_id=owner.conversation_id,
                    previous_turn_id=revision.source_message_id,
                    current_turn_id=next_turn_id,
                    allow_failed=closed,
                    complete_assistant_id=assistant_id
                    if response_id is not None
                    else None,
                    untraced=True,
                )
                != assistant_id
            ):
                raise ValueError("completed_turn_discard_owner")
            seen_turns.add(revision.source_message_id)
            next_turn_id = revision.source_message_id
        if self._closed_turn_assistant(
            cursor,
            conversation_id=owner.conversation_id,
            previous_turn_id=origin.turn_id,
            current_turn_id=next_turn_id,
            allow_failed=closed,
        ) != (
            witness.closed_assistant_message_id
            if closed
            else witness.discarded_assistant_message_id
        ):
            raise ValueError("completed_turn_discard_owner")
    boundary_events = cursor.execute(
        """SELECT call_id, sequence FROM console_trace_events
                WHERE segment_id = ? AND event_type = 'call_boundary'
                  AND call_id IN (?, ?) ORDER BY sequence LIMIT 3""",
        (segment_id, origin.call_id, chain_terminal.call_id),
    ).fetchall()
    if tuple(row[0] for row in boundary_events) != tuple(
        dict.fromkeys((origin.call_id, chain_terminal.call_id))
    ):
        raise ValueError("completed_tool_turn_lineage")
    origin_event_sequence = boundary_events[0][1]
    terminal_event_sequence = boundary_events[-1][1]
    previous_policy = self.repository.get_policy(cursor, terminal.policy_id)
    current_policy = (
        None
        if current_policy_id is None
        else self.repository.get_policy(cursor, current_policy_id)
    )
    if (
        previous_policy is None
        or current_policy is None
        or (
            previous_policy.credential_filter_version,
            previous_policy.pii_redaction_enabled,
            previous_policy.pii_ruleset_revision_id,
        )
        != (
            current_policy.credential_filter_version,
            current_policy.pii_redaction_enabled,
            current_policy.pii_ruleset_revision_id,
        )
    ):
        raise ValueError("completed_tool_turn_policy")
    origin_surface_head = (
        None
        if origin.surface_node_id is None
        else self.repository.get_surface_node(cursor, origin.surface_node_id)
    )
    origin_head = (
        self._call_message_tail(cursor, origin)
        if restoring_source
        else origin_surface_head
    )
    if (
        origin_head is None
        or origin_surface_head is None
        or (
            plan.start_sequence != origin_head.sequence
            if restoring_source
            else plan.start_sequence > origin_head.sequence + 1
            if project_transition
            else plan.start_sequence != origin_head.sequence + 1
        )
        or plan.end_sequence > tail.sequence
        or not 1
        <= plan.end_sequence - plan.start_sequence + 1
        <= trace.MAX_SURFACE_REPLACEMENT_SPAN
    ):
        raise ValueError("completed_tool_turn_range")
    if restoring_source:
        source = self.repository.get_semantic_revision(
            cursor, witness.source_revision_id
        )
        pin = cursor.execute(
            "SELECT semantic_revision_id FROM console_trace_events "
            "WHERE call_id = ? AND event_type = 'call_boundary'",
            (origin.call_id,),
        ).fetchall()
        if (
            source is None
            or len(pin) != 1
            or pin[0][0] != witness.source_revision_id
            or source.source_message_id != origin.turn_id
            or source.source_conversation_id != owner.conversation_id
            or source.normalized_role != "user"
            or origin_head.component_kind != "active_request"
            or origin_head.artifact_id is None
        ):
            raise ValueError("completed_turn_source")
    assistant = (
        None
        if witness.assistant_revision_id is None
        else self.repository.get_semantic_revision(
            cursor, witness.assistant_revision_id
        )
    )
    user = self.repository.get_semantic_revision(cursor, witness.user_revision_id)
    link = self.repository.get_response_link(cursor, terminal.call_id)
    if (
        user is None
        or (
            not (source_after_failure or unanswered)
            and (
                assistant is None
                or link is None
                or link.link_kind != "revision"
                or link.verification_outcome != "verified_equal"
                or link.semantic_revision_id != witness.assistant_revision_id
                or assistant.normalized_role != "assistant"
                or assistant.source_conversation_id != owner.conversation_id
            )
        )
        or user.normalized_role != "user"
        or user.source_conversation_id != owner.conversation_id
        or user.source_message_id != current_turn_id
        or not witness.matches_descriptors(descriptors)
        or len(values) != witness.descriptor_count
    ):
        raise ValueError("completed_tool_turn_revision")
    if witness.replay_row_count:
        checkpoint_value = self._resolve_reference_value(
            cursor,
            ("continuation", "revision", witness.assistant_revision_id),
            owner_id=owner_id,
        )
        if trace._artifact_bytes(values[-1]) != trace._artifact_bytes(checkpoint_value):
            raise ValueError("completed_replay_attachment")
        start = (
            witness.descriptor_count
            - 1
            - (witness.project_context_count or 0)
            - 2
            - witness.replay_row_count
        )
        expected_replay = self._completed_replay_values(
            cursor, owner_id, witness.assistant_revision_id
        )
        if len(expected_replay) != witness.replay_row_count or trace._artifact_bytes(
            expected_replay
        ) != trace._artifact_bytes(values[start : start + witness.replay_row_count]):
            raise ValueError("completed_replay_order")
    for descriptor, value in zip(descriptors, values, strict=True):
        if type(descriptor) is not trace.SavedRevisionTraceProvenance:
            # The admitted current user and the verified assistant's typed
            # thinking replay may carry provider artifacts. Their exact
            # sources, response link and policy are checked above and at
            # the preparation/final-value boundary.
            continue
        expected = self._resolve_reference_value(
            cursor,
            ("message", "revision", descriptor.revision_id),
            owner_id=owner_id,
        )
        if trace._artifact_bytes(expected) != trace._artifact_bytes(value):
            raise ValueError("completed_tool_turn_value")
    if any(
        not isinstance(value, trace.Mapping)
        or value.get("role") != "user"
        or value.get("_chatbook_ephemeral_origin") != "project_instructions"
        or (
            descriptor.policy.credential_filter_version,
            descriptor.policy.pii_redaction_enabled,
            descriptor.policy.pii_ruleset_revision_id,
        )
        != (
            current_policy.credential_filter_version,
            current_policy.pii_redaction_enabled,
            current_policy.pii_ruleset_revision_id,
        )
        for descriptor, value in zip(
            descriptors[
                witness.descriptor_count
                - int(witness.replay_row_count > 0)
                - (witness.project_context_count or 0) : witness.descriptor_count
                - int(witness.replay_row_count > 0)
            ],
            values[
                witness.descriptor_count
                - int(witness.replay_row_count > 0)
                - (witness.project_context_count or 0) : witness.descriptor_count
                - int(witness.replay_row_count > 0)
            ],
            strict=True,
        )
    ):
        raise ValueError("completed_project_context_value")
    projection = self._surface_projection(cursor, segment_id, tail)
    suffix = tuple(
        (sequence, key)
        for sequence, key in projection.entries
        if plan.start_sequence <= sequence <= plan.end_sequence
    )
    if any(
        sequence > plan.end_sequence
        and trace._surface_reference_domain(key) != "provider_continuations"
        for sequence, key in projection.entries
    ):
        raise ValueError("completed_tool_turn_range")
    if project_transition and not restoring_source:
        prefix = tuple(
            key
            for sequence, key in projection.entries
            if sequence < plan.start_sequence
            and trace._surface_reference_domain(key) == "messages_payload"
        )
        prior_user = (
            self.repository.get_semantic_revision(cursor, prefix[-1][2])
            if prefix and prefix[-1][1] == "revision"
            else None
        )
        if (
            prior_user is None
            or prior_user.normalized_role != "user"
            or prior_user.source_conversation_id != owner.conversation_id
            or prior_user.source_message_id != terminal.turn_id
        ):
            raise ValueError("completed_project_turn_anchor")
    replay_continuations = set()
    if witness.replay_row_count:
        from tldw_chatbook.Chat.provider_continuation import (
            parse_provider_continuation_json,
        )

        complete = self._completed_replay_trace_checkpoint(
            checkpoint_value, previous_policy, known_credentials=known_credentials
        )
        nodes = self.repository.read_lineage_surface_nodes(
            cursor,
            segment_id=segment_id,
            start_sequence=plan.start_sequence,
            end_sequence=plan.end_sequence,
        )
        for node in nodes:
            key = self._node_reference_key(node)
            if key[0] not in {"continuation", trace.CONTINUATION_DELTA_COMPONENT}:
                continue
            prior = parse_provider_continuation_json(
                self._resolve_reference_values(cursor, (key,), owner_id=owner_id)[key]
            )
            if (
                any(
                    getattr(prior, field) != getattr(complete, field)
                    for field in (
                        "provider",
                        "protocol",
                        "model",
                        "api_base_url",
                        "selected_provider",
                    )
                )
                or prior.rounds != complete.rounds[: len(prior.rounds)]
                or prior.checkpoint_revision > complete.checkpoint_revision
                or any(
                    call.state not in {"completed", "failed"}
                    for round_ in prior.rounds
                    for call in round_.calls
                )
            ):
                raise ValueError("completed_replay_prior_attachment")
            replay_continuations.add(node.sequence)
        active_continuations = [
            key for sequence, key in suffix if sequence in replay_continuations
        ]
        if len(active_continuations) != 1:
            raise ValueError("completed_replay_prior_attachment")
        prior = parse_provider_continuation_json(
            self._resolve_reference_values(
                cursor, tuple(active_continuations), owner_id=owner_id
            )[active_continuations[0]]
        )
        if prior.rounds != complete.rounds:
            raise ValueError("completed_replay_prior_attachment")
    if tuple(
        sorted({sequence for sequence, _ in suffix} | replay_continuations)
    ) != tuple(range(plan.start_sequence, plan.end_sequence + 1)) or any(
        sequence not in replay_continuations
        and (
            key[0]
            not in (
                {"active_request"}
                if restoring_source and sequence == origin_head.sequence
                else {"project_instruction", "tool_call", "tool_result"}
                if project_transition
                else {"tool_call", "tool_result"}
            )
            or key[1] != "artifact"
            or (
                sequence <= origin_surface_head.sequence
                and key[0] != "project_instruction"
                and not (restoring_source and sequence == origin_head.sequence)
            )
        )
        for sequence, key in suffix
    ):
        raise ValueError("completed_tool_turn_range")
    # Every removed node must have been appended under a durable call from
    # this run; generic artifact rows or later unrelated reservations do not
    # establish that ownership. The output is bounded by the removed span.
    rows = cursor.execute(
        """SELECT n.sequence, c.owner_id, c.segment_id, c.turn_id, c.run_id,
                      c.policy_id, c.route_identity, c.call_sequence
                 FROM console_trace_surface_nodes n
                 LEFT JOIN console_trace_surface_replacements r ON r.replacement_node_id = n.node_id
                      AND r.segment_id = n.segment_id
                 JOIN console_trace_events e ON e.segment_id = n.segment_id AND (
                      (e.surface_node_id = n.node_id AND e.event_type = 'surface_append')
                      OR (? AND e.surface_replacement_id = r.replacement_id AND e.event_type = 'surface_replace'))
                 JOIN console_trace_calls c ON c.call_id = (
                     SELECT b.call_id FROM console_trace_events b
                      WHERE b.segment_id = e.segment_id AND b.event_type = 'call_boundary'
                        AND b.sequence < e.sequence ORDER BY b.sequence DESC LIMIT 1)
                WHERE n.segment_id = ? AND n.sequence BETWEEN ? AND ?
                ORDER BY n.sequence LIMIT 257""",
        (
            bool(witness.replay_row_count),
            segment_id,
            plan.start_sequence,
            plan.end_sequence,
        ),
    ).fetchall()
    if tuple(row[0] for row in rows) != tuple(
        sorted({sequence for sequence, _ in suffix} | replay_continuations)
    ) or any(
        tuple(row[1:6])
        != (
            owner_id,
            segment_id,
            terminal.turn_id,
            chain_terminal.run_id,
            terminal.policy_id,
        )
        or (
            (row[6], row[7]) != (origin.route_identity, 0)
            if (
                restoring_source
                and row[0] == origin_head.sequence
                or project_transition
                and row[0] <= origin_surface_head.sequence
            )
            else row[6] != "tool_loop"
            or not 1 <= row[7] <= chain_terminal.call_sequence
        )
        for row in rows
    ):
        raise ValueError("completed_tool_turn_lineage")
    foreign = cursor.execute(
        """SELECT 1 FROM console_trace_events e
                 JOIN console_trace_calls c ON c.call_id = e.call_id
                WHERE e.segment_id = ? AND e.event_type = 'call_boundary'
                  AND e.sequence > ? AND e.sequence <= ?
                  AND (c.owner_id != ? OR c.run_id != ? OR c.turn_id != ? OR c.policy_id != ?)
                LIMIT 1""",
        (
            segment_id,
            origin_event_sequence,
            terminal_event_sequence,
            owner_id,
            chain_terminal.run_id,
            terminal.turn_id,
            terminal.policy_id,
        ),
    ).fetchone()
    if foreign is not None:
        raise ValueError("completed_tool_turn_lineage")


def completed_tool_turn_parent(
    self,
    cursor: sqlite3.Cursor,
    *,
    owner_id: str,
    segment_id: str,
    tail: SurfaceNodeRecord,
    projection: _SurfaceProjection,
    route: str,
    policy_id: str,
    retained_descriptors: Mapping[int, TraceProvenance],
) -> object:
    """Build a cold compound parent entirely from committed references."""
    from tldw_chatbook.Chat import console_trace_service as trace

    policy = self.repository.get_policy(cursor, policy_id)
    if policy is None:
        raise ValueError("trace_policy_mismatch")
    entries = []
    domains = []
    tool_loop = []
    message_ordinal = 0
    for sequence, key in projection.entries:
        domain = trace._surface_reference_domain(key)
        if sequence in retained_descriptors:
            descriptor = retained_descriptors[sequence]
        elif key[1] == "revision":
            descriptor = trace.SavedRevisionTraceProvenance(key[2])
            if domain == "provider_continuations":
                descriptor = trace.DerivedTraceProvenance(
                    trace.TraceTransformKind.CONTINUATION_ATTACHMENT,
                    inputs=(descriptor,),
                )
        elif key[1] == "artifact":
            descriptor = trace.ProviderArtifactTraceProvenance(
                trace.TraceProvenanceSource.CONTINUATION
                if key[0] == trace.CONTINUATION_DELTA_COMPONENT
                else trace.TraceProvenanceSource(key[0]),
                policy,
            )
        else:
            raise ValueError("completed_tool_turn_parent_unavailable")
        entries.append((sequence, descriptor))
        domains.append((sequence, domain))
        if domain == "messages_payload":
            if key[0] in {"tool_call", "tool_result"}:
                tool_loop.append(message_ordinal)
            message_ordinal += 1
    # Resolve every retained reference, including the removed tool suffix;
    # the incoming prefix cannot substitute for unavailable durable bytes.
    self._resolve_reference_values(
        cursor, tuple(key for _, key in projection.entries), owner_id=owner_id
    )
    structure, policies = trace._surface_projection_metadata(
        item for _, item in entries
    )
    capability = self._promote_surface_capability(
        owner_id=owner_id,
        segment_id=segment_id,
        head_id=tail.node_id,
        route=route,
        root=projection.root,
        descriptors=trace._DescriptorRoot(
            None, base=tuple(entries), base_domains=tuple(domains)
        ),
        next_sequence=tail.sequence + 1,
        tool_loop=tuple(tool_loop),
        active_domains=dict(domains),
        lineage_domains=dict(projection.lineage_domains),
        surface_structure=structure,
        surface_policies=policies,
    )
    assert capability is not None
    return capability
