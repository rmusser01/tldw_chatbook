"""Capture-only live native custody; never provider execution authority."""

from __future__ import annotations

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
