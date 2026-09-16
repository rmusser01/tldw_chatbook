"""Explicit v2 hook declaration and result contracts."""

import copy
import json

import pytest


def _command(event="PreToolUse", **overrides):
    value = {
        "id": "review",
        "event": event,
        "type": "command",
        "argv": ["review"],
        "effects": [],
    }
    value.update(overrides)
    return value


def test_mixed_transformer_runs_in_transformation_phase():
    from tldw_chatbook.Agents.hooks_v2.validation import handler_phase, parse_handlers

    handlers = parse_handlers([_command(effects=["updated_input", "deny"])])
    assert handler_phase(handlers[0]) == "transform"
    guards = parse_handlers([_command(effects=["deny"])])
    assert handler_phase(guards[0]) == "validate"
    assert (
        handler_phase(parse_handlers([_command()])[0], dependency_required=True)
        == "validate"
    )


def test_valid_declaration_control():
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    assert parse_handlers([_command()])[0].id == "review"
    assert parse_handlers([_command(argv=["review", ""])])[0].argv == ("review", "")


@pytest.mark.parametrize(
    ("event", "effects"),
    [
        ("SessionStart", ["context", "deny"]),
        ("UserPromptSubmit", ["deny", "context"]),
        ("PreToolUse", ["updated_input", "deny", "context"]),
        ("ApprovalRequested", []),
        ("PostToolUse", ["context"]),
        ("PostToolUseFailure", ["context"]),
        ("SubagentStart", ["deny", "child_limits", "context"]),
        ("SubagentStop", ["context"]),
        ("PreCompact", ["context"]),
        ("PostCompact", ["context"]),
        ("Stop", ["continuation", "stop_continuations"]),
        ("Interrupt", []),
        ("SessionEnd", []),
    ],
)
def test_all_events_accept_only_their_declared_effects(event, effects):
    from tldw_chatbook.Agents.hooks_v2.validation import EFFECTS, parse_handlers

    handler = parse_handlers([_command(event, effects=effects)])[0]
    assert handler.effects == frozenset(effects)
    illegal = next(
        (
            effect
            for effect in (
                "deny",
                "context",
                "updated_input",
                "child_limits",
                "continuation",
                "stop_continuations",
            )
            if effect not in EFFECTS[event]
        ),
        None,
    )
    if illegal:
        with pytest.raises(ValueError):
            parse_handlers([_command(event, effects=[illegal])])


@pytest.mark.parametrize(
    "event", ["ApprovalRequested", "Stop", "Interrupt", "SessionEnd"]
)
@pytest.mark.parametrize("flag", ["required", "require_context"])
def test_observation_and_stop_boundaries_reject_requirements(event, flag):
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    with pytest.raises(ValueError):
        parse_handlers([_command(event, **{flag: True})])


@pytest.mark.parametrize(
    "effects,required,dependency,expected",
    [
        ([], False, False, "observe"),
        ([], True, False, "validate"),
        ([], False, True, "validate"),
        (["context"], False, False, "context"),
        (["context"], True, False, "validate"),
        (["deny"], False, False, "validate"),
        (["updated_input"], False, False, "transform"),
        (["updated_input", "deny", "context"], True, True, "transform"),
    ],
)
def test_declaration_phase_order(effects, required, dependency, expected):
    from tldw_chatbook.Agents.hooks_v2.validation import handler_phase, parse_handlers

    handler = parse_handlers([_command(effects=effects, required=required)])[0]
    assert handler_phase(handler, dependency_required=dependency) == expected


@pytest.mark.parametrize(
    "event", ["ApprovalRequested", "Stop", "Interrupt", "SessionEnd"]
)
def test_forbidden_incoming_dependency_requirement_is_refused(event):
    from tldw_chatbook.Agents.hooks_v2.validation import handler_phase, parse_handlers

    handler = parse_handlers([_command(event)])[0]
    assert handler_phase(handler) == "observe"
    with pytest.raises(ValueError):
        handler_phase(handler, dependency_required=True)
    assert (
        handler_phase(parse_handlers([_command()])[0], dependency_required=True)
        == "validate"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"requires": ["something"]},
        {"owner_installation_id": "forged"},
        {"effects": ["deny", "deny"]},
        {"id": "bad id"},
        {"event": "NoSuchEvent"},
        {"argv": []},
        {"argv": ["ok", "bad\x00arg"]},
        {"timeout_seconds": True},
        {"timeout_seconds": 61},
        {"input": {}},
        {"required": "true"},
    ],
)
def test_invalid_or_type_inappropriate_declaration_rejects_whole_batch(change):
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    second = _command(id="second")
    second.update(change)
    with pytest.raises(ValueError):
        parse_handlers([_command(), second])


def test_duplicate_ids_native_envelope_and_caller_input_is_unchanged():
    from tldw_chatbook.Agents.hooks_v2.validation import (
        parse_handlers,
        parse_native_handlers,
    )

    original = [_command(effects=["deny"])]
    snapshot = copy.deepcopy(original)
    parsed = parse_handlers(original)
    original[0]["effects"].clear()
    assert parsed[0].effects == frozenset({"deny"})
    assert parsed[0].model_dump_json().count('"deny"') == 1
    assert snapshot[0]["effects"] == ["deny"]
    with pytest.raises(ValueError):
        parse_handlers([_command(), _command()])
    assert parse_native_handlers({"version": 2, "hooks": snapshot}) == (
        parse_handlers(snapshot)[0],
    )
    with pytest.raises(ValueError):
        parse_native_handlers({"version": 3, "hooks": snapshot})


def test_native_raw_file_rejects_duplicate_keys_before_object_validation():
    from tldw_chatbook.Agents.hooks_v2.validation import decode_native_handlers

    valid = b'{"version":2,"hooks":[{"id":"a","event":"Stop","type":"command","argv":["review"],"effects":[]}]}'
    assert decode_native_handlers(valid)[0].id == "a"
    with pytest.raises(ValueError):
        decode_native_handlers(b'{"version":2,"version":2,"hooks":[]}')
    with pytest.raises(ValueError):
        decode_native_handlers(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        decode_native_handlers(b" ")
    with pytest.raises(ValueError):
        decode_native_handlers(b"[" * 1000 + b"]" * 1000)
    from tldw_chatbook.Agents.hooks_v2.validation import parse_native_handlers

    with pytest.raises(ValueError):
        parse_native_handlers(
            {"version": 2, "hooks": [_command(id=f"hook{i}") for i in range(65)]}
        )


def test_matcher_matrix_and_declared_variable_env():
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    valid = [
        _command("SessionStart", match={"reason": ["resume"]}),
        _command(
            "PreToolUse", match={"tool_id": ["local:fs_*"], "provider": ["local"]}
        ),
        _command("PostToolUse", match={"operation": ["file.read"]}),
        _command("PostToolUseFailure", match={"reason": ["timeout"]}),
    ]
    for declaration in valid:
        assert parse_handlers([declaration])[0].match
    with pytest.raises(ValueError):
        parse_handlers([_command("ApprovalRequested", match={"tool_id": ["x"]})])
    with pytest.raises(ValueError):
        parse_handlers([_command("PreToolUse", match={"reason": ["x"]})])
    handler = parse_handlers(
        [_command(env={"http_proxy": "literal", "API_TOKEN": {"variable": "TOKEN"}})],
        declared_variable_names={"TOKEN"},
    )[0]
    assert handler.env["http_proxy"] == "literal"
    with pytest.raises(ValueError):
        parse_handlers([_command(env={"API_TOKEN": {"variable": "TOKEN"}})])
    for env in (
        {"PLUGIN_ROOT": "x"},
        {"BAD": {"variable": "NOPE"}},
        {"bad-key": "x"},
        {"X": "bad\x00value"},
    ):
        with pytest.raises(ValueError):
            parse_handlers([_command(env=env)], declared_variable_names={"TOKEN"})


@pytest.mark.parametrize(
    "event",
    [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "ApprovalRequested",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "PreCompact",
        "PostCompact",
        "Stop",
        "Interrupt",
        "SessionEnd",
    ],
)
@pytest.mark.parametrize("key", ["tool_id", "provider", "operation", "reason"])
def test_every_event_matcher_key_uses_closed_matrix(event, key):
    from tldw_chatbook.Agents.hooks_v2.validation import MATCH_KEYS, parse_handlers

    declaration = _command(event, match={key: ["value"]})
    if key in MATCH_KEYS.get(event, frozenset()):
        assert parse_handlers([declaration])[0].match[key] == ("value",)
    else:
        with pytest.raises(ValueError):
            parse_handlers([declaration])


def _result_handler(event="PreToolUse", effects=None, **flags):
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    return parse_handlers([_command(event, effects=effects or [], **flags)])[0]


def test_raw_decoder_rejects_duplicate_keys_nonfinite_trailing_and_oversize():
    from tldw_chatbook.Agents.hooks_v2.validation import decode_result, parse_result

    handler = _result_handler()
    assert decode_result(b"", handler).decision == "pass"
    assert parse_result({"version": 2, "decision": "pass"}, handler).decision == "pass"
    for raw in (
        b'{"version":2,"decision":"pass","decision":"deny"}',
        b'{"version":2,"decision":"pass","extra":{"x":1,"x":2}}',
        b'{"version":2,"decision":"pass"}{}',
        b'{"version":2,"decision":NaN}',
        b"[]",
        b"\xff",
        b" " * (16 * 1024 + 1),
    ):
        with pytest.raises(ValueError):
            decode_result(raw, handler)


def test_context_success_and_lifetime_policy():
    from tldw_chatbook.Agents.hooks_v2.validation import parse_result

    required = _result_handler("PreCompact", ["context"], require_context=True)
    valid = {
        "version": 2,
        "decision": "pass",
        "context": [{"text": "compaction hint", "lifetime": "turn"}],
    }
    assert parse_result(valid, required).context[0].lifetime == "turn"
    for invalid in (
        {"version": 2, "decision": "pass"},
        {
            "version": 2,
            "decision": "pass",
            "context": [{"text": "  ", "lifetime": "turn"}],
        },
        {
            "version": 2,
            "decision": "pass",
            "context": [{"text": "hint", "lifetime": "runtime"}],
        },
        {
            "version": 2,
            "decision": "pass",
            "context": [{"text": "x" * 4097, "lifetime": "turn"}],
        },
    ):
        with pytest.raises(ValueError):
            parse_result(invalid, required)
    session = _result_handler("SessionStart", ["context"])
    assert parse_result(
        {
            "version": 2,
            "decision": "pass",
            "context": [{"text": "session", "lifetime": "runtime"}],
        },
        session,
    ).context
    with pytest.raises(ValueError):
        parse_result(
            {
                "version": 2,
                "decision": "pass",
                "context": [{"text": "session", "lifetime": "turn"}],
            },
            session,
        )
    assert (
        parse_result(
            {"version": 2, "decision": "pass"}, _result_handler(required=True)
        ).decision
        == "pass"
    )


def test_result_effects_and_closed_nested_shapes():
    from tldw_chatbook.Agents.agent_models import MAX_RUN_CONTROL_STEPS
    from tldw_chatbook.Agents.hooks_v2.validation import parse_result

    child = _result_handler("SubagentStart", ["child_limits"])
    baseline = {"version": 2, "decision": "pass"}
    assert (
        parse_result(
            {**baseline, "child_limits": {"tool_ids": []}}, child
        ).child_limits["tool_ids"]
        == ()
    )
    assert parse_result(
        {
            **baseline,
            "child_limits": {
                "tool_ids": ["local:fs_read"],
                "budget_caps": {"max_total_tokens": 0, "max_subagents": 0},
            },
        },
        child,
    ).child_limits
    for limits in (
        {},
        {"tool_ids": ["x", "x"]},
        {"budget_caps": {}},
        {"budget_caps": {"budget_warning_fraction": 0.8}},
        {"budget_caps": {"max_steps": 0}},
        {"budget_caps": {"max_model_turns": True}},
        {"budget_caps": {"max_wall_seconds": float("nan")}},
        {"budget_caps": {"max_tool_result_chars": -1}},
        {"budget_caps": {"max_steps": MAX_RUN_CONTROL_STEPS + 1}},
    ):
        with pytest.raises(ValueError):
            parse_result({**baseline, "child_limits": limits}, child)
    assert parse_result(
        {**baseline, "child_limits": {"budget_caps": {"max_wall_seconds": 2.5}}}, child
    ).child_limits
    with pytest.raises(ValueError):
        parse_result(
            {
                **baseline,
                "child_limits": {"budget_caps": {"max_wall_seconds": 10**1000}},
            },
            child,
        )
    stop = _result_handler("Stop", ["continuation", "stop_continuations"])
    assert (
        parse_result(
            {
                **baseline,
                "continuation": {"message": "next"},
                "stop_continuations": True,
            },
            stop,
        ).continuation["message"]
        == "next"
    )
    for continuation in (
        {"message": ""},
        {"message": "x", "owner": "forged"},
        {"message": "😀" * 1025},
    ):
        with pytest.raises(ValueError):
            parse_result({**baseline, "continuation": continuation}, stop)
    for invalid in (
        {**baseline, "stop_continuations": False},
        {**baseline, "owner_run_id": "forged"},
        {**baseline, "decision": "deny"},
    ):
        with pytest.raises(ValueError):
            parse_result(invalid, stop)


def _event(event="PreToolUse", data=None):
    from tldw_chatbook.Agents.hooks_v2.validation import parse_event

    if data is None:
        data = (
            {"tool_id": "local:fs_read", "provider": "local"}
            if event in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
            else {}
        )
        if event == "SessionStart":
            data = {"reason": "resume"}
        if event == "PostToolUseFailure":
            data["reason"] = "timeout"
    return parse_event(
        {
            "protocol_version": 2,
            "event_id": "e1",
            "event": event,
            "timestamp": "2026-09-16T00:00:00Z",
            "runtime_session_id": "s1",
            "initiator": "manual",
            "origin": "user",
            "causal_chain_id": "c1",
            "causal_depth": 0,
            "data": data,
        }
    )


def test_host_event_matchers_distinguish_unavailable_from_nonmatch():
    from tldw_chatbook.Agents.hooks_v2.matching import (
        UnsupportedEventField,
        matches_handler,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    event = _event()
    tool_only = parse_handlers([_command(match={"tool_id": ["local:fs_*"]})])[0]
    operation = parse_handlers([_command(match={"operation": ["file.read"]})])[0]
    assert matches_handler(tool_only, event)
    with pytest.raises(UnsupportedEventField):
        matches_handler(operation, event)
    combined = parse_handlers(
        [_command(match={"tool_id": ["other"], "operation": ["file.read"]})]
    )[0]
    with pytest.raises(UnsupportedEventField):
        matches_handler(combined, event)
    assert not matches_handler(
        parse_handlers([_command(match={"tool_id": ["other"]})])[0], event
    )
    assert matches_handler(
        parse_handlers([_command("SessionStart", match={"reason": ["resume"]})])[0],
        _event("SessionStart"),
    )
    with pytest.raises(ValueError):
        _event("PreToolUse", {"tool_id": "local:fs_read"})
    with pytest.raises(ValueError):
        _event("PostToolUseFailure", {"tool_id": "x", "provider": "local"})


def test_mcp_templates_preserve_type_and_refuse_missing_paths():
    from tldw_chatbook.Agents.hooks_v2.matching import (
        UnsupportedEventField,
        expand_input,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    raw = {
        "id": "template",
        "event": "PreToolUse",
        "type": "mcp_tool",
        "server": "server",
        "tool": "inspect",
        "effects": [],
        "input": {
            "id": "${data.tool_id}",
            "arguments": "${data.original_arguments}",
            "label": "tool ${data.tool_id}",
        },
    }
    handler = parse_handlers([raw])[0]
    event = _event(
        data={
            "tool_id": "local:fs_read",
            "provider": "local",
            "original_arguments": {"path": "a"},
        }
    )
    assert expand_input(handler, event)["arguments"] == {"path": "a"}
    with pytest.raises(UnsupportedEventField):
        expand_input(handler, _event())
    raw["input"] = {"bad": "${data.undocumented}"}
    with pytest.raises(ValueError):
        parse_handlers([raw])


def test_large_event_is_refused_before_whole_object_serialization(monkeypatch):
    from tldw_chatbook.Agents.hooks_v2 import validation

    assert _event("UserPromptSubmit", {"prompt": "small"}).data["prompt"] == "small"
    original = validation.json.dumps

    def guarded_dumps(value, *args, **kwargs):
        if isinstance(value, (dict, list)):
            raise TypeError("whole JSON object serialized before byte refusal")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(validation.json, "dumps", guarded_dumps)
    with pytest.raises(ValueError):
        _event("UserPromptSubmit", {"prompt": "x" * (1024 * 1024)})


def test_event_input_accepts_exact_byte_limit_then_refuses_one_more():
    from tldw_chatbook.Agents.hooks_v2.validation import INPUT_BYTES, parse_event

    value = {
        "protocol_version": 2,
        "event_id": "e1",
        "event": "UserPromptSubmit",
        "timestamp": "2026-09-16T00:00:00Z",
        "runtime_session_id": "s1",
        "initiator": "manual",
        "origin": "user",
        "causal_chain_id": "c1",
        "causal_depth": 0,
        "data": {"prompt": ""},
    }
    overhead = len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    value["data"]["prompt"] = "x" * (INPUT_BYTES - overhead)
    assert parse_event(value).data["prompt"] == value["data"]["prompt"]
    value["data"]["prompt"] += "x"
    with pytest.raises(ValueError):
        parse_event(value)


def test_repeated_typed_template_is_bounded_before_materialization(monkeypatch):
    from tldw_chatbook.Agents.hooks_v2 import matching
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    event = _event(
        data={
            "tool_id": "local:fs_read",
            "provider": "local",
            "original_arguments": {"text": "x" * 8192},
        }
    )
    raw = {
        "id": "template",
        "event": "PreToolUse",
        "type": "mcp_tool",
        "server": "server",
        "tool": "inspect",
        "effects": [],
        "input": {str(i): "${data.original_arguments}" for i in range(180)},
    }
    handler = parse_handlers([raw])[0]
    accepted = parse_handlers(
        [{**raw, "input": {str(i): "${data.original_arguments}" for i in range(120)}}]
    )[0]
    assert len(matching.expand_input(accepted, event)) == 120
    original = matching._json_tree

    def bounded_before_expand(value, **kwargs):
        assert value is handler.input, "template was materialized before bounding"
        return original(value, **kwargs)

    monkeypatch.setattr(matching, "_json_tree", bounded_before_expand)
    with pytest.raises(ValueError):
        matching.expand_input(handler, event)
