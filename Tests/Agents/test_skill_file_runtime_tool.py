"""skill_file: the fourth runtime tool -- bindings, schema pin, dispatch.

Model: Tests/Agents/test_skill_tool_spawn.py's fake chat_call/registry
scaffolding (AgentService + AgentRunsDB + a scripted fence-protocol
provider). skill_file is NOT a ToolProvider -- its schema is pinned into
runtime_schemas (never disclosure-gated) and its authorization lives on the
per-run SkillFileBindings object, never config.allowed_tools.
"""

import asyncio
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RUN_DONE,
    RUNTIME_TOOL_NAMES,
    RunBudget,
    SKILL_FILE_TOOL_NAME,
    SkillFileBindings,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _skill_file_fence(skill_name, path):
    return _fence(SKILL_FILE_TOOL_NAME, {"skill_name": skill_name, "path": path})


def _base_config():
    return AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(),
    )


def _registry_with_builtins():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def _next_provider_turn_contains(calls, expected):
    return any(
        expected in str(message.get("content", ""))
        for message in calls[1]["messages_payload"]
    )


def _assert_sanitized_receipt(db, run_id, *, outcome):
    run = db.get_run(run_id)
    results = [step for step in run["steps"] if step["kind"] == "tool_result"]
    assert len(results) == 1
    receipt = results[0]
    assert receipt["result"] == ""
    assert receipt["field_states"]["result"] == "omitted"
    assert receipt["summary"] == "skill_file recorded"
    assert receipt["tool_outcome"] == outcome
    return receipt


def _pinned_bindings(reader, *, digest="digest-a"):
    return SkillFileBindings(
        authorized={"demo"},
        reader=reader,
        definition_digests={"demo": digest},
        current_definition_digest=lambda _name: digest,
    )
# --- Step 1 unit tests (brief's exact contract) -----------------------------


def test_skill_file_is_a_runtime_tool_name():
    assert SKILL_FILE_TOOL_NAME == "skill_file"
    # collision exclusion rides existing consumers
    assert SKILL_FILE_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_bindings_object_shape():
    b = SkillFileBindings(authorized=set(), reader=None)
    b.authorized.add("demo")
    assert "demo" in b.authorized


# --- Loop-level tests, run through AgentService (only it wires runtime_
# schemas / the reader closure -- the loop itself just dispatches by name). --


def test_skill_file_schema_offered_first_turn_and_authorized_read_succeeds(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    read_calls = []

    def reader(skill_name, path):
        read_calls.append((skill_name, path))
        return {"content": "REF", "truncated": False, "size": 3}

    bindings = _pinned_bindings(reader)

    script = [
        {
            "choices": [
                {"message": {"content": _skill_file_fence("demo", "references/api.md")}}
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert read_calls == [("demo", "references/api.md")]

    # Active from the FIRST provider call, never disclosure-gated -- unlike
    # a ToolProvider entry, which would need find_tools/load_tools first.
    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert SKILL_FILE_TOOL_NAME in first_system_content

    # The tool payload remains available to the live loop even though raw
    # tool content is deliberately omitted from durable run history.
    assert _next_provider_turn_contains(calls, "REF")
    _assert_sanitized_receipt(db, run_id, outcome="success")


def test_skill_file_revalidates_the_admitted_definition_before_every_read(tmp_path):
    current = {"digest": "digest-a", "body": "REFERENCE_A"}
    read_calls = []

    def reader(skill_name, path):
        read_calls.append((skill_name, path, current["body"]))
        return {"content": current["body"], "truncated": False, "size": 11}

    def run_read(bindings, suffix):
        calls = []
        script = [
            {
                "choices": [
                    {
                        "message": {
                            "content": _skill_file_fence(
                                "demo", "references/api.md"
                            )
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Done."}}]},
        ]
        def chat_call(**kwargs):
            calls.append(kwargs)
            return script.pop(0)

        service = AgentService(
            AgentRunsDB(tmp_path / f"runs-{suffix}.db", client_id="t"),
            _registry_with_builtins(),
            chat_call=chat_call,
            skill_file_bindings=bindings,
        )
        run_id, outcome = service.run_turn(
            conversation_id=f"c-{suffix}",
            messages=[{"role": "user", "content": "go"}],
            config=_base_config(),
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        _assert_sanitized_receipt(service.db, run_id, outcome="failed" if suffix == "old" else "success")
        return calls

    old_bindings = SkillFileBindings(authorized={"demo"}, reader=reader)
    old_bindings.definition_digests = {"demo": "digest-a"}
    old_bindings.current_definition_digest = lambda _name: current["digest"]

    current.update(digest="digest-b", body="REFERENCE_B")
    refused = run_read(old_bindings, "old")

    assert _next_provider_turn_contains(refused, "skill_definition_changed")
    assert read_calls == []
    assert "demo" not in old_bindings.authorized

    new_bindings = SkillFileBindings(authorized={"demo"}, reader=reader)
    new_bindings.definition_digests = {"demo": "digest-b"}
    new_bindings.current_definition_digest = lambda _name: current["digest"]

    accepted = run_read(new_bindings, "new")

    assert _next_provider_turn_contains(accepted, "REFERENCE_B")
    assert read_calls == [("demo", "references/api.md", "REFERENCE_B")]


def test_skill_file_definition_mismatch_revokes_even_without_a_reader(tmp_path):
    calls = []
    bindings = SkillFileBindings(
        authorized={"demo"},
        reader=None,
        definition_digests={"demo": "digest-a"},
        current_definition_digest=lambda _name: "digest-b",
    )
    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _skill_file_fence("demo", "references/api.md")
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(
        AgentRunsDB(tmp_path / "runs.db", client_id="t"),
        _registry_with_builtins(),
        chat_call=chat_call,
        skill_file_bindings=bindings,
    )

    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    _assert_sanitized_receipt(service.db, run_id, outcome="failed")
    assert _next_provider_turn_contains(calls, "skill_definition_changed")
    assert "demo" not in bindings.authorized


def test_skill_file_unauthorized_name_is_refused(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    def reader(skill_name, path):
        raise AssertionError("reader must not be called for an unauthorized name")

    # "demo" is active in this run; "other" is not -- the model asks for
    # "other" anyway (e.g. a stale/hallucinated skill name).
    bindings = _pinned_bindings(reader)

    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _skill_file_fence("other", "references/api.md")
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert _next_provider_turn_contains(
        calls, "ERROR: skill_file: 'other' is not active in this run"
    )
    _assert_sanitized_receipt(db, run_id, outcome="failed")


def test_skill_file_reader_returning_non_mapping_fails_the_call_not_the_run(
    tmp_path,
):
    """A reader is caller-supplied (the bridge's asyncio.run adapter over
    SkillsScopeService.read_skill_file); a malformed/misbehaving reader that
    returns a bare string (not a dict-like result) must fail only THAT tool
    call -- not crash the whole run via an uncaught AttributeError from
    `.get` on a non-mapping."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    def bad_reader(skill_name, path):
        return "not a dict"

    bindings = _pinned_bindings(bad_reader)

    script = [
        {
            "choices": [
                {"message": {"content": _skill_file_fence("demo", "references/api.md")}}
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert _next_provider_turn_contains(
        calls, "ERROR: skill_file: reader returned invalid result"
    )
    _assert_sanitized_receipt(db, run_id, outcome="failed")


def test_skill_file_empty_authorized_schema_absent_and_falls_through(tmp_path):
    # Qodo/PR#814: schema pinning (~agent_service.py:356-360) already
    # requires bindings.authorized to be non-empty, but the OLD
    # LoopDeps.read_skill_file wiring only checked bindings is not None --
    # so bindings with an EMPTY authorized set (no skill forked yet in this
    # run) still reached the named-refusal dispatch for a hallucinated
    # call, leaking the tool's existence to an undisclosed name. Wiring
    # must use the SAME predicate as schema pinning: empty authorized falls
    # through to the generic "Tool not permitted" path, identical to
    # bindings=None.
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    def reader(skill_name, path):
        raise AssertionError("reader must not be called with empty authorized")

    bindings = SkillFileBindings(authorized=set(), reader=reader)

    script = [
        {
            "choices": [
                {"message": {"content": _skill_file_fence("demo", "references/api.md")}}
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert SKILL_FILE_TOOL_NAME not in first_system_content

    # Falls through to the SAME permission-gate path any other undisclosed/
    # disallowed tool name hits -- not the skill_file-specific
    # "'demo' is not active in this run" refusal.
    assert _next_provider_turn_contains(calls, "Tool not permitted: skill_file")
    _assert_sanitized_receipt(db, run_id, outcome="blocked")


def test_skill_file_bindings_none_schema_absent_and_falls_through(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    script = [
        {
            "choices": [
                {"message": {"content": _skill_file_fence("demo", "references/api.md")}}
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    # No skill_file_bindings passed at all -- the feature was never
    # configured for this run.
    service = AgentService(db, reg, chat_call=chat_call)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert SKILL_FILE_TOOL_NAME not in first_system_content

    # Falls through to the SAME permission-gate path any other undisclosed/
    # disallowed tool name hits -- not a skill_file-specific refusal.
    assert _next_provider_turn_contains(calls, "Tool not permitted: skill_file")
    _assert_sanitized_receipt(db, run_id, outcome="blocked")


# --- Step 1 e2e: real LocalSkillsService, no fake reader --------------------


def test_skill_file_e2e_fork_reads_its_own_reference_file(tmp_path):
    """End-to-end through the REAL read seam: a real `LocalSkillsService`
    (file-marker trust arrangement mirrored from Tests/Skills/
    test_read_skill_file.py's `_svc` -- no keyring, no policy_enforcer, so
    `_enforce` and `_require_trusted_skill` both no-op) backs a skill whose
    bundle has `references/api.md`. The bindings reader is the real sync
    adapter (`asyncio.run` over the async service call), not a fake -- this
    is the seam the bridge itself uses in production. Drives the same
    scripted-model harness as the unit tests above: turn 1 calls skill_file,
    turn 2 finishes."""
    svc = LocalSkillsService(
        store_dir=tmp_path / "skills_store",
        allow_untrusted_without_trust_service=True,
    )
    asyncio.run(svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n"))
    skill_dir = svc._skill_dir("demo")
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "references" / "api.md").write_text("# api docs\n", encoding="utf-8")

    def reader(skill_name, path):
        return asyncio.run(svc.read_skill_file(skill_name, path))

    bindings = _pinned_bindings(reader)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    script = [
        {
            "choices": [
                {"message": {"content": _skill_file_fence("demo", "references/api.md")}}
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]

    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    assert _next_provider_turn_contains(calls, "# api docs")
    _assert_sanitized_receipt(db, run_id, outcome="success")


@pytest.mark.parametrize(
    "mode",
    [
        "standalone",
        "ordinary_user",
        "pruned_plugin",
        "truncated_plugin",
        "warning_plugin",
    ],
)
def test_host_plugin_context_survives_only_whole_in_final_agent_payload(tmp_path, mode):
    from dataclasses import replace

    from tldw_chatbook.Agents.agent_models import PluginContextText
    from tldw_chatbook.Chat.console_history_budget import ToolResultPruneSettings
    from tldw_chatbook.Plugins.context import check_send_context, instruction_block

    block = instruction_block(
        "installation", "skill:demo", "revision", "PRIVATE_FILE:" + "x" * 6800
    )
    file_count = 6 if mode != "ordinary_user" else 0
    supplied = block if mode.endswith("plugin") else str(block)
    payloads = []
    live_payloads = []

    def chat_call(**kwargs):
        live_payloads.append(kwargs["messages_payload"])
        payloads.append(check_send_context(kwargs["messages_payload"]))
        index = len(payloads) - 1
        text = (
            _skill_file_fence("demo", f"file-{index}.md")
            if index < file_count
            else "done"
        )
        return {"choices": [{"message": {"content": text}}]}

    db = AgentRunsDB(tmp_path / "origin-runs.sqlite", client_id="t")
    service = AgentService(
        db,
        _registry_with_builtins(),
        chat_call=chat_call,
        skill_file_bindings=_pinned_bindings(
            lambda _name, _path: {"content": supplied}
        ),
        tool_result_pruning=(
            ToolResultPruneSettings(
                keep_recent_turns=0,
                min_result_chars=1,
                head_chars=10,
                min_reclaim_chars=1,
            )
            if mode == "pruned_plugin"
            else None
        ),
    )
    config = replace(_base_config(), budget=replace(RunBudget(), max_steps=80))
    if mode == "warning_plugin":
        config = replace(
            config, budget=replace(config.budget, budget_warning_fraction=0.01)
        )
    if mode == "truncated_plugin":
        config = replace(
            config, budget=replace(config.budget, max_tool_result_chars=100)
        )
    user_text = str(block) * 6 if mode == "ordinary_user" else "go"
    try:
        run_id, outcome = service.run_turn(
            conversation_id="origin",
            messages=[{"role": "user", "content": user_text}],
            config=config,
            api_endpoint="llama_cpp",
        )
        if mode == "warning_plugin":
            assert outcome.status == "error", outcome
            assert (
                len(payloads) == 5
            ), "warning erased a live plugin block's attribution"
            assert any(
                "budget" in str(row["content"]).lower()
                and type(row["content"]) is str
                for payload in live_payloads
                for row in payload
                if "PRIVATE_FILE:" in str(row.get("content"))
            )
            return
        assert outcome.status == RUN_DONE, outcome
        assert len(payloads) == file_count + 1
        assert all(
            not isinstance(row.get("content"), PluginContextText)
            for payload in payloads
            for row in payload
        )
        if mode == "pruned_plugin":
            assert any("omitted whole" in str(payload) for payload in payloads)
            assert all("PRIVATE_FILE:" not in str(payload) for payload in payloads)
        elif mode == "truncated_plugin":
            assert "cannot fit whole" in str(payloads[-1])
            assert "PRIVATE_FILE:" not in str(payloads[-1])
        else:
            assert str(payloads[-1]).count("PRIVATE_FILE:") == 6
        assert "origins" not in json.dumps(db.get_run(run_id), default=str)
    finally:
        db.close()


@pytest.mark.parametrize("native", [False, True])
def test_live_plugin_attribution_formatting_and_json_replay(native):
    from copy import deepcopy

    from tldw_chatbook.Agents.agent_models import (
        PluginContextText,
        ToolCall,
        carry_plugin_context,
    )
    from tldw_chatbook.Agents.agent_runtime import _append_tool_result
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.context import check_send_context, instruction_block

    block = instruction_block("installation", "skill:demo", "revision", "x" * 6800)
    assembled = carry_plugin_context("Before\n" + str(block) + "\nBundled files", block)
    messages = []
    for index in range(5):
        _append_tool_result(
            messages,
            ToolCall("skill_file", {}, call_id=str(index) if native else ""),
            assembled,
        )
    assert all(isinstance(row["content"], PluginContextText) for row in messages)
    with pytest.raises(PluginUnavailable, match="send_too_large"):
        check_send_context(deepcopy(messages))
    assert len(check_send_context(messages[-4:])) == 4  # per-send, not lifetime
    replay = json.loads(json.dumps(messages))
    assert len(check_send_context(replay)) == 5  # plain text cannot forge attribution
    assert all(type(row["content"]) is str for row in replay)


@pytest.mark.parametrize("resume", [False, True])
def test_plugin_continuation_keeps_live_carrier_and_plain_checkpoint(resume):
    from dataclasses import replace

    from Tests.Agents.test_provider_continuation_runtime import (
        _checkpoint,
        _deps,
        _native_turn,
        _pending_call,
    )
    from tldw_chatbook.Agents.agent_models import (
        ModelTurn,
        PluginContextText,
        ToolCall,
        ToolResult,
    )
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop
    from tldw_chatbook.Agents.tool_catalog import SKILL_FILE_TOOL_SCHEMA
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationRestoreTarget,
        ContinuationResult,
    )
    from tldw_chatbook.Plugins.context import check_send_context, instruction_block

    block = instruction_block(
        "installation", "skill:demo", "revision", "PRIVATE_FILE:" + "x" * 6800
    )
    args = {"skill_name": "demo", "path": "reference.md"}
    call = ToolCall(
        "skill_file", args, "file-call", json.dumps(args, separators=(",", ":"))
    )
    pending = _checkpoint(_pending_call("file-call", name="skill_file", args=args))
    completed_call = replace(
        pending.rounds[0].calls[0],
        state="completed",
        result=ContinuationResult(str(block)),
    )
    final = replace(
        pending,
        checkpoint_revision=4,
        state="complete",
        rounds=(replace(pending.rounds[0], calls=(completed_call,)),),
    )
    events, payloads = [], []

    def expand(checkpoint):
        native = _native_turn((call,), checkpoint).assistant_message
        rows = [native]
        result = checkpoint.rounds[0].calls[0].result
        if result is not None:
            rows.append(
                {"role": "tool", "tool_call_id": call.call_id, "content": result.value}
            )
        return rows

    def call_model(messages, _active):
        payloads.append(list(messages))
        if len(payloads) == 1 and not resume:
            return _native_turn((call,), pending)
        return ModelTurn(text="done", provider_continuation=final)

    deps = _deps(
        [],
        order=[],
        persist=events.append,
        invoke=lambda _call: pytest.fail("runtime reader owns this"),
        expand=expand,
    )
    deps.call_model = call_model
    deps.read_skill_file = lambda _skill, _path: ToolResult(ok=True, content=block)
    restore = (
        {
            "restore_provider_continuation": pending,
            "restore_provider_target": ContinuationRestoreTarget(
                "deepseek",
                "deepseek-v4-flash",
                "responses",
                "https://api.deepseek.com/v1",
            ),
            "resume_provider_continuation": True,
        }
        if resume
        else {}
    )
    outcome = run_agent_loop(
        _base_config(), [], [SKILL_FILE_TOOL_SCHEMA], deps, **restore
    )
    assert outcome.status == RUN_DONE, outcome
    live = next(row["content"] for row in payloads[-1] if row.get("role") == "tool")
    assert isinstance(live, PluginContextText)
    assert live == block
    assert all(
        type(call.result.value) is str
        for event in events
        if hasattr(event, "checkpoint")
        for round_ in event.checkpoint.rounds
        for call in round_.calls
        if call.result is not None
    )
    assert type(check_send_context(payloads[-1])[-1]["content"]) is str
