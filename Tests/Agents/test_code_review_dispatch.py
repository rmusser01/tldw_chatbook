"""Review slots execute through the real service, gateway and tool threads."""

import hashlib
import json
import threading
import time

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.conftest import join_fleet_children
from Tests.Agents.test_agent_service import fence, native_call, provider_reply
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import AgentConfig, AgentDefinition, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.code_review_models import ReviewerChoice, ReviewTeam
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.run_context import current_run_actor
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def dispatch_case(review_case, monkeypatch, tmp_path):
    c = review_case
    from tldw_chatbook.Utils import token_counter
    real_window = token_counter.resolve_context_window
    monkeypatch.setattr(token_counter, "resolve_context_window", lambda provider, model, **kwargs: token_counter.ContextWindowResolution(32768, "scripted model metadata", True) if model == "verify" or model.startswith(("review-model-", "fallback-model")) else real_window(provider, model, **kwargs))
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    c.db = AgentRunsDB(tmp_path / "dispatch.db", client_id="dispatch-test")
    verifier = c.db.create_agent_definition(
        AgentDefinition(
            name="verifier", instructions="Verify", provider="openai", model="verify"
        )
    )
    c.team = c.team.model_copy(update={"verifier_preset_id": verifier})
    monkeypatch.setattr(agent_service, "load_agents_routing_config", lambda: c.routing)
    c.calls = []
    c.errors = []
    c.child_scripts = {}
    c.lock = threading.Lock()
    c.native = False
    c.review_adapters = {}
    from tldw_chatbook.Agents.code_review_service import CodeReviewService

    native_for_run = CodeReviewService.for_run

    def observe_adapter(service, actor, **kwargs):
        adapter = native_for_run(service, actor, **kwargs)
        c.review_adapters[actor.run_id] = adapter
        return adapter

    monkeypatch.setattr(CodeReviewService, "for_run", observe_adapter)

    def reply(name, args):
        return (
            {"content": None, "tool_calls": [native_call(name, args)]}
            if c.native
            else fence(name, args)
        )

    c.reply = reply

    def gateway(**kwargs):
        actor = current_run_actor()
        with c.lock:
            c.calls.append((actor, kwargs))
            script = (
                c.parent_script
                if actor.kind == "primary"
                else c.child_scripts.setdefault(actor.run_id, list(c.child_script))
            )
            if not script:
                c.errors.append("script exhausted: " + actor.run_id)
                raise AssertionError(c.errors[-1])
            item = script.pop(0)
        return provider_reply(item() if callable(item) else item)

    def run(
        *,
        reviewers=1,
        focuses=(),
        inline=False,
        replay=False,
        native=False,
        child_script=None,
        prepare_hook=None,
        parent_script=None,
        verify=False,
        limit=16000,
        ledger=None,
        destinations=None,
        preparation_inline=None,
        hooks_engine=None,
        on_step=None,
        work_chain_id=None,
        run_log_enabled=None,
    ):
        c.native = native
        ids = []
        for i in range(reviewers):
            ids.append(
                c.db.create_agent_definition(
                    AgentDefinition(
                        name=f"review-{i}",
                        instructions=f"PRESET-{i}",
                        provider=destinations[i] if destinations else "openai",
                        model=f"review-model-{i}",
                        fallback_models=getattr(c, "fallbacks", [()] * reviewers)[i],
                        params=(("temperature", 0.2 + i * 0.3),),
                    )
                )
            )
        c.team = ReviewTeam(
            reviewers=tuple(
                ReviewerChoice(preset_id=x, focus=focuses[i] if focuses else "")
                for i, x in enumerate(ids)
            ),
            verifier_preset_id=c.team.verifier_preset_id,
        )
        c.primary = c.service(limit=limit)
        from tldw_chatbook.Agents.code_review_tool_provider import (
            CodeReviewToolProvider,
        )

        c.provider = CodeReviewToolProvider(c.primary)
        if preparation_inline is not None:
            monkeypatch.setattr(
                agent_service,
                "_setting",
                lambda key, default: (
                    (1 if preparation_inline else 3)
                    if key == "max_live_subagents"
                    else default
                ),
            )
        prepared = c.prepare()
        c.prepared = prepared
        if verify:
            from tldw_chatbook.Agents.code_review_service import request_fingerprints

            c.primary.verification_grant = c.issuer.issue(
                c.owner,
                session_id="chat",
                acceptance_id="accepted",
                stage="verify",
                source_epoch=0,
                **request_fingerprints(c.working_tree_target, c.team),
            )
        if prepare_hook:
            prepare_hook(c)
        c.child_script = child_script or [
            reply(
                "code_review_read",
                {
                    "review_id": prepared.review_id,
                    "view": "source",
                    "source_query": {
                        "op": "list",
                        "relative_dir": ".",
                        "side": "worktree",
                    },
                },
            ),
            "review complete",
        ]
        c.parent_script = [
            reply(
                "spawn_subagent",
                {"review_id": prepared.review_id, "review_slot_id": f"R{i + 1}"},
            )
            for i in range(reviewers)
        ]
        if verify:
            c.parent_script = [
                reply(
                    "spawn_subagent",
                    {"review_id": prepared.review_id, "review_slot_id": "V1"},
                )
            ]
        if parent_script:
            c.parent_script = parent_script(c)
        if replay:
            c.parent_script.append(
                reply(
                    "spawn_subagent",
                    {"review_id": prepared.review_id, "review_slot_id": "R1"},
                )
            )
        if not inline:
            c.parent_script.append(reply("wait_agents", {}))
        c.parent_script.append("finished")
        registry = ToolCatalogRegistry()
        registry.register_provider(c.provider)
        if getattr(c, "process_provider", None) is not None:
            registry.register_provider(c.process_provider)
        if c.primary.native_coordinator is not None:
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
            from tldw_chatbook.MCP.permission_store import EffectiveToolState

            registry.register_provider(
                LocalToolProvider(
                    workspace_root=c.root,
                    admitted_roots=(c.authority.admitted,),
                    resolve_state=lambda tool: EffectiveToolState(
                        state="allow", origin="tool_override"
                    ),
                )
            )
        monkeypatch.setattr(
            agent_service,
            "_setting",
            lambda key, default: (
                run_log_enabled
                if key == "run_log_enabled" and run_log_enabled is not None
                else (1 if inline else 3)
                if key == "max_live_subagents"
                else default
            ),
        )
        service = AgentService(
            db=c.db,
            runtime_capacity=getattr(c, "runtime_capacity", None),
            work_chain_id=work_chain_id,
            registry=registry,
            chat_call=gateway,
            on_step=on_step,
            app_config=c.app_config,
            project_instruction_context=ledger,
            guard_tool_calls=getattr(c, "guard", None),
            hooks_v2_engine=hooks_engine,
            hooks_v2_session_id="review-hooks",
            hooks_v2_turn_id="review-turn",
            fleet_coordinator=None
            if getattr(c, "use_factory", False) or inline
            else FleetCoordinator(max_live=3, clock=time.monotonic),
        )
        c.runtime = service
        c.initial_registry = registry
        _, outcome = service.run_turn(
            conversation_id="chat",
            requested_run_id="primary",
            messages=[{"role": "user", "content": "PRIMARY-PEER-CANARY"}],
            config=AgentConfig(
                model="parent",
                system_prompt="Primary",
                personal_context_block="PERSONAL-CANARY",
                allowed_tools=(
                    "spawn_subagent",
                    "code_review_read",
                    "code_review_submit",
                )
                + (("fs_read",) if c.primary.native_coordinator is not None else ())
                + (
                    ("shell_exec",)
                    if getattr(c, "process_provider", None) is not None
                    else ()
                ),
                native_tools=native,
                budget=RunBudget(
                    max_steps=150,
                    max_model_turns=80,
                    max_subagents=2,
                    max_tool_result_chars=limit,
                ),
            ),
            api_endpoint="openai",
        )
        join_fleet_children(service)
        assert not c.errors
        c.outcome = outcome
        return c

    c.run = run
    code_review_helpers.install_batched_verifier(c, monkeypatch)
    code_review_helpers.install_check_handoff(c)
    yield c
    c.db.close()


@pytest.mark.parametrize("kind", ["working_tree", "commit_range"])
@pytest.mark.parametrize("verify", [False, True])
def test_child_task_identifies_captured_target_without_private_context(
    dispatch_case, kind, verify
):
    c = dispatch_case
    if kind == "commit_range":
        c.working_tree_target = c.committed_target
    c.run(native=True, verify=verify, child_script=["done"])
    child = next(request for actor, request in c.calls if actor.kind == "subagent")
    task = next(
        row["content"] for row in child["messages_payload"] if row["role"] == "user"
    )
    with c.store.lease(c.prepared.review_id) as context:
        assert context.preparation.target.kind == kind
        metadata = task.splitlines()[1]
        assert metadata == (
            f"Review {c.prepared.review_id}; slot {'V1' if verify else 'R1'}; "
            f"target {context.target_digest}; target_kind={kind}."
        )
    assert "Read submission_revision" in task
    assert "expected_revision" in task
    assert "code_review_prepare" not in task
    assert "Prepare the accepted target" not in task
    assert "Present each original report" not in task
    if verify:
        assert task.startswith("Verification requires an accepted verification stage")
        assert "Submit confirmed, refuted or unresolved dispositions" in task
    else:
        assert task.startswith("Review independently")
        assert "trigger" in task and "impact" in task and "uncertainty" in task
        assert "kind=coverage" in task
    text = json.dumps(child)
    assert str(c.root) not in text
    assert "PERSONAL-CANARY" not in text
    assert "PRIMARY-PEER-CANARY" not in text
    assert "report_to_supervisor" not in text
    assert "send_to_peer" not in text


def test_reviewers_share_stored_brief_and_fingerprint_with_additional_focus(
    dispatch_case,
):
    c = dispatch_case.run(
        reviewers=2, focuses=("FOCUS-ONLY-R1", ""), native=True, child_script=["done"]
    )
    children = [request for actor, request in c.calls if actor.kind == "subagent"]
    assert len(children) == 2
    with c.store.lease(c.prepared.review_id) as context:
        brief = context.preparation.guide_text
        snapshots = [
            json.loads(item.path.read_bytes())
            for item in context._files.values()
            if item.category == "report" and b'"slots"' in item.path.read_bytes()
        ]
        assert snapshots[0]["guide"]["text"] == brief
    assert brief.startswith("Review independently")
    assert "FOCUS-ONLY-R1" not in brief
    for child in children:
        task = next(
            row["content"] for row in child["messages_payload"] if row["role"] == "user"
        )
        assert task.splitlines()[0] == brief
        assert ("Additional focus: FOCUS-ONLY-R1" in task) == (
            child["model"] == "review-model-0"
        )
    from tldw_chatbook.Agents.run_context import use_run_actor

    with use_run_actor(c.actor):
        envelopes = c.primary.read(c.prepared.review_id, "summary").records[0]["slots"]
        guide = c.primary.guide("review")
    assert {
        slot["guide_fingerprint"]
        for slot in envelopes
        if slot["slot_id"].startswith("R")
    } == {hashlib.sha256(brief.encode()).hexdigest()}
    assert "code_review_prepare" in guide["text"]
    assert "spawn_subagent" in guide["text"]
    assert "Present every original first" in guide["text"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_scoped_spawn_dispatches_once_and_binds_real_tool_thread(
    dispatch_case, native, inline
):
    c = dispatch_case.run(replay=True, native=native, inline=inline)
    children = [
        (actor, request) for actor, request in c.calls if actor.kind == "subagent"
    ]
    assert children, c.outcome
    assert len({actor.run_id for actor, _ in children}) == 1
    assert {request["model"] for _, request in children} == {"review-model-0"}
    assert children[0][1]["temp"] == 0.2
    pages = model_pages(children[-1][1])
    assert pages and pages[-1]["review_id"] == c.prepared.review_id
    assert pages[-1]["submission_revision"] == 0
    assert all(row["status"] == "done" for row in c.db.list_runs("chat"))


@pytest.mark.parametrize("native", [False, True])
def test_concurrent_slots_keep_independent_routes_and_catalogs(
    dispatch_case, monkeypatch, native
):
    c = dispatch_case
    child_services = []
    parent_caches = []
    original_fork = AgentService._fork_review_child

    def observe_fork(service, *args, **kwargs):
        parent_caches.append(dict(service._tool_protocol_cache))
        child = original_fork(service, *args, **kwargs)
        child_services.append(child)
        return child

    monkeypatch.setattr(AgentService, "_fork_review_child", observe_fork)
    barrier = threading.Barrier(2, timeout=5)

    def first():
        barrier.wait()
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "source",
                "source_query": {"op": "list", "relative_dir": ".", "side": "worktree"},
            },
        )

    c.run(reviewers=2, native=native, child_script=[first, "done"])
    children = [(a, r) for a, r in c.calls if a.kind == "subagent"]
    assert len({a.run_id for a, _ in children}) == 2
    assert {(r["api_endpoint"], r["model"], r["temp"]) for _, r in children} == {
        ("openai", "review-model-0", 0.2),
        ("openai", "review-model-1", 0.5),
    }
    assert (
        c.initial_registry.resolve_owner_for_name("code_review_prepare")[1]
        is c.provider
    )
    assert c.provider.service.actor == c.actor
    assert len({id(s._tool_protocol_cache) for s in [c.runtime, *child_services]}) == 3
    assert len({id(s.registry) for s in [c.runtime, *child_services]}) == 3
    assert c.runtime._tool_protocol_cache == parent_caches[0]
    assert all(row["status"] == "done" for row in c.db.list_runs("chat"))


@pytest.mark.parametrize("inline", [False, True])
def test_verifier_mode_follows_actual_launch_and_only_reads_source(
    dispatch_case, inline
):
    c = dispatch_case.run(inline=inline, verify=True)
    children = [(a, r) for a, r in c.calls if a.kind == "subagent"]
    assert children
    actor = children[0][0]
    with c.store.lease(c.prepared.review_id) as context:
        binding = c.store._bound(context, actor.run_id)
    mode = "unavailable_inline" if inline else "primary_handoff"
    assert binding.check_feedback_mode == mode
    assert mode in json.dumps(children[0][1]["messages_payload"])
    if inline:
        assert "wait_agents" not in json.dumps(c.calls[0][1].get("tools", []))
    assert all(row["status"] == "done" for row in c.db.list_runs("chat"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", ""),
        ("agent", None),
        ("provider", ""),
        ("model", "parent"),
        ("isolation", None),
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_scoped_wire_mixtures_refuse_before_spending(
    dispatch_case, field, value, native
):
    def script(c):
        args = {"review_id": c.prepared.review_id, "review_slot_id": "R1", field: value}
        return [
            c.reply("spawn_subagent", args),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "R1"},
            ),
        ]

    c = dispatch_case.run(native=native, parent_script=script)
    assert len({a.run_id for a, _ in c.calls if a.kind == "subagent"}) == 1
    assert "invalid_review_spawn" in json.dumps(c.calls[-1][1]["messages_payload"])


def test_scoped_schema_is_exclusive_and_advertised(dispatch_case):
    c = dispatch_case.run(native=True)
    schema = next(
        t["function"]["parameters"]
        for t in c.calls[0][1]["tools"]
        if t["function"]["name"] == "spawn_subagent"
    )
    assert "review_id" in schema["properties"]
    from tldw_chatbook.Agents.tool_catalog import SPAWN_TOOL_SCHEMA, review_spawn_schema

    assert "oneOf" in review_spawn_schema(SPAWN_TOOL_SCHEMA).parameters


def model_pages(request, tool="code_review_read"):
    result = []
    for row in request["messages_payload"]:
        content = row.get("content")
        prefix = f"Tool result for {tool}: "
        if isinstance(content, str) and (
            row.get("role") == "tool" or content.startswith(prefix)
        ):
            try:
                page = json.loads(content.removeprefix(prefix))
            except ValueError:
                continue
            if isinstance(page, dict) and "review_id" in page:
                result.append(page)
    return result


@pytest.mark.parametrize("native", [False, True])
def test_concurrent_submissions_have_independent_streams(dispatch_case, native):
    from Tests.Agents.code_review_helpers import finding_payload

    c = dispatch_case
    barrier = threading.Barrier(2, timeout=5)

    def submit():
        with c.store.lease(c.prepared.review_id) as context:
            ref, source = next(iter(context.sources.items()))
        finding = finding_payload()
        finding["locations"] = [{"source_ref": ref, "start_line": 1, "end_line": 1}]
        finding["evidence"] = [
            dict(finding["locations"][0], kind="source", digest=source.digest)
        ]
        barrier.wait()
        return c.reply(
            "code_review_submit",
            dict(
                review_id=c.prepared.review_id,
                kind="findings",
                records=[finding],
                expected_revision=0,
                submission_id="same-id",
            ),
        )

    c.run(reviewers=2, native=native, child_script=[submit, "done"])
    with c.store.lease(c.prepared.review_id) as context:
        assert set(context.original_index) == {"R1-F1", "R2-F1"}
        assert [
            s.revision for (slot, _), s in context.streams.items() if slot != "primary"
        ] == [1, 1]
    for actor, request in c.calls:
        if actor.kind == "subagent" and any(
            row.get("role") == "tool"
            or str(row.get("content", "")).startswith(
                "Tool result for code_review_submit"
            )
            for row in request["messages_payload"]
        ):
            assert "submission_revision" in json.dumps(request["messages_payload"])
            assert "ERROR" not in json.dumps(request["messages_payload"])


def test_frozen_destinations_and_definitions_survive_preference_edits(dispatch_case):
    from tldw_chatbook.Chat.Chat_Deps import ChatModelUnavailableError

    c = dispatch_case
    c.app_config["custom_endpoints"] = {
        name: {
            "display_name": name,
            "family": "llama_cpp",
            "base_url": f"http://127.0.0.1:{port}",
            "params": {"top_k": port},
        }
        for name, port in (
            ("review-a", 8081),
            ("review-b", 8082),
            ("fallback-a", 8083),
            ("fallback-b", 8084),
        )
    }
    c.fallbacks = [
        (("custom-ep:fallback-a", "fallback-model-a"),),
        (("custom-ep:fallback-b", "fallback-model-b"),),
    ]
    barrier = threading.Barrier(2, timeout=5)

    def first():
        barrier.wait()
        raise ChatModelUnavailableError("model unavailable", provider="llama_cpp")

    def edit(c):
        for i, choice in enumerate(c.team.reviewers):
            c.db.update_agent_definition(
                choice.preset_id,
                AgentDefinition(
                    name=f"review-{i}",
                    instructions="EDITED-CANARY",
                    provider="openai",
                    model="parent",
                ),
            )
        c.app_config["custom_endpoints"]["review-a"]["params"]["top_k"] = 99

    c.run(
        reviewers=2,
        destinations=("custom-ep:review-a", "custom-ep:review-b"),
        prepare_hook=edit,
        child_script=[first, "done"],
    )
    children = [r for a, r in c.calls if a.kind == "subagent"]
    assert {
        (r["api_endpoint"], r["model"], r["topk"], r["temp"]) for r in children
    } == {
        ("custom-ep:review-a", "review-model-0", 8081, 0.2),
        ("custom-ep:review-b", "review-model-1", 8082, 0.5),
        ("custom-ep:fallback-a", "fallback-model-a", 8083, 0.2),
        ("custom-ep:fallback-b", "fallback-model-b", 8084, 0.5),
    }
    assert "EDITED-CANARY" not in json.dumps(children)
    assert c.outcome.subagents_spawned == 2


def test_direct_scoped_mixtures_are_refused_before_reserving(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    original = agent_service.run_agent_loop
    results = []

    def loop(config, messages, schemas, deps):
        if config.review_binding is None:
            for extra in (
                {"task": "fresh"},
                {"agent": None},
                {"provider": ""},
                {"model": ""},
                {"isolation": None},
            ):
                results.append(
                    deps.spawn(
                        review_id=c.prepared.review_id, review_slot_id="R1", **extra
                    )
                )
        return original(config, messages, schemas, deps)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    c.run()
    assert len(results) == 5 and all(r.error == "invalid_review_spawn" for r in results)
    assert c.outcome.subagents_spawned == 1


def test_current_destination_authority_refuses_without_spending(dispatch_case):
    c = dispatch_case

    def refuse(c):
        c.primary.destination_check = lambda target: "destination_revoked"

    c.run(prepare_hook=refuse)
    assert not [r for a, r in c.calls if a.kind == "subagent"]
    assert c.outcome.subagents_spawned == 0
    assert "destination_revoked" in json.dumps(c.calls[-1][1]["messages_payload"])
    with c.store.lease(c.prepared.review_id) as context:
        assert not context.launch_claims


def test_same_stream_conflict_has_one_unchanged_retry_then_finite_failure(
    dispatch_case,
):
    c = dispatch_case

    def submit(key, revision):
        from Tests.Agents.code_review_helpers import finding_payload

        with c.store.lease(c.prepared.review_id) as context:
            ref, source = next(iter(context.sources.items()))
        finding = finding_payload()
        finding["local_key"] = key
        finding["locations"] = [{"source_ref": ref, "start_line": 1, "end_line": 1}]
        finding["evidence"] = [
            dict(finding["locations"][0], kind="source", digest=source.digest)
        ]
        return c.reply(
            "code_review_submit",
            dict(
                review_id=c.prepared.review_id,
                kind="findings",
                records=[finding],
                expected_revision=revision,
                submission_id=key,
            ),
        )

    c.run(
        child_script=[
            lambda: submit("first", 0),
            lambda: submit("second", 0),
            lambda: submit("second", 0),
            "explicitly unresolved",
        ]
    )
    children = [r for a, r in c.calls if a.kind == "subagent"]
    assert len(children) == 4
    assert (
        sum(
            message.get("content", "").startswith(
                "Tool result for code_review_submit: ERROR: submission_revision_conflict"
            )
            for message in children[-1]["messages_payload"]
        )
        == 2
    )
    with c.store.lease(c.prepared.review_id) as context:
        assert set(context.original_index) == {"R1-F1"}
        assert (
            next(
                stream for (slot, _), stream in context.streams.items() if slot == "R1"
            ).revision
            == 1
        )


def test_frozen_fallback_is_selected_by_provider_event(dispatch_case):
    from tldw_chatbook.Chat.Chat_Deps import ChatModelUnavailableError

    c = dispatch_case
    c.app_config["custom_endpoints"] = {
        "fallback": {
            "display_name": "Fallback",
            "family": "llama_cpp",
            "base_url": "http://127.0.0.1:8090",
            "params": {"top_k": 55},
        }
    }
    c.fallbacks = [(("custom-ep:fallback", "fallback-model"),)]

    def edit(c):
        c.db.update_agent_definition(
            c.team.reviewers[0].preset_id,
            AgentDefinition(
                name="review-0",
                instructions="EDITED",
                provider="openai",
                model="parent",
                fallback_models=(("openai", "wrong"),),
            ),
        )
        c.app_config["custom_endpoints"]["fallback"]["params"]["top_k"] = 99

    def fail():
        raise ChatModelUnavailableError("model unavailable", provider="openai")

    c.run(prepare_hook=edit, child_script=[fail, "done"])
    children = [(a, r) for a, r in c.calls if a.kind == "subagent"]
    assert [(r["api_endpoint"], r["model"]) for a, r in children] == [
        ("openai", "review-model-0"),
        ("custom-ep:fallback", "fallback-model"),
    ], c.db.get_run(children[0][0].run_id)
    assert children[-1][1]["topk"] == 55
    state = c.db.get_run_fallback_state(children[0][0].run_id)
    assert state["active_index"] == 1
    from tldw_chatbook.Agents.run_context import use_run_actor

    with use_run_actor(c.actor):
        slot = c.primary.read(c.prepared.review_id, "summary").records[0]["slots"][0]
    assert slot["requested"] == {"provider": "openai", "model": "review-model-0"}
    assert slot["actual_routes"] == (
        {"provider": "openai", "model": "review-model-0"},
        {"provider": "custom-ep:fallback", "model": "fallback-model"},
    )
    assert slot["active_route"] == slot["actual_routes"][-1]


@pytest.mark.parametrize(
    "prepared_inline,launched_inline", [(True, False), (False, True)]
)
def test_mode_changes_after_preparation_use_dispatch_setting(
    dispatch_case, prepared_inline, launched_inline
):
    c = dispatch_case
    # Preparation has no coordinator or frozen execution-mode authority.
    c.run(inline=launched_inline, preparation_inline=prepared_inline, verify=True)
    actor = next(a for a, r in c.calls if a.kind == "subagent")
    with c.store.lease(c.prepared.review_id) as context:
        binding = c.store._bound(context, actor.run_id)
    assert binding.check_feedback_mode == (
        "unavailable_inline" if launched_inline else "primary_handoff"
    )


def test_live_child_budget_counts_ordinary_spawn_before_scoped(dispatch_case):
    c = dispatch_case

    def script(c):
        return [
            c.reply("spawn_subagent", {"task": "ordinary child"}),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "R1"},
            ),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "R2"},
            ),
        ]

    c.run(reviewers=2, parent_script=script, child_script=["done"])
    assert len({a.run_id for a, r in c.calls if a.kind == "subagent"}) == 2
    assert c.outcome.subagents_spawned == 2
    assert "child_budget_unavailable" in json.dumps(c.calls[-1][1]["messages_payload"])


def test_unknown_scoped_review_returns_refusal_without_cleanup_exception(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    original = agent_service.run_agent_loop
    results = []

    def loop(config, messages, schemas, deps):
        if config.review_binding is None:
            results.append(deps.spawn(review_id="missing-review", review_slot_id="R1"))
        return original(config, messages, schemas, deps)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    c.run()
    assert len(results) == 1 and not results[0].ok
    assert c.outcome.subagents_spawned == 1


@pytest.mark.parametrize("inline", [False, True])
def test_revocation_at_attachment_settles_precreated_run_without_dispatch(
    dispatch_case, monkeypatch, inline
):
    c = dispatch_case
    attempts = []

    def revoke(c):
        original = c.primary.attach_review_run

        def attach(claim, run_id):
            attempts.append(run_id)
            c.issuer.close(c.owner)
            return original(claim, run_id)

        monkeypatch.setattr(c.primary, "attach_review_run", attach)

    c.run(inline=inline, replay=True, prepare_hook=revoke)
    assert len(attempts) == 1
    assert not [a for a, _ in c.calls if a.kind == "subagent"]
    assert c.outcome.status == "done"
    assert c.outcome.subagents_spawned == 0
    assert c.db.get_run(attempts[0])["status"] == "error"
    if not inline:
        assert c.runtime._fleet.live_count() == 0


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_accepted_child_failure_keeps_charge_and_cannot_relaunch(
    dispatch_case, monkeypatch, inline, native
):
    from dataclasses import replace

    from tldw_chatbook.Agents.agent_models import SpawnAdmissionReplay

    c = dispatch_case
    receipts = []
    original_loop = agent_service.run_agent_loop

    def loop(config, messages, schemas, deps):
        if config.review_binding is None:
            original_spawn = deps.spawn

            def record(*args, **kwargs):
                result = original_spawn(*args, **kwargs)
                receipts.append(result)
                return result

            deps = replace(deps, spawn=record)
        return original_loop(config, messages, schemas, deps)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)

    def fail():
        raise RuntimeError("scripted accepted provider failure")

    c.run(inline=inline, replay=True, native=native, child_script=[fail])
    children = [a for a, _ in c.calls if a.kind == "subagent"]
    assert len(children) == 1
    assert c.outcome.subagents_spawned == 1
    assert c.db.get_run(children[0].run_id)["status"] == "error"
    with c.store.lease(c.prepared.review_id) as context:
        assert context.launch_claims["R1"].run_id == children[0].run_id
        assert sum(slot == "R1" for slot, _ in context.streams) == 1

    assert len(receipts) == 2 and isinstance(receipts[1], SpawnAdmissionReplay)
    if inline:
        assert not receipts[0].ok and not receipts[1].ok
        assert receipts[0].error == receipts[1].error
        assert children[0].run_id in receipts[0].error
        assert (
            "error" in receipts[0].error
            and "review_launch_pending" not in receipts[1].error
        )
        parent = next(r for a, r in reversed(c.calls) if a.kind == "primary")
        assert json.dumps(parent["messages_payload"]).count(children[0].run_id) == 2


@pytest.mark.parametrize("started_before_error", [False, True])
def test_fleet_thread_start_refusal_retries_only_when_definitely_unstarted(
    dispatch_case, monkeypatch, started_before_error
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = dispatch_case
    original_start = threading.Thread.start
    refused = []

    def start(thread):
        if thread.name.startswith("fleet-") and not refused:
            with c.store.lease(c.prepared.review_id) as context:
                claim = context.launch_claims["R1"]
                refused.append((claim.run_id, claim.binding))
            if started_before_error:
                original_start(thread)
            raise RuntimeError("scripted fleet start failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    c.run(replay=True)
    children = [a for a, _ in c.calls if a.kind == "subagent"]
    assert children, c.outcome
    assert len({a.run_id for a in children}) == 1
    assert c.outcome.subagents_spawned == 1
    assert c.runtime._fleet.live_count() == 0
    old_run, old_binding = refused[0]
    with c.store.lease(c.prepared.review_id) as context:
        assert sum(slot == "R1" for slot, _ in context.streams) == 1
        if started_before_error:
            assert children[0].run_id == old_run
            assert context.launch_claims["R1"].run_id == old_run
        else:
            assert children[0].run_id != old_run
            assert c.db.get_run(old_run)["status"] == "error"
            assert old_run not in c.store._tab.runs
            assert ("R1", old_binding.attempt_id) not in context.streams
            with pytest.raises(ReviewRefused, match="stale_attempt"):
                c.store.bind_attempt(old_run, old_binding)
