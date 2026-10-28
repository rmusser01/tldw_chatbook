"""Review children have a closed capability set in schemas and dispatch."""

import json

import pytest

from Tests.Agents import test_code_review_dispatch as dispatch_fixtures

dispatch_case = dispatch_fixtures.dispatch_case
review_case = dispatch_fixtures.review_case
source_case = dispatch_fixtures.source_case

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("native", [False, True])
def test_review_child_never_receives_personal_or_primary_context(dispatch_case, native):
    c = dispatch_case.run(native=native)
    children = [request for actor, request in c.calls if actor.kind == "subagent"]
    assert children, c.outcome
    text = json.dumps(children)
    assert "PERSONAL-CANARY" not in text
    assert "PRIMARY-PEER-CANARY" not in text
    assert "PRESET-0" in text
    assert "report_to_supervisor" not in text
    assert "send_to_peer" not in text


@pytest.mark.parametrize("native", [False, True])
def test_all_guessed_runtime_and_external_capabilities_are_denied(
    dispatch_case, native
):
    from tldw_chatbook.Agents.agent_models import RUNTIME_TOOL_NAMES

    c = dispatch_case
    forbidden = sorted(set(RUNTIME_TOOL_NAMES) - {"find_tools", "load_tools"}) + [
        "fs_read",
        "fs_write",
        "shell_exec",
        "python",
        "virtual_cli",
        "library_search",
        "profile_read",
    ]

    def first():
        calls = [
            {"name": name, "args": {"task": "nested", "path": "/tmp/canary"}}
            for name in forbidden
        ]
        if native:
            from Tests.Agents.test_agent_service import native_call

            return {
                "content": None,
                "tool_calls": [
                    native_call(x["name"], x["args"], f"deny-{i}")
                    for i, x in enumerate(calls)
                ],
            }
        from Tests.Agents.test_agent_service import fence

        return "\n".join(fence(x["name"], x["args"]) for x in calls)

    if native:
        script = [first, "done"]
    else:
        from Tests.Agents.test_agent_service import fence

        script = [
            fence(name, {"task": "nested", "path": "/tmp/canary"}) for name in forbidden
        ] + ["done"]
    c.run(native=native, child_script=script)
    children = [r for a, r in c.calls if a.kind == "subagent"]
    assert len(children) == (2 if native else len(forbidden) + 1)
    result_text = json.dumps(children[-1]["messages_payload"])
    assert result_text.count("role_denied") == len(forbidden)
    schemas = json.dumps(children[0].get("tools", []))
    assert all('"name": "' + name + '"' not in schemas for name in forbidden)


@pytest.mark.parametrize("native", [False, True])
def test_nested_guidance_precedes_complete_source_pages(
    dispatch_case, monkeypatch, native
):
    from Tests.Agents.test_code_review_dispatch import model_pages
    from Tests.Agents.test_project_instruction_runtime import _snapshot
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.project_instruction_runtime import (
        InstructionActivationLedger,
    )

    c = dispatch_case
    nested = c.root / "nested"
    nested.mkdir()
    (c.root / "AGENTS.md").write_text("root-secret")
    (nested / "AGENTS.md").write_text("NESTED-GUIDANCE-CANARY")
    text = "".join(f"evidence line {i:03d}\n" for i in range(80))
    (nested / "helper.py").write_text(text)
    c.working_tree_target = c.working_tree_target.model_copy(
        update={"context_paths": ("nested/helper.py",)}
    )
    ledger = InstructionActivationLedger(_snapshot(c.root), nested_max_bytes=4096)
    delivered = []
    original_mark = ledger.mark_payload_sent

    def observe(receipt, rows):
        delivered.append(receipt)
        return original_mark(receipt, rows)

    monkeypatch.setattr(ledger, "mark_payload_sent", observe)
    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: False if key == "run_log_enabled" else default,
    )
    pages = []

    def drive():
        from tldw_chatbook.Agents.run_context import current_run_actor

        actor = current_run_actor()
        request = next(r for a, r in reversed(c.calls) if a == actor)
        seen = model_pages(request)
        args = {"review_id": c.prepared.review_id, "view": "source"}
        if not seen:
            args["source_query"] = {
                "op": "resolve",
                "relative_path": "nested/helper.py",
                "side": "worktree",
            }
        else:
            page = seen[-1]
            if page.get("source_ref"):
                pages.append(page)
                if not page["cursor"]:
                    return "done"
                args["cursor"] = page["cursor"]
            else:
                args["document_ref"] = page["records"][0]["source_ref"]
        return c.reply("code_review_read", args)

    c.run(native=native, limit=1024, ledger=ledger, child_script=[drive] * 40)
    assert pages and "".join(p["text"] for p in pages) == text, json.dumps(
        [r["messages_payload"] for a, r in c.calls if a.kind == "subagent"], indent=2
    )
    assert all(
        len(json.dumps(p, ensure_ascii=False, separators=(",", ":"))) <= 1024
        for p in pages
    )
    assert all(p["submission_revision"] == 0 and p["revision"] == 0 for p in pages)
    assert len({p["cursor"] for p in pages if p["cursor"]}) == len(pages) - 1
    children = [r for a, r in c.calls if a.kind == "subagent"]
    guidance_index = next(
        i
        for i, r in enumerate(children)
        if "NESTED-GUIDANCE-CANARY" in json.dumps(r["messages_payload"])
    )
    evidence_index = next(
        i
        for i, r in enumerate(children)
        if "evidence line 000" in json.dumps(r["messages_payload"])
    )
    assert guidance_index < evidence_index
    import hashlib

    digest = hashlib.sha256(b"NESTED-GUIDANCE-CANARY").hexdigest()
    assert sum(digest in receipt.source_digests for receipt in delivered) == 1
    assert "NESTED-GUIDANCE-CANARY" not in json.dumps(
        [c.db.get_run(row["id"]) for row in c.db.list_runs("chat")], default=str
    )


def test_cycle_projection_preserves_call_identity_without_bodies(dispatch_case):
    from tldw_chatbook.Agents.agent_models import ToolCall

    provider = dispatch_case.provider
    variants = [
        {"cursor": "one"},
        {"cursor": "two"},
        {"document_ref": "source", "source_range": {"start_line": 1, "end_line": 2}},
        {"source_query": {"op": "search", "text": "secret-body"}},
    ]
    projections = [
        provider.project_tool_record("cycle", ToolCall("code_review_read", args), None)
        for args in variants
    ]
    assert len({repr(p.arguments) for p in projections}) == len(variants)
    assert projections[0] == provider.project_tool_record(
        "cycle", ToolCall("code_review_read", variants[0]), None
    )
    assert "secret-body" not in repr(projections)


@pytest.mark.parametrize("native", [False, True])
def test_oversized_source_line_is_explicit_incomplete_coverage(dispatch_case, native):
    c = dispatch_case
    c.helper.write_text("INDIVISIBLE-CANARY" * 200 + "\n")

    def read():
        with c.store.lease(c.prepared.review_id) as context:
            ref = next(
                key
                for key, source in context.source_state.documents.items()
                if source.relative_path == "helpers.py" and source.side == "worktree"
            )
        c.oversized_ref = ref
        return c.reply(
            "code_review_read",
            {"review_id": c.prepared.review_id, "view": "source", "document_ref": ref},
        )

    def coverage():
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "coverage",
                "records": [
                    {
                        "schema_version": 1,
                        "inspected_refs": [],
                        "gaps": [
                            {
                                "reason": "source_span_too_large_for_tool_result",
                                "source_ref": c.oversized_ref,
                            }
                        ],
                        "empty_findings": False,
                    }
                ],
                "expected_revision": 0,
                "submission_id": "incomplete",
            },
        )

    c.run(native=native, limit=1024, child_script=[read, coverage, "incomplete"])
    children = [r for a, r in c.calls if a.kind == "subagent"]
    assert "source_span_too_large_for_tool_result" in json.dumps(
        children[-1]["messages_payload"]
    )
    assert "INDIVISIBLE-CANARY" not in json.dumps(children)
    with c.store.lease(c.prepared.review_id) as context:
        stream = next(s for (slot, _), s in context.streams.items() if slot == "R1")
        assert stream.revision == 1 and "incomplete" in stream.submissions


@pytest.mark.parametrize("native", [False, True])
def test_admitted_guidance_does_not_bypass_later_guard(dispatch_case, native):
    from Tests.Agents.test_project_instruction_runtime import _snapshot
    from tldw_chatbook.Agents.project_instruction_runtime import (
        InstructionActivationLedger,
    )

    c = dispatch_case
    nested = c.root / "nested"
    nested.mkdir()
    (c.root / "AGENTS.md").write_text("root-secret")
    (nested / "AGENTS.md").write_text("GUIDANCE-IS-NOT-PERMISSION")
    (nested / "helper.py").write_text("FORBIDDEN-SOURCE-CANARY")
    c.working_tree_target = c.working_tree_target.model_copy(
        update={"context_paths": ("nested/helper.py",)}
    )
    ledger = InstructionActivationLedger(_snapshot(c.root), nested_max_bytes=4096)
    observed = []

    def guard(calls, run_id):
        if run_id != "primary":
            observed.append(ledger.activation_revision)
            return {call.call_id or call.name: "permission-denied" for call in calls}
        return {}

    c.guard = guard

    def read():
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "source",
                "source_query": {
                    "op": "resolve",
                    "relative_path": "nested/helper.py",
                    "side": "worktree",
                },
            },
        )

    c.run(native=native, ledger=ledger, child_script=[read, read, "blocked"])
    children = [r for a, r in c.calls if a.kind == "subagent"]
    assert "GUIDANCE-IS-NOT-PERMISSION" in json.dumps(children)
    assert "permission-denied" in json.dumps(children[-1]["messages_payload"])
    assert "FORBIDDEN-SOURCE-CANARY" not in json.dumps(children)
    assert observed and all(value > 0 for value in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_hook_narrowed_child_pages_and_submission_remain_complete(
    dispatch_case, native
):
    import asyncio

    from Tests.Agents.code_review_helpers import finding_payload
    from Tests.Agents.test_code_review_dispatch import model_pages
    from Tests.Agents.test_hooks_v2_execution import command
    from tldw_chatbook.Agents.activation import worker_guard
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.run_context import current_run_actor

    c = dispatch_case
    source_text = "".join(f"hook-narrowed evidence {i:03d}\n" for i in range(80))
    c.helper.write_text(source_text)
    hook = command(
        'print(\'{"version":2,"decision":"pass","child_limits":{"budget_caps":{"max_tool_result_chars":1024}}}\')',
        name="SubagentStart",
        effects=["child_limits"],
        required=True,
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    pages = []
    submitted = False

    def drive():
        nonlocal submitted
        if submitted:
            return "done"
        actor = current_run_actor()
        request = next(r for a, r in reversed(c.calls) if a == actor)
        seen = model_pages(request)
        args = {"review_id": c.prepared.review_id, "view": "source"}
        if not seen:
            args["source_query"] = {
                "op": "resolve",
                "relative_path": "helpers.py",
                "side": "worktree",
            }
        else:
            page = seen[-1]
            if page.get("source_ref"):
                pages.append(page)
                if page["cursor"]:
                    args["cursor"] = page["cursor"]
                else:
                    with c.store.lease(c.prepared.review_id) as context:
                        source = context.sources[page["source_ref"]]
                    finding = finding_payload()
                    finding["locations"] = [
                        {
                            "source_ref": page["source_ref"],
                            "start_line": 1,
                            "end_line": 1,
                        }
                    ]
                    finding["evidence"] = [
                        dict(
                            finding["locations"][0], kind="source", digest=source.digest
                        )
                    ]
                    submitted = True
                    return c.reply(
                        "code_review_submit",
                        dict(
                            review_id=c.prepared.review_id,
                            kind="findings",
                            records=[finding],
                            expected_revision=0,
                            submission_id="hook-bounded",
                        ),
                    )
            else:
                args["document_ref"] = page["records"][0]["source_ref"]
        return c.reply("code_review_read", args)

    try:
        await asyncio.to_thread(
            worker_guard(c)(c.run),
            native=native,
            hooks_engine=engine,
            limit=16000,
            child_script=[drive] * 40,
        )
    finally:
        await engine.close()
    assert pages and "".join(p["text"] for p in pages) == source_text
    assert len(pages) > 1
    assert all(
        len(json.dumps(p, ensure_ascii=False, separators=(",", ":"))) <= 1024
        for p in pages
    )
    rows = [c.db.get_run(row["id"]) for row in c.db.list_runs("chat")]
    assert (
        next(r for r in rows if r["agent_kind"] == "primary")["budget"][
            "max_tool_result_chars"
        ]
        == 16000
    )
    assert (
        next(r for r in rows if r["agent_kind"] == "subagent")["budget"][
            "max_tool_result_chars"
        ]
        == 1024
    )
    assert c.primary.result_limit == 16000
    last = next(r for a, r in reversed(c.calls) if a.kind == "subagent")
    acks = []
    for row in last["messages_payload"]:
        content = row.get("content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(
                    content.removeprefix("Tool result for code_review_submit: ")
                )
            except ValueError:
                continue
            if isinstance(parsed, dict) and "accepted_ids" in parsed:
                acks.append(parsed)
    assert len(acks) == 1 and acks[0]["accepted_ids"] == ["R1-F1"]
    assert acks[0]["submission_revision"] == 1 and acks[0]["cursor"]
    with c.store.lease(c.prepared.review_id) as context:
        assert len(context.original_index) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_hook_narrowed_submission_refuses_before_unfittable_ack(
    dispatch_case, native
):
    import asyncio

    from Tests.Agents.code_review_helpers import coverage_payload
    from Tests.Agents.test_hooks_v2_execution import command
    from tldw_chatbook.Agents.activation import worker_guard
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    c = dispatch_case
    hook = command(
        'print(\'{"version":2,"decision":"pass","child_limits":{"budget_caps":{"max_tool_result_chars":96}}}\')',
        name="SubagentStart",
        effects=["child_limits"],
        required=True,
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())

    def submit():
        with c.store.lease(c.prepared.review_id) as context:
            ref = next(iter(context.sources))
        record = coverage_payload() | {"inspected_refs": [ref]}
        return c.reply(
            "code_review_submit",
            dict(
                review_id=c.prepared.review_id,
                kind="coverage",
                records=[record],
                expected_revision=0,
                submission_id="too-large-ack",
            ),
        )

    try:
        await asyncio.to_thread(
            worker_guard(c)(c.run),
            native=native,
            hooks_engine=engine,
            child_script=[submit, "done"],
        )
    finally:
        await engine.close()
    last = next(r for a, r in reversed(c.calls) if a.kind == "subagent")
    assert "review_response_capacity_unavailable" in json.dumps(
        last["messages_payload"]
    )
    assert "accepted_ids" not in json.dumps(last["messages_payload"])
    with c.store.lease(c.prepared.review_id) as context:
        stream = next(s for (slot, _), s in context.streams.items() if slot == "R1")
        assert stream.revision == 0 and not stream.submissions
        assert context.publication_sequence == 0


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_child_request_advertises_only_its_host_role(dispatch_case, native, verify):
    c = dispatch_case.run(native=native, verify=verify)
    children = [request for actor, request in c.calls if actor.kind == "subagent"]
    assert children, c.outcome
    views = (
        ["source", "batch", "verification_batch", "checkpoint"]
        if verify
        else ["source"]
    )
    kinds = (
        ["dispositions", "check_proposals", "verifier_complete"]
        if verify
        else ["findings", "coverage", "reviewer_complete"]
    )
    for request in children:
        if native:
            schemas = {
                tool["function"]["name"]: tool["function"] for tool in request["tools"]
            }
            assert (
                not {"code_review_guide", "code_review_prepare", "code_review_release"}
                & schemas.keys()
            )
            assert (
                schemas["code_review_read"]["parameters"]["properties"]["view"]["enum"]
                == views
            )
            assert (
                schemas["code_review_submit"]["parameters"]["properties"]["kind"][
                    "enum"
                ]
                == kinds
            )
            assert (
                "code_review_prepare" not in schemas["code_review_read"]["description"]
            )
        else:
            assert request.get("tools") is None
            instructions = " ".join(request["messages_payload"][0]["content"].split())
            assert request["messages_payload"][0]["role"] == "system"
            assert "```tool_call" in instructions
            assert "your reply MUST START with the fence as its first" in instructions
            assert '"name": "code_review_read"' in instructions
            assert '"name": "code_review_submit"' in instructions
            assert (
                '"view": { "enum": ' + " ".join(json.dumps(views, indent=2).split())
                in instructions
            )
            assert (
                '"kind": { "enum": ' + " ".join(json.dumps(kinds, indent=2).split())
                in instructions
            )
            assert (
                "Use submission_revision from your page or acknowledgment as expected_revision."
                in instructions
            )
            assert "code_review_guide" not in instructions
            assert "code_review_prepare" not in instructions
            assert "code_review_release" not in instructions
    pages = dispatch_fixtures.model_pages(children[-1])
    assert pages and pages[-1]["records"]
    assert pages[-1]["submission_revision"] == 0
