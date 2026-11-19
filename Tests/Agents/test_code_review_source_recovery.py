"""Invalid captured ranges remain refused but permit bounded verifier recovery."""

import json
from copy import deepcopy

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_code_review_completion_recovery import invoke
from Tests.Agents.test_code_review_conflict_recovery import (
    stream_for,
    submit,
    verifier_summary,
)
from Tests.Agents.test_code_review_verification import seed
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_STUCK,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
pytestmark = pytest.mark.bootstrap_profile


def resolve(provider, review_id, side="worktree"):
    result = invoke(
        provider,
        "read",
        review_id,
        view="source",
        source_query={
            "op": "resolve",
            "relative_path": "main.py",
            "side": side,
        },
    )
    assert result.ok
    return json.loads(result.content)["records"][0]["source_ref"]


@pytest.mark.parametrize("lines", [0, 7])
@pytest.mark.parametrize("role,slot", [("reviewer", "R1"), ("verifier", "V1")])
def test_invalid_range_feedback_retries_same_captured_document(
    review_case, lines, role, slot
):
    c = review_case
    captured = "captured line\n" * lines
    c.file.write_text(captured)
    prepared = c.prepare()
    with c.participant(prepared, role=role, slot=slot) as provider:
        ref = resolve(provider, prepared.review_id)
        # Feedback describes captured bytes, never a newer mutable file.
        c.file.write_text("newer mutable line\n" * 20)
        refused = invoke(
            provider,
            "read",
            prepared.review_id,
            view="source",
            document_ref=ref,
            source_range={"start_line": 1, "end_line": 10},
        )
        assert not refused.ok and refused.error == "invalid_source_range"
        assert refused.content, "A range refusal must identify a usable bounded retry"
        feedback = json.loads(refused.content)
        assert feedback["line_count"] == lines
        assert feedback["retry"] == {
            "review_id": prepared.review_id,
            "view": "source",
            "document_ref": ref,
        }
        assert ("checkpoint" in feedback) == (role == "verifier")
        assert (
            "newer mutable" not in refused.content
            and "captured line" not in refused.content
        )
        recovered = provider.invoke("code_review:code_review_read", feedback["retry"])
        assert recovered.ok, recovered.error
        page = json.loads(recovered.content)
        assert page["text"] == captured and page["cursor"] is None
        if lines:
            assert (page["start_line"], page["end_line"]) == (1, 7)


@pytest.mark.parametrize("failure", ["foreign", "permission", "capacity"])
def test_range_recovery_cannot_expose_unadmitted_or_oversized_feedback(
    review_case, failure
):
    c = review_case
    c.file.write_text("captured line\n" * 7)
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        ref = resolve(provider, prepared.review_id)
        if failure == "foreign":
            ref = "not-an-admitted-reference"
        elif failure == "permission":
            provider.service.source_permission = lambda *args: "permission_denied"
        else:
            provider.service.result_limit = 32
        result = invoke(
            provider,
            "read",
            prepared.review_id,
            view="source",
            document_ref=ref,
            source_range={"start_line": 1, "end_line": 10},
        )
        assert not result.ok and result.content == ""
        if failure == "capacity":
            assert result.error == "invalid_source_range"
        elif failure == "permission":
            assert result.error == "permission_denied"
        else:
            assert result.error == "invalid_source_reference"


@pytest.fixture
def accepted_case(review_case):
    c = review_case
    c.file.write_text("captured line\n" * 7)
    prepared = seed(c, 4)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        provider.service.verifier_input_tokens = 10000
        provider.service.verifier_estimate = lambda text: len(text) // 4
        batch = invoke(provider, "read", prepared.review_id, view="batch")
        assert batch.ok
        records = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": "confirmed",
                "reason": "captured evidence",
                "evidence": row["claim"]["evidence"],
            }
            for row in json.loads(batch.content)["records"]
        ]
        case = c, prepared, provider, records
        assert len(records) == 4 and submit(case, records).ok
        yield case


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("recover", [False, True])
def test_range_feedback_reaches_model_and_completes_without_relaxing_failure_guard(
    accepted_case, native, recover
):
    c, prepared, provider, records = case = accepted_case
    ref = resolve(provider, prepared.review_id)
    base_ref = resolve(provider, prepared.review_id, "base")
    stream = stream_for(case)
    decisions = deepcopy(stream.verification.dispositions)
    accepted_ack = submit(case, records).content
    requests, results, reviewed = [], [], []
    feedback = None

    def model(messages, active):
        nonlocal feedback
        step = len(requests)
        requests.append(deepcopy(messages))
        name = "code_review_read"
        if step == 0 or not recover:
            args = {
                "review_id": prepared.review_id,
                "view": "source",
                "document_ref": ref,
                "source_range": {"start_line": 1, "end_line": 10},
            }
            if not recover and step == 1:
                args["document_ref"] = base_ref
            elif not recover and step == 2:
                args["source_query"] = None
        elif step == 1:
            assert results[-1].error == "invalid_source_range"
            assert results[-1].content, "Missing actionable source-range feedback"
            assert any(
                results[-1].content in row.get("content", "") for row in messages
            )
            feedback = json.loads(results[-1].content)
            args = feedback["retry"]
        elif step == 2:
            assert results[-1].ok
            page = json.loads(results[-1].content)
            assert page["text"] == "captured line\n" * 7 and page["cursor"] is None
            args = feedback["checkpoint"]
        elif step == 3:
            assert results[-1].ok
            page = json.loads(results[-1].content)
            assert page["records"][0]["all_units_disposed"]
            assert not page["records"][0]["pending_unit_ids"]
            name = "code_review_submit"
            args = {
                "review_id": prepared.review_id,
                "kind": "verifier_complete",
                "records": [],
                "expected_revision": page["submission_revision"],
                "submission_id": "complete",
            }
        else:
            assert step == 4 and results[-1].ok
            return ModelTurn(text="Verification complete")
        call = ToolCall(name, args, f"call-{step}")
        if not native:
            return ModelTurn(text=fence(name, args))
        return ModelTurn(
            text="",
            tool_calls=(call,),
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            },
        )

    def dispatch(call):
        result = provider.invoke("code_review:" + call.name, call.args)
        results.append(result)
        return result

    outcome = run_agent_loop(
        AgentConfig(
            model="offline",
            system_prompt="Verify",
            native_tools=native,
            allowed_tools=("code_review_read", "code_review_submit"),
            budget=RunBudget(max_steps=24),
        ),
        [{"role": "user", "content": "Complete the accepted verification"}],
        [
            provider.load_schema("code_review:" + name)
            for name in ("code_review_read", "code_review_submit")
        ],
        LoopDeps(
            call_model=model,
            invoke_tool=dispatch,
            spawn=lambda *args: pytest.fail("No child launches"),
            find_tools=lambda query: [],
            load_schemas=lambda *args: [],
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            review_tool_calls=lambda calls: reviewed.extend(calls) or {},
        ),
    )
    assert stream.verification.dispositions == decisions and not stream.rejected
    assert submit(case, records).content == accepted_ack
    assert verifier_summary(case)["seal_receipt"] is None
    if not recover:
        assert outcome.status == RUN_STUCK and len(results) == len(reviewed) == 3
        assert any("failed 3 times in a row" in step.summary for step in outcome.steps)
        assert not stream.completion_requested and stream.revision == 1
        return
    assert outcome.status == RUN_DONE and len(results) == len(reviewed) == 4
    assert stream.revision == 2 and stream.completion_requested
    provider.service.capture_review_output(
        provider.service.actor.run_id, "offline verification", terminal=True
    )
    assert verifier_summary(case)["seal_receipt"] is None
    provider.service.settle_review_run(
        provider.service.actor.run_id,
        terminal_status="done",
        receipt_id="offline-terminal",
    )
    assert verifier_summary(case)["report_status"] == "sealed"
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert len(expanded) == len({row["original_id"] for row in expanded}) == 4
