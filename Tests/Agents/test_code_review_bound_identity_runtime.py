"""Scripted DeepSeek omission through real requests, tools, settlement and DB."""

import inspect
import json
from unittest.mock import patch

import pytest

from Tests.Agents import code_review_helpers, test_code_review_dispatch
from Tests.Agents.test_code_review_reports import coverage, finding, summary
from Tests.LLM_Calls.test_groq_openrouter_migration_characterization import (
    _nonstreaming_ok_response,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.run_context import current_run_actor
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.Chat_Functions import chat_api_call

pytestmark = pytest.mark.bootstrap_profile
source_case = code_review_helpers.source_case
review_case = code_review_helpers.review_case
dispatch_case = test_code_review_dispatch.dispatch_case


@pytest.mark.parametrize("ending", ["complete", "cancel", "exception"])
def test_missing_review_id_completes_or_retains_honest_aborted_accounting(
    dispatch_case, monkeypatch, ending
):
    c = dispatch_case
    c.app_config["api_settings"]["deepseek"] = {"api_key": "fixture-key"}
    # Captured diagnostic token buckets, then two synthetic completion turns.
    usage = iter(
        [
            (2158, 120, 1536),
            (2454, 123, 2176),
            (2860, 2448, 2560),
            (5372, 320, 5120),
            (12, 20, 0),
            (12, 20, 0),
        ]
    )
    requests, totals, submissions = [], [], []
    coverage_accepted = False
    native_init = agent_service.AgentService.__init__

    def init(service, *args, **kwargs):
        scripted = kwargs["chat_call"]

        def gateway(**call):
            if current_run_actor().kind != "subagent":
                return scripted(**call)
            if coverage_accepted and ending == "exception":
                raise RuntimeError("diagnostic stopped before next dispatch")
            response = scripted(**call)
            choice = response["choices"][0]
            message = choice["message"]
            message["role"] = "assistant"
            choice["index"] = 0
            for index, tool in enumerate(message.get("tool_calls", [])):
                tool["index"] = index
            choice["finish_reason"] = (
                "tool_calls" if message.get("tool_calls") else "stop"
            )
            prompt, completion, cached = next(usage)
            response["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_cache_hit_tokens": cached,
                "prompt_cache_miss_tokens": prompt - cached,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
            wire = {
                key: value
                for key, value in call.items()
                if key in inspect.signature(chat_api_call).parameters
            }
            wire.update(api_key="fixture-key", streaming=False)
            with patch(
                "requests.Session.post",
                return_value=_nonstreaming_ok_response(response),
            ) as post:
                result = chat_api_call(**wire)
            requests.append(post.call_args.kwargs["json"])
            totals.append(result["usage"]["total_tokens"])
            return result

        kwargs["chat_call"] = gateway
        native_init(service, *args, **kwargs)

    monkeypatch.setattr(agent_service.AgentService, "__init__", init)
    native_turn = agent_service.AgentService.run_turn

    def run_turn(service, **kwargs):
        kwargs["should_cancel"] = lambda: coverage_accepted and ending == "cancel"
        return native_turn(service, **kwargs)

    monkeypatch.setattr(agent_service.AgentService, "run_turn", run_turn)
    native_invoke = ToolCatalogRegistry.invoke_by_name

    def invoke(registry, name, args, **kwargs):
        nonlocal coverage_accepted
        result = native_invoke(registry, name, args, **kwargs)
        if name == "code_review_submit":
            submissions.append((args.copy(), result))
            if args["kind"] == "coverage":
                coverage_accepted = result.ok
        return result

    monkeypatch.setattr(ToolCatalogRegistry, "invoke_by_name", invoke)

    def submit(kind, records, revision):
        args = {
            "kind": kind,
            "records": records,
            "expected_revision": revision,
            "submission_id": kind,
        }
        if kind == "findings":
            args["review_id"] = c.prepared.review_id
        return c.reply("code_review_submit", args)

    def read_source():
        with c.store.lease(c.prepared.review_id) as context:
            ref = next(iter(context.sources))
        return c.reply("code_review_read", {"view": "source", "document_ref": ref})

    script = [
        lambda: c.reply(
            "code_review_read",
            {
                "view": "source",
                "source_query": {"op": "list", "relative_dir": ".", "side": "worktree"},
            },
        ),
        read_source,
        lambda: submit(
            "findings", [finding(c, c.prepared, key=str(i)) for i in range(3)], 0
        ),
        lambda: submit("coverage", [coverage(c, c.prepared, empty=False)], 1),
    ]
    if ending == "complete":
        script += [
            lambda: submit(
                "reviewer_complete", [coverage(c, c.prepared, empty=False)], 2
            ),
            "Original review complete",
        ]
    c.run(native=True, inline=True, destinations=("deepseek",), child_script=script)
    assert coverage_accepted and all(result.ok for _, result in submissions)
    assert "review_id" not in submissions[1][0]
    assert json.loads(submissions[0][1].content)["submission_revision"] == 1
    assert json.loads(submissions[1][1].content)["submission_revision"] == 2
    with c.store.lease(c.prepared.review_id) as context:
        assert len(context.original_index) == 3
    slot = summary(c, c.prepared)["slots"][0]
    child = c.db.get_run(slot["run_id"])
    assert (
        child["status"]
        == {"complete": "done", "cancel": "cancelled", "exception": "error"}[ending]
    )
    assert sum(totals) == (15919 if ending == "complete" else 15855)
    assert child["budget_tokens"] == (None if ending == "exception" else sum(totals))
    assert slot["submission_revision"] == (3 if ending == "complete" else 2)
    assert not slot["rejected_submissions"]
    assert slot["report_status"] == ("sealed" if ending == "complete" else "incomplete")
    assert bool(slot["seal_receipt"]) == (ending == "complete")
    assert len(requests) == (6 if ending == "complete" else 4)
    if ending == "complete":
        assert slot["capture_receipt"] and slot["terminal_receipt"]
        tools = [row for row in requests[-1]["messages"] if row["role"] == "tool"]
        assert '"submission_revision":3' in tools[-1]["content"]
