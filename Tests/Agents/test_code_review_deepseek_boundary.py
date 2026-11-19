"""Scripted DeepSeek wire replies through native review completion and accounting."""

import inspect
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


@pytest.mark.parametrize("recover", [True, False])
def test_interrupted_deepseek_turn_retains_usage_and_cannot_seal_by_itself(
    dispatch_case, monkeypatch, recover
):
    c = dispatch_case
    c.app_config["api_settings"]["deepseek"] = {"api_key": "fixture-key"}
    requests = []
    native_init = agent_service.AgentService.__init__

    def init(service, *args, **kwargs):
        scripted = kwargs["chat_call"]

        def gateway(**call):
            response = scripted(**call)
            if current_run_actor().kind != "subagent":
                return response
            choice = response["choices"][0]
            message = choice["message"]
            message["role"] = "assistant"
            choice["index"] = 0
            for index, tool_call in enumerate(message.get("tool_calls", [])):
                tool_call["index"] = index
            choice["finish_reason"] = (
                "length"
                if message.get("reasoning_content")
                else "tool_calls"
                if message.get("tool_calls")
                else "stop"
            )
            response["usage"] = {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
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
            return result

        kwargs["chat_call"] = gateway
        native_init(service, *args, **kwargs)

    monkeypatch.setattr(agent_service.AgentService, "__init__", init)
    interrupted = {"content": None, "reasoning_content": "PRIVATE-THINKING-CANARY"}
    child_script = [interrupted, interrupted.copy()]
    if recover:
        child_script = [
            interrupted,
            lambda: c.reply(
                "code_review_read",
                {
                    "review_id": c.prepared.review_id,
                    "view": "source",
                    "source_query": {
                        "op": "list",
                        "relative_dir": ".",
                        "side": "worktree",
                    },
                },
            ),
            lambda: c.reply("load_tools", {"ids": ["code_review:code_review_submit"]}),
            lambda: c.reply(
                "code_review_submit",
                {
                    "review_id": c.prepared.review_id,
                    "kind": "findings",
                    "records": [finding(c, c.prepared)],
                    "expected_revision": 0,
                    "submission_id": "finding",
                },
            ),
            lambda: c.reply(
                "code_review_submit",
                {
                    "review_id": c.prepared.review_id,
                    "kind": "reviewer_complete",
                    "records": [coverage(c, c.prepared, empty=False)],
                    "expected_revision": 1,
                    "submission_id": "complete",
                },
            ),
            "Original review complete",
        ]
    c.run(
        native=True, inline=True, destinations=("deepseek",), child_script=child_script
    )
    slot = summary(c, c.prepared)["slots"][0]
    assert slot["report_status"] == ("sealed" if recover else "incomplete")
    assert slot["terminal_status"] == ("done" if recover else "stuck")
    assert slot["submission_revision"] == (2 if recover else 0)
    assert not slot["rejected_submissions"]
    assert len(requests) == (6 if recover else 2)
    child = c.db.get_run(slot["run_id"])
    assert child["budget_tokens"] == 32 * len(requests)
    assert "PRIVATE-THINKING-CANARY" not in repr(child)
    if recover:
        tool_results = [
            row for row in requests[-1]["messages"] if row["role"] == "tool"
        ]
        assert tool_results and '"submission_revision":2' in tool_results[-1]["content"]


@pytest.mark.parametrize("abort", ["cancel", "exception"])
def test_aborted_source_read_distinguishes_settled_from_unknown_accounting(
    dispatch_case, monkeypatch, abort
):
    """Replay the captured 3-turn/4-call failure without any provider network."""
    c = dispatch_case
    c.app_config["api_settings"]["deepseek"] = {"api_key": "fixture-key"}
    failed = False
    usages = iter([(2162, 120), (2449, 112), (2820, 255)])
    responses = []
    native_init = agent_service.AgentService.__init__

    def init(service, *args, **kwargs):
        scripted = kwargs["chat_call"]

        def gateway(**call):
            if current_run_actor().kind != "subagent":
                return scripted(**call)
            if failed:
                raise RuntimeError("diagnostic stopped before next dispatch")
            response = scripted(**call)
            prompt, completion = next(usages)
            response["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
            responses.append(response)
            return response

        kwargs["chat_call"] = gateway
        native_init(service, *args, **kwargs)

    monkeypatch.setattr(agent_service.AgentService, "__init__", init)
    native_turn = agent_service.AgentService.run_turn

    def run_turn(service, **kwargs):
        kwargs["should_cancel"] = lambda: failed and abort == "cancel"
        return native_turn(service, **kwargs)

    monkeypatch.setattr(agent_service.AgentService, "run_turn", run_turn)
    native_invoke = ToolCatalogRegistry.invoke_by_name

    def invoke(registry, name, args, **kwargs):
        nonlocal failed
        result = native_invoke(registry, name, args, **kwargs)
        if result.error == "invalid_source_side":
            failed = True
        return result

    monkeypatch.setattr(ToolCatalogRegistry, "invoke_by_name", invoke)

    def list_side(side):
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "source",
                "source_query": {"op": "list", "relative_dir": ".", "side": side},
            },
        )

    def read_base():
        with c.store.lease(c.prepared.review_id) as context:
            ref = next(
                doc.source_ref
                for doc in context.source_state.documents.values()
                if doc.relative_path == "main.py" and doc.side == "base"
            )
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "source",
                "document_ref": ref,
            },
        )

    def two_sides():
        response = list_side("worktree")
        call = list_side("head")["tool_calls"][0]
        call["id"] = "head-call"
        response["tool_calls"].append(call)
        return response

    c.run(
        native=True,
        inline=True,
        destinations=("deepseek",),
        child_script=[lambda: list_side("base"), read_base, two_sides],
    )
    slot = summary(c, c.prepared)["slots"][0]
    child = c.db.get_run(slot["run_id"])
    assert failed and len(responses) == 3
    assert child["status"] == ("cancelled" if abort == "cancel" else "error")
    assert child["budget_tokens"] == (7918 if abort == "cancel" else None)
    assert slot["report_status"] == "incomplete" and slot["seal_receipt"] is None
    assert slot["submission_revision"] == 0 and not slot["rejected_submissions"]
