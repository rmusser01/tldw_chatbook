"""Negative controls for the project-chat UAT's independent evidence checks."""

from __future__ import annotations

import copy
import json
import urllib.request

import pytest


def evidence():
    return {
        "code": "fixture-unique-9471",
        "alias": "folder-abcdef123456",
        "final_text": "Project summary: fixture-unique-9471",
        "persisted_text": "Project summary: fixture-unique-9471",
        "painted_text": "Project summary: fixture-unique-9471",
        "steps": [
            {
                "kind": "tool_call",
                "tool_name": "fs_read",
                "args": {"root_alias": "folder-abcdef123456", "path": "README.md"},
            },
            {
                "kind": "tool_result",
                "tool_name": "fs_read",
                "tool_outcome": "success",
                "result": "1: Verification code: fixture-unique-9471",
            },
        ],
    }


def test_actual_read_and_all_three_answer_views_are_required():
    from Tests.uat_project_chat_support import assert_project_evidence

    assert_project_evidence(**evidence())


@pytest.mark.parametrize("view", ["final_text", "persisted_text", "painted_text"])
def test_code_only_in_tool_result_cannot_pass_final_answer(view):
    from Tests.uat_project_chat_support import assert_project_evidence

    row = evidence()
    row[view] = "I cannot inspect your project."
    with pytest.raises(AssertionError, match=view):
        assert_project_evidence(**row)


@pytest.mark.parametrize(
    "defect", ["wrong-root", "empty", "failed", "no-read", "wrong-path"]
)
def test_canned_final_answer_cannot_replace_real_project_evidence(defect):
    from Tests.uat_project_chat_support import assert_project_evidence

    row = evidence()
    if defect == "wrong-root":
        row["steps"][0]["args"]["root_alias"] = "scratch"
    elif defect == "wrong-path":
        row["steps"][0]["args"]["path"] = "decoy.md"
    elif defect == "empty":
        row["steps"][1]["result"] = ""
    elif defect == "failed":
        row["steps"][1]["tool_outcome"] = "failed"
    else:
        row["steps"] = []
    with pytest.raises(AssertionError, match="bound README"):
        assert_project_evidence(**row)


def request_payload(native=False):
    schemas = [
        {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
        for name in ("find_tools", "load_tools")
    ]
    return {
        "model": "uat-project-model",
        "messages": [
            {
                "role": "system",
                "content": "Bound root: folder-abcdef123456\n"
                + ("\n".join(json.dumps(s) for s in schemas) if not native else ""),
            }
        ],
        "tools": [{"type": "function", "function": s} for s in schemas]
        if native
        else [],
    }


def test_native_variant_rejects_schema_names_only_in_prose():
    from Tests.uat_project_chat_support import ProjectChatModel

    with pytest.raises(AssertionError, match="native schemas"):
        ProjectChatModel("native").reply(request_payload())


def test_unexpected_generation_does_not_advance_without_tool_result():
    from Tests.uat_project_chat_support import ProjectChatModel

    model = ProjectChatModel("fenced")
    payload = request_payload()
    assert "find_tools" in model.reply(payload)["content"]
    with pytest.raises(AssertionError, match="find_tools result"):
        model.reply(copy.deepcopy(payload))


@pytest.mark.loopback_network
def test_metadata_probes_do_not_consume_model_steps_and_server_retires():
    from Tests.uat_project_chat_support import project_chat_server

    with project_chat_server("fenced") as server:
        for route in (
            "/health",
            "/v1/models",
            "/props",
            "/props?model=uat-project-model&autoload=false",
        ):
            with urllib.request.urlopen(server.url + route, timeout=3) as response:
                assert response.status == 200
        assert server.model.requests == []
        worker = server.thread
    assert not worker.is_alive()


@pytest.mark.loopback_network
def test_owned_port_guard_records_even_a_caught_refusal():
    import socket
    from Tests.uat_project_chat_support import only_owned_server, project_chat_server

    with project_chat_server("fenced") as server:
        with only_owned_server(server.port) as attempts:
            with socket.socket() as client:
                with pytest.raises(OSError, match="UAT destination refused"):
                    client.connect(("127.0.0.1", 1))
            with urllib.request.urlopen(server.url + "/health", timeout=3) as response:
                assert response.status == 200
        assert attempts == [("127.0.0.1", 1)]


def test_fenced_tool_names_in_user_prose_do_not_disclose_schemas():
    from Tests.uat_project_chat_support import ProjectChatModel

    payload = request_payload()
    payload["messages"][0]["role"] = "user"
    with pytest.raises(AssertionError, match="find_tools schema absent"):
        ProjectChatModel("fenced").reply(payload)


@pytest.mark.parametrize("read_result", ["", "README has no verification evidence"])
def test_model_cannot_use_an_earlier_message_to_fabricate_the_read_answer(read_result):
    from Tests.uat_project_chat_support import ProjectChatModel

    payload = request_payload()
    model = ProjectChatModel("fenced")
    for name, result in (
        ("find_tools", "fs_list fs_read"),
        ("load_tools", "loaded: fs_list, fs_read"),
        ("fs_list", "README.md"),
    ):
        payload["messages"].append(model.reply(payload))
        payload["messages"].append(
            {"role": "user", "content": f"Tool result for {name}: {result}"}
        )
        if name == "load_tools":
            payload["messages"][0]["content"] += "\n" + "\n".join(
                json.dumps(
                    {
                        "name": tool,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "root_alias": {"type": "string"},
                            },
                        },
                    }
                )
                for tool in ("fs_list", "fs_read")
            )
    payload["messages"].append(model.reply(payload))
    payload["messages"].append(
        {"role": "assistant", "content": "An old answer: Verification code: stale-code"}
    )
    payload["messages"].append(
        {"role": "user", "content": f"Tool result for fs_read: {read_result}"}
    )
    with pytest.raises(AssertionError, match="README read did not supply"):
        model.reply(payload)


def test_rejected_provider_request_is_retained_for_failure_diagnostics():
    from Tests.uat_project_chat_support import ProjectChatModel

    model = ProjectChatModel("native")
    payload = request_payload()
    with pytest.raises(AssertionError):
        model.reply(payload)
    assert model.received == [payload]
    assert model.requests == []


@pytest.mark.parametrize("model_id", [None, "wrong-model"])
def test_script_rejects_wrong_or_omitted_model(model_id):
    from Tests.uat_project_chat_support import ProjectChatModel

    payload = request_payload()
    if model_id is None:
        payload.pop("model", None)
    else:
        payload["model"] = model_id
    with pytest.raises(AssertionError, match="model identity"):
        ProjectChatModel("fenced").reply(payload)


@pytest.mark.parametrize(
    "stage,tool",
    [
        ("find", "find_tools"),
        ("load", "load_tools"),
        ("list", "fs_list"),
        ("read", "fs_read"),
    ],
)
def test_script_cannot_compensate_for_missing_argument_schemas(stage, tool):
    from Tests.uat_project_chat_support import ProjectChatModel

    model = ProjectChatModel("native")
    # Isolate validation of the response about to be emitted; end-to-end cases
    # independently verify the complete sequence and tool provenance.
    model.sequence = [stage]
    payload = request_payload(native=True)
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": tool,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(AssertionError, match="argument schema"):
        model.reply(payload)


@pytest.mark.parametrize(
    "defect", ["missing-anchor", "wrong-anchor", "wrong-conversation", "wrong-message"]
)
def test_run_and_saved_answer_must_share_the_expected_owner(defect):
    from Tests.uat_project_chat_support import assert_project_ownership

    run = {"assistant_message_id": "answer-1", "conversation_id": "chat-1"}
    saved = {"id": "answer-1", "conversation_id": "chat-1"}
    assert_project_ownership(run, saved, "chat-1", "answer-1")
    if defect == "missing-anchor":
        run["assistant_message_id"] = None
    elif defect == "wrong-anchor":
        run["assistant_message_id"] = "answer-2"
    elif defect == "wrong-conversation":
        saved["conversation_id"] = "chat-2"
    else:
        saved["id"] = "answer-2"
    with pytest.raises(AssertionError, match="owner"):
        assert_project_ownership(run, saved, "chat-1", "answer-1")
