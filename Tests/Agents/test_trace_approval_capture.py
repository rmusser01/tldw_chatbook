"""Real AgentService/approval seams produce durable causal Trace steps."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import ReadFileTool
from tldw_chatbook.Tools.tool_executor import Tool


class _CredentialResultTool(Tool):
    @property
    def name(self) -> str:
        return "credential_result"

    @property
    def description(self) -> str:
        return "Return a test result."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs) -> dict:
        return {
            "text": (
                "ghp_" + "a" * 36 + " "
                "AKIA" + "A" * 16 + " "
                "eyJabcdefghij.abcdefghij.abcdefghij "
                "-----BEGIN PRIVATE KEY-----"
            )
        }


class _HiddenReasoningResultTool(_CredentialResultTool):
    @property
    def name(self) -> str:
        return "hidden_reasoning_result"

    async def execute(self, **_kwargs) -> dict:
        return {"text": "chain of thought: private internal plan"}


class _PathResultTool(_CredentialResultTool):
    @property
    def name(self) -> str:
        return "path_result"

    async def execute(self, **_kwargs) -> dict:
        return {
            "text": (
                "/private/var/db/secrets.txt ~/private.txt "
                r"C:\Users\alice\secret.txt \\server\share\secret.txt"
            )
        }


def _native_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.mark.parametrize(
    ("decision", "decision_kind", "terminal_kind"),
    (
        ("approve_once", "approval_approved", "tool_succeeded"),
        ("deny", "approval_denied", None),
    ),
)
def test_real_agent_service_and_approval_hook_capture_ordered_lifecycle(
    tmp_path, monkeypatch, decision: str, decision_kind: str, terminal_kind: str
) -> None:
    target = tmp_path / "evidence.txt"
    target.write_text("credential=sk-durable-secret", encoding="utf-8")

    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as workspace_roots

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: tmp_path)
    monkeypatch.setattr(
        file_tools,
        "allowed_file_roots",
        lambda **_kwargs: (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        workspace_roots, "allowed_file_roots", lambda **_kwargs: (tmp_path.resolve(),)
    )

    gate = BuiltinToolGate(service=None)
    provider = BuiltinToolProvider(gate=gate)
    provider._tools["read_file"] = ReadFileTool()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    approval_rounds = []

    def decide(pending):
        approval_rounds.append(tuple(row.llm_name for row in pending))
        return {row.call_id or row.llm_name: decision for row in pending}

    review_hook = build_tool_review_hook(gate, provider, None, decide)
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            _native_call("read_file", {"file_path": str(target)})
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def fake_provider(**_kwargs):
        return replies.pop(0)

    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            registry,
            chat_call=fake_provider,
            review_tool_calls=review_hook,
            review_state_scope=gate.stamp_scope,
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "read it"}],
            config=AgentConfig(
                model="gpt-4o",
                system_prompt="system",
                allowed_tools=("read_file",),
                native_tools=True,
            ),
            api_endpoint="openai",
        )

        assert outcome.status == "done"
        assert approval_rounds == [("read_file",)]
        durable = db.get_run(run_id)
        assert "sk-durable-secret" not in repr(durable)
        kinds = [step["kind"] for step in durable["steps"]]
        assert kinds.index("model_request_started") < kinds.index("tool_proposed")
        assert kinds.index("tool_proposed") < kinds.index("approval_requested")
        assert kinds.index("approval_requested") < kinds.index(decision_kind)
        if terminal_kind is None:
            assert "tool_execution_started" not in kinds
            assert "tool_failed" not in kinds
        else:
            assert kinds.index(decision_kind) < kinds.index(terminal_kind)

        snapshot = derive_trajectory(
            [],
            {},
            [],
            [],
            [],
            agent_runs=[durable],
            agent_steps=[
                {**step, "run_id": run_id, "conversation_id": "conv-1"}
                for step in durable["steps"]
            ],
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        joined_kinds = [record.kind for record in records]
        expected = [
            "model_request_started",
            "model_response_completed",
            "model",
            "tool_proposed",
            "approval_requested",
            decision_kind,
        ]
        if terminal_kind is not None:
            expected.extend(["tool_execution_started", terminal_kind])
        positions = [joined_kinds.index(kind) for kind in expected]
        assert positions == sorted(positions)
        proposal = next(record for record in records if record.kind == "tool_proposed")
        assert proposal.parent_event_id == f"agent-run:{run_id}"
        assert proposal.field_states["args"] == "omitted"
        assert proposal.sensitivity == "tool_content"
        assert str(target) not in repr(proposal)
    finally:
        db.close()


@pytest.mark.parametrize("diagnostic_fails", (False, True))
def test_lifecycle_capture_failure_is_contained_and_diagnosed_when_writable(
    tmp_path, monkeypatch, diagnostic_fails: bool
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    registry = ToolCatalogRegistry()
    real_insert = db.insert_steps_at_indices

    def fail_lifecycle(run_id, rows):
        kind = rows[0][1].get("kind")
        if kind == "model_request_started" or (
            diagnostic_fails and kind == "capture_failed"
        ):
            raise RuntimeError("SECRET_CAPTURE_FAILURE")
        return real_insert(run_id, rows)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_lifecycle)
    try:
        service = AgentService(
            db,
            registry,
            chat_call=lambda **_kwargs: {"choices": [{"message": {"content": "done"}}]},
            review_tool_calls=lambda _calls, _run_id: {},
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "go"}],
            config=AgentConfig(model="model", system_prompt="system", allowed_tools=()),
            api_endpoint="openai",
        )

        assert outcome.status == "done"
        durable_steps = db.get_run(run_id)["steps"]
        kinds = [step["kind"] for step in durable_steps]
        assert ("capture_failed" in kinds) is (not diagnostic_fails)
        if not diagnostic_fails:
            diagnostic = next(
                step for step in durable_steps if step["kind"] == "capture_failed"
            )
            assert diagnostic["field_states"]["payload"] == "capture_failed"
            event_ids = {
                f"agent-step:{run_id}:{step['index']}" for step in durable_steps
            } | {f"agent-run:{run_id}"}
            assert diagnostic["parent_event_id"] in event_ids
            assert diagnostic["source_event_id"] is None
        assert "SECRET_CAPTURE_FAILURE" not in repr(db.get_run(run_id))
    finally:
        db.close()


def test_repeated_same_tool_calls_keep_distinct_safe_durable_correlation(
    tmp_path,
) -> None:
    gate = BuiltinToolGate(service=None)
    provider = BuiltinToolProvider(gate=gate)
    provider._tools["read_file"] = ReadFileTool()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    secret_a = str(tmp_path / "SECRET_A.txt")
    secret_b = str(tmp_path / "SECRET_B.txt")
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            _native_call("read_file", {"file_path": secret_a}, "a"),
                            _native_call("read_file", {"file_path": secret_b}, "b"),
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]
    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            registry,
            chat_call=lambda **_kwargs: replies.pop(0),
            review_tool_calls=lambda calls, _run_id: {
                call.call_id: "deny" for call in calls
            },
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "read both"}],
            config=AgentConfig(
                model="model",
                system_prompt="system",
                allowed_tools=("read_file",),
                native_tools=True,
            ),
            api_endpoint="openai",
        )
        assert outcome.status == "done"
        durable = db.get_run(run_id)["steps"]
        proposals = [step for step in durable if step["kind"] == "tool_proposed"]
        assert {step["call_id"] for step in proposals} == {"a", "b"}
        assert len({step["parent_event_id"] for step in proposals}) == 1
        decisions = [step for step in durable if step["kind"] == "approval_denied"]
        assert {step["call_id"] for step in decisions} == {"a", "b"}
        assert all(step["parent_event_id"] for step in decisions)
        assert secret_a not in repr(durable) and secret_b not in repr(durable)
    finally:
        db.close()


def test_generic_tool_credentials_are_scrubbed_at_durable_agent_step_boundary(
    tmp_path,
) -> None:
    gate = BuiltinToolGate(service=None)
    provider = BuiltinToolProvider(gate=gate)
    provider._tools["credential_result"] = _CredentialResultTool()
    provider._tools["hidden_reasoning_result"] = _HiddenReasoningResultTool()
    provider._tools["path_result"] = _PathResultTool()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            _native_call("credential_result", {}, "credentials"),
                            _native_call("hidden_reasoning_result", {}, "reasoning"),
                            _native_call("path_result", {}, "paths"),
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]
    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            registry,
            chat_call=lambda **_kwargs: replies.pop(0),
            review_tool_calls=lambda calls, _run_id: {
                call.call_id: "proceed" for call in calls
            },
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "run it"}],
            config=AgentConfig(
                model="model",
                system_prompt="system",
                allowed_tools=(
                    "credential_result",
                    "hidden_reasoning_result",
                    "path_result",
                ),
                native_tools=True,
            ),
            api_endpoint="openai",
        )

        assert outcome.status == "done"
        durable = db.get_run(run_id)["steps"]
        serialized = repr(durable)
        for secret_fragment in (
            "ghp_",
            "AKIA",
            "eyJabcdefghij",
            "BEGIN PRIVATE KEY",
            "private internal plan",
            "/private/var/db/secrets.txt",
            "~/private.txt",
            r"C:\Users\alice\secret.txt",
            r"\\server\share\secret.txt",
        ):
            assert secret_fragment not in serialized
        result_step = next(
            step
            for step in durable
            if step.get("tool_name") == "credential_result"
            and step.get("field_states", {}).get("result") == "redacted"
        )
        assert result_step["result"]
        assert "summarized" not in repr(result_step["field_states"])
        hidden_step = next(
            step
            for step in durable
            if step.get("tool_name") == "hidden_reasoning_result"
            and step.get("field_states", {}).get("result") == "omitted"
        )
        assert hidden_step["result"] == ""
        assert hidden_step["field_states"]["result"] == "omitted"
        path_step = next(
            step
            for step in durable
            if step.get("tool_name") == "path_result"
            and step.get("field_states", {}).get("result") == "omitted"
        )
        assert path_step["result"] == ""
        assert path_step["sensitivity"] == "path"
    finally:
        db.close()


def test_denied_call_cancelled_before_dispatch_is_not_approval_revoked(tmp_path) -> None:
    gate = BuiltinToolGate(service=None)
    provider = BuiltinToolProvider(gate=gate)
    provider._tools["credential_result"] = _CredentialResultTool()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    cancel = {"set": False}

    def deny_and_cancel(calls, _run_id):
        cancel["set"] = True
        return {call.call_id: "deny" for call in calls}

    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            registry,
            chat_call=lambda **_kwargs: {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [_native_call("credential_result", {})],
                        }
                    }
                ]
            },
            review_tool_calls=deny_and_cancel,
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "run it"}],
            config=AgentConfig(
                model="model",
                system_prompt="system",
                allowed_tools=("credential_result",),
                native_tools=True,
            ),
            api_endpoint="openai",
            should_cancel=lambda: cancel["set"],
        )

        assert outcome.status == "cancelled"
        kinds = [step["kind"] for step in db.get_run(run_id)["steps"]]
        approval_lifecycle = [
            kind
            for kind in kinds
            if kind.startswith("tool_") or kind.startswith("approval_")
        ]
        assert approval_lifecycle == [
            "tool_proposed",
            "approval_requested",
            "approval_denied",
        ]
        assert "approval_revoked" not in kinds
    finally:
        db.close()


def test_agent_service_actual_request_assembly_captures_safe_context_chain(tmp_path) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            ToolCatalogRegistry(),
            chat_call=lambda **_kwargs: {"choices": [{"message": {"content": "done"}}]},
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[
                {
                    "role": "user",
                    "content": "project credential=sk-project-secret",
                    EPHEMERAL_ORIGIN_KEY: "project",
                }
            ],
            config=AgentConfig(
                model="model",
                system_prompt="system credential=sk-system-secret",
                workspace_context_note="workspace credential=sk-workspace-secret",
                allowed_tools=(),
            ),
            api_endpoint="openai",
        )

        assert outcome.status == "done"
        durable = db.get_run(run_id)["steps"]
        lifecycle = [
            step for step in durable if step["kind"].startswith("agent_run_")
        ]
        assert [step["kind"] for step in lifecycle] == [
            "agent_run_created",
            "agent_run_started",
            "agent_run_completed",
        ]
        assert len({step["index"] for step in durable}) == len(durable)
        assert all(step["index"] >= 10_000_000 for step in lifecycle)
        attached = next(step for step in durable if step["kind"] == "context_attached")
        injected = next(step for step in durable if step["kind"] == "context_injected")
        assert attached["parent_event_id"] == f"agent-run:{run_id}"
        assert injected["parent_event_id"] == f"agent-step:{run_id}:{attached['index']}"
        assert injected["source_event_id"] == f"agent-step:{run_id}:{attached['index']}"
        assert attached["field_states"]["content"] == "omitted"
        assert attached["sensitivity"] == "system_context"
        serialized = repr(durable)
        for secret in ("sk-project-secret", "sk-system-secret", "sk-workspace-secret"):
            assert secret not in serialized
        assert all(name in attached["summary"] for name in ("project", "system", "workspace"))
        snapshot = derive_trajectory(
            [],
            {},
            [],
            [],
            [],
            agent_runs=[db.get_run(run_id)],
            agent_steps=[
                {**step, "run_id": run_id, "conversation_id": "conv-1"}
                for step in durable
            ],
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        ordered = [record.kind for record in records]
        assert ordered.index("agent_run_created") < ordered.index(
            "agent_run_started"
        )
        assert ordered.index("agent_run_started") < ordered.index(
            "context_attached"
        )
        assert ordered.index("context_attached") < ordered.index("context_injected")
        assert ordered.index("context_injected") < ordered.index(
            "model_request_started"
        )
        model_started = next(
            record for record in records if record.kind == "model_request_started"
        )
        injected_event_id = f"agent-step:{run_id}:{injected['index']}"
        assert model_started.parent_event_id == injected_event_id
        assert model_started.source_event_id == injected_event_id
    finally:
        db.close()


def test_agent_service_post_response_cancel_persists_causal_observation(tmp_path) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    flags = iter((False, True))
    try:
        service = AgentService(
            db,
            ToolCatalogRegistry(),
            chat_call=lambda **_kwargs: {"choices": [{"message": {"content": "done"}}]},
            review_tool_calls=lambda _calls, _run_id: {},
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "go"}],
            config=AgentConfig(model="model", system_prompt="system", allowed_tools=()),
            api_endpoint="openai",
            should_cancel=lambda: next(flags, True),
        )

        assert outcome.status == "cancelled"
        durable = db.get_run(run_id)["steps"]
        request = next(step for step in durable if step["kind"] == "model_request_started")
        completed = next(
            step for step in durable if step["kind"] == "model_response_completed"
        )
        cancelled = next(step for step in durable if step["kind"] == "model_cancelled")
        assert cancelled["parent_event_id"] == f"agent-step:{run_id}:{completed['index']}"
        assert cancelled["source_event_id"] == f"agent-step:{run_id}:{request['index']}"
        snapshot = derive_trajectory(
            [],
            {},
            [],
            [],
            [],
            agent_runs=[db.get_run(run_id)],
            agent_steps=[
                {**step, "run_id": run_id, "conversation_id": "conv-1"}
                for step in durable
            ],
        )
        kinds = [record.kind for turn in snapshot.turns for record in turn.records]
        assert kinds.index("model_response_completed") < kinds.index("model_cancelled")
    finally:
        db.close()
