"""Automatic nested project context stays inside the provider-only channel."""

from __future__ import annotations

import json
import time

from loguru import logger

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider, _default_specs
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionSnapshot,
    ProjectInstructionResolver,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ProjectInstructionActivationEvent,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState


SENTINEL = "CHATBOOK_AGENTS_SENTINEL_7d1e9c"
EXPLICIT_READ = "EXPLICIT_AGENTS_READ_RETAINS_CONTENT"
MODEL_QUOTATION = "MODEL_QUOTATION_RETAINS_CONTENT"
CALLBACK_SENTINEL = "CALLBACK_FAILURE_SECRET_/private/project/AGENTS.md"


class _Resolution:
    provider = "Groq"
    execution_key = "groq"


class _ProviderSpy:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.messages = []

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.messages.append([dict(row) for row in messages])
        for item in self.scripts.pop(0):
            yield item


def _native_fs_read():
    return ProviderToolCalls(
        tool_calls=(
            {
                "id": "read-1",
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "arguments": json.dumps({"path": "pkg/data.txt"}),
                },
            },
        )
    )


def test_automatic_nested_body_is_absent_from_every_durable_and_diagnostic_surface(
    tmp_path,
):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text(SENTINEL, encoding="utf-8")
    (nested / "data.txt").write_text(EXPLICIT_READ, encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    provider = _ProviderSpy(
        [[_native_fs_read()], [_native_fs_read()], [MODEL_QUOTATION]]
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="read it")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=provider,
    )
    local = LocalToolProvider(
        workspace_root=root,
        specs=[spec for spec in _default_specs(root) if spec.name == "fs_read"],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="global_default"
        ),
    )
    reviews = []
    events = []
    app_logs = []
    sink = logger.add(lambda record: app_logs.append(str(record)), level="DEBUG")
    try:
        _run_id, outcome = bridge.run_reply(
            conversation_id="conv",
            session_id=session.id,
            resolution=_Resolution(),
            assistant_message_id=assistant.id,
            model="model",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "read it"}],
            should_cancel=lambda: False,
            local_provider=local,
            review_tool_calls=lambda calls, _run_id: reviews.append(tuple(calls)) or {},
            startup_instruction_candidate=candidate,
            confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
            on_project_instruction_activation=events.append,
        )
    finally:
        logger.remove(sink)

    assert any(SENTINEL in repr(request) for request in provider.messages)
    forbidden = {
        "outcome": repr(outcome),
        "agent_runs_db": repr(db.list_runs("conv")),
        "transcript": repr(store.messages_for_session(session.id)),
        "review": repr(reviews),
        "events": repr(events),
        "application_log": repr(app_logs),
    }
    assert {name: value for name, value in forbidden.items() if SENTINEL in value} == {}
    assert events
    assert events[0].relative_sources == ("pkg/AGENTS.md",)
    assert events[0].scopes == ("pkg",)
    assert vars(events[0]) == {
        "relative_sources": ("pkg/AGENTS.md",),
        "scopes": ("pkg",),
        "outcome_codes": (),
    }
    # Explicit tool content remains available to the live run, while the
    # durable run summary intentionally keeps tool bodies out of storage.
    assert EXPLICIT_READ in repr(outcome.steps)
    assert EXPLICIT_READ not in repr(db.list_runs("conv"))
    assert MODEL_QUOTATION in repr(store.messages_for_session(session.id))


def test_activation_callback_failure_is_content_free_and_does_not_block_provider(
    tmp_path, monkeypatch
):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text(SENTINEL, encoding="utf-8")
    (nested / "data.txt").write_text("payload", encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    provider = _ProviderSpy([[_native_fs_read()], [_native_fs_read()], ["done"]])
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="read it")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=provider,
    )
    local = LocalToolProvider(
        workspace_root=root,
        specs=[spec for spec in _default_specs(root) if spec.name == "fs_read"],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="global_default"
        ),
    )
    marks = []
    real_mark = bridge_module.InstructionActivationLedger.mark_payload_sent

    def track_mark(ledger, receipt, rows):
        marks.append(receipt)
        return real_mark(ledger, receipt, rows)

    monkeypatch.setattr(
        bridge_module.InstructionActivationLedger, "mark_payload_sent", track_mark
    )

    def fail_callback(_event):
        raise RuntimeError(CALLBACK_SENTINEL)

    app_logs = []
    sink = logger.add(lambda record: app_logs.append(str(record)), level="DEBUG")
    try:
        _run_id, outcome = bridge.run_reply(
            conversation_id="conv",
            session_id=session.id,
            resolution=_Resolution(),
            assistant_message_id=assistant.id,
            model="model",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "read it"}],
            should_cancel=lambda: False,
            local_provider=local,
            startup_instruction_candidate=candidate,
            confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
            on_project_instruction_activation=fail_callback,
        )
    finally:
        logger.remove(sink)

    assert outcome.status == "done"
    assert len(marks) == 1
    assert len(provider.messages) == 3
    durable = repr(
        (
            outcome,
            db.list_runs("conv"),
            store.messages_for_session(session.id),
            app_logs,
        )
    )
    assert CALLBACK_SENTINEL not in durable
    assert SENTINEL not in durable


def test_accepted_primary_run_replaces_prior_session_activation_events(tmp_path):
    controller = object.__new__(ConsoleChatController)
    controller._project_instruction_display = {}
    controller._project_instruction_activation_events = {
        "session": [
            ProjectInstructionActivationEvent(
                relative_sources=("old/AGENTS.md",), scopes=("old",)
            )
        ]
    }
    snapshot = InstructionSnapshot(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=time.time_ns(),
        startup_source=None,
        global_outcomes=(),
        primary_delivery=InstructionChainDelivery((), ()),
        warning_codes=(),
    )

    controller._remember_project_instruction_delivery("session", snapshot)
    controller._record_project_instruction_activation(
        "session",
        ProjectInstructionActivationEvent(
            relative_sources=("new/AGENTS.md",), scopes=("new",)
        ),
    )

    events = controller.project_instruction_activation_events("session")
    assert [event.scopes for event in events] == [("new",)]
