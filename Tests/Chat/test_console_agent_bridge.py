"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""

import asyncio
import contextlib
import copy
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

import tldw_chatbook.Agents.agent_service as agent_service_module
from tldw_chatbook.Chat import console_agent_bridge
import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from tldw_chatbook.Chat.console_agent_bridge import (
    CHANGE_KIND_SUBAGENT_POST_TURN,
    CHANGE_KIND_TURN,
    CHANGE_KIND_TURN_CONCURRENT_SUBAGENT,
    CONSOLE_AGENT_OPERATING_PROMPT,
    FIND_LOAD_DISCOVERY_HINT,
    ConsoleAgentBridge,
    SubAgentSummary,
    _StreamingModelAdapter,
    _append_to_last_user_message,
    _openai_usage_from_provider_call,
    compose_agent_system_prompt,
    format_agent_step_marker,
    format_todo_marker,
    inject_resume_agent_markers,
    _BridgeSkillRunner,
    _compose_run_allowed_tools,
    _compose_run_registry_and_allowed,
    _non_colliding_mcp_names,
    _subagent_summaries_from_fleet,
    _WARNED_SHADOWED_MCP_NAMES,
    shadowed_mcp_names,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleThinkingCompatibilityError,
)
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    USER_DENIED_REFUSAL as CONTROLLER_USER_DENIED_REFUSAL,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_display_state import format_diff_feedback_disclosure
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    ProviderProprietaryThinkingEvidence,
    ProviderThinkingDelta,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD,
    LOAD_TOOLS_NAME,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    TERMINAL_RUN_STATUSES,
    STEP_ERROR,
    STEP_MODEL,
    STEP_SPAWN,
    STEP_TOOL_CALL,
    STEP_TOOL_RESULT,
    AgentStep,
    AgentDefinition,
    RunOutcome,
    SkillFileBindings,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.tool_catalog import (
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider, _default_specs
from tldw_chatbook.Agents.project_instruction_resolver import ProjectInstructionResolver
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Workspaces.change_turn_tracker import TurnChangeRecord

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX


@pytest.fixture(autouse=True)
def _reset_shadowed_mcp_warning_dedup():
    """Reset the shadowed-MCP-name log-once set so tests are order-independent.

    Mirrors ``Tests/Internal_Prompts/conftest.py``'s
    ``resolver._warned_ids.clear()`` idiom for the identical reason:
    ``_compose_run_registry_and_allowed``'s shadowed-name warning is now
    deduped per name for the life of the process (finding 8, substrate
    review). Without this reset, whichever test in this file runs FIRST
    for a given shadowed name would win, and every later test asserting
    the same warning would silently observe nothing logged.
    """
    _WARNED_SHADOWED_MCP_NAMES.clear()
    yield
    _WARNED_SHADOWED_MCP_NAMES.clear()


class _FakeMCPProvider:
    """Minimal ``ToolProvider`` double standing in for a composed
    ``MCPToolProvider`` (T3) -- these bridge-level tests only need the
    catalog/invoke seam, not the real service/approval plumbing."""

    def __init__(self, entries):
        # entries: iterable of (name, description) pairs
        self._entries = list(entries)
        self.invoke_calls: list[tuple[str, dict]] = []
        self.stamp_scope_calls = 0
        self.list_catalog_calls = 0

    def list_catalog(self):
        self.list_catalog_calls += 1
        return [
            ToolCatalogEntry(
                id=name, name=name, one_line_description=desc, source="mcp"
            )
            for name, desc in self._entries
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description="",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"mcp-result:{tool_id}")

    @contextlib.contextmanager
    def stamp_scope(self, run_id):
        # C1 (probe-verified security regression): stands in for
        # MCPToolProvider.stamp_scope -- a no-op snapshot/restore here since
        # this fake carries no per-turn stamp state of its own, just a call
        # counter so bridge-level wiring tests can assert `run_reply` threads
        # it through to AgentService(review_state_scope=...). PR2a Task 5:
        # the scope takes the run id whose slice it guards.
        self.stamp_scope_calls += 1
        yield


class _ResultMCPProvider(_FakeMCPProvider):
    """MCP-shaped provider returning one exact structured result."""

    def __init__(self, result: ToolResult):
        super().__init__([("collision_tool", "Return the test payload")])
        self._result = result

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return self._result


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index.

    Each scripted entry is a list of chunks, where a chunk is either a
    plain ``str`` (streamed text, as before) or a ``ProviderToolCalls``
    sentinel (native tool-calls, yielded as the final item of that turn).
    ``tools_seen`` records the ``tools=`` kwarg passed on each call, in
    call order, so tests can assert whether/what was forwarded.
    """

    def __init__(self, scripts):
        self._scripts = list(
            scripts
        )  # each entry: list of str and/or ProviderToolCalls
        self.calls = 0
        self.tools_seen = []
        self.messages_seen = []

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.tools_seen.append(tools)
        self.messages_seen.append([dict(message) for message in messages])
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class _FleetChunkGateway:
    """A gateway that addresses scripts by AGENT, not by arrival order.

    PR2a Task 6.5: `_ChunkGateway` indexes its scripts by call count
    (`self._scripts[self.calls]`), which stops being deterministic once the
    fleet is ON by default -- a spawned child runs on its own thread and
    its turn can land before, after, or between the parent's. This keeps
    the same chunk-list-per-turn shape but keeps ONE queue for the primary
    agent and one per child, so each turn's script reaches the agent it was
    written for.

    Children are identified exactly as ``_StreamingModelAdapter.
    _is_subagent`` identifies them -- a system prompt starting with the
    sub-agent prompt -- so the addressing follows the production contract
    rather than a test-only convention.

    Args:
        parent_script: turns for the primary agent, in order.
        child_script: turns for THE child, in order. These suites spawn
            exactly one, and its task text is an implementation detail of
            whatever spawned it (a skill's rendered prompt, say), so
            addressing on "is a child" is both sufficient and stabler than
            addressing on that text.
    """

    def __init__(self, parent_script, child_script=()):
        self._parent = list(parent_script)
        self._child = list(child_script)
        self.calls = 0
        self.tools_seen = []
        self.child_calls = 0
        self._lock = threading.Lock()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        is_child = system.startswith(SUBAGENT_PROMPT_PREFIX)
        with self._lock:
            self.tools_seen.append(tools)
            self.calls += 1
            if is_child:
                self.child_calls += 1
                assert self._child, "child script exhausted"
                chunks = self._child.pop(0)
            else:
                assert self._parent, "parent script exhausted"
                chunks = self._parent.pop(0)
        # Resolved OUTSIDE the lock: a turn may be a zero-arg callable that
        # BLOCKS on an Event, which is how a test pins an interleaving; the
        # lock would serialize the very concurrency under test.
        if callable(chunks):
            chunks = chunks()
        for chunk in chunks:
            yield chunk


def _join_fleet_threads(timeout=5.0):
    """Block until every live fleet child thread has fully finished.

    `AgentService` names them ``fleet-<handle>``. "Fully finished" is the
    point: a child's own run row goes terminal slightly BEFORE its thread
    unwinds, so joining the thread -- not polling the DB -- is what
    guarantees any context manager wrapping that run has already exited.
    """
    for thread in list(threading.enumerate()):
        if thread.name.startswith("fleet-"):
            thread.join(timeout)


@pytest.mark.parametrize(
    ("case", "agent_messages", "supersede"),
    [
        ("text", [{"role": "user", "content": "text"}], False),
        (
            "multimodal",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA=="},
                        },
                    ],
                }
            ],
            False,
        ),
        ("retry", [{"role": "user", "content": "retry"}], True),
        ("regenerate", [{"role": "user", "content": "regenerate"}], True),
        ("continue", [{"role": "user", "content": "continue"}], True),
    ],
)
def test_all_agent_dispatch_shapes_receive_one_startup_rider(
    tmp_path, case, agent_messages, supersede
):
    from tldw_chatbook.Agents.project_instruction_resolver import (
        InstructionSource,
        StartupInstructionCandidate,
    )
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    gateway = _ChunkGateway([["done"]])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / case, gateway
    )
    source = InstructionSource(
        canonical_path=tmp_path / "AGENTS.md",
        relative_path="AGENTS.md",
        scope=".",
        kind="standard",
        body="BRIDGE_STARTUP_SENTINEL",
        byte_count=23,
        digest="a" * 64,
    )
    candidate = StartupInstructionCandidate(
        binding_id="b",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=source,
        outcomes=(),
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
        agent_messages=agent_messages,
        supersede_previous=supersede,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    assert outcome.status == "done", outcome.steps
    rows = [
        row
        for row in gateway.messages_seen[0]
        if "BRIDGE_STARTUP_SENTINEL" in str(row.get("content", ""))
    ]
    assert len(rows) == 1
    assert rows[0][EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert "BRIDGE_STARTUP_SENTINEL" not in str(
        [message.content for message in store.messages_for_session(session.id)]
    )


class _SignalChunkGateway(_ChunkGateway):
    """Scripted gateway that records the out-of-band signal by identity."""

    def __init__(self, scripts):
        super().__init__(scripts)
        self.signals_seen = []
        self.signal_states_seen = []

    async def stream_chat(self, resolution, messages, tools=None, signals=None):
        self.signals_seen.append(signals)
        self.signal_states_seen.append(signals.synthetic_fallback_emitted)
        async for chunk in super().stream_chat(
            resolution,
            messages,
            tools=tools,
        ):
            yield chunk


def _test_resolution(**over):
    """Minimal REAL-shaped resolution for the shared harness.

    TASK-16270: PR #1612 widened ``_StreamingModelAdapter``'s implicit
    ``resolution`` contract past an opaque token — ``.provider`` and
    ``.model`` are now read unconditionally after every streamed turn
    (usage accounting), so ``object()`` stand-ins no longer satisfy it.
    Constructing the PRODUCTION ``ConsoleProviderResolution`` dataclass
    here (instead of a hand-rolled stand-in) keeps this fixture from
    silently drifting the next time the contract widens: a new required
    field fails loudly in every suite that shares ``_run``.
    """
    fields = dict(provider="TestProvider", base_url="", model=None, ready=True)
    fields.update(over)
    return ConsoleProviderResolution(**fields)


def _native_resolution():
    """A real resolution whose execution_key resolves to a native-capable provider."""
    return _test_resolution(provider="Groq", execution_key="groq")


class _FencedResolution:
    provider = "Custom"
    execution_key = "custom"
    model = "test-model"
    max_tokens = 10


def _native_calls(name, args, call_id="c1"):
    return ProviderToolCalls(
        tool_calls=(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
        )
    )


def _bridge(tmp_path, scripts, native_tools_enabled=None):
    return _bridge_with_gateway(
        tmp_path,
        _ChunkGateway(scripts),
        native_tools_enabled=native_tools_enabled,
    )


def _bridge_with_gateway(tmp_path, gateway, native_tools_enabled=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=native_tools_enabled,
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, store, session, assistant_id, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=_test_resolution(),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    kwargs.update(over)
    # run_reply returns (run_id, outcome); these tests assert on the outcome.
    _run_id, outcome = bridge.run_reply(**kwargs)
    return outcome


def _tool_messages(store, session_id: str) -> list[ConsoleChatMessage]:
    return [
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.TOOL
    ]


def _resume_tool_messages(
    db: AgentRunsDB, conversation_id: str = "conv-1"
) -> list[ConsoleChatMessage]:
    return [
        message
        for _anchor, block in ConsoleAgentBridge(
            agent_runs_db=db, store=None, provider_gateway=None
        ).resume_marker_messages(conversation_id)
        for message in block
    ]


def _activity_marker_signature(
    messages: list[ConsoleChatMessage],
) -> list[tuple[str, ConsoleActivityPresentation | None, str | None]]:
    return [
        (message.content, message.activity_presentation, message.tool_output_full)
        for message in messages
    ]


def test_nested_project_instructions_defer_whole_batch_before_review_and_execution(
    tmp_path,
):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / "AGENTS.md").write_text("ROOT_GUIDANCE", encoding="utf-8")
    (nested / "AGENTS.md").write_text("NESTED_GUIDANCE", encoding="utf-8")
    (nested / "data.txt").write_text("payload", encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    first_calls = ProviderToolCalls(
        tool_calls=(
            {
                "id": "read-1",
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "arguments": json.dumps({"path": "pkg/data.txt"}),
                },
            },
            {
                "id": "list-1",
                "type": "function",
                "function": {
                    "name": "fs_list",
                    "arguments": json.dumps({"path": "pkg"}),
                },
            },
        )
    )
    gateway = _ChunkGateway([[first_calls], [first_calls], ["done"]])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "run", gateway
    )
    local = LocalToolProvider(
        workspace_root=root,
        specs=[
            spec for spec in _default_specs(root) if spec.name in {"fs_read", "fs_list"}
        ],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="global_default"
        ),
    )
    reviews = []

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
        local_provider=local,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
        review_tool_calls=lambda calls, _run_id: (
            reviews.append(tuple(c.name for c in calls)) or {}
        ),
    )

    assert outcome.status == "done", outcome.steps
    assert reviews == [("fs_read", "fs_list")]
    second = gateway.messages_seen[1]
    assert [row["tool_call_id"] for row in second if row["role"] == "tool"] == [
        "read-1",
        "list-1",
    ]
    context_rows = [
        row for row in second if "NESTED_GUIDANCE" in str(row.get("content", ""))
    ]
    assert len(context_rows) == 1
    assert second.index(context_rows[0]) > max(
        index for index, row in enumerate(second) if row["role"] == "tool"
    )
    assert all("NESTED_GUIDANCE" not in str(step.result) for step in outcome.steps)


@pytest.mark.parametrize(
    ("transformed_tokens", "expected_status", "expected_calls"),
    [(90, "done", 3), (91, "error", 1)],
)
def test_fenced_nested_delivery_counts_exact_transformed_payload_before_mark(
    tmp_path, monkeypatch, transformed_tokens, expected_status, expected_calls
):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("FENCED_NESTED_GUIDANCE", encoding="utf-8")
    (nested / "data.txt").write_text("payload", encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    call = _fence("fs_read", {"path": "pkg/data.txt"})
    gateway = _ChunkGateway([[call], [call], ["done"]])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "run", gateway
    )
    local = LocalToolProvider(
        workspace_root=root,
        specs=[spec for spec in _default_specs(root) if spec.name == "fs_read"],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="global_default"
        ),
    )
    events = []
    monkeypatch.setattr(agent_service_module, "get_model_token_limit", lambda *_: 100)
    monkeypatch.setattr(agent_service_module, "_count_model_messages", lambda *_: 10)
    monkeypatch.setattr(
        bridge_module, "get_model_token_limit", lambda *_: 100, raising=False
    )
    monkeypatch.setattr(
        bridge_module,
        "_count_model_messages",
        lambda *_: transformed_tokens,
        raising=False,
    )
    marks = []
    real_mark = bridge_module.InstructionActivationLedger.mark_payload_sent

    def track_mark(ledger, receipt, rows):
        marks.append(receipt)
        return real_mark(ledger, receipt, rows)

    monkeypatch.setattr(
        bridge_module.InstructionActivationLedger, "mark_payload_sent", track_mark
    )

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_FencedResolution(),
        local_provider=local,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
        on_project_instruction_activation=events.append,
    )

    assert outcome.status == expected_status
    assert gateway.calls == expected_calls
    if expected_status == "done":
        delivered = gateway.messages_seen[1]
        assert all(row["role"] != "tool" for row in delivered)
        assert "Tool results:\n```tool_results\n" in delivered[-1]["content"]
        assert "\n```\n\nProject instruction context:\n" in delivered[-1]["content"]
        assert "FENCED_NESTED_GUIDANCE" in delivered[-1]["content"]
        assert events
        assert len(marks) == 1
    else:
        assert events == []
        assert marks == []


def test_opaque_tool_arguments_do_not_activate_nested_project_instructions(tmp_path):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / "AGENTS.md").write_text("ROOT_GUIDANCE", encoding="utf-8")
    (nested / "AGENTS.md").write_text("NESTED_GUIDANCE", encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    calculator = _native_calls(
        "calculator", {"expression": "len('pkg/data.txt')"}, "calc-1"
    )
    gateway = _ChunkGateway([[calculator], ["done"]])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "run", gateway
    )

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )

    assert outcome.status == "done", outcome.steps
    assert all("NESTED_GUIDANCE" not in repr(rows) for rows in gateway.messages_seen)


def test_nested_resolution_rejects_backdated_root_replacement_after_consent(tmp_path):
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / "AGENTS.md").write_text("ROOT_GUIDANCE", encoding="utf-8")
    (nested / "data.txt").write_text("original", encoding="utf-8")
    dispatch_started = time.time_ns()
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=dispatch_started,
    )
    read = _native_calls("fs_read", {"path": "pkg/data.txt"}, "read-1")
    gateway = _ChunkGateway([[read], [read], ["done"]])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "run", gateway
    )
    local = LocalToolProvider(
        workspace_root=root,
        specs=[spec for spec in _default_specs(root) if spec.name == "fs_read"],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="global_default"
        ),
    )

    def replace_after_consent_check(_snapshot):
        root.rename(tmp_path / "displaced")
        (root / "pkg").mkdir(parents=True)
        replacement = root / "pkg" / "AGENTS.md"
        replacement.write_text("BACKDATED_REPLACEMENT_SECRET", encoding="utf-8")
        (root / "pkg" / "data.txt").write_text("replacement", encoding="utf-8")
        backdated = dispatch_started - 1_000_000_000
        os.utime(replacement, ns=(backdated, backdated))
        return "proceed"

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
        local_provider=local,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=replace_after_consent_check,
    )

    assert outcome.status == "done", outcome.steps
    assert all(
        "BACKDATED_REPLACEMENT_SECRET" not in repr(rows)
        for rows in gateway.messages_seen
    )
    assert any("resolution_failed" in repr(rows) for rows in gateway.messages_seen)


def test_parent_and_child_share_activation_but_each_receive_nested_revision(
    tmp_path, monkeypatch
):
    original_setting = agent_service_module._setting
    monkeypatch.setattr(
        agent_service_module,
        "_setting",
        lambda key, default: (
            1
            if key == agent_service_module.MAX_LIVE_SUBAGENTS_KEY
            else original_setting(key, default)
        ),
    )
    root = tmp_path / "workspace"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / "AGENTS.md").write_text("ROOT_GUIDANCE", encoding="utf-8")
    (nested / "AGENTS.md").write_text("NESTED_GUIDANCE", encoding="utf-8")
    (nested / "data.txt").write_text("payload", encoding="utf-8")
    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="f" * 64,
        max_bytes=32768,
        dispatch_started_wall_ns=time.time_ns(),
    )
    read = _native_calls("fs_read", {"path": "pkg/data.txt"}, "read-1")
    gateway = _ChunkGateway(
        [
            [_native_calls(SPAWN_TOOL_NAME, {"task": "inspect pkg"}, "spawn-1")],
            [read],
            [read],
            ["child done"],
            [read],
            [read],
            ["parent done"],
        ]
    )
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path / "run", gateway
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

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
        local_provider=local,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
        review_tool_calls=lambda calls, _run_id: (
            reviews.append(tuple(c.name for c in calls)) or {}
        ),
        on_project_instruction_activation=events.append,
    )

    assert outcome.status == "done", outcome.steps
    nested_visibility = [
        any("NESTED_GUIDANCE" in str(row.get("content", "")) for row in rows)
        for rows in gateway.messages_seen
    ]
    assert nested_visibility == [False, False, True, True, False, True, True]
    assert reviews.count(("fs_read",)) == 2
    assert reviews.count((SPAWN_TOOL_NAME,)) == 1
    assert len(events) == 1
    assert events[0].relative_sources == ("pkg/AGENTS.md",)


@pytest.mark.asyncio
@pytest.mark.parametrize("force_character", [False, True])
async def test_plain_and_character_forced_plain_never_resolve_project_instructions(
    monkeypatch, force_character
):
    from types import SimpleNamespace

    from tldw_chatbook.Agents.project_instruction_resolver import (
        ProjectInstructionResolver,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    gateway = _ChunkGateway([["plain done"]])
    store = ConsoleChatStore()
    session = store.ensure_session()
    if force_character:
        session.assistant_kind = "character"
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    class ExplodingBridge:
        def run_reply(self, **_kwargs):
            raise AssertionError("character sessions must stay plain")

    monkeypatch.setattr(
        ProjectInstructionResolver,
        "resolve_startup",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("plain sends must not read AGENTS.md")
        ),
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=ExplodingBridge(),
        agent_runtime_enabled=force_character,
    )
    resolution = SimpleNamespace(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=128,
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="gpt-4o-mini",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )
    configuration = controller.resolve_turn_configuration_snapshot(session.id)
    authority = await controller._capture_turn_library_authority(
        session.id, configuration
    )
    turn_context = controller._finalize_turn_execution_context(
        configuration, authority, resolution
    )
    result = await controller._stream_assistant_response_inner(
        resolution=resolution,
        provider_messages=[{"role": "user", "content": user.content}],
        assistant_message_id=assistant.id,
        turn_context=turn_context,
    )
    assert result.accepted is True
    assert gateway.calls == 1


def test_compose_prepends_session_prompt_then_agent_prompt():
    composed = compose_agent_system_prompt("You are Ada.")
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT


def test_compose_appends_discovery_hint_only_when_find_load_offered():
    # Default (direct disclosure): no hint — find/load is not the live mode.
    assert FIND_LOAD_DISCOVERY_HINT not in compose_agent_system_prompt("You are Ada.")
    assert FIND_LOAD_DISCOVERY_HINT not in compose_agent_system_prompt("")
    # Past the threshold the caller flags find/load mode: the hint is
    # appended after the operating prompt, session prompt still first.
    composed = compose_agent_system_prompt("You are Ada.", offer_find_load=True)
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert composed.endswith(FIND_LOAD_DISCOVERY_HINT)
    blank = compose_agent_system_prompt("", offer_find_load=True)
    assert blank.startswith(CONSOLE_AGENT_OPERATING_PROMPT)
    assert blank.endswith(FIND_LOAD_DISCOVERY_HINT)


def test_no_tool_message_streams_final_answer_like_today(tmp_path):
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Tok", "yo."]])
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done" and outcome.final_text == "Tokyo."
    assert store.get_message(aid).content == "Tokyo."
    # No tool markers were appended.
    roles = [m.role for m in store.messages_for_session(session.id)]
    assert ConsoleMessageRole.TOOL not in roles


def test_usage_accounting_failure_never_flips_a_streamed_run_to_error(
    tmp_path, monkeypatch
):
    """TASK-16270: usage accounting is pure observability. A run that
    streamed its answer successfully and then fails INSIDE usage
    extraction must complete ``done`` with the usage simply missing (plus
    a logged warning naming the failure) — never flip to
    ``status='error'``."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("usage extraction exploded")

    monkeypatch.setattr(console_agent_bridge, "_openai_usage_from_provider_call", _boom)
    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        bridge, _db, store, session, aid = _bridge(tmp_path, [["Tok", "yo."]])
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            resolution=ConsoleProviderResolution(
                provider="TestProvider", base_url="", model=None, ready=True
            ),
        )
    finally:
        logger.remove(sink_id)

    assert outcome.status == "done"
    assert outcome.final_text == "Tokyo."
    assert store.get_message(aid).content == "Tokyo."
    # Usage is simply missing — never attached to the message.
    assert store.get_message(aid).usage is None
    assert any("usage accounting failed" in m for m in warnings)


def test_a_genuine_provider_failure_still_classifies_as_error(tmp_path):
    """The TASK-16270 wrap covers ONLY usage accounting: a failure in the
    model call / stream itself must still land ``status='error'``."""

    class _ExplodingGateway:
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            yield "Tok"
            raise RuntimeError("provider connection dropped")

    bridge, _db, store, session, aid = _bridge_with_gateway(
        tmp_path, _ExplodingGateway()
    )
    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=ConsoleProviderResolution(
            provider="TestProvider", base_url="", model=None, ready=True
        ),
    )
    assert outcome.status == "error"
    assert any(step.kind == "error" for step in outcome.steps)


def test_tool_turn_renders_a_tool_marker_not_prose(tmp_path):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: leading fence
        ["It is ", "42."],  # turn 2: final answer
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a tool turn must drop a TOOL marker"
    assert any("calculator" in marker.content for marker in tool_rows)
    # The fenced tool JSON never streamed into the assistant answer.
    assert FENCE_OPEN not in store.get_message(aid).content
    assert store.get_message(aid).content == "It is 42."


def test_joined_continuation_never_logs_private_tool_arguments_or_results(
    tmp_path, monkeypatch
):
    """Operational bridge logs must not serialize restored private fields."""
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    async def private_result(self, expression):
        assert expression == "PRIVATE_ARGUMENT_CANARY"
        return {"value": "PRIVATE_RESULT_CANARY"}

    monkeypatch.setattr(CalculatorTool, "execute", private_result)
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("PRIVATE_REASONING_CANARY",),
                calls=(
                    ContinuationCall(
                        call_id="PRIVATE_CALL_ID_CANARY",
                        name="calculator",
                        arguments='{"expression":"PRIVATE_ARGUMENT_CANARY"}',
                        state="pending",
                    ),
                ),
            ),
        ),
    )
    db = AgentRunsDB(tmp_path / "private-resume.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    assistant.provider_continuation = checkpoint
    assistant.provider_continuation_message_version = 1
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway([["finished"]]),
    )
    captured: list[str] = []
    sink = logger.add(lambda record: captured.append(str(record)), level="DEBUG")
    target = ContinuationRestoreTarget(
        provider="moonshot",
        model="kimi-k2",
        protocol="chat_completions",
        api_base_url="https://api.moonshot.ai/v1",
    )
    try:
        outcome = _run(
            bridge,
            store,
            session,
            assistant.id,
            resolution=_test_resolution(
                provider="Moonshot",
                execution_key="moonshot",
                model="kimi-k2",
                base_url="https://api.moonshot.ai/v1",
            ),
            restore_provider_continuation=checkpoint,
            restore_provider_target=target,
            # TASK-16270: PR #1612 widened the resume contract — an ACTIVE
            # checkpoint now rides as a continuation group and requires an
            # owner (production always passes assistant_message_id and
            # derives the owner key from the continuation target).
            continuation_target=target,
            expand_provider_continuation=lambda _checkpoint: [],
            resume_provider_continuation=True,
        )
    finally:
        logger.remove(sink)

    assert any(step.kind == STEP_TOOL_RESULT for step in outcome.steps)
    joined = "\n".join(captured)
    assert "agent_kind=primary tool=calculator" in joined
    assert "PRIVATE_ARGUMENT_CANARY" not in joined
    assert "PRIVATE_RESULT_CANARY" not in joined
    assert "PRIVATE_CALL_ID_CANARY" not in joined
    assert "PRIVATE_REASONING_CANARY" not in joined


class _RefusingBuiltinGate:
    """A `BuiltinToolGate` double that refuses every call -- proves
    `run_reply`'s `builtin_gate=` argument is the SAME object
    `BuiltinToolProvider.invoke()` ends up checking, end to end."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def check(self, tool, run_id):
        self.checked.append(tool.name)
        return f"disabled for test: {tool.name}"


def test_run_reply_threads_builtin_gate_end_to_end(tmp_path):
    """task-545/T6: a `builtin_gate` handed to `run_reply` must be the
    exact instance the run's own `BuiltinToolProvider.invoke()` consults --
    a second, independently-built gate would silently desync from
    whatever the caller's review hook already decided (the core risk this
    task's wiring exists to avoid)."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["it was refused."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    gate = _RefusingBuiltinGate()
    outcome = _run(bridge, store, session, aid, builtin_gate=gate)
    assert outcome.status == "done"
    assert gate.checked == ["calculator"]
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a refused tool call still drops a TOOL marker"
    assert any(
        "disabled for test: calculator" in marker.content for marker in tool_rows
    )
    assert store.get_message(aid).content == "it was refused."


def test_run_reply_threads_session_workspace_id_end_to_end(tmp_path, monkeypatch):
    """task-6 (settings-workspaces-folder-roots spec Sec3): the RUNNING
    session's own ``workspace_id`` must reach the run's
    ``BuiltinToolProvider``, so a builtin tool observes it (via
    ``workspace_file_roots.run_workspace``) while it executes -- not
    whatever workspace happens to be active in the UI by the time the
    call actually fires. Verified by monkeypatching ``CalculatorTool.
    execute`` (the real built-in tool the scripted fence calls) to record
    the ContextVar it sees, since the bridge builds its own
    ``BuiltinToolProvider`` internally with no seam for a test double."""
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    observed: list[str | None] = []
    real_execute = CalculatorTool.execute

    async def _recording_execute(self, expression):
        observed.append(wfr.current_run_workspace_id())
        return await real_execute(self, expression)

    monkeypatch.setattr(CalculatorTool, "execute", _recording_execute)

    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is 42."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="ws-session-42")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts)
    )
    # Any non-None `builtin_gate` forces the fresh-build branch that
    # actually looks up and threads the session's workspace_id (see
    # `run_reply`'s own docstring) -- the shared/no-gate fast path's
    # provider is built once at bridge-construction time, before any
    # session exists, so it is out of scope for a per-session binding.
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome.status == "done"
    assert observed == ["ws-session-42"]
    assert wfr.current_run_workspace_id() is None  # cleared after the run


def test_run_reply_threads_captured_scratch_end_to_end(tmp_path, monkeypatch):
    """The live run's file dispatch observes its captured private sandbox."""
    import tldw_chatbook.config as config
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Tools.file_operation_tools import ReadFileTool

    real_setting = config.get_cli_setting

    def enable_read_file(section, key, default=None):
        if section == "tools" and key == "read_file_enabled":
            return True
        return real_setting(section, key, default)

    observed: list[Path | None] = []
    real_execute = ReadFileTool.execute

    async def recording_execute(self, file_path, **kwargs):
        observed.append(wfr.current_run_sandbox_root())
        return await real_execute(self, file_path, **kwargs)

    monkeypatch.setattr(config, "get_cli_setting", enable_read_file)
    monkeypatch.setattr(ReadFileTool, "execute", recording_execute)

    scratch = tmp_path / "chat-a"
    scratch.mkdir()
    marker = scratch / "marker.txt"
    marker.write_text("chat-a", encoding="utf-8")
    scripts = [[_fence("read_file", {"file_path": str(marker)})], ["done"]]
    bridge, _db, store, session, aid = _bridge(tmp_path / "run", scripts)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
        scratch_root=scratch,
        scratch_lease=lambda: contextlib.nullcontext(scratch),
    )

    assert outcome.status == "done"
    assert observed == [scratch.resolve()]
    assert wfr.current_run_sandbox_root() is None


class _RecordingGateway:
    """Records each turn's system prompt (messages[0]) and answers 'ok'."""

    def __init__(self):
        self.systems: list[str] = []

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.systems.append(str(messages[0].get("content", "")) if messages else "")
        yield "ok"


def test_run_reply_appends_workspace_note_for_a_non_default_workspace(
    tmp_path, monkeypatch
):
    """A session bound to a non-default workspace must carry the workspace
    note into the primary agent's system prompt -- even on the fast path with
    no builtin_gate, since run_reply resolves the workspace up front rather
    than only inside the provider-gated branch."""
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    ws_registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="bridge-note-test")
    )
    ws_registry.ensure_default_workspace()
    ws_registry.create_workspace(workspace_id="ws-note-1", name="Notes Workspace")
    monkeypatch.setattr(wfr, "_registry_factory", lambda: ws_registry)

    gateway = _RecordingGateway()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="ws-note-1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    outcome = _run(
        bridge, store, session, assistant.id, session_system_prompt="BASE PROMPT"
    )

    assert outcome.status == "done"
    assert gateway.systems, "expected a primary model call"
    primary = gateway.systems[0]
    assert primary.startswith("BASE PROMPT")
    assert "NOT running in the default workspace" in primary
    assert "Notes Workspace" in primary


def test_run_reply_adds_no_workspace_note_for_the_default_workspace(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID

    gateway = _RecordingGateway()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(workspace_id=DEFAULT_WORKSPACE_ID)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    outcome = _run(
        bridge, store, session, assistant.id, session_system_prompt="BASE PROMPT"
    )

    assert outcome.status == "done"
    assert gateway.systems, "expected a primary model call"
    assert "NOT running in the default workspace" not in gateway.systems[0]


def test_run_reply_refuses_write_file_in_an_ephemeral_session_end_to_end(
    tmp_path, monkeypatch
):
    """F4 (final-review): agent tool calls are a 9th, ungated local-write
    sink -- an ordinary reply in a temporary Console session can compose
    and dispatch `write_file` (a gateable built-in) exactly like any other
    session, independently of the Console UI action-id registry in
    `Chat/console_ephemeral.py`. Verified end-to-end through `run_reply`,
    mirroring `test_run_reply_threads_session_workspace_id_end_to_end`'s
    own pattern (the bridge builds its own `BuiltinToolProvider` internally
    with no seam for a test double) -- the RUNNING session's `ephemeral`
    flag must reach it via `_store.session_is_ephemeral`.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Tools.file_operation_tools import WriteFileTool

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    called = {"n": 0}

    async def _recording_execute(self, **kwargs):
        called["n"] += 1
        return {"success": True}

    monkeypatch.setattr(WriteFileTool, "execute", _recording_execute)

    scripts = [
        [_fence("write_file", {"file_path": "note.txt", "content": "hi"})],
        ["I could not save that."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts)
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome.status == "done"
    assert called["n"] == 0, "write_file must never execute in a temporary chat"
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a refused tool call still drops a TOOL marker"
    assert any("temporary chat" in marker.content for marker in tool_rows)

    # CONTROL: the identical scripted call executes normally outside a
    # temporary chat.
    called["n"] = 0
    db2 = AgentRunsDB(tmp_path / "runs2.db", client_id="t")
    store2 = ConsoleChatStore()
    normal_session = store2.create_session()
    store2.append_message(normal_session.id, role=ConsoleMessageRole.USER, content="hi")
    normal_assistant = store2.append_message(
        normal_session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    scripts2 = [
        [_fence("write_file", {"file_path": "note.txt", "content": "hi"})],
        ["Saved."],
    ]
    bridge2 = ConsoleAgentBridge(
        agent_runs_db=db2, store=store2, provider_gateway=_ChunkGateway(scripts2)
    )
    outcome2 = _run(
        bridge2,
        store2,
        normal_session,
        normal_assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome2.status == "done"
    assert called["n"] == 1, "write_file must execute normally outside a temporary chat"


def test_leaked_prose_before_disobedient_fence_is_reset_not_garbled(tmp_path):
    # Finding A repro: a disobedient turn streams prose live, THEN a tool
    # fence, in the same response. The gate has already forwarded the prose
    # to the store by the time the loop classifies the turn as a tool call.
    # That leaked prose must not survive to garble the real final answer
    # that streams onto the same assistant message afterward.
    scripts = [
        [
            "Let me check that ",
            "for you.\n```tool_call\n",
            '{"name": "calculator", "arguments": {"expression": "6*7"}}\n```',
        ],
        ["42."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    assert outcome.final_text == "42."
    assert store.get_message(aid).content == "42."


# -- Task 5: native provider tool-calls through the streaming adapter,
# plus the [console] native_tool_calls kill-switch. --


def test_native_tool_call_round_trip_streams_final_answer(tmp_path):
    bridge, db, store, session, aid = _bridge(
        tmp_path, [[_native_calls("get_current_datetime", {})], ["It is ", "now."]]
    )
    outcome = _run(bridge, store, session, aid, resolution=_native_resolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "It is now."
    gateway = bridge._gateway
    assert gateway.tools_seen[0] is not None  # tools= sent on turn 1
    names = [t["function"]["name"] for t in gateway.tools_seen[0]]
    assert "get_current_datetime" in names
    kinds = [step["kind"] for step in db.list_runs("conv-1")[0]["steps"]]
    assert "tool_call" in kinds and "tool_result" in kinds
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a native tool turn must drop a TOOL marker too"
    assert any("get_current_datetime" in marker.content for marker in tool_rows)


def test_native_multi_call_round_without_summary_emits_no_planning(
    tmp_path,
) -> None:
    calls = ProviderToolCalls(
        tool_calls=(
            {
                "id": "calc-1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "6*7"}),
                },
            },
            {
                "id": "calc-2",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "7*8"}),
                },
            },
        )
    )
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [[calls], ["done."]],
    )

    outcome = _run(bridge, store, session, aid, resolution=_native_resolution())
    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)

    assert outcome.status == "done"
    assert [marker.activity_presentation.kind for marker in live] == [
        "tool",
        "tool",
    ]
    assert not any(marker.activity_presentation.kind == "planning" for marker in live)
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)


def test_unsafe_model_summary_emits_no_planning_with_resume_parity(
    tmp_path,
) -> None:
    scripts = [
        [
            "<analysis>PRIVATE_REASONING_CANARY</analysis>\n",
            _fence("calculator", {"expression": "6*7"}),
        ],
        ["done."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)
    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)

    assert outcome.status == "done"
    assert all(marker.activity_presentation.kind != "planning" for marker in live)
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)
    assert "PRIVATE_REASONING_CANARY" not in repr(_activity_marker_signature(live))
    assert "PRIVATE_REASONING_CANARY" not in repr(_activity_marker_signature(resumed))


def test_persona_buddy_tool_step_uses_real_on_step_and_releases_result(tmp_path):
    """The bridge's real step callback brackets tool execution exactly."""
    buddy = PersonaBuddyController()

    class RecordingAdapter(PersonaBuddyConsoleAdapter):
        def __init__(self):
            super().__init__(buddy)
            self.observed: list[tuple[str, int]] = []
            self.released_runs: list[str] = []

        def tool_step(self, run_id, sequence, kind):
            result = super().tool_step(run_id, sequence, kind)
            if kind in {"tool_call", "tool_result", "error"}:
                self.observed.append((kind, self.active_owner_count("tool")))
            return result

        def release_run(self, run_id):
            self.released_runs.append(run_id)
            super().release_run(run_id)

    sink = RecordingAdapter()
    gateway = _ChunkGateway(
        [[_native_calls("get_current_datetime", {})], ["It is now."]]
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        buddy_sink=sink,
    )

    outcome = _run(
        bridge, store, session, assistant.id, resolution=_native_resolution()
    )

    assert outcome.status == "done"
    assert sink.observed[:2] == [("tool_call", 1), ("tool_result", 0)]
    assert sink.released_runs
    assert sink.active_owner_count("tool") == 0


def test_native_leaked_prose_is_reset_before_final_answer(tmp_path):
    """Prose streamed before the ProviderToolCalls arrives must not survive
    (Finding-A parity with the fence path)."""
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [["Let me check. ", _native_calls("get_current_datetime", {})], ["Done."]],
    )
    outcome = _run(bridge, store, session, aid, resolution=_native_resolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "Done."


def test_native_kill_switch_off_stays_on_fence_path(tmp_path):
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [[_fence("get_current_datetime", {})], ["Done."]],
        native_tools_enabled=lambda: False,
    )
    outcome = _run(bridge, store, session, aid, resolution=_native_resolution())
    assert outcome.status == "done"
    assert bridge._gateway.tools_seen[0] is None  # no tools= despite groq


def test_captured_native_tools_override_beats_later_callback_change(tmp_path):
    enabled = True
    bridge, _db, store, session, aid = _bridge(
        tmp_path,
        [[_native_calls("get_current_datetime", {})], ["Done."]],
        native_tools_enabled=lambda: enabled,
    )
    enabled = False

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=_native_resolution(),
        native_tools_enabled=True,
    )

    assert outcome.status == "done"
    assert bridge._gateway.tools_seen[0] is not None


def test_multi_turn_run_reuses_one_event_loop_across_chat_call_turns(tmp_path):
    """PR #629 Fix 1(c) (Gemini HIGH x2 + Qodo-8): ``_StreamingModelAdapter.
    chat_call`` used to bridge every turn via its own ``asyncio.run()`` --
    a fresh loop per turn, and therefore (per the gateway's per-loop
    ``_active_http_client`` swap) a client swap/churn on every single turn
    of a run. ``run_reply`` must create ONE event loop per invocation and
    reuse it for every turn -- the tool-call turn and the final-answer turn
    here -- so at most one swap happens per run."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: tool call
        ["It is ", "42."],  # turn 2: final answer
    ]
    seen_loops = []

    class _LoopSpyGateway(_ChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            seen_loops.append(asyncio.get_running_loop())
            async for chunk in super().stream_chat(resolution, messages, tools=tools):
                yield chunk

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_LoopSpyGateway(scripts)
    )
    outcome = _run(bridge, store, session, assistant.id)

    assert outcome.status == "done"
    assert len(seen_loops) == 2, "both turns of this run must reach the gateway"
    assert seen_loops[0] is seen_loops[1], (
        "every turn of one run_reply invocation must share the same event "
        "loop -- a fresh loop per turn is exactly the per-turn churn Fix "
        "1(c) removes"
    )
    assert seen_loops[0].is_closed(), (
        "the run's shared loop must be closed once run_reply returns"
    )


def test_a_concurrent_child_runs_alongside_the_parent_on_its_own_loop(tmp_path):
    """PR2a Task 6.5: a fleet child's turn must survive overlapping the
    parent's. PR3a-1 Task 1: on a lifeline of its OWN, not the turn's.

    Found by probe when the fleet default was flipped, and it failed every
    overlapping child: `chat_call` drove the shared loop with
    `run_until_complete` from whichever thread was calling, which is only
    sound while the whole run tree is single-threaded. A loop may be driven
    by exactly one thread, so the second concurrent caller got
    ``RuntimeError: This event loop is already running``, its coroutine was
    dropped un-awaited, and the child's run row persisted `error` -- silent
    from the parent's side, which just saw a failed sub-agent. `chat_call`
    now submits with `run_coroutine_threadsafe` to a loop running on its
    own thread.

    The gateway awaits inside every turn so the parent and the child really
    are in flight together (a turn that yields without awaiting can slip
    through the window); the child additionally blocks until the parent has
    entered its own next turn, so the overlap is pinned rather than raced
    for. PR #629 Fix 1(c) is re-asserted here too, now scoped to the agent
    it was always about: the PRIMARY agent still runs every turn of the run
    on ONE loop, so the gateway's per-loop client is swapped at most once
    per run. The child's own loop is the price PR3a-1 pays for a child that
    can outlive its turn -- measured in that task's report.
    """
    parent_in_flight = threading.Event()
    parent_loops = []
    child_loops = []

    class _OverlappingGateway(_FleetChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                child_loops.append(asyncio.get_running_loop())
            else:
                parent_in_flight.set()
                parent_loops.append(asyncio.get_running_loop())
            # Yield control so the other agent's coroutine can be scheduled
            # onto this same loop while this one is still open.
            await asyncio.sleep(0.01)
            async for chunk in super().stream_chat(resolution, messages, tools=tools):
                yield chunk

    def gated_child_turn():
        assert parent_in_flight.wait(5), "the parent never started a turn"
        return ["child answer"]

    gateway = _OverlappingGateway(
        [
            [_fence("spawn_subagent", {"task": "child work"})],
            [_fence("wait_agents", {})],
            ["parent final"],
        ],
        [gated_child_turn],
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-overlap")

    assert outcome.status == "done"
    assert outcome.final_text == "parent final"
    # The child completed -- it did NOT die on the loop.
    child_runs = [
        r for r in db.list_runs("conv-overlap") if r["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1
    assert child_runs[0]["status"] == RUN_DONE
    assert child_runs[0]["result"] == "child answer"
    # ... and its answer really reached the parent, through wait_agents.
    assert "child answer" in str(outcome.steps)
    # PR3a-1 Task 1: the PRIMARY agent still runs every one of its turns on
    # the ONE per-turn loop (Fix 1(c) -- at most one httpx client swap per
    # run), and that loop is still closed when the turn returns.
    assert len(parent_loops) == 3
    assert len({id(loop) for loop in parent_loops}) == 1
    assert parent_loops[0].is_closed()
    # The child no longer shares it. It owns a lifeline of its own from
    # birth -- so the turn's teardown cannot kill a call it has in flight
    # (see `test_a_fleet_child_completes_its_model_call_after_the_turn_
    # loop_is_gone`) -- and that lifeline is torn down when the CHILD
    # finishes, which by here it has.
    assert len(child_loops) == 1
    assert child_loops[0] is not parent_loops[0]
    assert child_loops[0].is_closed()


async def _await_event(event: threading.Event, timeout: float) -> bool:
    """Await a threading.Event without blocking the loop it is awaited on.

    A bare ``event.wait()`` inside an async gateway double would block that
    coroutine's whole event loop, which would mask the very cross-loop
    behaviour these tests pin (on the pre-fix code the parent and the child
    share one loop, and a blocked loop is indistinguishable from a dead one).
    """
    deadline = time.monotonic() + timeout
    while not event.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class _CrossTurnGateway:
    """Parent answers WITHOUT collecting; the child's turn outlives it.

    The parent's final-answer turn waits until the child is actually inside
    a ``stream_chat`` call, and the child's turn then waits until the test
    says the spawning turn has fully returned -- so the child's model call
    is guaranteed to be completed after ``run_reply``'s own loop is closed.
    """

    def __init__(self, turn_over: threading.Event):
        self._turn_over = turn_over
        self.child_in_flight = threading.Event()
        self._parent_turns = [
            [_fence("spawn_subagent", {"task": "outlive the turn"})],
            ["parent final"],
        ]
        self._lock = threading.Lock()
        self.parent_loops = []
        self.child_loops = []

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        loop = asyncio.get_running_loop()
        if system.startswith(SUBAGENT_PROMPT_PREFIX):
            self.child_loops.append(loop)
            self.child_in_flight.set()
            assert await _await_event(self._turn_over, 30.0), (
                "the spawning turn never returned"
            )
            yield "child answer after the turn"
            return
        with self._lock:
            chunks = self._parent_turns.pop(0)
            is_final_turn = not self._parent_turns
        self.parent_loops.append(loop)
        if is_final_turn:
            # Pin the interleave: the parent may not finish its turn until
            # the child is demonstrably mid-model-call.
            assert await _await_event(self.child_in_flight, 30.0), (
                "the child never reached the gateway"
            )
        for chunk in chunks:
            yield chunk


def test_a_fleet_child_completes_its_model_call_after_the_turn_loop_is_gone(
    tmp_path, monkeypatch
):
    """PR3a-1 Task 1: a fleet child must own its model-call lifeline.

    Before this task every agent of a run -- primary and children alike --
    bridged to the provider through the ONE loop `run_reply` builds per
    invocation, which `run_reply`'s `finally` stops, joins and CLOSES. The
    teardown comment stated the invariant that made that safe: "by this
    point `run_turn` has already settled every fleet child ... so nothing
    should still be submitting." PR 3a makes children outlive their
    spawning turn, so that sentence stops being true and the dependency is
    actively DESTROYED under a live child: its `run_coroutine_threadsafe`
    future is never scheduled again and its turn can never complete.

    The fix is not a transfer at settle time; a child owns its own loop and
    driver thread FROM BIRTH, torn down when the child finishes. This test
    asserts the observable consequence -- the child's persisted run row
    reaches a terminal status with its real answer -- with the turn's loop
    verified closed before the child's model call is even released.
    """
    turn_over = threading.Event()
    gateway = _CrossTurnGateway(turn_over)
    # Task 1 proves only that a child CAN survive; end-of-turn settling
    # (which cancels and then abandons stragglers) is Task 2's to change,
    # so it is bypassed HERE, in the test, and not in production code.
    monkeypatch.setattr(AgentService, "_settle_fleet", lambda self, *a, **k: None)
    # Bound the failure mode. Without a per-child lifeline the child's
    # future is submitted onto the turn's loop and is never completed once
    # that loop closes, so the production `_CHAT_CALL_TIMEOUT_SECONDS`
    # backstop of an hour would turn a regression into a CI hang. The
    # fixed path needs milliseconds, so 30s can never pre-empt it.
    monkeypatch.setattr(console_agent_bridge, "_CHAT_CALL_TIMEOUT_SECONDS", 30.0)
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-cross-turn")

        assert outcome.status == "done"
        assert outcome.final_text == "parent final"
        # The turn is over and its loop really is gone ...
        assert gateway.parent_loops, "the parent never reached the gateway"
        assert gateway.parent_loops[0].is_closed()
        # ... while the child is still mid-call on a lifeline of its own.
        assert gateway.child_loops, "the child never reached the gateway"
        assert gateway.child_loops[0] is not gateway.parent_loops[0]
        assert not gateway.child_loops[0].is_closed()
        # Release the child: from here it has to finish a model call with
        # the spawning turn's loop already closed.
        turn_over.set()
        _join_fleet_threads(timeout=30.0)
    finally:
        # Never leave a wedged child blocking the whole suite's teardown.
        turn_over.set()

    child_runs = [
        r for r in db.list_runs("conv-cross-turn") if r["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1
    assert child_runs[0]["status"] == RUN_DONE
    assert child_runs[0]["result"] == "child answer after the turn"


def test_provider_stream_signal_survives_primary_tool_and_final_turns(tmp_path):
    signals = ConsoleProviderStreamSignals()
    gateway = _SignalChunkGateway(
        [
            [_fence("calculator", {"expression": "6*7"})],
            ["It is ", "42."],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert gateway.signals_seen == [signals, signals]
    assert all(item is signals for item in gateway.signals_seen)


def test_provider_stream_signal_survives_subagent_turns(tmp_path):
    signals = ConsoleProviderStreamSignals()
    gateway = _SignalChunkGateway(
        [
            [_fence("spawn_subagent", {"task": "compute 1+1"})],
            ["2"],
            ["Done: 2."],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    # PR3a-1 Task 2: the sub-agent outlives the turn by default, so its
    # provider call -- the third one this test is counting -- may still be
    # in flight when `run_reply` returns. The subject here is that the
    # SAME signals object reaches a child's call, not when it does.
    _join_fleet_threads()
    assert outcome.status == "done"
    assert gateway.signals_seen == [signals, signals, signals]
    assert all(item is signals for item in gateway.signals_seen)


def test_provider_stream_signal_is_never_reset_by_bridge(tmp_path):
    signals = ConsoleProviderStreamSignals()
    signals.mark_synthetic_fallback()
    gateway = _SignalChunkGateway([["Already marked."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert gateway.signals_seen == [signals]
    assert gateway.signal_states_seen == [True]
    assert signals.synthetic_fallback_emitted is True


def test_qwencloud_terminal_usage_reaches_agent_native_budget_without_fallback(
    tmp_path,
):
    usage = {
        "input_tokens": 9,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 12,
    }

    def chat_api_call(**_kwargs):
        return iter(
            (
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    signals = ConsoleProviderStreamSignals()
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model="qwen3.8-max",
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert outcome.final_text == "answer"
    assert outcome.total_tokens == 12
    assert store.get_message(aid).content == "answer"
    assert signals.synthetic_fallback_emitted is False
    assert signals.usage_payloads() == [usage]


# TASK-18603: `expected_total` is now CACHE-WEIGHTED, not the flat sum.
# 6,656 of these input tokens are cache reads, billed at 0.3 vs 3.0 per
# mtok for claude-sonnet-4-6 -- a tenth -- so they consume a tenth of the
# run budget instead of full price. 3,571 uncached + 6,656*0.1 + 727 output
# = 4,963.6, rounded up. The flat totals this used to assert (10,954 /
# 11,065) are still what `_usage_total_tokens` reports and still what the
# provider billed in RAW tokens; they were just never what the run cost.
#
# The third bucket (TASK-18607): the gateway's normalization still folds
# `cache_creation_input_tokens` into `prompt_tokens` -- readers of the flat
# sum are unchanged -- but now ALSO preserves it as
# `prompt_tokens_details.cache_creation_tokens`, so the budget prices a
# cache WRITE at its real published rate (3.75 vs 3.0 per mtok on
# claude-sonnet-4-6 -- 1.25x) instead of 1.0x, matching the
# Anthropic-native path. The 111-token case therefore adds
# ceil(111 * 1.25) worth of budget, not 111.
@pytest.mark.parametrize(
    ("cache_creation_input_tokens", "expected_total"),
    [(0, 4_964), (111, 5_103)],
    ids=("cache-read", "cache-read-and-creation"),
)
def test_anthropic_split_usage_reaches_agent_budget_with_cache_buckets(
    tmp_path,
    monkeypatch,
    cache_creation_input_tokens,
    expected_total,
):
    estimator_calls: list[str] = []
    captured_usage: list[dict] = []

    def fail_estimator(*_args, **_kwargs):
        estimator_calls.append("called")
        raise AssertionError("provider usage must bypass the local estimator")

    monkeypatch.setattr(agent_service, "count_tokens_messages", fail_estimator)
    monkeypatch.setattr(agent_service, "estimate_tokens", fail_estimator)
    real_usage_total_tokens = agent_service._usage_total_tokens

    def capture_usage(response):
        captured_usage.append(response["usage"])
        return real_usage_total_tokens(response)

    monkeypatch.setattr(agent_service, "_usage_total_tokens", capture_usage)
    input_usage = {
        "input_tokens": 3_571,
        "cache_read_input_tokens": 6_656,
        "cache_creation_input_tokens": cache_creation_input_tokens,
    }
    output_usage = {"output_tokens": 727}

    def chat_api_call(**_kwargs):
        return iter(
            (
                {"choices": [], "usage": input_usage},
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                    "usage": output_usage,
                },
            )
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    signals = ConsoleProviderStreamSignals()
    resolution = ConsoleProviderResolution(
        provider="Anthropic",
        base_url="",
        model="claude-sonnet-4-6",
        ready=True,
        readiness_key="anthropic",
        execution_key="anthropic",
        api_key="anthropic-test-key",
        streaming=True,
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model="claude-sonnet-4-6",
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert outcome.final_text == "answer"
    assert outcome.total_tokens == expected_total
    assert estimator_calls == []
    expected_details: dict = {"cached_tokens": 6_656}
    if cache_creation_input_tokens:
        expected_details["cache_creation_tokens"] = cache_creation_input_tokens
    assert captured_usage == [
        {
            "prompt_tokens": 3_571 + 6_656 + cache_creation_input_tokens,
            "prompt_tokens_details": expected_details,
            "completion_tokens": 727,
            # Still the RAW total the provider billed in tokens -- this
            # asserts the gateway's normalization: TASK-18607 adds the
            # write bucket to the DETAILS only; the flat sums are
            # unchanged. Only what the BUDGET counts it as changed.
            "total_tokens": 3_571 + 6_656 + cache_creation_input_tokens + 727,
        }
    ]
    assert signals.synthetic_fallback_emitted is False
    assert signals.usage_payloads() == [{**input_usage, **output_usage}]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"tokens": 12},
        {"prompt_tokens": "not-a-number", "completion_tokens": -5},
        {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    ],
)
def test_provider_usage_handoff_omits_absent_or_nonpositive_payloads(payload):
    assert (
        _openai_usage_from_provider_call(
            payload,
            provider="openai",
            model="gpt-4.1",
        )
        is None
    )


@pytest.mark.parametrize(
    "count_key",
    [
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ],
)
def test_budget_usage_handoff_rejects_each_malformed_top_level_count(count_key):
    payload = {
        "prompt_tokens": 1,
        "input_tokens": 1,
        "completion_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        count_key: "1",
    }

    assert (
        _openai_usage_from_provider_call(
            payload,
            provider="openai",
            model="gpt-4.1",
        )
        is None
    )


@pytest.mark.parametrize(
    ("details_key", "details"),
    [
        ("prompt_tokens_details", None),
        ("input_tokens_details", []),
        ("input_token_details", {"cached_tokens": "1"}),
        ("completion_tokens_details", {"cached_tokens": False}),
        ("output_tokens_details", {"reasoning_tokens": 1.5}),
        ("output_token_details", {"reasoning_tokens": -1}),
    ],
)
def test_budget_usage_handoff_rejects_each_malformed_details_shape_or_count(
    details_key,
    details,
):
    payload = {
        "prompt_tokens": 1,
        "input_tokens": 1,
        "completion_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        details_key: details,
    }

    assert (
        _openai_usage_from_provider_call(
            payload,
            provider="openai",
            model="gpt-4.1",
        )
        is None
    )


def test_openai_usage_handoff_preserves_chat_completion_cache_details():
    assert _openai_usage_from_provider_call(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
        provider="openai",
        model="gpt-4.1",
    ) == {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 80},
        "completion_tokens": 20,
        "total_tokens": 120,
    }


def test_normalized_usage_round_trips_cache_write_bucket():
    """Native and gateway-normalized usage parse to identical buckets.

    TASK-18607 AC#3/#4: normalization must preserve Anthropic's write
    bucket so the native and normalized shapes account identically, while
    the flat ``prompt_tokens`` sum -- what the cost ticker and persistence
    read -- keeps the full billed total.
    """
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    native = {
        "input_tokens": 3_571,
        "cache_read_input_tokens": 6_656,
        "cache_creation_input_tokens": 111,
        "output_tokens": 727,
    }
    normalized = _openai_usage_from_provider_call(
        native, provider="anthropic", model="claude-sonnet-4-6"
    )
    assert normalized["prompt_tokens"] == 3_571 + 6_656 + 111
    assert normalized["prompt_tokens_details"] == {
        "cached_tokens": 6_656,
        "cache_creation_tokens": 111,
    }
    direct = ProviderUsage.from_provider_payload(
        native, provider="anthropic", model="claude-sonnet-4-6"
    )
    round_tripped = ProviderUsage.from_provider_payload(
        normalized, provider="anthropic", model="claude-sonnet-4-6"
    )
    assert round_tripped == direct


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True, "output_tokens": 3, "total_tokens": 4},
        {"input_tokens": "9", "output_tokens": 3, "total_tokens": 12},
        {"input_tokens": 9.5, "output_tokens": 3, "total_tokens": 12},
        {"input_tokens": -1, "output_tokens": 3, "total_tokens": 2},
        {"input_tokens": 9, "output_tokens": 3, "total_tokens": "12"},
        {
            "input_tokens": 9,
            "input_tokens_details": {"cached_tokens": False},
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 1.5},
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_tokens_details": [],
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "cache_read_input_tokens": "2",
            "cache_creation_input_tokens": 1.5,
            "output_tokens": 3,
        },
        {
            "input_tokens": 9,
            "input_token_details": {"cached_tokens": 0, "audio_tokens": True},
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_token_details": {
                "cached_tokens": 2,
                "cached_tokens_details": [],
            },
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 3,
            "output_tokens_details": {"accepted_prediction_tokens": "1"},
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_token_details": {"text_tokens": 9.5},
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "output_tokens": 3,
            "output_token_details": {"image_tokens": -1},
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "input_token_details": {
                "cached_tokens": 2,
                "cached_tokens_details": {"audio_tokens": "1"},
            },
            "output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "input_tokens": 9,
            "output_tokens": 3,
            "output_tokens_details": {"rejected_prediction_tokens": False},
            "total_tokens": 12,
        },
    ],
    ids=(
        "bool",
        "numeric-string",
        "fractional-float",
        "negative",
        "malformed-total",
        "nested-bool",
        "nested-float",
        "malformed-details-shape",
        "anthropic-cache-fields",
        "boolean-audio",
        "malformed-cached-details-shape",
        "numeric-string-accepted-prediction",
        "fractional-singular-text",
        "negative-singular-image",
        "malformed-nested-cached-count",
        "boolean-rejected-prediction",
    ),
)
def test_malformed_streamed_usage_uses_agent_estimator_instead_of_coercion(
    tmp_path,
    monkeypatch,
    usage,
):
    estimator_calls: list[str] = []

    def count_messages(*_args, **_kwargs):
        estimator_calls.append("messages")
        return 40

    def count_text(*_args, **_kwargs):
        estimator_calls.append("text")
        return 2

    monkeypatch.setattr(agent_service, "count_tokens_messages", count_messages)
    monkeypatch.setattr(agent_service, "estimate_tokens", count_text)

    def chat_api_call(**_kwargs):
        return iter(
            (
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    signals = ConsoleProviderStreamSignals()
    resolution = ConsoleProviderResolution(
        provider="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1",
        ready=True,
        readiness_key="openai",
        execution_key="openai",
        api_key="openai-test-key",
        streaming=True,
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model="gpt-4.1",
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert outcome.total_tokens == 42
    assert estimator_calls == ["messages", "text"]
    # The raw aggregate remains tolerant/unchanged for persistence and cost
    # consumers; only the AgentService budget handoff fails closed.
    assert signals.usage_payloads() == [usage]


def test_exact_zero_streamed_usage_counts_remain_authoritative(
    tmp_path,
    monkeypatch,
):
    usage = {
        "input_tokens": 0,
        "input_tokens_details": {
            "cached_tokens": 0,
            "audio_tokens": 0,
            "text_tokens": 0,
            "image_tokens": 0,
            "cached_tokens_details": {
                "audio_tokens": 0,
                "text_tokens": 0,
                "image_tokens": 0,
                "vendor_metadata": "preserved",
            },
        },
        "input_token_details": {
            "cached_tokens": 0,
            "audio_tokens": 0,
            "text_tokens": 0,
            "image_tokens": 0,
            "cached_tokens_details": {
                "audio_tokens": 0,
                "text_tokens": 0,
                "image_tokens": 0,
            },
        },
        "output_tokens": 3,
        "output_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "text_tokens": 0,
            "image_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
            "vendor_metadata": "preserved",
        },
        "output_token_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "text_tokens": 0,
            "image_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "total_tokens": 3,
    }

    def fail_estimator(*_args, **_kwargs):
        raise AssertionError("valid exact-integer usage must remain authoritative")

    monkeypatch.setattr(agent_service, "count_tokens_messages", fail_estimator)
    monkeypatch.setattr(agent_service, "estimate_tokens", fail_estimator)

    def chat_api_call(**_kwargs):
        return iter(
            (
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model="qwen3.8-max",
    )

    assert outcome.status == "done"
    assert outcome.total_tokens == 3


def test_concurrent_subagent_adapter_calls_keep_terminal_usage_call_scoped(tmp_path):
    terminal_seen = threading.Barrier(3)
    release_terminal = threading.Event()
    usage_by_prompt = {"alpha": 12, "beta": 34}

    def chat_api_call(**kwargs):
        prompt = kwargs["messages_payload"][-1]["content"]
        total = usage_by_prompt[prompt]

        def stream():
            yield {"choices": [{"delta": {"content": prompt}}]}
            yield {
                "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                "usage": {
                    "input_tokens": total - 3,
                    "input_tokens_details": {"cached_tokens": total // 6},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": total,
                },
            }
            terminal_seen.wait(timeout=5)
            release_terminal.wait(timeout=5)

        return stream()

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    signals = ConsoleProviderStreamSignals()
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    adapter = _StreamingModelAdapter(
        store=store,
        provider_gateway=gateway,
        resolution=resolution,
        assistant_message_id=assistant.id,
        should_cancel=lambda: False,
        loop=loop,
        native_tools=False,
        provider_stream_signals=signals,
    )
    responses: dict[str, dict] = {}
    errors: list[BaseException] = []

    def call(prompt: str) -> None:
        try:
            responses[prompt] = adapter.chat_call(
                messages_payload=[
                    {"role": "system", "content": SUBAGENT_PROMPT_PREFIX},
                    {"role": "user", "content": prompt},
                ]
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    callers = [
        threading.Thread(target=call, args=(prompt,)) for prompt in usage_by_prompt
    ]
    try:
        for caller in callers:
            caller.start()
        terminal_seen.wait(timeout=5)
        assert sorted(
            payload["total_tokens"] for payload in signals.usage_payloads()
        ) == [12, 34]
        release_terminal.set()
        for caller in callers:
            caller.join(timeout=5)
    finally:
        release_terminal.set()
        for caller in callers:
            caller.join(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()

    assert errors == []
    assert all(not caller.is_alive() for caller in callers)
    assert {
        prompt: response["usage"]["total_tokens"]
        for prompt, response in responses.items()
    } == usage_by_prompt
    assert responses["alpha"]["usage"] == {
        "prompt_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens": 3,
        "completion_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 12,
    }
    assert sorted(payload["total_tokens"] for payload in signals.usage_payloads()) == [
        12,
        34,
    ]


def test_provider_stream_signal_omission_preserves_legacy_gateway_signature(tmp_path):
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Unchanged."]])

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    assert bridge._gateway.calls == 1
    assert store.get_message(aid).content == "Unchanged."


def test_spawn_renders_marker_and_persists_linked_subagent(tmp_path):
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn
        ["Done: ", "2."],  # primary final
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    # PR3a-1 Task 2: the child outlives the turn, so its run ROW may not
    # exist yet when `run_reply` returns. This test is about the marker
    # and the lineage, not about when the row lands.
    _join_fleet_threads()
    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-1") == 1
    live_markers = _tool_messages(store, session.id)
    resumed_markers = _resume_tool_messages(db)
    spawn_markers = [
        marker for marker in live_markers if "sub-agent" in marker.content.lower()
    ]
    assert spawn_markers
    assert [marker.activity_presentation.kind for marker in live_markers] == [
        "spawn",
        "tool",
    ]
    assert not any(
        marker.activity_presentation.kind == "planning" for marker in live_markers
    )
    live_signature = _activity_marker_signature(live_markers)
    resumed_signature = _activity_marker_signature(resumed_markers)
    assert live_signature[:1] == resumed_signature[:1]
    assert [item[1:] for item in live_signature] == [
        item[1:] for item in resumed_signature
    ]
    child = next(row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent")
    assert f"run:{child['id']}" not in live_signature[-1][0]
    assert f"run:{child['id']}" in resumed_signature[-1][0]
    assert live_signature[-1][0].split(": compute 1+1", 1)[1] == (
        resumed_signature[-1][0].split(": compute 1+1", 1)[1]
    )
    snap = bridge.live_snapshot("conv-1")
    assert any(s.text for s in snap.subagents)


def test_tool_marker_with_brackets_renders_literally_not_escaped(tmp_path):
    # Both TOOL-marker consumers (console_transcript.py's Content.assemble
    # and chat_screen.py's legacy Text(...) fallback) render markup-off, so
    # a bracketed task/result must survive as literal text -- not as a
    # backslash-escaped sequence (`fetch \[docs]`) that a markup parser
    # would need to consume but never runs.
    scripts = [
        [_fence("spawn_subagent", {"task": "fetch [docs]"})],  # primary turn 1
        ["ok"],  # sub-agent turn
        ["Done."],  # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    spawn_markers = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()
    ]
    assert spawn_markers
    assert "[docs]" in spawn_markers[0].content
    assert "\\[docs]" not in spawn_markers[0].content


def test_supersede_marks_previous_primary_and_tree(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [["one."], ["two."]])
    _run(bridge, store, session, aid)  # first run
    first = db.list_runs("conv-1")[0]
    assert first["status"] == "done"
    # Second run supersedes the previous primary.
    aid2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
    _run(bridge, store, session, aid2, supersede_previous=True)
    prior = db.get_run(first["id"])
    assert prior["status"] == "superseded"


def test_stop_persists_cancelled(tmp_path):
    # A long tool loop; cancel flips after the first step.
    scripts = [[_fence("calculator", {"expression": "1"})], ["never reached"]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    flags = iter([False, True])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"


def test_stop_before_first_chunk_persists_cancelled_not_error(tmp_path):
    """Plan-B agent-runtime gate Finding 1: reproduces the live-repro'd race
    where Stop is clicked before the (slow) provider has streamed a single
    chunk. The controller's ``stop_active_run`` marks the assistant message
    "stopped" and flips ``should_cancel`` *before* the first chunk arrives;
    when it finally does, ``append_stream_chunk`` must drop it silently
    (store-level fix) instead of raising, so the run settles as
    "cancelled" (AgentRunsDB) rather than "error" with a step-log message
    of "Cannot append stream chunks to a stopped message.\""""
    scripts = [["late", " chunk", " arrives", " anyway."]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    # Mirror ConsoleChatController.stop_active_run(): the message is
    # finalized to "stopped" up front, before any chunk streamed. The
    # first should_cancel() poll (at the top of the loop, before the model
    # call) still returns False -- the run genuinely starts -- then flips
    # True from the second poll onward, mirroring how ``_stop_requested``
    # was already True by the time the slow provider's first chunk
    # finally arrived inside ``_consume()``.
    store.mark_message_stopped(aid)
    flags = iter([False])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))

    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
    from tldw_chatbook.Agents.agent_models import STEP_ERROR

    assert not any(s.kind == STEP_ERROR for s in outcome.steps)
    # The message stays exactly as Stop left it -- no late content leaked in.
    stored = store.get_message(aid)
    assert stored.status == "stopped"
    assert stored.content == ""


def test_stop_mid_final_answer_persists_cancelled_and_store_agrees(tmp_path):
    # Finding B: a Stop that lands mid a plain final-answer stream (no tool
    # call to dispatch) must not be downgraded to "done" -- the outcome
    # status, the persisted AgentRunsDB row, and the store's own streamed
    # content must all agree that the run was cancelled.
    scripts = [["Par", "tial", " answer."]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    flags = iter([False, True])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
    assert store.get_message(aid).content == outcome.final_text


# -- Plan-B agent-runtime gate Finding 2: rail summary re-derived from
# AgentRunsDB after a restart, when this bridge instance has no in-process
# live-run record for the conversation. --


def test_historical_snapshot_idle_when_conversation_never_ran(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snap = bridge.historical_snapshot("conv-never-seen")
    assert snap.status == "idle"
    assert snap.steps == ()
    assert snap.subagents == ()


def test_historical_snapshot_derives_status_steps_and_subagents_from_db(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        primary_id,
        [
            {
                "index": 0,
                "kind": "model",
                "summary": "The capital of France is Paris.",
                "tool_name": "",
                "args": None,
                "result": "",
                "created_at": "",
            },
        ],
    )
    db.set_status(primary_id, "done", result="The capital of France is Paris.")
    sub_id = db.create_run(
        conversation_id="conv-1",
        agent_kind="subagent",
        task="research pricing",
        parent_run_id=primary_id,
    )
    db.set_status(sub_id, "done", result="done researching")

    # Fresh bridge instance -- simulates an app restart: `_live` starts empty.
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.live_snapshot("conv-1").status == "idle"

    snap = bridge.historical_snapshot("conv-1")
    assert snap.status == "done"
    assert len(snap.steps) == 1
    assert "Paris" in snap.steps[0].text
    assert snap.steps[0].agent_kind == "primary"
    assert len(snap.subagents) == 1
    assert snap.subagents[0].text == "research pricing"
    assert snap.subagents[0].status == "done"
    # PR2b Task 4: a historical/resumed sub-agent row now carries its own
    # permanent run id -- the rail's per-row click-through
    # (`ConsoleAgentController._console_agent_drilldown_target_run_id`)
    # resolves a clicked row directly to its run, and a resumed row with an
    # empty id could never be drilled into.
    assert snap.subagents[0].run_id == sub_id


def test_historical_snapshot_ignores_subagents_of_other_runs(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(primary_id, "done", result="ok")
    other_primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(other_primary_id, "superseded")
    db.create_run(
        conversation_id="conv-1",
        agent_kind="subagent",
        task="orphaned",
        parent_run_id=other_primary_id,
    )

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snap = bridge.historical_snapshot("conv-1")
    assert snap.status == "done"
    assert snap.subagents == ()


def test_historical_snapshot_caches_per_conversation_not_hit_every_call(
    tmp_path, monkeypatch
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(primary_id, "done", result="ok")

    calls = []
    original = db.list_runs

    def spy(conversation_id, *args, **kwargs):
        calls.append(conversation_id)
        return original(conversation_id, *args, **kwargs)

    monkeypatch.setattr(db, "list_runs", spy)
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    first = bridge.historical_snapshot("conv-1")
    second = bridge.historical_snapshot("conv-1")
    assert first == second
    assert len(calls) == 1  # the 0.2s rail poll must not re-hit the DB

    # A different conversation is a separate cache entry.
    bridge.historical_snapshot("conv-2")
    assert len(calls) == 2


# -- Plan-B final-review Medium-1: inline transcript TOOL markers re-derive
# from AgentRunsDB on resume, the same way the rail already does. --


def test_format_agent_step_marker_matches_each_live_marker_shape():
    assert (
        format_agent_step_marker(STEP_SPAWN, summary="research pricing")
        == "⤷ spawned sub-agent: research pricing"
    )
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="calculator", result="42")
        == "⚙ calculator → 42"
    )
    assert format_agent_step_marker(STEP_ERROR, summary="boom") == "⚠ boom"
    # Quiet tool-catalog steps and plain model steps never produce a marker.
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="find_tools", result="[]")
        is None
    )
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="load_tools", result="[]")
        is None
    )
    assert format_agent_step_marker(STEP_MODEL, summary="The answer is 42.") is None


def test_long_tool_result_marker_is_collapsed_with_truncation_affordance():
    """TASK-350: a spawn_subagent whose result IS the full answer must not be
    dumped verbatim into the TOOL marker — it duplicated the assistant bubble
    word-for-word. Collapse it to a preview and mark the truncation plus how much
    is hidden, so the marker reads as provenance, not a second copy."""
    answer = "### Understanding SQLite WAL. " + "word " * 300  # long answer
    marker = format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="spawn_subagent", result=answer
    )
    assert marker.startswith("\u2699 spawn_subagent \u2192 ### Understanding SQLite")
    assert "\u2026" in marker  # ellipsis marks the cut
    assert "chars)" in marker  # explicit "how much more" affordance
    assert len(marker) < len(answer) // 2  # collapsed, not a full duplicate


def test_short_tool_result_marker_is_unchanged():
    # Short results stay verbatim — no ellipsis, no affordance.
    marker = format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="calculator", result="42"
    )
    assert marker == "\u2699 calculator \u2192 42"
    assert "\u2026" not in marker


def test_step_truncation_cuts_on_word_boundary_with_ellipsis():
    from tldw_chatbook.Chat.console_agent_bridge import _truncate_step_text

    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = _truncate_step_text(text, limit=24)
    assert "\u2026" in out and "(+" in out and "chars)" in out
    preview = out.split("\u2026", 1)[0].strip()
    # every token shown is a whole word from the source — never a mid-word clip
    assert preview
    assert all(tok in text.split() for tok in preview.split())


def test_step_truncation_cuts_on_newline_and_tab_boundaries():
    """Qodo #3: markdown/structured results split on newlines/tabs, not just
    spaces — the boundary search must treat any whitespace as a token break so a
    fenced/heading result is not clipped mid-token."""
    from tldw_chatbook.Chat.console_agent_bridge import _truncate_step_text

    text = "### Heading\n\nsome body text that keeps going well past the limit here"
    out = _truncate_step_text(text, limit=15)
    assert out.split("\u2026", 1)[0] == "### Heading"  # cut at the newline, not "so"


@pytest.mark.parametrize(
    ("tasks", "expected_lines"),
    [
        (
            [
                {
                    "id": "10101",
                    "version": 1,
                    "content": "write tests",
                    "status": "pending",
                },
                {
                    "id": "20202",
                    "version": 1,
                    "content": "implement",
                    "status": "pending",
                },
                {
                    "id": "30303",
                    "version": 1,
                    "content": "commit",
                    "status": "pending",
                },
            ],
            ("[ ] write tests", "[ ] implement", "[ ] commit"),
        ),
        (
            [
                {
                    "id": "10101",
                    "version": 1,
                    "content": "write tests",
                    "status": "pending",
                },
                {
                    "id": "20202",
                    "version": 2,
                    "content": "implement safely",
                    "status": "completed",
                },
                {
                    "id": "30303",
                    "version": 1,
                    "content": "commit",
                    "status": "pending",
                },
            ],
            ("[ ] write tests", "[x] implement safely", "[ ] commit"),
        ),
        (
            [
                {
                    "id": "10101",
                    "version": 1,
                    "content": "write tests",
                    "status": "pending",
                },
                {
                    "id": "30303",
                    "version": 1,
                    "content": "commit",
                    "status": "pending",
                },
            ],
            ("[ ] write tests", "[ ] commit"),
        ),
    ],
    ids=("create", "update", "delete"),
)
def test_format_todo_marker_renders_task_snapshots_in_creation_order(
    tasks, expected_lines
):
    text = format_todo_marker(tasks)

    assert text == "\n".join(
        ["☰ Tasks (0 in progress):", *(f"  {line}" for line in expected_lines)]
    )


def test_format_todo_marker_uses_sanitized_active_form_for_in_progress_task():
    text = format_todo_marker(
        [
            {
                "id": "10101",
                "version": 8,
                "content": "implement",
                "status": "in_progress",
                "activeForm": "implementing\x1bright",
            }
        ]
    )

    assert text == "☰ Tasks (1 in progress):\n  [~] implementing right"


def test_format_todo_marker_does_not_render_protocol_ids_or_versions():
    text = format_todo_marker(
        [
            {
                "id": "867530912345",
                "version": 741852963,
                "content": "ship safely",
                "status": "completed",
            }
        ]
    )

    assert "867530912345" not in text
    assert "741852963" not in text
    assert "id" not in text.casefold()
    assert "version" not in text.casefold()


def test_format_todo_marker_empty_task_snapshot_reads_as_cleared():
    assert format_todo_marker([]) == "☰ Tasks cleared"


def test_format_todo_marker_truncates_long_item_text():
    # Same 200-char convention as step-marker summaries (_summarize).
    long_content = "y" * 300
    text = format_todo_marker([{"content": long_content, "status": "pending"}])
    assert text == f"☰ Tasks (0 in progress):\n  [ ] {'y' * 200}"


def test_format_todo_marker_sanitizes_before_truncating_display_text():
    content = f"{'y' * 198}\r\nYZ"
    text = format_todo_marker([{"content": content, "status": "pending"}])

    assert text == f"☰ Tasks (0 in progress):\n  [ ] {'y' * 198} Y"


@pytest.mark.parametrize("control", ["\x00", "\x1f", "\x7f", "\x80", "\x9f"])
def test_format_todo_marker_replaces_terminal_controls_without_mutating_input(
    control,
):
    tasks = [
        {
            "id": "10101",
            "version": 1,
            "content": f"left{control}right",
            "status": "pending",
        }
    ]
    before = copy.deepcopy(tasks)

    text = format_todo_marker(tasks)

    assert text == "☰ Tasks (0 in progress):\n  [ ] left right"
    assert control not in text
    assert tasks == before


def test_format_todo_marker_flattens_line_breaks_and_replaces_tab():
    # Markers stay one line per task; CRLF is one line break and tab is one control.
    text = format_todo_marker(
        [
            {
                "content": "first\nsecond\r\nthird\rfour\tfive",
                "status": "pending",
            }
        ]
    )
    assert text == ("☰ Tasks (0 in progress):\n  [ ] first second third four five")


def test_format_todo_marker_preserves_input_equality_and_aliases():
    first = {
        "id": "10101",
        "version": 3,
        "content": "first\x00task",
        "status": "pending",
    }
    second = {
        "id": "20202",
        "version": 4,
        "content": "second task",
        "status": "completed",
    }
    tasks = [first, second]
    tasks_alias = tasks
    first_alias = first
    before = copy.deepcopy(tasks)

    format_todo_marker(tasks)

    assert tasks == before
    assert tasks is tasks_alias
    assert tasks[0] is first_alias


def test_append_todo_marker_appends_tool_message_to_store(tmp_path):
    bridge, _db, store, session, _aid = _bridge(tmp_path, [])
    bridge.append_todo_marker(
        session.id, [{"content": "ship it", "status": "in_progress"}]
    )
    tool_messages = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert [m.content for m in tool_messages] == [
        "☰ Tasks (1 in progress):\n  [~] ship it"
    ]
    assert tool_messages[0].activity_presentation == ConsoleActivityPresentation(
        "tasks", "Tasks updated", "done"
    )


def test_resume_marker_messages_reproduces_live_markers_after_simulated_restart(
    tmp_path,
):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: leading fence
        ["It is ", "42."],  # turn 2: final answer
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live_tool_contents = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live_tool_contents  # sanity: the live run actually left a marker

    # A fresh bridge instance -- simulates an app restart -- must re-derive
    # byte-identical marker text purely from AgentRunsDB.
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    blocks = fresh_bridge.resume_marker_messages("conv-1")
    resumed_markers = [m for _anchor, block in blocks for m in block]
    assert [m.content for m in resumed_markers] == live_tool_contents
    assert [m.activity_presentation for m in resumed_markers] == [
        ConsoleActivityPresentation("tool", "calculator", "success"),
    ]
    live_markers = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert [m.activity_presentation for m in live_markers] == [
        m.activity_presentation for m in resumed_markers
    ]


@pytest.mark.parametrize(
    "content",
    [
        "ERROR: harmless successful payload",
        CONTROLLER_USER_DENIED_REFUSAL.format(name="collision_tool"),
    ],
)
def test_successful_tool_payload_collisions_stay_success_live_and_resumed(
    tmp_path, content: str
) -> None:
    scripts = [
        [_fence("collision_tool", {})],
        ["done"],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        mcp_provider=_ResultMCPProvider(ToolResult(ok=True, content=content)),
    )

    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)
    tool_step = next(step for step in outcome.steps if step.kind == STEP_TOOL_RESULT)
    persisted_step = next(
        step
        for step in db.list_runs("conv-1")[0]["steps"]
        if step["kind"] == STEP_TOOL_RESULT
    )
    assert tool_step.tool_outcome == "success"
    assert persisted_step["tool_outcome"] == "success"
    assert live[-1].activity_presentation.status == "success"
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ToolResult(ok=False, error="ordinary dispatch failure"), "failed"),
        (
            ToolResult.blocked("tool execution is disabled by the kill switch"),
            "blocked",
        ),
    ],
)
def test_structured_tool_failure_status_has_live_resume_parity(
    tmp_path, result: ToolResult, expected: str
) -> None:
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [[_fence("collision_tool", {})], ["done"]],
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        mcp_provider=_ResultMCPProvider(result),
    )

    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)
    tool_step = next(step for step in outcome.steps if step.kind == STEP_TOOL_RESULT)
    assert tool_step.tool_outcome == expected
    assert live[-1].activity_presentation.status == expected
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)


def _planning_markers_for_attributed_steps(
    events: list[tuple[AgentStep, str]],
    *,
    actual_thinking_round_ordinals: frozenset[int] = frozenset(),
) -> list[ConsoleChatMessage]:
    deriver = bridge_module._PendingPrimaryPlanningDeriver()
    return [
        marker
        for step, agent_kind in events
        if (
            marker := deriver.observe(
                step,
                agent_kind,
                actual_thinking_round_ordinals=actual_thinking_round_ordinals,
            )
        )
        is not None
    ]


@pytest.mark.parametrize(
    ("events", "expected_content"),
    [
        (
            [
                (AgentStep(0, STEP_MODEL, summary="Checking."), "primary"),
                (AgentStep(1, STEP_TOOL_CALL, tool_name="fs_read"), "primary"),
                (AgentStep(2, STEP_TOOL_RESULT, tool_name="fs_read"), "primary"),
                (AgentStep(3, STEP_MODEL, summary="Final answer."), "primary"),
            ],
            ["Checking."],
        ),
        (
            [
                (AgentStep(0, STEP_MODEL, summary="Delegating."), "primary"),
                (AgentStep(1, STEP_SPAWN, summary="research"), "primary"),
            ],
            ["Delegating."],
        ),
        (
            [
                (AgentStep(0, STEP_MODEL, summary="Preparing call."), "primary"),
                (
                    AgentStep(
                        1,
                        STEP_TOOL_RESULT,
                        tool_name="fs_write",
                        result="denied",
                    ),
                    "primary",
                ),
            ],
            ["Preparing call."],
        ),
        (
            [
                (AgentStep(0, STEP_MODEL, summary="Two checks."), "primary"),
                (AgentStep(1, STEP_TOOL_CALL, tool_name="first"), "primary"),
                (AgentStep(2, STEP_TOOL_RESULT, tool_name="first"), "primary"),
                (AgentStep(3, STEP_TOOL_CALL, tool_name="second"), "primary"),
                (AgentStep(4, STEP_TOOL_RESULT, tool_name="second"), "primary"),
            ],
            ["Two checks."],
        ),
        (
            [(AgentStep(0, STEP_MODEL, summary="Final only."), "primary")],
            [],
        ),
        (
            [
                (AgentStep(0, STEP_MODEL, summary="Will fail."), "primary"),
                (AgentStep(1, STEP_ERROR, summary="provider failed"), "primary"),
            ],
            [],
        ),
    ],
)
def test_pending_primary_planning_marker_sequence_rules(
    events: list[tuple[AgentStep, str]], expected_content: list[str]
) -> None:
    markers = _planning_markers_for_attributed_steps(events)

    assert [marker.content for marker in markers] == expected_content
    assert all(
        marker.activity_presentation
        == ConsoleActivityPresentation("planning", "Planning", "done")
        for marker in markers
    )


def test_actual_thinking_suppresses_only_its_owned_planning_round() -> None:
    events = [
        (AgentStep(0, STEP_MODEL, summary="Actual round."), "primary"),
        (AgentStep(1, STEP_TOOL_CALL, tool_name="first"), "primary"),
        (AgentStep(2, STEP_TOOL_RESULT, tool_name="first"), "primary"),
        (AgentStep(3, STEP_MODEL, summary="Planning-only round."), "primary"),
        (AgentStep(4, STEP_TOOL_CALL, tool_name="second"), "primary"),
        (AgentStep(5, STEP_TOOL_RESULT, tool_name="second"), "primary"),
        (AgentStep(6, STEP_MODEL, summary="Final answer."), "primary"),
    ]

    markers = _planning_markers_for_attributed_steps(
        events,
        actual_thinking_round_ordinals=frozenset({0}),
    )

    assert [marker.content for marker in markers] == ["Planning-only round."]
    assert markers[0].activity_round_ordinal == 1
    assert markers[0].activity_presentation == ConsoleActivityPresentation(
        "planning", "Planning", "done"
    )


@pytest.mark.parametrize(
    "evidence",
    [
        ProviderThinkingDelta(
            text="actual reasoning",
            provider="llama_cpp",
            model="reasoner",
            protocol="chat_completions",
            source_format="start_anchored_think",
        ),
        ProviderProprietaryThinkingEvidence(
            provider="moonshot",
            model="kimi",
            protocol="chat_completions",
            source_format="reasoning_content",
        ),
    ],
    ids=["displayable", "proprietary"],
)
def test_live_actual_thinking_suppresses_only_its_model_round(
    tmp_path,
    evidence: ProviderThinkingDelta | ProviderProprietaryThinkingEvidence,
) -> None:
    bridge, _db, store, session, assistant_id = _bridge(
        tmp_path,
        [
            [
                evidence,
                "First round plan.\n",
                _fence("calculator", {"expression": "1 + 1"}),
            ],
            [
                "Second round plan.\n",
                _fence("calculator", {"expression": "2 + 2"}),
            ],
            ["Done."],
        ],
    )

    outcome = _run(bridge, store, session, assistant_id)

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_DONE
    assert assistant.thinking is not None
    assert [block.round_ordinal for block in assistant.thinking.blocks] == [0]
    markers = _tool_messages(store, session.id)
    assert [marker.activity_presentation.kind for marker in markers] == [
        "tool",
        "planning",
        "tool",
    ]
    assert [marker.activity_round_ordinal for marker in markers] == [0, 1, 1]


def test_resume_suppresses_planning_from_exact_selected_envelope_rounds(
    tmp_path,
) -> None:
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(
        conversation_id="conv-1",
        agent_kind="primary",
        assistant_message_id="assistant-1",
    )
    db.append_steps(
        run_id,
        [
            vars(AgentStep(0, STEP_MODEL, summary="Actual first round.")),
            vars(AgentStep(1, STEP_TOOL_CALL, tool_name="first")),
            vars(AgentStep(2, STEP_TOOL_RESULT, tool_name="first", result="one")),
            vars(AgentStep(3, STEP_MODEL, summary="Planning second round.")),
            vars(AgentStep(4, STEP_TOOL_CALL, tool_name="second")),
            vars(AgentStep(5, STEP_TOOL_RESULT, tool_name="second", result="two")),
            vars(AgentStep(6, STEP_MODEL, summary="Final answer.")),
        ],
    )
    db.set_status(run_id, RUN_DONE, result="Final answer.")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    block = bridge.resume_marker_messages(
        "conv-1",
        thinking_round_ordinals_by_assistant_message_id={"assistant-1": frozenset({0})},
    )[0][1]

    assert [marker.activity_presentation.kind for marker in block] == [
        "tool",
        "planning",
        "tool",
    ]
    assert [marker.activity_round_ordinal for marker in block] == [0, 1, 1]


def test_subagent_steps_do_not_flush_or_clear_pending_primary_planning() -> None:
    events = [
        (AgentStep(0, STEP_MODEL, summary="Primary preamble."), "primary"),
        (AgentStep(0, STEP_MODEL, summary="Child private turn."), "subagent"),
        (AgentStep(1, STEP_TOOL_CALL, tool_name="child_tool"), "subagent"),
        (AgentStep(2, STEP_TOOL_RESULT, tool_name="child_tool"), "subagent"),
        (AgentStep(1, STEP_TOOL_CALL, tool_name="primary_tool"), "primary"),
    ]

    markers = _planning_markers_for_attributed_steps(events)

    assert [marker.content for marker in markers] == ["Primary preamble."]


def test_live_callback_interleaving_preserves_primary_planning_and_resume_sequence(
    tmp_path,
    monkeypatch,
) -> None:
    primary_steps = [
        AgentStep(0, STEP_MODEL, summary="Primary preamble."),
        AgentStep(1, STEP_TOOL_CALL, tool_name="primary_tool"),
        AgentStep(
            2,
            STEP_TOOL_RESULT,
            tool_name="primary_tool",
            result="primary result",
        ),
        AgentStep(3, STEP_MODEL, summary="Final answer."),
    ]
    callback_events = [
        (primary_steps[0], "primary", "primary-run"),
        (
            AgentStep(0, STEP_MODEL, summary="Child internal turn."),
            "subagent",
            "child-run",
        ),
        (
            AgentStep(1, STEP_TOOL_CALL, tool_name="child_tool"),
            "subagent",
            "child-run",
        ),
        (
            AgentStep(2, STEP_TOOL_RESULT, tool_name="child_tool", result="child"),
            "subagent",
            "child-run",
        ),
        (primary_steps[1], "primary", "primary-run"),
        (primary_steps[2], "primary", "primary-run"),
        (primary_steps[3], "primary", "primary-run"),
    ]

    class _InterleavingAgentService:
        def __init__(self, db, _registry, *, on_step, **_kwargs):
            self._db = db
            self._on_step = on_step

        def run_turn(self, *, conversation_id, **_kwargs):
            run_id = self._db.create_run(
                conversation_id=conversation_id,
                agent_kind="primary",
            )
            for step, agent_kind, attributed_run_id in callback_events:
                self._on_step(step, agent_kind, attributed_run_id)
            # Child steps belong to their own run and are intentionally absent
            # from the persisted primary sequence. Resume therefore performs
            # primary-only look-ahead while live receives the interleaving.
            self._db.append_steps(run_id, [vars(step) for step in primary_steps])
            self._db.set_status(run_id, "done", result="Final answer.")
            return run_id, RunOutcome("done", primary_steps, final_text="Final answer.")

        def fleet_snapshot(self):
            return []

        def live_subagent_handles(self):
            return []

    monkeypatch.setattr(bridge_module, "AgentService", _InterleavingAgentService)
    bridge, db, store, session, aid = _bridge(tmp_path, [])

    outcome = _run(bridge, store, session, aid)
    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)

    assert outcome.status == "done"
    assert [marker.activity_presentation.kind for marker in live] == [
        "planning",
        "tool",
    ]
    assert live[0].content == "Primary preamble."
    assert not any("child" in marker.content.lower() for marker in live)
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)


def test_planning_live_resume_marker_order_content_and_presentation_parity(
    tmp_path,
) -> None:
    scripts = [
        [
            "I will calculate this safely.\n",
            _fence("calculator", {"expression": "6*7"}),
        ],
        ["It is 42."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)
    live = [
        message
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.TOOL
    ]
    resumed = [
        message
        for _anchor, block in ConsoleAgentBridge(
            agent_runs_db=db, store=None, provider_gateway=None
        ).resume_marker_messages("conv-1")
        for message in block
    ]

    assert outcome.status == "done"
    assert [message.activity_presentation.kind for message in live] == [
        "planning",
        "tool",
    ]
    assert live[0].content == "I will calculate this safely."
    assert [message.content for message in resumed] == [
        message.content for message in live
    ]
    assert [message.activity_presentation for message in resumed] == [
        message.activity_presentation for message in live
    ]
    assert [message.tool_output_full for message in resumed] == [
        message.tool_output_full for message in live
    ]


def test_resume_step_markers_attach_presentation_for_every_known_step_shape(
    tmp_path,
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_SPAWN,
                "summary": "research",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
            {
                "index": 1,
                "kind": STEP_TOOL_RESULT,
                "summary": "",
                "tool_name": "fs_write",
                "result": "ERROR: disk exploded",
                "args": None,
                "created_at": "",
            },
            {
                "index": 2,
                "kind": STEP_ERROR,
                "summary": "provider failed",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
            {
                "index": 3,
                "kind": console_agent_bridge.STEP_APPROVAL_TIMEOUT,
                "summary": "30",
                "tool_name": "fs_edit",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    markers = bridge.resume_marker_messages("conv-1")[0][1]

    assert [m.activity_presentation for m in markers] == [
        ConsoleActivityPresentation("spawn", "Sub-agent", "done"),
        ConsoleActivityPresentation("tool", "fs_write", "failed"),
        ConsoleActivityPresentation("warning", "Error", "failed"),
        ConsoleActivityPresentation("warning", "fs_edit", "blocked"),
    ]


def test_live_and_resume_change_marker_inventory_has_content_and_metadata_parity(
    tmp_path,
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=None)
    inventory = [
        (
            CHANGE_KIND_TURN,
            TurnChangeRecord(root="/turn", files_changed=1, adds=2, dels=3),
        ),
        (
            CHANGE_KIND_SUBAGENT_POST_TURN,
            TurnChangeRecord(root="/post", files_changed=2, adds=4, dels=5),
        ),
        (
            CHANGE_KIND_TURN_CONCURRENT_SUBAGENT,
            TurnChangeRecord(root="/concurrent", files_changed=3, adds=6, dels=7),
        ),
        (
            CHANGE_KIND_TURN,
            TurnChangeRecord(root="/failed", tracking_error="snapshot failed"),
        ),
    ]
    for kind, record in inventory:
        # One real change window belongs to one run. Keeping these separate
        # mirrors production and prevents resume's intentional per-run
        # same-kind aggregation from inventing a fixture-only difference.
        run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
        bridge._append_change_markers(  # noqa: SLF001 - targeted builder contract
            session.id, run_id, [record], kind=kind
        )
        db.record_change_snapshot(
            run_id=run_id,
            root=record.root,
            baseline_sha=record.baseline_sha,
            end_sha=record.end_sha,
            files_changed=record.files_changed,
            adds=record.adds,
            dels=record.dels,
            tracking_error=record.tracking_error,
            kind=kind,
        )
        db.set_status(run_id, "done", result="ok")

    live = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    resumed = [
        marker
        for _anchor, block in ConsoleAgentBridge(
            agent_runs_db=db, store=None, provider_gateway=None
        ).resume_marker_messages("conv-1")
        for marker in block
    ]

    assert [(m.content, m.activity_presentation) for m in resumed] == [
        (m.content, m.activity_presentation) for m in live
    ]
    assert [m.activity_presentation for m in live] == [
        ConsoleActivityPresentation("changes", "Changes", "done"),
        ConsoleActivityPresentation("changes", "Sub-agent changes", "done"),
        ConsoleActivityPresentation("changes", "Changes", "done"),
        ConsoleActivityPresentation("warning", "Concurrent sub-agent", "done"),
        ConsoleActivityPresentation("warning", "Change tracking", "failed"),
    ]


def test_live_and_resume_diff_feedback_disclosure_has_metadata_parity(tmp_path):
    bridge, db, store, session, assistant_id = _bridge(tmp_path, [["answer"]])
    annotated_run = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(annotated_run, "done", result="prior")
    note_id = _add_note(db, annotated_run, note="rename this variable")

    _run(bridge, store, session, assistant_id)

    delivered = db.notes_for_run(annotated_run)
    assert delivered[0]["id"] == note_id
    assert delivered[0]["delivered_at"] is not None
    live = [
        m
        for m in store.messages_for_session(session.id)
        if "Diff feedback attached" in m.content
    ]
    resumed = [
        m
        for _anchor, block in ConsoleAgentBridge(
            agent_runs_db=db, store=None, provider_gateway=None
        ).resume_marker_messages("conv-1")
        for m in block
        if "Diff feedback attached" in m.content
    ]

    assert len(live) == len(resumed) == 1
    assert (resumed[0].content, resumed[0].activity_presentation) == (
        live[0].content,
        live[0].activity_presentation,
    )
    assert live[0].activity_presentation == ConsoleActivityPresentation(
        "feedback", "Feedback delivered", "done"
    )


def test_resume_marker_messages_surfaces_assistant_message_id_anchor(tmp_path):
    """Task 3: each block is paired with the run's ``assistant_message_id``
    (``None`` for a legacy/pre-Phase-C run that never recorded one) so the
    placement layer can anchor by id instead of guessing ordinally."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    anchored = db.create_run(
        conversation_id="conv-1",
        agent_kind="primary",
        assistant_message_id="asst-anchor",
    )
    db.append_steps(
        anchored,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(anchored, "done", result="ok")
    legacy = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        legacy,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "legacy",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(legacy, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 2
    assert blocks[0][0] == "asst-anchor"
    assert "boom" in blocks[0][1][0].content
    assert blocks[1][0] is None
    assert "legacy" in blocks[1][1][0].content


def test_resume_marker_messages_orders_blocks_chronologically_oldest_first(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    first = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        first,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(first, "done", result="4")
    second = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        second,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "timed out",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(second, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 2
    assert "calculator" in blocks[0][1][0].content
    assert "timed out" in blocks[1][1][0].content


def test_resume_marker_messages_skips_superseded_runs(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    superseded = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        superseded,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "old attempt",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(superseded, "superseded")
    kept = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        kept,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "final attempt",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(kept, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 1
    assert "final attempt" in blocks[0][1][0].content


def _add_note(db, run_id, *, path="a.py", header="@@ -1,1 +1,1 @@", note="n"):
    return db.add_change_note(
        run_id=run_id,
        root="/workspace",
        path=path,
        hunk_index=0,
        hunk_header=header,
        hunk_excerpt="+x",
        note=note,
    )


def test_resume_marker_messages_re_derives_diff_feedback_disclosure_after_marker(
    tmp_path,
):
    """Task 6 (spec §4): a run with snapshots AND delivered notes yields
    its marker row(s) AND, after them, a disclosure row whose text equals
    ``format_diff_feedback_disclosure`` over the delivered notes."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="4")
    note_id = _add_note(db, run_id, note="use the cached value here")
    db.mark_notes_delivered([note_id], delivered_by_run_id=run_id)
    delivered = db.notes_for_run(run_id)
    assert delivered[0]["delivered_at"] is not None

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    assert "calculator" in block[0].content
    assert block[-1].content == format_diff_feedback_disclosure(delivered)
    assert "use the cached value here" in block[-1].content
    assert block[-1].role is ConsoleMessageRole.TOOL
    assert block[-1].change_review_run_id is None


def test_resume_marker_messages_groups_two_delivery_batches_in_delivery_order(
    tmp_path,
):
    """Two separate deliveries (two distinct ``delivered_at`` stamps) yield
    two disclosure rows, oldest delivery first."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="ok")
    first_id = _add_note(db, run_id, path="a.py", note="first batch note")
    db.mark_notes_delivered([first_id], delivered_by_run_id=run_id)
    second_id = _add_note(db, run_id, path="b.py", note="second batch note")
    db.mark_notes_delivered([second_id], delivered_by_run_id=run_id)
    delivered = db.notes_for_run(run_id)
    delivered_ats = {n["id"]: n["delivered_at"] for n in delivered}
    assert delivered_ats[first_id] <= delivered_ats[second_id]

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    disclosure_rows = [m for m in block if "Diff feedback attached" in m.content]
    assert len(disclosure_rows) == 2
    assert "first batch note" in disclosure_rows[0].content
    assert "second batch note" in disclosure_rows[1].content


def test_resume_marker_messages_omits_disclosure_for_pending_only_notes(tmp_path):
    """Pending (undelivered) notes yield NO disclosure row on resume."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="ok")
    _add_note(db, run_id, note="still pending")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    assert not any("Diff feedback attached" in m.content for m in block)
    assert not any("still pending" in m.content for m in block)


def test_resume_marker_messages_heals_disclosure_when_live_append_never_happened(
    tmp_path,
):
    """Task 5's live seam stamps ``delivered_at`` then appends the
    disclosure row in one ``try`` -- if the append fails after the stamp
    lands, live never shows a row. This is the designed healer: a fresh
    resume (no prior live TOOL row in the store at all) must still surface
    the disclosure purely from the DB stamp."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(run_id, "done", result="ok")  # no steps: a plain answer
    note_id = _add_note(db, run_id, note="healed disclosure")
    # Stamped as if live append then failed -- delivered_by_run_id is this
    # same run's id, matching what the real Task 5 stamp call now passes.
    db.mark_notes_delivered([note_id], delivered_by_run_id=run_id)

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    assert len(block) == 1  # no marker-worthy steps -- only the healed row
    assert block[0].content == format_diff_feedback_disclosure(db.notes_for_run(run_id))
    assert "healed disclosure" in block[0].content


def test_resume_marker_messages_disclosure_survives_run_with_no_snapshot_rows(
    tmp_path,
):
    """A run can have delivered notes but NO change_snapshots rows at all
    (change tracking failed or was never configured for that turn -- note
    delivery does not depend on tracking having succeeded). The disclosure
    row must still surface; it is not gated on ``snap_rows``."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="4")
    note_id = _add_note(db, run_id, note="no snapshot for this turn")
    db.mark_notes_delivered([note_id], delivered_by_run_id=run_id)
    # Sanity: no change_snapshots rows exist for this run/conversation.
    assert db.change_snapshots_for_conversation("conv-1") == []

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    assert "calculator" in block[0].content
    assert block[-1].content == format_diff_feedback_disclosure(
        db.notes_for_run(run_id)
    )
    assert "no snapshot for this turn" in block[-1].content


def test_resume_marker_messages_disclosure_anchors_at_the_delivering_run(tmp_path):
    """Fix-round CRITICAL finding: a note annotated against run A's diff
    but DELIVERED on run B's later completion must resume with its
    disclosure row after B's own marker row(s) -- not A's. Live emission
    places the row at whichever run's completion actually did the
    stamping; re-derivation must match that, matching
    ``format_diff_feedback_disclosure`` byte-for-byte, or a resumed
    transcript would show the disclosure attached to the wrong turn (or,
    under the pre-fix anchor-run grouping, not at all if A had since been
    superseded)."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_a = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_a,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_a, "done", result="4")
    note_id = _add_note(db, run_a, note="annotated on a, delivered on b")

    run_b = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_b,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_b, "done", result="ok")
    db.mark_notes_delivered([note_id], delivered_by_run_id=run_b)
    delivered = db.notes_for_run(run_a)

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 2
    block_a = blocks[0][1]
    block_b = blocks[1][1]
    assert "calculator" in block_a[0].content
    assert not any("Diff feedback attached" in m.content for m in block_a)
    assert "boom" in block_b[0].content
    assert block_b[-1].content == format_diff_feedback_disclosure(delivered)
    assert "annotated on a, delivered on b" in block_b[-1].content
    assert block_b[-1].change_review_run_id is None


def test_resume_marker_messages_batch_spanning_two_anchor_runs_stays_one_row(
    tmp_path,
):
    """Fix-round finding: one live delivery batch containing notes
    annotated against TWO different runs' diffs must stay ONE disclosure
    row -- at the delivering run's position -- not fragment into one row
    per anchor run."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_a = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(run_a, "done", result="ok")
    note_a = _add_note(db, run_a, path="a.py", note="note on a")

    run_b = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(run_b, "done", result="ok")
    note_b = _add_note(db, run_b, path="b.py", note="note on b")

    run_c = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_c,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "final turn",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_c, "done", result="ok")
    # ONE call -- both notes delivered together, by run_c's completion.
    db.mark_notes_delivered([note_a, note_b], delivered_by_run_id=run_c)

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 3
    block_a, block_b, block_c = (b for _anchor, b in blocks)
    assert not any("Diff feedback attached" in m.content for m in block_a)
    assert not any("Diff feedback attached" in m.content for m in block_b)
    disclosure_rows = [m for m in block_c if "Diff feedback attached" in m.content]
    assert len(disclosure_rows) == 1
    assert "note on a" in disclosure_rows[0].content
    assert "note on b" in disclosure_rows[0].content


def test_resume_marker_messages_legacy_null_delivered_by_run_id_falls_back_to_annotated_run(
    tmp_path,
):
    """A note stamped delivered before ``delivered_by_run_id`` existed (or
    by any caller that omits it) carries NULL there -- there is no way to
    recover which run delivered it, so resume falls back to the note's own
    (annotated) run's position, exactly like the original Task 6
    mechanism. Verified against a conversation with a LATER run present,
    so a wrong "always use the newest/last run" fallback would be
    detectable."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_a = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_a,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_a, "done", result="4")
    note_id = _add_note(db, run_a, note="legacy stamp, no deliverer recorded")
    db.mark_notes_delivered([note_id])  # no delivered_by_run_id -- legacy shape

    run_b = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_b,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_b, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 2
    block_a = blocks[0][1]
    block_b = blocks[1][1]
    assert "calculator" in block_a[0].content
    assert block_a[-1].content == format_diff_feedback_disclosure(
        db.notes_for_run(run_a)
    )
    assert "legacy stamp, no deliverer recorded" in block_a[-1].content
    assert not any("Diff feedback attached" in m.content for m in block_b)


def test_resume_marker_messages_survives_delivered_notes_read_failure(
    tmp_path, monkeypatch
):
    """Fix-round CRITICAL C1: a ``change_notes`` read failure must not
    break conversation resume entirely -- guarded the same way the
    ``change_snapshots_for_conversation`` fetch immediately above it
    already is ("resume must not die on this"). A run's own step markers
    must still come back even if the disclosure lookup blows up."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(run_id, "done", result="4")
    note_id = _add_note(db, run_id, note="should not crash resume")
    db.mark_notes_delivered([note_id], delivered_by_run_id=run_id)

    def _boom(self, conversation_id):
        raise RuntimeError("change_notes DB is on fire")

    monkeypatch.setattr(AgentRunsDB, "delivered_notes_for_conversation", _boom)

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 1
    block = blocks[0][1]
    assert "calculator" in block[0].content
    assert not any("Diff feedback attached" in m.content for m in block)


def _tool_marker(text: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content=text, status="complete"
    )


def test_inject_resume_agent_markers_places_block_after_matching_assistant_message():
    """Task 3: placement follows the block's anchor id (matched against
    each message's ``persisted_message_id``), not block order/ordinal
    position. The blocks below are listed in the OPPOSITE order of their
    anchors -- the block for the SECOND assistant reply ("asst-2") comes
    first in the list -- so an ordinal placement would put it after "42."
    (wrong); id-anchoring must still put it after "ok." (right)."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="ok.",
            status="complete",
            persisted_message_id="asst-2",
        ),
    ]
    blocks = [
        ("asst-2", [_tool_marker("⚠ retry")]),
        ("asst-1", [_tool_marker("⚙ calculator → 42")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    roles = [(m.role, m.content) for m in result]
    assert roles == [
        (ConsoleMessageRole.USER, "hi"),
        (ConsoleMessageRole.ASSISTANT, "42."),
        (ConsoleMessageRole.TOOL, "⚙ calculator → 42"),
        (ConsoleMessageRole.USER, "again"),
        (ConsoleMessageRole.ASSISTANT, "ok."),
        (ConsoleMessageRole.TOOL, "⚠ retry"),
    ]


def test_inject_resume_agent_markers_drops_block_whose_anchor_matches_no_active_path_message():
    """A block anchored to an assistant_message_id that isn't in the
    active-path ``messages`` (the run's reply lives on another branch) must
    be hidden entirely -- never appended anywhere -- rather than leak an
    off-branch tool trace onto the visible transcript."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
    ]
    blocks = [("asst-OFF-PATH", [_tool_marker("⚙ calculator → 999")])]

    result = inject_resume_agent_markers(messages, blocks)

    assert [m.content for m in result] == ["hi", "42."]
    assert not any(m.role is ConsoleMessageRole.TOOL for m in result)


def test_inject_resume_agent_markers_null_anchor_blocks_place_ordinally_when_no_id_claims():
    """Legacy (pre-Phase-C) runs never recorded an assistant_message_id --
    their blocks carry a ``None`` anchor and keep the old ordinal
    placement: Nth null block <-> Nth assistant reply."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    blocks = [
        (None, [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ retry")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    roles = [(m.role, m.content) for m in result]
    assert roles == [
        (ConsoleMessageRole.USER, "hi"),
        (ConsoleMessageRole.ASSISTANT, "42."),
        (ConsoleMessageRole.TOOL, "⚙ calculator → 42"),
        (ConsoleMessageRole.USER, "again"),
        (ConsoleMessageRole.ASSISTANT, "ok."),
        (ConsoleMessageRole.TOOL, "⚠ retry"),
    ]


def test_inject_resume_agent_markers_mixed_id_and_null_anchors_null_skips_claimed_assistant():
    """Mixed case: one block is id-anchored to the FIRST assistant reply: the
    remaining null (legacy) block's ordinal fallback must skip that already
    id-claimed reply and land on the next unclaimed one, not double up on
    "42."."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    blocks = [
        ("asst-1", [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ legacy retry")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    roles = [(m.role, m.content) for m in result]
    assert roles == [
        (ConsoleMessageRole.USER, "hi"),
        (ConsoleMessageRole.ASSISTANT, "42."),
        (ConsoleMessageRole.TOOL, "⚙ calculator → 42"),
        (ConsoleMessageRole.USER, "again"),
        (ConsoleMessageRole.ASSISTANT, "ok."),
        (ConsoleMessageRole.TOOL, "⚠ legacy retry"),
    ]


def test_inject_resume_agent_markers_appends_leftover_block_when_more_runs_than_replies():
    """Preserves the old leftover-append behavior, now for NULL-anchored
    blocks: a legacy run with no corresponding assistant reply left in the
    active path is appended at the end rather than dropped."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"
        ),
    ]
    blocks = [
        (None, [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ orphan run")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    assert [m.content for m in result] == [
        "hi",
        "42.",
        "⚙ calculator → 42",
        "⚠ orphan run",
    ]


def test_inject_resume_agent_markers_skips_empty_blocks():
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    result = inject_resume_agent_markers(messages, [(None, []), ("some-id", [])])
    assert [m.content for m in result] == ["ok."]


def test_inject_resume_agent_markers_is_idempotent_no_duplicates_on_second_call():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
    ]
    blocks = [("asst-1", [_tool_marker("⚙ calculator → 42")])]

    once = inject_resume_agent_markers(messages, blocks)
    twice = inject_resume_agent_markers(once, blocks)

    tool_rows = [m.content for m in twice if m.role is ConsoleMessageRole.TOOL]
    assert tool_rows == ["⚙ calculator → 42"]
    assert len(once) == len(twice)


def test_inject_resume_agent_markers_leaves_live_session_with_markers_untouched():
    """A session that already carries live markers (this bridge ran the
    turn in-process rather than resuming) must be left byte-for-byte
    unchanged if this function is (defensively) called on it again."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        _tool_marker("⚙ calculator → 42"),
    ]
    blocks = [("asst-1", [_tool_marker("⚙ calculator → 42")])]

    result = inject_resume_agent_markers(messages, blocks)

    assert result == messages


def test_resume_injects_markers_matching_live_format_end_to_end(tmp_path):
    """Fresh store + populated AgentRunsDB -> resuming reconstructs a
    transcript whose TOOL markers match live-format byte-for-byte, placed
    after the answer they belong to, and a second resume onto the
    already-injected transcript adds nothing more."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is ", "42."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live_messages = store.messages_for_session(session.id)
    live_tool_contents = [
        m.content for m in live_messages if m.role is ConsoleMessageRole.TOOL
    ]

    # Simulate resume after a restart: the "ChaChaNotes-only" transcript
    # never carries markers (they persist=False), then inject markers
    # derived fresh from the DB via a brand-new bridge instance.
    chachanotes_only = [
        ConsoleChatMessage(role=m.role, content=m.content, status="complete")
        for m in live_messages
        if m.role is not ConsoleMessageRole.TOOL
    ]
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = inject_resume_agent_markers(
        chachanotes_only, fresh_bridge.resume_marker_messages("conv-1")
    )

    resumed_tool_contents = [
        m.content for m in resumed if m.role is ConsoleMessageRole.TOOL
    ]
    assert resumed_tool_contents == live_tool_contents
    assistant_index = next(
        i for i, m in enumerate(resumed) if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert resumed[assistant_index + 1].role is ConsoleMessageRole.TOOL

    resumed_again = inject_resume_agent_markers(
        resumed, fresh_bridge.resume_marker_messages("conv-1")
    )
    assert len(resumed_again) == len(resumed)


# -- Task 12: per-run spawn-wired skill executor + run allow-list composition --


class _FakeSkillsService:
    """Minimal async skills service: one trusted, model-invocable skill."""

    def __init__(self, *, skill_name="code-review", allowed_tools=None, blocked=False):
        self.skill_name = skill_name
        self.allowed_tools = allowed_tools
        self.blocked = blocked
        self.execute_calls = []
        self.get_context_calls = 0

    async def get_context(self, *, mode="local"):
        self.get_context_calls += 1
        return {
            "available_skills": [
                {
                    "name": self.skill_name,
                    "description": "Review a diff",
                    "argument_hint": "[diff]",
                    "trust_blocked": False,
                    "disable_model_invocation": False,
                },
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append(args)
        if self.blocked:
            raise SkillTrustBlockedError(
                skill_name=name,
                reason_code="quarantined_modified",
                trust_status="quarantined_modified",
            )
        return {
            "skill_name": name,
            "rendered_prompt": f"Review this: {args}",
            "allowed_tools": self.allowed_tools,
            "execution_mode": "inline",
        }


def test_skill_tool_call_routes_through_run_scoped_spawn(tmp_path):
    """A model-invoked skill tool runs as a budget-counted sub-agent of THIS
    run -- not SkillToolProvider.invoke (which raises by design), and not an
    unbounded/uncancellable bespoke path. The rendered skill prompt becomes
    the sub-agent's task, the sub-agent turn goes through the same scripted
    gateway as any other spawned sub-agent, and a TOOL marker records the
    call in the transcript exactly like any other tool call."""
    scripts = [
        [_fence("code-review", {"args": "the diff"})],  # primary calls the skill
        ["Looks fine to me."],  # sub-agent turn
        ["All done."],  # primary final
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-skill")

    assert outcome.status == "done"
    assert skills_service.execute_calls == ["the diff"]
    assert db.count_subagent_runs("conv-skill") == 1
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("code-review" in row.content for row in tool_rows)

    db_path = db.db_path
    db.close()
    reopened = AgentRunsDB(db_path, client_id="trace-skill-reload")
    runs = reopened.list_runs("conv-skill", include_superseded=True)
    parent = next(row for row in runs if row["agent_kind"] == "primary")
    child = next(row for row in runs if row["agent_kind"] == "subagent")
    causes = [
        step
        for step in parent["steps"]
        if f"agent-step:{parent['id']}:{step['index']}" == child["spawn_event_id"]
    ]
    assert len(causes) == 1
    assert causes[0]["kind"] == STEP_TOOL_CALL
    assert causes[0]["tool_name"] == "code-review"

    agent_steps = [
        {**step, "run_id": row["id"], "conversation_id": "conv-skill"}
        for row in runs
        for step in row["steps"]
    ]
    snapshot = derive_trajectory(
        messages=[],
        usage_by_id={},
        traj_rows=[],
        variant_sets=[],
        compaction_records=[],
        agent_runs=runs,
        agent_steps=agent_steps,
    )
    event_ids = [
        record.event_id for turn in snapshot.turns for record in turn.records
    ]
    assert event_ids.index(child["spawn_event_id"]) < event_ids.index(
        f"agent-run:{child['id']}"
    )
    reopened.close()


def test_skill_trust_blocked_refuses_without_spawning(tmp_path):
    """A skill whose trust was revoked between catalog build and model call
    refuses (re-verified at render time by execute_skill) -- no sub-agent is
    ever spawned, so the run tree never grows for a blocked call."""
    scripts = [
        [_fence("code-review", {"args": "x"})],
        ["I could not review that."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(blocked=True)
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-blocked")

    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-blocked") == 0


# -- task-4 (skills-fork-reachability): _BridgeSkillRunner grants its own
# name skill_file authorization pre-spawn and appends a "Bundled files"
# pointer block to the rendered task text whenever execute_skill reports
# reference_files -- unit-level, directly against _BridgeSkillRunner, so
# these don't need a full run_reply/model round trip. --


class _FakeSkillsServiceWithRefs(_FakeSkillsService):
    """Same fake as above, but execute_skill also reports a bundle manifest."""

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append(args)
        return {
            "skill_name": name,
            "rendered_prompt": f"Review this: {args}",
            "allowed_tools": self.allowed_tools,
            "execution_mode": "inline",
            "reference_files": [
                {"path": "references/api.md", "size": 120, "is_text": True},
                {"path": "assets/logo.png", "size": 2048, "is_text": False},
            ],
        }


def test_bridge_skill_runner_grants_own_name_and_appends_bundle_block_before_spawn():
    skills_service = _FakeSkillsServiceWithRefs()
    bindings = SkillFileBindings(authorized=set())
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
        skill_file_bindings=bindings,
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        # The name must already be authorized by the time spawn actually
        # runs -- so the spawned child's very first turn can read its own
        # bundle -- not merely by the time .run() returns.
        assert "code-review" in bindings.authorized
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    result = runner.run("code-review", "the diff", spawn)

    assert result.ok
    assert spawn_calls == [
        "Review this: the diff\n\nBundled files (readable via skill_file): "
        "references/api.md (120 bytes), assets/logo.png (2048 bytes, binary)"
    ]
    assert "code-review" in bindings.authorized


def test_bridge_skill_runner_no_reference_files_body_unchanged_still_authorizes():
    skills_service = _FakeSkillsService()  # no reference_files key at all
    bindings = SkillFileBindings(authorized=set())
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
        skill_file_bindings=bindings,
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    runner.run("code-review", "the diff", spawn)

    assert spawn_calls == ["Review this: the diff"]
    assert "code-review" in bindings.authorized


def test_bridge_skill_runner_bindings_none_is_byte_identical_legacy_behavior():
    # reference_files IS present in the execute_skill result, but with no
    # skill_file_bindings wired at all (legacy/non-bridge construction) the
    # block must never be appended and nothing must crash.
    skills_service = _FakeSkillsServiceWithRefs()
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    runner.run("code-review", "the diff", spawn)

    assert spawn_calls == ["Review this: the diff"]


def test_run_reply_wires_one_skill_file_bindings_to_both_service_and_runner(
    tmp_path,
):
    """run_reply must construct exactly ONE SkillFileBindings per run and
    hand the SAME object to both AgentService and the _BridgeSkillRunner --
    never two independently-seeded copies (which would let the runner's
    pre-spawn grant silently fail to reach the loop's authorization check)."""
    captured = {}
    real_runner_init = _BridgeSkillRunner.__init__
    real_service_init = AgentService.__init__

    def spy_runner_init(self, **kwargs):
        captured["runner_bindings"] = kwargs.get("skill_file_bindings")
        real_runner_init(self, **kwargs)

    def spy_service_init(self, *args, **kwargs):
        captured["service_bindings"] = kwargs.get("skill_file_bindings")
        real_service_init(self, *args, **kwargs)

    scripts = [
        [_fence("code-review", {"args": "the diff"})],
        ["Looks fine to me."],
        ["All done."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    with (
        patch.object(_BridgeSkillRunner, "__init__", spy_runner_init),
        patch.object(AgentService, "__init__", spy_service_init),
    ):
        outcome = _run(
            bridge, store, session, assistant.id, conversation_id="conv-bindings"
        )

    assert outcome.status == "done"
    assert captured["runner_bindings"] is not None
    assert captured["runner_bindings"] is captured["service_bindings"]


def test_run_reply_seeds_turn_bindings_into_shared_object(tmp_path):
    """Task 5: the turn's already-resolved `$skill` binding names (splice
    output the CONTROLLER computed for the triggering turn) must be seeded
    into THIS run's SkillFileBindings.authorized -- the SAME shared object
    handed to both AgentService and the _BridgeSkillRunner (Task 4) -- so
    the primary agent's very first turn can already read that skill's own
    bundle via skill_file. Reuses the sibling wiring test's spy idiom to
    prove the seed lands on the one shared object, not a second copy."""
    captured = {}
    real_runner_init = _BridgeSkillRunner.__init__
    real_service_init = AgentService.__init__

    def spy_runner_init(self, **kwargs):
        captured["runner_bindings"] = kwargs.get("skill_file_bindings")
        real_runner_init(self, **kwargs)

    def spy_service_init(self, *args, **kwargs):
        captured["service_bindings"] = kwargs.get("skill_file_bindings")
        real_service_init(self, *args, **kwargs)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway([["Tokyo."]]),
        skills_service=skills_service,
    )

    with (
        patch.object(_BridgeSkillRunner, "__init__", spy_runner_init),
        patch.object(AgentService, "__init__", spy_service_init),
    ):
        outcome = _run(
            bridge,
            store,
            session,
            assistant.id,
            conversation_id="conv-turn-bindings",
            turn_skill_bindings=("code-review",),
        )

    assert outcome.status == "done"
    assert captured["runner_bindings"] is not None
    assert "code-review" in captured["runner_bindings"].authorized
    # Seeded onto the ONE shared object -- never two independently-seeded
    # copies (Task 4's invariant, re-verified here under a non-empty seed).
    assert captured["runner_bindings"] is captured["service_bindings"]


# -- _append_to_last_user_message (Qodo round, TASK-17611): direct unit
# tests for the shared attach helper `run_reply`'s two attach seams both
# call -- previously only exercised indirectly through `run_reply` itself
# (see `test_run_reply_appends_bundle_block_copy_safely` just below, and
# the diff-feedback-focused suite in `Tests/Chat/
# test_console_diff_feedback_delivery.py`). ---------------------------


def test_append_to_last_user_message_attaches_to_the_last_eligible_message():
    """Multiple user messages present -- the block lands on the LAST one,
    not the first."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    result, attached = _append_to_last_user_message(messages, "BLOCK")
    assert attached is True
    assert result[-1]["content"] == "second\n\nBLOCK"
    assert result[0]["content"] == "first"


def test_append_to_last_user_message_skips_list_content_and_picks_earlier_str_message():
    """The LAST message is role=="user" but carries LIST content (a
    vision/attachment turn) -- ineligible as a carrier, so the backward
    scan continues past it and attaches to an EARLIER string-content user
    message instead."""
    messages = [
        {"role": "user", "content": "earlier string"},
        {"role": "user", "content": [{"type": "text", "text": "vision turn"}]},
    ]
    result, attached = _append_to_last_user_message(messages, "BLOCK")
    assert attached is True
    assert result[1]["content"] == [{"type": "text", "text": "vision turn"}]
    assert result[0]["content"] == "earlier string\n\nBLOCK"


def test_append_to_last_user_message_falsy_block_is_a_noop():
    """A falsy (empty-string) block never scans at all -- the input list
    is returned unchanged, same object identity."""
    messages = [{"role": "user", "content": "hi"}]
    result, attached = _append_to_last_user_message(messages, "")
    assert attached is False
    assert result is messages


def test_append_to_last_user_message_no_carrier_leaves_list_untouched():
    """No eligible carrier anywhere (a system message plus a list-content
    user message) -- `attached` is False and the caller's list/dicts are
    completely untouched, same object identity."""
    messages = [
        {"role": "system", "content": "no user turn here"},
        {"role": "user", "content": [{"type": "text", "text": "vision only"}]},
    ]
    result, attached = _append_to_last_user_message(messages, "BLOCK")
    assert attached is False
    assert result is messages
    assert result[0]["content"] == "no user turn here"
    assert result[1]["content"] == [{"type": "text", "text": "vision only"}]


def test_append_to_last_user_message_copy_on_write():
    """The caller's own list and its message dict are never mutated: when
    it attaches, a NEW list is returned with a NEW dict at the matched
    index -- the original list/dict stay exactly as they were."""
    original_message = {"role": "user", "content": "hi"}
    messages = [original_message]

    result, attached = _append_to_last_user_message(messages, "BLOCK")

    assert attached is True
    assert result is not messages
    assert len(messages) == 1
    assert messages[0] is original_message
    assert original_message["content"] == "hi"
    assert result[0] is not original_message
    assert result[0]["content"] == "hi\n\nBLOCK"


def test_append_to_last_user_message_stacks_two_sequential_calls_in_order():
    """`run_reply`'s bundle-block and diff-feedback-block attach seams call
    this helper twice, back to back, on the same message -- the second
    call's block must land after the first's, both after the original
    content, and the caller's original list/dict must stay untouched
    after BOTH calls, not just the first."""
    original_message = {"role": "user", "content": "hi"}
    messages = [original_message]

    after_bundle, attached_1 = _append_to_last_user_message(messages, "BUNDLE")
    after_feedback, attached_2 = _append_to_last_user_message(after_bundle, "FEEDBACK")

    assert attached_1 is True
    assert attached_2 is True
    assert after_feedback[-1]["content"] == "hi\n\nBUNDLE\n\nFEEDBACK"
    assert messages == [original_message]
    assert messages[0] is original_message
    assert original_message["content"] == "hi"


def test_run_reply_appends_bundle_block_copy_safely(tmp_path):
    """Task 5: the turn's pre-rendered "Bundled files" block is appended to
    a NEW list + NEW dict for the last role=="user" entry -- the caller's
    own `agent_messages` list and its message dict are never mutated. A
    future refactor that switches to in-place `message["content"] += ...`
    must fail assertions (b) below."""
    real_run_turn = AgentService.run_turn

    def _spy(captured):
        def spy_run_turn(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return real_run_turn(self, **kwargs)

        return spy_run_turn

    block = "Bundled files (readable via skill_file): notes.md (1 bytes)"

    # -- (a) + (b): a non-empty block is appended copy-safely -------------
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Tokyo."]])
    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured = {}

    with patch.object(AgentService, "run_turn", _spy(captured)):
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            conversation_id="conv-bundle-block",
            agent_messages=agent_messages,
            turn_bundle_block=block,
        )

    assert outcome.status == "done"
    run_messages = captured["messages"]
    assert run_messages is not None
    # (a) the run actually received the block appended after "\n\n" on the
    # last user entry.
    assert run_messages[-1]["content"] == f"hi\n\n{block}"
    assert run_messages[-1]["role"] == "user"
    # (b) non-mutation contract: the caller's own list length and the dict
    # identity at that index are unchanged, and the original dict's content
    # carries no trace of the block. This is exactly what would break if
    # the append were switched to in-place `message["content"] += ...`.
    assert len(agent_messages) == 1
    assert agent_messages[0] is original_user_message
    assert original_user_message["content"] == "hi"
    assert "Bundled files" not in original_user_message["content"]
    # A new list AND a new dict were built for the run -- not the caller's.
    assert run_messages is not agent_messages
    assert run_messages[-1] is not original_user_message

    # -- (c) sibling case: an empty block appends nothing, no-op path -----
    bridge2, _db2, store2, session2, aid2 = _bridge(tmp_path, [["Tokyo."]])
    agent_messages2 = [{"role": "user", "content": "hi"}]
    captured2 = {}

    with patch.object(AgentService, "run_turn", _spy(captured2)):
        outcome2 = _run(
            bridge2,
            store2,
            session2,
            aid2,
            conversation_id="conv-bundle-block-empty",
            agent_messages=agent_messages2,
            turn_bundle_block="",
        )

    assert outcome2.status == "done"
    assert captured2["messages"][-1]["content"] == "hi"
    # No block to append: the frozen first-request plan remains value-identical
    # while keeping the caller-owned list untouched.
    assert captured2["messages"] == agent_messages2
    assert agent_messages2 == [{"role": "user", "content": "hi"}]


def test_no_skills_service_leaves_shared_registry_path_untouched(tmp_path):
    """The no-skills-service path (skills_service=None, the default) must
    stay byte-identical to the pre-Task-12 behavior: no get_context call, no
    skill_runner wiring, the bridge's own shared registry/allow-list used."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["Tokyo."]])
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done" and outcome.final_text == "Tokyo."


def test_compose_run_allowed_tools_includes_eligible_skill_names():
    """Pure per-run allow-list: builtins, then eligible skill names, then
    spawn -- a trust-blocked or model-invocation-disabled skill is excluded."""
    context = {
        "available_skills": [
            {
                "name": "code-review",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
            {
                "name": "needs-review",
                "trust_blocked": True,
                "disable_model_invocation": False,
            },
            {
                "name": "user-only",
                "trust_blocked": False,
                "disable_model_invocation": True,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
    assert allowed == (
        "calculator",
        "get_current_datetime",
        "code-review",
        SPAWN_TOOL_NAME,
    )


def test_compose_run_allowed_tools_empty_context_is_builtins_plus_spawn():
    allowed = _compose_run_allowed_tools({}, ("calculator",))
    assert allowed == ("calculator", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_builtin_shadows_same_named_skill():
    """Task 11 review note 2: a skill named the same as a builtin must never
    become a distinct, skill-routable tool -- the builtin always wins. The
    allow-list carries the name exactly once (from builtins), never twice."""
    context = {
        "available_skills": [
            {
                "name": "calculator",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
    assert allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_runtime_tool_name_shadows_same_named_skill():
    """Qodo finding 4 (PR #636 bot review): `_non_colliding_skill_entries`
    used to filter a skill's name only against `BuiltinToolProvider` names,
    not the loop's own in-loop runtime handler names (`find_tools`/
    `load_tools`/`spawn_subagent` -- `agent_models.RUNTIME_TOOL_NAMES`). A
    skill front-matter'd with one of those names would be advertised in the
    run's catalog/allow-list, then get hijacked by the loop's own
    name-based dispatch (`agent_runtime.run_agent_loop` checks
    `call.name == FIND_TOOLS_NAME` etc. before any registry/skill routing),
    making the skill permanently unreachable while still occupying a
    catalog slot with a misleading schema. The allow-list must exclude it
    exactly like a builtin-name collision does."""
    context = {
        "available_skills": [
            {
                "name": "find_tools",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
    assert allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)


def test_compose_run_registry_excludes_skill_named_like_a_runtime_tool():
    """Same collision, verified against the actual registry/allow-list this
    run would use: the skill must not appear as a distinct catalog entry
    (under either FIND_TOOLS_NAME or LOAD_TOOLS_NAME or SPAWN_TOOL_NAME)."""
    context = {
        "available_skills": [
            {
                "name": LOAD_TOOLS_NAME,
                "description": "d",
                "argument_hint": "",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    registry, allowed_tools, builtin_names, _local_names = (
        _compose_run_registry_and_allowed(context)
    )
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names) :]
    catalog_entries = [(entry.name, entry.source) for entry in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "skill") not in catalog_entries


# -- P5-T6: MCPToolProvider registration + collision precedence --


def test_compose_run_registry_and_allowed_includes_mcp_entries_when_eligible():
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    )
    assert "mcp__srv_a__search" in allowed_tools
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("mcp__srv_a__search", "mcp") in catalog_entries
    result = registry.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert result.ok is True
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]


def test_compose_run_registry_and_allowed_absent_mcp_provider_is_unchanged():
    """`mcp_provider=None` (the default) must not add anything -- the
    pre-P5-T6 no-MCP behavior stays byte-identical."""
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({})
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert len(registry.list_catalog()) == 2


class _FakeBuiltinGateForRegistry:
    """Minimal `BuiltinToolGate` double -- only `.check()` is exercised by
    `BuiltinToolProvider.invoke()`."""

    def __init__(self, refuse: bool) -> None:
        self.refuse = refuse
        self.checked: list[str] = []

    def check(self, tool, run_id):
        self.checked.append(tool.name)
        return f"disabled for test: {tool.name}" if self.refuse else None


def test_compose_run_registry_and_allowed_threads_builtin_gate_into_the_provider():
    """task-545/T6: `builtin_gate=` must reach the freshly-built
    `BuiltinToolProvider` -- NOT a second, independently-built gate --
    else a decision the caller's review hook stamped on that gate would
    never be visible to `invoke()`."""
    gate = _FakeBuiltinGateForRegistry(refuse=True)
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, builtin_gate=gate)
    )
    result = registry.invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is False
    assert result.error == "disabled for test: calculator"
    assert gate.checked == ["calculator"]


def test_compose_run_registry_and_allowed_no_builtin_gate_is_unchanged():
    """`builtin_gate=None` (the default) must not alter the pre-task-545
    no-skills/no-MCP behavior -- the provider builds its own lazy gate."""
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({})
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    result = registry.invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is True


class _WorkspaceProbeTool:
    """A minimal Tool double that reports the run workspace bound around it."""

    name = "probe_workspace"
    description = "records the bound run workspace"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        from tldw_chatbook.Tools import workspace_file_roots as wfr

        return {"workspace": wfr.current_run_workspace_id()}


def test_compose_run_registry_and_allowed_threads_workspace_id_into_the_provider():
    """task-6 (settings-workspaces-folder-roots spec Sec3): `workspace_id=`
    must reach the freshly-built `BuiltinToolProvider` so its `invoke()`
    binds the run's workspace around every tool call."""
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            workspace_id="ws-compose",
        )
    )
    # registry._providers[0] is the BuiltinToolProvider this call just built
    # (see _compose_run_registry_and_allowed's own body) -- poke a probe
    # tool onto it exactly as the constructor-level test does, since the
    # real calculator/datetime tools have no way to report the ContextVar.
    registry._providers[0]._tools["probe_workspace"] = _WorkspaceProbeTool()
    result = registry.invoke_by_name("probe_workspace", {})
    assert result.ok, result.error
    assert '"workspace": "ws-compose"' in result.content


def test_compose_run_registry_and_allowed_no_workspace_id_is_unchanged():
    """`workspace_id=None` (the default) must not alter the pre-task-6
    behavior -- the provider leaves the run workspace unbound."""
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
        )
    )
    registry._providers[0]._tools["probe_workspace"] = _WorkspaceProbeTool()
    result = registry.invoke_by_name("probe_workspace", {})
    assert result.ok, result.error
    assert '"workspace": null' in result.content


def test_run_registry_binds_builtin_provider_to_captured_scratch(tmp_path):
    scratch = tmp_path / "chat-a"
    scratch.mkdir()

    registry, _allowed, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            workspace_id="workspace-default",
            scratch_root=scratch,
            scratch_lease=lambda: contextlib.nullcontext(scratch),
        )
    )

    provider = registry._providers[0]
    assert provider.sandbox_root == scratch.resolve()


@pytest.mark.parametrize("missing", ["root", "lease"])
def test_run_registry_rejects_incomplete_scratch_authority(tmp_path, missing):
    scratch = tmp_path / "chat-a"
    scratch.mkdir()
    kwargs = {
        "scratch_root": scratch,
        "scratch_lease": lambda: contextlib.nullcontext(scratch),
    }
    kwargs["scratch_root" if missing == "root" else "scratch_lease"] = None

    with pytest.raises(ValueError, match="supplied together"):
        _compose_run_registry_and_allowed({}, **kwargs)


def test_two_console_runs_cannot_dispatch_across_scratch_roots(tmp_path):
    from tldw_chatbook.Tools.file_operation_tools import ReadFileTool

    root_a = tmp_path / "chat-a"
    root_b = tmp_path / "chat-b"
    root_a.mkdir()
    root_b.mkdir()
    marker = root_a / "marker.txt"
    marker.write_text("chat-a", encoding="utf-8")
    registry_b, _allowed, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            workspace_id="workspace-default",
            scratch_root=root_b,
            scratch_lease=lambda: contextlib.nullcontext(root_b),
        )
    )
    registry_b._providers[0]._tools["read_file"] = ReadFileTool()

    result = registry_b.invoke_by_name(
        "read_file",
        {"file_path": str(marker)},
    )

    assert result.ok is False
    assert "outside" in str(result.error).lower()


class _StubWriteFileTool:
    """Minimal Tool double named ``write_file`` -- ``_WorkspaceProbeTool``
    above hardcodes its own name (``probe_workspace``), which would make
    `registry.invoke_by_name("write_file", ...)` fail to resolve since the
    catalog entry it produces is keyed by ``.name``, not by the provider's
    internal dict key it was poked under."""

    name = "write_file"
    description = "stub"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"ok": True}


def test_compose_run_registry_and_allowed_threads_ephemeral_into_the_provider():
    """F4 (final-review): `ephemeral=` must reach the freshly-built
    `BuiltinToolProvider` so its `invoke()` refuses the write-shaped
    built-ins for a temporary session. Mirrors ``..._threads_workspace_id_
    into_the_provider`` exactly."""
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            ephemeral=True,
        )
    )
    registry._providers[0]._tools["write_file"] = _StubWriteFileTool()
    result = registry.invoke_by_name("write_file", {})
    assert result.ok is False
    assert "temporary chat" in result.error


def test_compose_run_registry_and_allowed_no_ephemeral_is_unchanged():
    """`ephemeral=False` (the default) must not alter pre-F4 behavior --
    the provider dispatches the tool normally."""
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
        )
    )
    registry._providers[0]._tools["write_file"] = _StubWriteFileTool()
    result = registry.invoke_by_name("write_file", {})
    assert result.ok, result.error


# -- the temporary-session tool gate at the shared choke point --
#
# Every refusal assertion below carries its allowed-normally control in the
# SAME test. That is the dangerous direction here: a guard that fired
# unconditionally would break MCP and skill tools for every user of the app,
# and each "assert refused" half would still be perfectly green.


def _stub_tool(tool_name: str):
    """A minimal always-succeeding ``Tool`` double registered under a name.

    ``_StubWriteFileTool`` above hardcodes one name; the gate tests need a
    handful (the three write-shaped built-ins and the four gateable
    read-only ones, none of which are registered by default because their
    ``[tools]`` gates ship off).
    """

    class _Stub:
        name = tool_name
        description = "stub"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return {"ok": True}

    return _Stub()


class _FakeSourcedProvider:
    """``ToolProvider`` double that reports an arbitrary catalog ``source``.

    Lets the source-based policy be exercised for skills (whose real
    provider deliberately raises from ``invoke``) and for a source no one
    has whitelisted, without inventing a whole provider per case.
    """

    def __init__(self, source, entries):
        self._source = source
        self._entries = list(entries)
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=name, name=name, one_line_description="d", source=self._source
            )
            for name in self._entries
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description="",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"result:{tool_id}")


_SKILL_CONTEXT = {
    "available_skills": [
        {
            "name": "tidy_repo",
            "description": "Tidy the repository",
            "argument_hint": "",
            "trust_blocked": False,
            "disable_model_invocation": False,
        },
    ],
}


def test_invoke_by_name_refuses_an_mcp_tool_only_in_a_temporary_run():
    """Requirements 1 and 5, together: the CHOKE POINT is load-bearing on
    its own, not merely a backstop to the allow-list.

    Both registries here are built by hand and the MCP provider registered
    straight onto them -- no allow-list, no ``_compose_run_registry_and_
    allowed`` filtering anywhere in the picture. So the refusal can only be
    coming from ``invoke_by_name`` itself, and the control proves the same
    call succeeds when the session is not temporary.
    """
    saved_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert control.ok is True, control.error
    assert saved_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]

    temp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert "mcp__srv_a__search" in refused.error
    # The provider was never reached -- refused before dispatch, not after.
    assert temp_provider.invoke_calls == []


def test_invoke_by_name_refuses_a_skill_tool_only_in_a_temporary_run():
    """Requirement 2, at the choke point, with the real ``SkillToolProvider``.

    That class's ``invoke`` raises by design (skill calls route through the
    run-scoped spawn executor), which makes the control unusually sharp: in
    a saved chat the call must still REACH the provider, and the only
    observable proof of that is the RuntimeError escaping. In a temporary
    chat the gate must intercept first and hand back a ToolResult.
    """
    entries = [
        {"name": "tidy_repo", "description": "Tidy the repository", "argument_hint": ""}
    ]

    saved = ToolCatalogRegistry()
    saved.register_provider(SkillToolProvider(entries))
    with pytest.raises(RuntimeError):
        saved.invoke_by_name("tidy_repo", {"args": ""})

    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(SkillToolProvider(entries))
    refused = temporary.invoke_by_name("tidy_repo", {"args": ""})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert "tidy_repo" in refused.error


def test_invoke_by_name_refuses_a_skill_sourced_tool_that_would_otherwise_succeed():
    """Requirement 2 again, with a skill-sourced provider that DOES return
    a successful result -- so the control half is a real, observed success
    rather than the real provider's by-design raise."""
    saved_provider = _FakeSourcedProvider("skill", ["tidy_repo"])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("tidy_repo", {"args": "x"})
    assert control.ok is True, control.error
    assert saved_provider.invoke_calls == [("tidy_repo", {"args": "x"})]

    temp_provider = _FakeSourcedProvider("skill", ["tidy_repo"])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("tidy_repo", {"args": "x"})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert temp_provider.invoke_calls == []


def test_invoke_by_name_refuses_a_provider_source_nobody_whitelisted():
    """Unknown capability fails toward not-writing: a provider added after
    this was written is gated on the day it is added, and the control shows
    it is otherwise dispatched completely normally."""
    saved_provider = _FakeSourcedProvider("some_provider_invented_in_2027", ["thing"])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("thing", {})
    assert control.ok is True, control.error

    temp_provider = _FakeSourcedProvider("some_provider_invented_in_2027", ["thing"])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("thing", {})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert temp_provider.invoke_calls == []


def test_temporary_run_keeps_read_only_builtins_and_still_refuses_write_shaped_ones():
    """Requirements 3 and 4 in one test: no over-blocking, no regression.

    The read-only half is the control that catches an unconditional guard;
    the saved-run half at the end is the control that catches a guard which
    ignores the flag entirely.
    """
    write_shaped = ("write_file", "create_note", "update_note")
    read_only = ("read_file", "list_directory", "glob_files", "grep_files")

    temporary, _allowed, _names, _local = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False), ephemeral=True
    )
    # The gateable built-ins ship behind `[tools]` gates that default to
    # off, so poke stubs under their real names rather than flipping user
    # config from a test.
    for name in write_shaped + read_only:
        temporary._providers[0]._tools[name] = _stub_tool(name)
    temporary.reset_catalog_cache()

    for name in write_shaped:
        result = temporary.invoke_by_name(name, {})
        assert result.ok is False, name
        assert "temporary chat" in result.error, name
    for name in read_only:
        result = temporary.invoke_by_name(name, {})
        assert result.ok, (name, result.error)
    calc = temporary.invoke_by_name("calculator", {"expression": "6*7"})
    assert calc.ok, calc.error

    saved, _allowed, _names, _local = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
    )
    for name in write_shaped:
        saved._providers[0]._tools[name] = _stub_tool(name)
    saved.reset_catalog_cache()
    for name in write_shaped:
        result = saved.invoke_by_name(name, {})
        assert result.ok, (name, result.error)


def test_temporary_run_never_advertises_mcp_or_skill_tools_but_a_saved_run_does():
    """Defense in depth (the allow-list half): the model is not offered a
    tool whose only possible outcome is a refusal. Control in the same
    test: the identical composition for a saved chat advertises both."""
    saved_mcp = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    saved, saved_allowed, _names, _local = _compose_run_registry_and_allowed(
        _SKILL_CONTEXT, mcp_provider=saved_mcp
    )
    assert "mcp__srv_a__search" in saved_allowed
    assert "tidy_repo" in saved_allowed
    assert {e.source for e in saved.list_catalog()} == {"builtin", "skill", "mcp"}

    temp_mcp = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    temporary, temp_allowed, _names, _local = _compose_run_registry_and_allowed(
        _SKILL_CONTEXT, mcp_provider=temp_mcp, ephemeral=True
    )
    assert "mcp__srv_a__search" not in temp_allowed
    assert "tidy_repo" not in temp_allowed
    assert temp_allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert {e.source for e in temporary.list_catalog()} == {"builtin"}
    # Never listed, so a stray call cannot resolve at all -- and the fake
    # is never reached either way.
    assert temporary.invoke_by_name("mcp__srv_a__search", {}).ok is False
    assert temp_mcp.invoke_calls == []


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_builtin():
    """Task 11 review note 2's shadowing precedent, extended to MCP: a
    builtin always wins a same-named MCP tool -- the name is carried
    exactly once (from the builtin), and invoking it never reaches the
    MCP fake."""
    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    )
    assert allowed_tools.count("calculator") == 1
    assert "mcp__srv_a__search" in allowed_tools
    result = registry.invoke_by_name("calculator", {"expression": "1+1"})
    assert result.ok is True
    assert mcp_provider.invoke_calls == []  # the builtin handled it, not the MCP fake


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_runtime_tool():
    """Qodo finding 4 (PR #636)'s shadowing precedent, extended to MCP: a
    tool named like one of the loop's own in-loop runtime handlers must
    never become a distinct, MCP-routable catalog entry."""
    mcp_provider = _FakeMCPProvider([(LOAD_TOOLS_NAME, "shadowing MCP tool")])
    registry, allowed_tools, builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    )
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names) :]
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "mcp") not in catalog_entries


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_skill():
    """A skill (registered before MCP) also wins a same-named MCP tool."""
    context = {
        "available_skills": [
            {
                "name": "code-review",
                "description": "Review a diff",
                "argument_hint": "",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    mcp_provider = _FakeMCPProvider(
        [("code-review", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(context, mcp_provider=mcp_provider)
    )
    assert allowed_tools.count("code-review") == 1
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("code-review", "skill") in catalog_entries
    assert ("code-review", "mcp") not in catalog_entries
    assert "mcp__srv_a__search" in allowed_tools


def test_compose_run_registry_and_allowed_all_mcp_names_colliding_skips_registration():
    """When every MCP entry collides, the provider is not registered at
    all -- no dangling catalog entries the model could never reach."""
    mcp_provider = _FakeMCPProvider([("calculator", "shadowing MCP tool")])
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("calculator", "mcp") not in catalog_entries


def test_non_colliding_mcp_names_pure_helper():
    mcp_provider = _FakeMCPProvider([("calculator", "x"), ("mcp__srv__y", "y")])
    assert _non_colliding_mcp_names(mcp_provider, {"calculator"}) == ("mcp__srv__y",)


def test_shadowed_mcp_names_reports_what_the_filter_drops():
    """A user's configured MCP tool must never vanish silently.

    Built-ins keep winning the collision -- inverting that would let a
    compromised server name-squat an audited built-in -- so the shadowing
    is surfaced instead. ``shadowed_mcp_names`` and
    ``_non_colliding_mcp_names`` must partition the same catalog: every
    entry appears in exactly one of the two results.
    """
    mcp_provider = _FakeMCPProvider([("read_file", "x"), ("weather", "y")])
    collision_names = {"read_file"}

    assert shadowed_mcp_names(mcp_provider, collision_names) == ("read_file",)
    assert _non_colliding_mcp_names(mcp_provider, collision_names) == ("weather",)


def test_compose_run_registry_and_allowed_warns_when_mcp_tool_is_shadowed():
    """``shadowed_mcp_names`` itself is a silent pure partition -- the
    user-visible half of this behavior is the warning logged from
    ``_compose_run_registry_and_allowed`` for each dropped name. Without it
    a user whose configured MCP tool stopped being offered has no way to
    discover a built-in silently claimed the name.

    caplog does not intercept loguru (this project's logger); attach a
    temporary loguru sink instead (mirrors
    ``Tests/Chat/test_console_chat_store.py``'s pattern).
    """
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert any("calculator" in message for message in messages), messages


def test_compose_run_registry_and_allowed_no_warning_without_mcp_collisions():
    """No MCP name collided with a builtin, so nothing should be logged --
    guards against a future refactor that logs unconditionally instead of
    only when a name is actually dropped."""
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search")])
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert not any("is shadowed by a built-in" in message for message in messages)


def test_compose_run_registry_and_allowed_walks_mcp_catalog_only_once():
    """Finding 8 (substrate review): ``_partition_mcp_catalog_by_collision``
    already computes both the non-colliding and shadowed sides in a single
    walk of ``mcp_provider.list_catalog()``, but the composition function
    used to call the two PUBLIC wrapper functions separately
    (``_non_colliding_mcp_names`` then ``shadowed_mcp_names``), each of
    which re-invoked the partition -- walking the catalog twice per run.
    """
    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )

    _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)

    assert mcp_provider.list_catalog_calls == 1


def test_compose_run_registry_and_allowed_warns_about_a_shadowed_name_only_once():
    """Finding 8 (substrate review): ``_compose_run_registry_and_allowed``
    runs once per Console message, so a naive per-call warning re-logs the
    identical shadowed-name message on every single turn of a long-running
    session. The warning must fire at most once per name for the life of
    the process.
    """
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        # Simulate three Console messages in the same session, each
        # re-composing the run registry.
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    calculator_warnings = [m for m in messages if "calculator" in m]
    assert len(calculator_warnings) == 1, messages


def test_compose_run_registry_and_allowed_warns_once_per_distinct_shadowed_name():
    """A dedup keyed on the wrong thing (e.g. "has anything ever been
    warned") would silently suppress a DIFFERENT shadowed name's very
    first warning -- this pins that the dedup is per-name. Both
    ``calculator`` and ``get_current_datetime`` are always-on builtins
    (see ``Tools.tool_executor``), so both genuinely collide.
    """
    from loguru import logger as loguru_logger

    first_provider = _FakeMCPProvider([("calculator", "x")])
    second_provider = _FakeMCPProvider([("get_current_datetime", "y")])
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=first_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=second_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert any("calculator" in m for m in messages)
    assert any("get_current_datetime" in m for m in messages)


def test_run_reply_routes_fence_call_to_mcp_provider(tmp_path):
    """End-to-end: a run with no skills service still registers an eligible
    MCP provider fresh (not the shared, construction-time registry) and
    dispatches a matching fence call to it."""
    scripts = [
        [_fence("mcp__srv_a__search", {"query": "weather"})],
        ["The weather is nice."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])

    outcome = _run(bridge, store, session, aid, mcp_provider=mcp_provider)

    assert outcome.status == "done"
    assert outcome.final_text == "The weather is nice."
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("mcp__srv_a__search" in row.content for row in tool_rows)


def test_run_reply_forwards_review_tool_calls_hook_to_agent_service(tmp_path):
    """`review_tool_calls=` must reach AgentService/the loop -- a batch
    verdict other than "proceed" skips dispatch and becomes the tool
    result, exactly like the T4 hook contract documents."""
    scripts = [
        [
            "I will request approval for this calculation.\n",
            _fence("calculator", {"expression": "6*7"}),
        ],
        ["done."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    captured_batches = []

    # PR2a Task 5: an AgentService-wired hook takes `(calls, run_id)`.
    def hook(calls, run_id):
        captured_batches.append(list(calls))
        return {"calculator": CONTROLLER_USER_DENIED_REFUSAL.format(name="calculator")}

    outcome = _run(bridge, store, session, aid, review_tool_calls=hook)

    assert outcome.status == "done"
    assert captured_batches and captured_batches[0][0].name == "calculator"
    live = _tool_messages(store, session.id)
    resumed = _resume_tool_messages(db)
    assert not any(step.kind == STEP_TOOL_CALL for step in outcome.steps)
    assert (
        next(
            step for step in outcome.steps if step.kind == STEP_TOOL_RESULT
        ).tool_outcome
        == "blocked"
    )
    assert [marker.activity_presentation.kind for marker in live] == [
        "planning",
        "tool",
    ]
    assert live[0].content == "I will request approval for this calculation."
    assert any("denied" in row.content.lower() for row in live)
    assert live[1].activity_presentation.status == "blocked"
    assert _activity_marker_signature(resumed) == _activity_marker_signature(live)


def test_run_reply_still_wires_stamp_scope_for_the_inline_kill_switch_path(
    tmp_path, monkeypatch
):
    """`run_reply` must still thread `mcp_provider.stamp_scope` through to
    `AgentService(review_state_scope=...)` whenever an MCP provider is
    composed for this run.

    This is the surviving half of the old
    `test_run_reply_wires_mcp_provider_stamp_scope_around_a_spawned_child`
    (C1, probe-verified security regression). Its guarantee -- one
    enter/exit of the composed provider's `stamp_scope()` around a spawned
    child -- is UNCHANGED on the inline path, which is what a user who sets
    `[agents] max_live_subagents = 1` gets, so the assertion is unchanged
    too; only the pinning of that path is new (before Task 6.5 it was the
    shipped default and needed no pinning). The concurrent path is the
    companion test below -- where holding this scope would be actively
    harmful, not merely unnecessary.
    """
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        ),
    )
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn (inline, so strictly ordered)
        ["Done: ", "2."],  # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])

    outcome = _run(bridge, store, session, aid, mcp_provider=mcp_provider)

    assert outcome.status == "done"
    assert mcp_provider.stamp_scope_calls == 1


class _StampingMCPProvider(_FakeMCPProvider):
    """A `_FakeMCPProvider` that models the REAL per-run stamp bookkeeping.

    `_FakeMCPProvider.stamp_scope` is a bare call counter, which cannot
    show what the scope actually DOES. This one mirrors
    `MCPToolProvider`'s two documented behaviours that matter here:

    * verdicts are keyed by ``(run_id, llm_name)``, so a run can only ever
      reach its own slice (PR2a Task 5);
    * ``stamp_scope(run_id)`` snapshots that run's slice on enter, CLEARS
      it, and restores the snapshot on exit.

    That second behaviour is exactly why the threaded spawn path must not
    take the scope: entering it mid-turn wipes the parent's live verdicts,
    and with concurrent siblings there is no LIFO order in which the
    restore could put them back correctly.
    """

    def __init__(self, entries, on_invoke=None):
        super().__init__(entries)
        self.stamps: dict[tuple[str, str], str] = {}
        self.refusals: list[str] = []
        # Runs at the top of invoke, BEFORE the stamp is consumed -- the
        # seam a test uses to pin what has happened by the time the parent
        # cashes in its verdict.
        self._on_invoke = on_invoke

    def stamp(self, run_id, llm_name, verdict):
        self.stamps[(run_id, llm_name)] = verdict

    def invoke(self, tool_id, args):
        if self._on_invoke is not None:
            self._on_invoke()
        # Fails CLOSED on a missing stamp, like the real gate: that is what
        # makes a wiped verdict observable rather than silently benign.
        run_id = current_run_id()
        if self.stamps.get((run_id, tool_id)) != "approve_once":
            self.refusals.append(tool_id)
            return ToolResult(ok=False, error=f"no stamped approval for {tool_id}")
        return super().invoke(tool_id, args)

    @contextlib.contextmanager
    def stamp_scope(self, run_id):
        self.stamp_scope_calls += 1
        snapshot = {k: v for k, v in self.stamps.items() if k[0] == run_id}
        self.stamps = {k: v for k, v in self.stamps.items() if k[0] != run_id}
        try:
            yield
        finally:
            self.stamps = {k: v for k, v in self.stamps.items() if k[0] != run_id}
            self.stamps.update(snapshot)


def test_run_reply_keeps_the_parents_mcp_verdict_across_a_concurrent_child(tmp_path):
    """The parent's own approval SURVIVES a concurrently-running child.

    Replaces `test_run_reply_wires_mcp_provider_stamp_scope_around_a_
    spawned_child`'s spawn-path half. That test pinned the scope being
    entered around a child, which PR2a Task 6 deliberately stopped doing on
    the threaded path -- not as a relaxation but because it is now unsafe:
    `stamp_scope` CLEARS the parent's slice on enter, so holding it across
    siblings would wipe verdicts the parent is still going to consume, and
    with no LIFO order the restore cannot repair it.

    The protection that replaces it is Task 5's per-run keying, and this
    asserts THAT, at the same bridge level.

    The interleaving is pinned, not hoped for, because only ONE ordering
    is dangerous: the child must still be live when the parent stamps, and
    must finish before the parent consumes. The child therefore blocks
    until the parent's own MCP invoke releases it, and that invoke joins
    the child's thread before reading the stamp. Under the mutation
    (re-wrapping the fleet path in `review_state_scope`) the child's scope
    is open across the parent's stamp, so its exit-restore rolls the
    parent's slice back to a snapshot taken before that stamp existed and
    the parent's own call is refused -- verified red both behaviourally
    (`refusals`) and structurally (`stamp_scope_calls`).
    """
    release_child = threading.Event()

    def gated_child_turn():
        assert release_child.wait(5), "the parent never released the child"
        return ["2"]

    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        [_fence("mcp__srv_a__search", {"query": "weather"})],  # parent's own call
        ["Done."],  # primary final
    ]
    child_script = [gated_child_turn]  # the child's single turn, on its own thread
    gateway = _FleetChunkGateway(scripts, child_script)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    def release_then_settle():
        # The parent is about to cash in its verdict. Let the child finish
        # first, and wait until its THREAD is gone -- that is the moment any
        # context manager wrapping the child's run would have exited.
        release_child.set()
        _join_fleet_threads()

    mcp_provider = _StampingMCPProvider(
        [("mcp__srv_a__search", "Search the web")], on_invoke=release_then_settle
    )

    # The parent's verdict, stamped under ITS run id, while the child is
    # still live. The review hook is what does this in production; here it
    # stamps directly so the test is about the scope, not about the hook.
    def review(calls, run_id):
        for call in calls:
            if call.name == "mcp__srv_a__search":
                mcp_provider.stamp(run_id, call.name, "approve_once")
        return {}

    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-fleet-stamp",
        mcp_provider=mcp_provider,
        review_tool_calls=review,
    )

    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-fleet-stamp") == 1  # the child really ran
    assert gateway.child_calls == 1  # ... concurrently, on its own thread
    # The parent's own approved call executed: its verdict outlived the
    # concurrent child.
    assert mcp_provider.refusals == []
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]
    # ... and the threaded child was NOT wrapped in the parent's scope,
    # which is what would have wiped that verdict.
    assert mcp_provider.stamp_scope_calls == 0


def test_skill_named_like_a_runtime_tool_never_shadows_it_at_invocation(tmp_path):
    """End-to-end: a skill front-matter'd as "find_tools" must not hijack
    the runtime's own find_tools meta-tool -- the real runtime dispatch
    still answers, and the skill is never invoked (it's excluded from the
    run's catalog/allow-list entirely, so no sub-agent is ever spawned for
    what looks like a skill call)."""
    scripts = [
        [_fence("find_tools", {"query": "anything"})],
        ["done."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(skill_name="find_tools")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-runtime-collide"
    )

    assert outcome.status == "done"
    assert skills_service.execute_calls == []  # the skill was never invoked
    assert db.count_subagent_runs("conv-runtime-collide") == 0


def test_skill_named_like_a_builtin_never_shadows_it_at_invocation(tmp_path):
    """End-to-end: a skill front-matter'd as "calculator" must not hijack
    calculator calls -- the real builtin still answers, and no sub-agent
    is spawned for what looks like a skill call."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is 42."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(skill_name="calculator")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-collide")

    assert outcome.status == "done"
    assert skills_service.execute_calls == []  # the skill was never invoked
    assert db.count_subagent_runs("conv-collide") == 0  # no sub-agent spawned
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("42" in row.content for row in tool_rows)


# -- Skills Phase-2 gate finding 1: discovery-heavy runs must not exhaust the
# bare engine step budget right after a successful skill call, before the
# final wrap-up reply (Task-14 gate scenario 5: "Find a skill that can
# shout, load it, and use it on: hello"). --


class _ManySkillsService:
    """Enough real skills to exceed DIRECT_DISCLOSE_THRESHOLD on their own
    (even before the 2 builtins _compose_run_registry_and_allowed always
    adds), so the catalog defers everything to find_tools/load_tools -- the
    same >threshold-skill shape that engaged progressive disclosure in the
    live gate capture."""

    def __init__(self):
        self.execute_calls = []

    async def get_context(self, *, mode="local"):
        names = ["shout"] + [f"filler{i}" for i in range(DIRECT_DISCLOSE_THRESHOLD)]
        return {
            "available_skills": [
                {
                    "name": n,
                    "description": f"{n} skill",
                    "argument_hint": "[args]",
                    "trust_blocked": False,
                    "disable_model_invocation": False,
                }
                for n in names
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append((name, args))
        return {
            "skill_name": name,
            "rendered_prompt": f"SHOUT[{args}]",
            "allowed_tools": None,
            "execution_mode": "inline",
        }


def _discovery_heavy_shout_scripts():
    """The gate's live shape, split per agent (PR2a Task 6.5).

    Mirrors the gate's live raw step log: find_tools({"query": "shout"})
    -> load_tools({"ids": ["skill:shout"]}) -> shout({"args": "hello"})
    -> the sub-agent's own turn -> the primary's final wrap-up reply.

    The ROUND COUNT is unchanged from before the fleet: a skill call runs
    its child INLINE and returns the skill's output, so there is no
    collection round (see `AgentService`'s `invoke_tool` skill branch).
    Only the SHAPE changed -- the five turns are addressed per agent (the
    primary's four, the child's one) rather than popped off one queue,
    which is how they should always have been written.

    Returns:
        (parent_script, child_script) for `_FleetChunkGateway`.
    """
    parent = [
        [_fence("find_tools", {"query": "shout"})],
        [_fence("load_tools", {"ids": ["skill:shout"]})],
        [_fence("shout", {"args": "hello"})],
        ["Shouted: HELLO"],  # primary final answer
    ]
    child = [["HELLO"]]  # sub-agent turn (never streamed to the store)
    return parent, child


def test_discovery_heavy_skill_run_completes_done_not_stuck(tmp_path):
    """Task-14 gate finding 1 repro: find_tools -> load_tools -> a skill
    call -> final answer needs exactly 10 primary-loop steps at minimum (3
    steps per tool round x 3 rounds, plus 1 final model turn -- see
    agent_runtime.run_agent_loop's per-round STEP_MODEL/STEP_TOOL_CALL/
    STEP_TOOL_RESULT accounting). The bare engine default
    (agent_models.RunBudget.max_steps == 8, pinned by
    test_agent_models.test_budget_defaults) is ONE ROUND short of that --
    it exhausts right after the skill's tool_result, one step before the
    wrap-up reply, even though every tool call already succeeded. The
    Console bridge must give this exact shape enough headroom to actually
    reach the final answer and persist `done`."""
    parent_script, child_script = _discovery_heavy_shout_scripts()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _ManySkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_FleetChunkGateway(parent_script, child_script),
        skills_service=skills_service,
    )

    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-discover"
    )

    assert outcome.status == "done"
    assert outcome.final_text == "Shouted: HELLO"
    assert skills_service.execute_calls == [("shout", "hello")]
    assert db.count_subagent_runs("conv-discover") == 1
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("shout" in row.content for row in tool_rows)


def test_console_run_budget_is_raised_above_the_bare_engine_default(tmp_path):
    """Pins the config-assembly override directly: a primary Console run's
    PERSISTED budget must sit strictly above the engine's own pure default
    (RunBudget().max_steps == 8 -- see test_agent_models.test_budget_defaults,
    which stays unchanged) -- with enough headroom (>= 16, per the counted
    10-step discovery-heavy floor above) to survive a real disclosure run,
    and a proportionally raised wall-clock allowance for the extra turns."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["hi there"]])
    _run(bridge, store, session, aid, conversation_id="conv-budget")
    run = db.list_runs("conv-budget")[0]
    assert run["agent_kind"] == "primary"
    assert run["budget"]["max_steps"] > 8
    assert run["budget"]["max_steps"] >= 16
    assert run["budget"]["max_wall_seconds"] > 240.0


def _make_bridge() -> ConsoleAgentBridge:
    store = MagicMock()
    store.messages_for_session.return_value = []
    return ConsoleAgentBridge(
        agent_runs_db=MagicMock(),
        store=store,
        provider_gateway=MagicMock(),
    )


def test_run_reply_returns_runoutcome_done():
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    with patch.object(AgentService, "run_turn", return_value=("run-1", outcome)):
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-1"
    assert result.status == RUN_DONE
    assert result.final_text == "done"


def test_agent_rounds_keep_thinking_paired_across_native_tool_calls(tmp_path) -> None:
    gateway = _ChunkGateway(
        [
            [
                ProviderThinkingDelta(
                    text="choose a tool",
                    provider="llama_cpp",
                    model="reasoner",
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                ),
                _native_calls("calculator", {"expression": "1 + 1"}),
            ],
            [
                ProviderThinkingDelta(
                    text="use the result",
                    provider="llama_cpp",
                    model="reasoner",
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                ),
                "The answer is 2.",
            ],
        ]
    )
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)
    live_tokens: list[int | None] = []
    terminal_tokens: list[int | None] = []
    original_replace = store.replace_message_thinking
    original_settle = store.settle_message_thinking

    def record_replace(message_id, envelope, *, generation_token=None):
        live_tokens.append(generation_token)
        return original_replace(
            message_id,
            envelope,
            generation_token=generation_token,
        )

    def record_settle(message_id, envelope, *, generation_token=None):
        terminal_tokens.append(generation_token)
        return original_settle(
            message_id,
            envelope,
            generation_token=generation_token,
        )

    store.replace_message_thinking = record_replace
    store.settle_message_thinking = record_settle

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        resolution=_native_resolution(),
    )

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_DONE
    assert assistant.content == "The answer is 2."
    assert assistant.thinking is not None
    assert [block.round_ordinal for block in assistant.thinking.blocks] == [0, 1]
    assert [block.text for block in assistant.thinking.blocks] == [
        "choose a tool",
        "use the result",
    ]
    assert [block.status for block in assistant.thinking.blocks] == [
        "complete",
        "complete",
    ]
    assert live_tokens and terminal_tokens
    assert set(live_tokens + terminal_tokens) == {live_tokens[0]}
    assert type(live_tokens[0]) is int


def test_agent_rounds_advance_for_fence_first_tool_calls(tmp_path) -> None:
    gateway = _ChunkGateway(
        [
            [
                ProviderThinkingDelta(
                    text="choose a tool",
                    provider="llama_cpp",
                    model="reasoner",
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                ),
                _fence("calculator", {"expression": "1 + 1"}),
            ],
            [
                ProviderThinkingDelta(
                    text="use the result",
                    provider="llama_cpp",
                    model="reasoner",
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                ),
                "The answer is 2.",
            ],
        ]
    )
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(bridge, store, session, assistant_id)

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_DONE
    assert assistant.thinking is not None
    assert [block.round_ordinal for block in assistant.thinking.blocks] == [0, 1]
    assert [block.text for block in assistant.thinking.blocks] == [
        "choose a tool",
        "use the result",
    ]


def test_agent_terminal_proprietary_evidence_never_enters_answer_text(tmp_path) -> None:
    gateway = _ChunkGateway(
        [
            [
                ProviderProprietaryThinkingEvidence(
                    provider="moonshot",
                    model="kimi",
                    protocol="chat_completions",
                    source_format="reasoning_content",
                ),
                "answer",
            ]
        ]
    )
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(bridge, store, session, assistant_id)

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_DONE
    assert assistant.content == "answer"
    assert assistant.thinking is not None
    assert assistant.thinking.blocks[0].visibility == "proprietary"
    assert not hasattr(assistant.thinking.blocks[0], "text")


def test_agent_answer_without_evidence_leaves_thinking_null(tmp_path) -> None:
    bridge, _db, store, session, assistant_id = _bridge(tmp_path, [["answer"]])

    outcome = _run(bridge, store, session, assistant_id)

    assert outcome.status == RUN_DONE
    assert store.get_message(assistant_id).thinking is None


def test_agent_provider_failure_settles_open_thinking_as_failed(tmp_path) -> None:
    class RaisingGateway(_ChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            yield ProviderThinkingDelta(
                text="unfinished",
                provider="llama_cpp",
                model="reasoner",
                protocol="chat_completions",
                source_format="start_anchored_think",
            )
            raise RuntimeError("provider failed")

    gateway = RaisingGateway([])
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(bridge, store, session, assistant_id)

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_ERROR
    assert assistant.thinking is not None
    assert assistant.thinking.blocks[0].status == "failed"


def test_agent_stop_after_thinking_does_not_pull_the_next_answer_chunk(
    tmp_path,
) -> None:
    bridge, _db, store, session, assistant_id = _bridge(
        tmp_path,
        [
            [
                ProviderThinkingDelta(
                    text="unfinished",
                    provider="llama_cpp",
                    model="reasoner",
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                ),
                "must not stream",
            ]
        ],
    )
    flags = iter([False, True])

    outcome = _run(
        bridge,
        store,
        session,
        assistant_id,
        should_cancel=lambda: next(flags, True),
    )

    assistant = store.get_message(assistant_id)
    assert outcome.status == RUN_CANCELLED
    assert assistant.content == ""
    assert assistant.thinking is not None
    assert assistant.thinking.blocks[0].status == "stopped"


def test_agent_late_thinking_after_controller_stop_is_durably_settled(
    tmp_path,
) -> None:
    """A detached bridge must durably hand late evidence to the stopped owner."""

    class GatedThinkingGateway:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            yield ProviderThinkingDelta(
                text="late but delivered",
                provider="llama_cpp",
                model="reasoner",
                protocol="chat_completions",
                source_format="start_anchored_think",
            )

    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "paired answer",
                    "reasoning_blocks": ["paired private state"],
                    "calls": [
                        {
                            "call_id": "paired-call",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "done",
                        }
                    ],
                }
            ],
        }
    )
    chat_db = CharactersRAGDB(tmp_path / "chat.sqlite", "late-thinking")
    runs_db = AgentRunsDB(tmp_path / "runs.sqlite", client_id="late-thinking")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(title="late thinking")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="question",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="paired answer",
            persist=True,
        )
        owned = store._message_or_raise(assistant.id)
        owned.provider_continuation = checkpoint
        assert store.persist_selected_generation(assistant.id) is True
        owned.status = "streaming"
        owned.assistant_generation_state = "streaming"

        gateway = GatedThinkingGateway()
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=store,
            provider_gateway=gateway,
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            agent_bridge=bridge,
        )
        cancel_event = threading.Event()
        controller._active_cancel_events[session.id] = cancel_event
        controller._active_assistant_message_ids[session.id] = assistant.id
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Thinking"),
            session_id=session.id,
        )
        result: dict[str, object] = {}

        def run_bridge() -> None:
            try:
                result["reply"] = bridge.run_reply(
                    conversation_id=session.persisted_conversation_id,
                    session_id=session.id,
                    resolution=_test_resolution(
                        model="reasoner",
                        thinking_stream_disposition="displayable",
                        thinking_round_trip_version=1,
                    ),
                    assistant_message_id=assistant.id,
                    model="reasoner",
                    session_system_prompt="",
                    agent_messages=[{"role": "user", "content": "question"}],
                    should_cancel=cancel_event.is_set,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                result["error"] = exc

        worker = threading.Thread(target=run_bridge, daemon=True)
        worker.start()
        assert gateway.entered.wait(timeout=3)
        assert controller.stop_active_run(record_user_stop=False) is True
        assert session.persisted_conversation_id is not None
        assert assistant.persisted_message_id is not None
        stopped_row = chat_db.get_message_by_id(assistant.persisted_message_id)
        assert stopped_row is not None
        assert stopped_row["assistant_generation_state"] == "stopped"
        assert stopped_row["thinking_blocks_json"] is None
        # Stop commits the visible terminal state while the detached worker
        # still owns late-evidence settlement authority.
        assert store._generation_runtime_counts() == (0, 0, 1)

        gateway.release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert "error" not in result
        _run_id, outcome = result["reply"]
        assert outcome.status == RUN_CANCELLED
        assert store._generation_runtime_counts() == (0, 0, 0)

        settled_row = chat_db.get_message_by_id(assistant.persisted_message_id)
        assert settled_row is not None
        settled_version = settled_row["version"]
        settled = store.get_message(assistant.id)
        assert settled.thinking is not None
        store.settle_message_thinking(assistant.id, settled.thinking)
        assert chat_db.get_message_by_id(assistant.persisted_message_id)["version"] == (
            settled_version
        )

        rows = chat_db.get_messages_for_conversation(
            session.persisted_conversation_id, limit=100
        )
        nodes = [
            ConsoleChatMessage(
                id=str(row["id"]),
                role=ConsoleMessageRole(str(row["role"])),
                content=str(row.get("content") or ""),
                persisted_message_id=str(row["id"]),
                parent_message_id=row.get("parent_message_id"),
            )
            for row in rows
        ]
        reloaded_store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
        reloaded_store.restore_persisted_session(
            title="late thinking reload",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=nodes,
            active_leaf_persisted_id=assistant.persisted_message_id,
        )
        reloaded = reloaded_store.get_message(assistant.persisted_message_id)
        assistants = [
            message
            for message in reloaded_store.messages_for_session(
                reloaded_store.active_session_id
            )
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert len(assistants) == 1
        assert reloaded.content == "paired answer"
        assert reloaded.assistant_generation_state == "stopped"
        assert reloaded.provider_continuation == checkpoint
        assert reloaded.thinking is not None
        assert reloaded.thinking.blocks[0].text == "late but delivered"
        assert reloaded.thinking.blocks[0].status == "stopped"
    finally:
        runs_db.close()
        chat_db.close_connection()


def test_agent_adapter_preflights_backend_before_provider_contact() -> None:
    class UnsupportedPersistence:
        pass

    class ProviderSpy:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            self.calls += 1
            yield "answer"

    store = ConsoleChatStore(persistence=UnsupportedPersistence())
    session = store.create_session(ephemeral=False)
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    gateway = ProviderSpy()
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    adapter = _StreamingModelAdapter(
        store=store,
        provider_gateway=gateway,
        resolution=_test_resolution(
            thinking_stream_disposition="displayable",
            thinking_round_trip_version=1,
        ),
        assistant_message_id=assistant.id,
        should_cancel=lambda: False,
        loop=loop,
        native_tools=False,
        thinking_capture=ThinkingCapture(assistant_owner_id=assistant.id),
    )
    try:
        with pytest.raises(
            ConsoleThinkingCompatibilityError,
            match="cannot preserve model thinking version 1",
        ):
            adapter.chat_call(messages_payload=[{"role": "user", "content": "hi"}])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()

    assert gateway.calls == 0


def test_run_reply_passes_private_continuation_sidecar_to_agent_service():
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("BRIDGE-PRIVATE-CANARY",),
                calls=(
                    ContinuationCall(
                        call_id="c",
                        name="lookup",
                        arguments="{}",
                        state="completed",
                    ),
                ),
            ),
        ),
    )
    sidecar = (ProviderContinuationSidecar("a-old", checkpoint),)
    target = ContinuationRestoreTarget(
        "deepseek",
        "deepseek-v4-flash",
        "responses",
        "https://api.deepseek.com/v1",
    )

    with patch.object(
        AgentService, "run_turn", return_value=("run-1", outcome)
    ) as run_turn:
        bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "assistant", "content": "old", "_owner": "a-old"}],
            should_cancel=lambda: False,
            continuation_sidecar=sidecar,
            continuation_target=target,
            continuation_owner_key="_owner",
        )

    assert run_turn.call_args.kwargs["continuation_sidecar"] is sidecar
    assert run_turn.call_args.kwargs["continuation_target"] is target
    assert run_turn.call_args.kwargs["continuation_owner_key"] == "_owner"


def test_resumed_sidecars_reach_normal_prepared_gateway_once(tmp_path) -> None:
    class CapturingGateway(ConsoleProviderGateway):
        def __init__(self) -> None:
            super().__init__(config_provider=lambda: {}, environ={})
            self.dispatched: list[object] = []

        async def stream_chat(self, resolution, messages, tools=None, **_kwargs):
            self.dispatched.append(messages)
            yield "finished"

    target = ContinuationRestoreTarget(
        "moonshot", "kimi-k3", "chat_completions", "https://api.moonshot.ai/v1"
    )
    prior = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="prior visible",
                reasoning_blocks=("PRIOR-PRIVATE-REASONING",),
                calls=(),
            ),
        ),
    )
    active = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("ACTIVE-PRIVATE-REASONING",),
                calls=(
                    ContinuationCall(
                        call_id="active-call",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state="pending",
                    ),
                ),
            ),
        ),
    )
    gateway = CapturingGateway()
    db = AgentRunsDB(tmp_path / "resume-prepared.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    aid = assistant.id
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
    )
    store.get_message(aid).provider_continuation = active
    store.get_message(aid).provider_continuation_message_version = 1
    resolution = ConsoleProviderResolution(
        provider="Moonshot",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k3",
        ready=True,
        readiness_key="moonshot",
        execution_key="moonshot",
        streaming=True,
        continuation_protocol="chat_completions",
    )

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model="kimi-k3",
        agent_messages=[
            {"role": "assistant", "content": "prior visible", "_owner": "prior"},
            {"role": "user", "content": "continue"},
        ],
        restore_provider_continuation=active,
        restore_provider_target=target,
        # TASK-16270: PR #1612 widened the resume contract — the ACTIVE
        # checkpoint now rides as its own continuation group, attached to
        # the canonical assistant row carrying the pending call's
        # tool_calls, so the expanded transcript must include that row.
        expand_provider_continuation=lambda _checkpoint: [
            {
                "role": "assistant",
                "content": "ACTIVE-CANONICAL-ONCE",
                "tool_calls": [
                    {
                        "id": "active-call",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            }
        ],
        resume_provider_continuation=True,
        continuation_sidecar=(ProviderContinuationSidecar("prior", prior),),
        continuation_target=target,
        continuation_owner_key="_owner",
    )

    # This foundation-only fake emits no terminal provider checkpoint, so the
    # runtime correctly rejects the turn after the request seam under test.
    assert outcome.status == "error"
    assert len(gateway.dispatched) == 1
    prepared = gateway.dispatched[0]
    assert isinstance(prepared, PreparedProviderRequest)
    # TASK-16270: since PR #1612 the ACTIVE checkpoint rides as its own
    # continuation group, owned by the assistant message being resumed —
    # alongside the prior sidecar group.
    assert [group.owner_message_id for group in prepared.continuation_groups] == [
        "prior",
        aid,
    ]
    without_private = gateway.prepare_chat_request(
        resolution,
        replace(
            prepared.semantic,
            compactable=tuple(
                replace(unit, continuation_groups=())
                for unit in prepared.semantic.compactable
            ),
            active_continuation_groups=(),
        ),
        continuation_target=target,
    )
    assert (
        prepared.accounting.total_input_tokens
        > without_private.accounting.total_input_tokens
    )
    assert (
        sum(
            row.get("content") == "ACTIVE-CANONICAL-ONCE"
            for row in prepared.messages_payload
        )
        == 1
    )


def test_run_reply_returns_runoutcome_error():
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")

    with patch.object(AgentService, "run_turn", return_value=("run-1", outcome)):
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-1"
    assert result.status == RUN_ERROR


def test_run_reply_returns_run_id_and_does_not_store_native_assistant_id():
    """run_reply exposes the primary run id to its caller but must NOT forward
    the native in-memory assistant_message_id into run_turn: create_run would
    store it, and that native id can never match any persisted_message_id, so an
    unfinished/crashed run would be left holding a stale non-null id. The run
    row therefore starts NULL (assistant_message_id omitted / None); the
    controller writes the durable persisted id onto the run on every terminal
    path later, via record_run_assistant_message."""
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    with patch.object(
        AgentService, "run_turn", return_value=("run-xyz", outcome)
    ) as run_turn:
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="native-a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-xyz"
    assert result is outcome
    # The native id is NOT forwarded -- the kwarg is omitted (or None), so
    # create_run leaves the run's assistant_message_id NULL at create time.
    assert run_turn.call_args.kwargs.get("assistant_message_id") is None


def test_native_tool_schemas_returns_builtin_tool_schemas():
    bridge = _make_bridge()

    schemas = bridge.native_tool_schemas()

    names = {schema["name"] for schema in schemas}
    assert "calculator" in names
    assert "get_current_datetime" in names
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema


# -- task-4 (skills-agent-install): the install_skill closure built in
# run_reply and threaded via AgentService(install_skill_tool=...). Order is
# load-bearing: enforce policy -> classify URL -> in-chat confirm -> install
# -> broad-catch wrap. --


def _install_skills_service():
    svc = _FakeSkillsService()

    def enforce_install_remote():
        return None

    async def import_skill_file(
        *a, **k
    ):  # not used (install_skill_from_url is patched)
        return {"name": "unused"}

    svc.enforce_install_remote = enforce_install_remote
    svc.import_skill_file = import_skill_file
    return svc


def test_install_skill_confirm_allow_installs(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    installed = []

    async def fake_install(url, *, scope_service, **kw):
        installed.append(url)
        return {
            "name": "demo",
            "trust_status": "quarantined_added",
            "trust_blocked": True,
        }

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Installed it."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    confirmed = []

    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-install",
        request_skill_install_confirm=lambda url: confirmed.append(url) or True,
    )
    assert outcome.status == "done"
    assert confirmed == ["https://github.com/o/r"]
    assert installed == ["https://github.com/o/r"]
    tool_msgs = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role == ConsoleMessageRole.TOOL
    ]
    assert any("demo" in c and "pending" in c.lower() for c in tool_msgs)


def test_install_skill_confirm_deny_does_not_install(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise AssertionError("install must not run when the user denies")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Okay, cancelled."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-deny",
        request_skill_install_confirm=lambda url: False,
    )
    assert outcome.status == "done"
    tool_msgs = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role == ConsoleMessageRole.TOOL
    ]
    assert any("declined" in c.lower() for c in tool_msgs)


def test_install_skill_malformed_url_never_prompts(tmp_path):
    """A URL that fails classification returns an error WITHOUT prompting."""
    prompted = []
    scripts = [
        [_fence("install_skill", {"url": "not-a-url"})],
        ["That URL is not valid."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-bad",
        request_skill_install_confirm=lambda url: prompted.append(url) or True,
    )
    assert outcome.status == "done"
    assert prompted == []  # classification failed before any prompt


def test_install_skill_collision_error_survives_turn(tmp_path, monkeypatch):
    """A bare ValueError('local_skill_exists:...') is wrapped, not fatal."""
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise ValueError("local_skill_exists:demo")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)
    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["It already exists."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-exists",
        request_skill_install_confirm=lambda url: True,
    )
    assert outcome.status == "done"  # turn survives the bare ValueError
    tool_msgs = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role == ConsoleMessageRole.TOOL
    ]
    assert any("local_skill_exists" in c for c in tool_msgs)


def test_install_skill_absent_without_confirm_callback(tmp_path):
    """No request_skill_install_confirm wired -> the tool is ABSENT, not auto-denied.

    A skills service alone is not enough to advertise install_skill: without a
    confirm callback, run_reply must never pin/dispatch the tool at all, so a
    model call to it falls through the same "Tool not permitted" path as any
    other undisclosed tool -- never the misleading "declined" message.
    """
    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["It seems that tool is unavailable."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-no-confirm",
        # request_skill_install_confirm intentionally omitted.
    )
    assert outcome.status == "done"
    tool_msgs = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role == ConsoleMessageRole.TOOL
    ]
    assert any("Tool not permitted: install_skill" in c for c in tool_msgs)
    assert not any("declined" in c.lower() for c in tool_msgs)


# --- task-628: combined per-turn state scopes -------------------------------


def test_combine_state_scopes_none_and_single_are_passthrough():
    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    assert _combine_state_scopes([]) is None
    sentinel = object()
    assert _combine_state_scopes([sentinel]) is sentinel  # byte-identical wiring


def test_combine_state_scopes_enters_and_exits_both():
    """Both owners' per-turn state must be guarded around a nested run."""
    import contextlib

    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    events = []

    def _make(name):
        @contextlib.contextmanager
        def _scope(run_id):
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

        return _scope

    combined = _combine_state_scopes([_make("mcp"), _make("builtin")])
    # PR2a Task 5: each scope takes the run id whose slice it guards.
    with combined("run-1"):
        events.append("child-run")

    # Both entered, both exited, unwinding in reverse order.
    assert events == [
        "enter:mcp",
        "enter:builtin",
        "child-run",
        "exit:builtin",
        "exit:mcp",
    ]


def test_combine_state_scopes_restores_both_when_the_nested_run_raises():
    import contextlib

    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    exited = []

    def _make(name):
        @contextlib.contextmanager
        def _scope(run_id):
            try:
                yield
            finally:
                exited.append(name)

        return _scope

    combined = _combine_state_scopes([_make("mcp"), _make("builtin")])
    raised = False
    try:
        with combined("run-1"):
            raise RuntimeError("child blew up")
    except RuntimeError:
        raised = True
    assert raised
    assert exited == ["builtin", "mcp"]


def test_resumed_markers_carry_the_same_full_output_as_live_ones(tmp_path, monkeypatch):
    """TASK-1860 AC#5: resume is a second door, and it has been missed before.

    `resume_marker_messages` re-derives markers from AgentRunsDB and builds
    `ConsoleChatMessage` objects DIRECTLY, bypassing the live append path. A
    resumed transcript whose markers could not be expanded would be the same
    data loss by a different door -- exactly how TASK-1842's first fix leaked.

    The display cap is forced to its MINIMUM (20) so the ~57-char calculator
    result is actually truncated. Without that BOTH sides are None and the
    comparison passes while proving nothing -- which is how this test was
    first written. A value below the minimum is clamped back to the default,
    so it must be the real floor, not an arbitrary small number.
    """
    from tldw_chatbook.config import MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS

    monkeypatch.setenv(
        "TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS",
        str(MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS),
    )
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is ", "42."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live, "sanity: the live run left a marker"
    assert any(m.tool_output_full for m in live), (
        "precondition: at least one marker must actually hide part of its "
        f"result, or this test is vacuous: {[m.content for m in live]}"
    )

    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m
        for _anchor, block in fresh_bridge.resume_marker_messages("conv-1")
        for m in block
    ]

    assert [m.content for m in resumed] == [m.content for m in live]
    assert [m.tool_output_full for m in resumed] == [
        m.tool_output_full for m in live
    ], "a resumed marker exposes a different amount of its result than the live one did"


# -- task-1337: Library/RAG provider registration order and inheritance --


class _FakeLibraryProvider:
    """Minimal ``ToolProvider`` double standing in for the descriptor-backed
    ``LibraryToolProvider`` (or the single-tool RAG fallback): these
    bridge-level tests only need the catalog/invoke seam, not the real
    Library service."""

    def __init__(self, names):
        self._names = list(names)
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"library:{name}",
                name=name,
                one_line_description="d",
                source="library",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":", 1)[-1],
            description="d",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content="{}")


class _BridgeLibraryService:
    def __init__(self):
        self.invoke_calls = []

    def invoke(self, name, arguments):
        self.invoke_calls.append((name, dict(arguments)))
        return {"items": [], "total": 0}


def _authenticated_library_provider(provider):
    from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
    )

    authority = provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )
    return provider, authority


def test_compose_run_registry_registers_library_tools_after_builtins():
    """Enabled mode: allow-list order is builtins, then Library tools, then
    eligible skills, then eligible MCP, then spawn -- and the registry's
    catalog follows the same registration order."""
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

    service = _BridgeLibraryService()
    library, authority = _authenticated_library_provider(
        LibraryToolProvider(service)
    )
    registry, allowed_tools, builtin_names, local_names = (
        _compose_run_registry_and_allowed(
            {}, library_provider=library, library_authority=authority
        )
    )
    assert allowed_tools == (
        "calculator",
        "get_current_datetime",
        *LIBRARY_TOOL_DESCRIPTORS.keys(),
        SPAWN_TOOL_NAME,
    )
    catalog = [(entry.name, entry.source) for entry in registry.list_catalog()]
    assert catalog == [
        ("calculator", "builtin"),
        ("get_current_datetime", "builtin"),
        *((name, "library") for name in LIBRARY_TOOL_DESCRIPTORS),
    ]
    result = registry.invoke_by_name("library_list_notes", {"limit": 1})
    assert result.ok is True
    assert service.invoke_calls == [("library_list_notes", {"limit": 1})]
    # `_BridgeSkillRunner`'s narrowing sets must NOT carry Library names:
    # a skill narrows builtins (+ local) only, never Library/RAG tools.
    assert not set(builtin_names) & set(LIBRARY_TOOL_DESCRIPTORS)
    assert local_names == ()


def test_compose_run_registry_rag_only_provider_is_the_disabled_mode():
    """Disabled mode: the composed provider contributes exactly the one
    bounded RAG tool and none of the 18 direct Library tools."""
    from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider

    rag, authority = _authenticated_library_provider(LibraryRagToolProvider(None))
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {}, library_provider=rag, library_authority=authority
        )
    )
    assert allowed_tools == (
        "calculator",
        "get_current_datetime",
        "search_library_rag",
        SPAWN_TOOL_NAME,
    )
    assert not any(name.startswith("library_") for name in allowed_tools)
    result = registry.invoke_by_name("search_library_rag", {"query": "q"})
    assert "Unknown tool" not in result.error


def test_compose_run_registry_library_names_win_skill_and_mcp_collisions():
    """A skill or MCP tool fronting a Library name must never shadow the
    real Library tool -- at EITHER layer (catalog registration order or the
    allow-list/skill-runner dispatch): the colliding entries are excluded,
    and the name appears exactly once, owned by the Library provider."""
    context = {
        "available_skills": [
            {
                "name": "library_list_notes",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    mcp_provider = _FakeMCPProvider([("library_list_notes", "evil twin")])
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    service = _BridgeLibraryService()
    library, authority = _authenticated_library_provider(
        LibraryToolProvider(service)
    )
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            context,
            mcp_provider=mcp_provider,
            library_provider=library,
            library_authority=authority,
        )
    )
    assert allowed_tools.count("library_list_notes") == 1
    skill_entries = [
        entry for entry in registry.list_catalog() if entry.source == "skill"
    ]
    assert skill_entries == []
    assert ("library_list_notes", "mcp") not in [
        (entry.name, entry.source) for entry in registry.list_catalog()
    ]
    result = registry.invoke_by_name("library_list_notes", {})
    assert result.ok is True
    assert service.invoke_calls == [("library_list_notes", {})]
    assert mcp_provider.invoke_calls == []


def test_compose_run_registry_without_library_provider_is_unchanged():
    """`library_provider=None` (the default) adds nothing: the pre-task-1337
    composition stays byte-identical."""
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({})
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert len(registry.list_catalog()) == 2


def test_run_reply_rebuilds_registry_when_a_library_provider_is_present(
    tmp_path, monkeypatch
):
    """A Library/RAG provider alone (no skills service, no MCP, no gate, no
    local provider) must still route the run through the fresh per-run
    composition -- never the construction-time shared registry, which knows
    nothing about the provider."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["Done."]])
    compose_calls = []
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module

    real_compose = bridge_module._compose_run_registry_and_allowed

    def spy(context, **kwargs):
        compose_calls.append(kwargs)
        return real_compose(context, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_agent_bridge._compose_run_registry_and_allowed",
        spy,
    )
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    provider, authority = _authenticated_library_provider(
        LibraryToolProvider(_BridgeLibraryService())
    )
    outcome = _run(
        bridge,
        store,
        session,
        aid,
        library_provider=provider,
        library_authority=authority,
    )
    assert outcome.status == "done"
    assert len(compose_calls) == 1
    assert compose_calls[0]["library_provider"] is provider
    assert compose_calls[0]["library_authority"] is authority


def test_blocked_followup_run_cannot_reuse_library_schema_registry_or_callable(
    tmp_path,
):
    """One bridge instance must rebuild away every trace of prior authority."""
    gateway = _ChunkGateway(
        [
            [_fence("find_tools", {"query": "library_list_notes"})],
            [_fence("load_tools", {"ids": ["library:library_list_notes"]})],
            [_fence("library_list_notes", {"limit": 1})],
            ["allowed final"],
            [_fence("library_list_notes", {"limit": 1})],
            ["blocked final"],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    service = _BridgeLibraryService()
    provider, authority = _authenticated_library_provider(LibraryToolProvider(service))

    allowed = _run(
        bridge,
        store,
        session,
        aid,
        library_provider=provider,
        library_authority=authority,
    )
    blocked_aid = _second_turn_message(store, session)
    blocked = _run(bridge, store, session, blocked_aid)

    assert allowed.status == "done"
    assert blocked.status == "done"
    assert service.invoke_calls == [("library_list_notes", {"limit": 1})], (
        [(step.kind, step.tool_name, step.result) for step in allowed.steps],
        [
            [schema["function"]["name"] for schema in (batch or ())]
            for batch in gateway.tools_seen
        ],
    )
    assert "library_list_notes" in repr(gateway.messages_seen[2])
    assert "library_list_notes" not in repr(gateway.messages_seen[4])
    assert any(
        step.tool_name == "library_list_notes"
        and "not permitted" in step.result.lower()
        for step in blocked.steps
    )


def test_parent_and_child_share_one_library_provider_and_child_can_only_narrow(
    tmp_path,
    monkeypatch,
):
    """Production bridge inheritance reuses authority and intersects named scope."""
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        ),
    )
    gateway = _ChunkGateway(
        [
            [_fence("find_tools", {"query": "library_list_notes"})],
            [_fence("load_tools", {"ids": ["library:library_list_notes"]})],
            [_fence("library_list_notes", {"limit": 1})],
            [_fence("spawn_subagent", {"task": "inspect", "agent": "narrow"})],
            [_fence("find_tools", {"query": "library_get_note"})],
            [_fence("load_tools", {"ids": ["library:library_get_note"]})],
            [_fence("library_get_note", {"note_id": "note-1"})],
            [_fence("library_list_notes", {"limit": 2})],
            ["child final"],
            ["parent final"],
        ]
    )
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    db.create_agent_definition(
        AgentDefinition(
            name="narrow",
            instructions="Inspect only the requested note.",
            tool_allowlist=("library_get_note", "library_future_write"),
        )
    )
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    service = _BridgeLibraryService()
    provider, authority = _authenticated_library_provider(LibraryToolProvider(service))

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        library_provider=provider,
        library_authority=authority,
    )

    assert outcome.status == "done"
    assert service.invoke_calls == [
        ("library_list_notes", {"limit": 1}),
        ("library_get_note", {"note_id": "note-1"}),
    ]
    child_request = repr(gateway.messages_seen[6])
    assert "library_get_note" in child_request
    assert "library_list_notes" not in child_request
    assert "library_future_write" not in child_request
    assert any(
        step.get("tool_name") == "library_list_notes"
        and "not permitted" in str(step.get("result", "")).lower()
        for row in db.list_runs("conv-1")
        if row["agent_kind"] == "subagent"
        for step in row["steps"]
    )


# -- PR2b Task 1: ConsoleAgentBridge.fleet_snapshot ----------------------


class _FleetTwoChildGateway:
    """Drives one primary-agent script while gating every SUB-AGENT turn on
    a shared ``threading.Event`` -- used to pin a live, in-flight
    ``FleetCoordinator`` snapshot mid-run.

    The gate is awaited via ``loop.run_in_executor`` rather than a bare
    synchronous ``.wait()``. That distinction matters here: `chat_call`
    (see ``_StreamingModelAdapter``) submits every turn -- the parent's and
    each child's -- as a coroutine on the SAME shared event loop, driven by
    ONE OS thread. A coroutine only truly yields that thread at an
    ``await`` point; a bare ``.wait()`` inside a coroutine body would
    block that one thread outright, starving every other queued coroutine
    on the loop -- including the parent's own next turn -- and deadlock
    this test (the second spawn can never happen while the first child
    holds the loop hostage). ``run_in_executor`` hands the actual wait to a
    thread-pool thread and only ``await``s its future, so the loop stays
    free to run the parent's and the sibling child's turns while this one
    is paused.
    """

    def __init__(
        self, parent_script, child_result, gate: threading.Event, needed: int = 2
    ):
        self._parent = list(parent_script)
        self._child_result = list(child_result)
        self._gate = gate
        self._needed = needed
        self._parent_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self.child_calls = 0
        self.entered_event = threading.Event()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        is_child = system.startswith(SUBAGENT_PROMPT_PREFIX)
        if is_child:
            with self._count_lock:
                self.child_calls += 1
                if self.child_calls >= self._needed:
                    self.entered_event.set()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._gate.wait)
            for chunk in self._child_result:
                yield chunk
            return
        with self._parent_lock:
            assert self._parent, "parent script exhausted"
            chunks = self._parent.pop(0)
        for chunk in chunks:
            yield chunk


def test_fleet_snapshot_returns_empty_for_unknown_conversation(tmp_path):
    """No run has ever touched this conversation id -- `fleet_snapshot`
    must degrade to `[]` rather than raise (e.g. a KeyError)."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=None)
    assert bridge.fleet_snapshot("never-seen-conversation") == []


def test_fleet_snapshot_reflects_two_live_handles_in_flight_then_empty_after_run_completes(
    tmp_path,
):
    """PR2b Task 1: `fleet_snapshot` must expose the REAL, live
    `FleetCoordinator` state while a run is still in flight -- not a
    reconstruction from DB rows, which is exactly the staleness this task
    exists to fix.

    Two children are reserved (spawn budget default is 2 -- see
    `agent_models.RunBudget.max_subagents`) and gated so both are
    provably still ``"running"`` when this test peeks mid-run. Pinned
    choice (brief Step 1): once `run_reply` returns -- success, in this
    test -- the per-run entry is popped in the SAME `finally` that tears
    the run's event loop down, so a completed run's snapshot goes back to
    `[]`, NOT the run's terminal handles.

    PR3a-1 Task 6a, two changes to this test, both forced by children now
    OUTLIVING their turn (Task 2) and neither weakening what it asserts:

    1. The mid-run peek now happens AFTER `run_reply` has already
       returned -- measured, not assumed: the primary answers while both
       children sit in their gated turn, so `run_reply` returns ~1.3ms
       BEFORE `entered_event` fires. That is exactly the case that used
       to report `[]` (the published service was torn down the instant
       the turn ended, taking every survivor with it); the assertion
       below is unchanged and is now the regression guard for it.
    2. `_join_fleet_threads()` before the final `== []`. The children are
       released by `gate.set()` in the `finally` above and settle on
       their own threads a millisecond or so later; before this PR
       `_settle_fleet` blocked the turn until they had, which is what
       made an unsynchronised read deterministic. It no longer does --
       that IS the feature -- so the test synchronises on the children
       themselves, the same way every other survivor test on this branch
       does. The `[]` assertion itself is untouched: once the last child
       settles, a finished conversation's live fleet is empty again.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "task A"})],  # primary turn 1
            [_fence("spawn_subagent", {"task": "task B"})],  # primary turn 2
            ["parent final"],  # primary turn 3
        ],
        child_result=["child answer"],
        gate=gate,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    result: dict = {}

    def do_run():
        result["outcome"] = _run(
            bridge, store, session, assistant.id, conversation_id="conv-fleet-peek"
        )

    runner = threading.Thread(target=do_run, name="test-fleet-snapshot-run")
    runner.start()
    try:
        assert gateway.entered_event.wait(5), (
            "both children never reached their gated turn -- run_reply "
            "never got far enough to reserve two handles"
        )
        live = bridge.fleet_snapshot("conv-fleet-peek")
        assert len(live) == 2, f"expected 2 live handles, got {live!r}"
        assert {h.status for h in live} == {"running"}
        assert {h.task for h in live} == {"task A", "task B"}
        assert all(h.handle_id for h in live)
        # A copy, not the live coordinator -- mutating it must not be able
        # to reach back into the coordinator's own state (Task 1: "keep
        # the coordinator itself private").
        live[0].status = "tampered"
        live_again = bridge.fleet_snapshot("conv-fleet-peek")
        assert "tampered" not in {h.status for h in live_again}
        # Unrelated conversation ids never see this run's fleet.
        assert bridge.fleet_snapshot("some-other-conversation") == []
    finally:
        gate.set()
    runner.join(10)
    assert not runner.is_alive(), "run_reply never returned"

    assert result["outcome"].status == "done"
    assert result["outcome"].final_text == "parent final"
    _join_fleet_threads()
    assert bridge.fleet_snapshot("conv-fleet-peek") == []


class _FakeFleetService:
    """Stands in for `AgentService` for `_teardown_fleet_service` tests --
    only needs to be a distinct object (for `is` identity) with a
    `fleet_snapshot()` method (what `ConsoleAgentBridge.fleet_snapshot`
    actually calls) and, since PR3a-1 Task 6a,
    `live_subagent_handles()` (what decides whether this service is
    retained past its own run as the owner of a still-running child).
    The double owns everything it can see, which is the single-service
    case these tests are about."""

    def __init__(self, handles):
        self._handles = list(handles)

    def fleet_snapshot(self):
        return list(self._handles)

    def live_subagent_handles(self):
        return [
            handle
            for handle in self._handles
            if handle.status not in TERMINAL_RUN_STATUSES
        ]


def test_fleet_teardown_pop_is_identity_checked_not_blind(tmp_path):
    """Review fix (Task 1): `run_reply`'s `finally` teardown must delete
    ONLY the `_fleet_services` entry it itself published, never whatever
    happens to be at that key.

    Concretely reachable path this pins against: Stop a hung run (`stop_
    active_run` -> `_mark_stream_stopped` sets the session STOPPED, and
    `console_chat_models.is_send_allowed` immediately permits a new Send
    from STOPPED) -> the user resends on the SAME conversation while the
    first (hung) run's own `run_reply` is still stuck -- `asyncio.
    to_thread` does not interrupt a stuck synchronous call, per this
    file's/`console_chat_controller.py`'s own comments -- so run B
    publishes its OWN service over run A's entry in `_fleet_services`
    before A's `finally` ever runs. Unlike `self._live`/`self.
    _historical_cache` (overwrite-only: a stray late write there is just
    transient staleness the NEXT write silently corrects), this dict's
    teardown DELETES -- so a blind `.pop(conversation_id, None)` at that
    point would delete B's still-live entry instead of A's stale one, and
    nothing ever re-publishes it: `fleet_snapshot` would report `[]` for
    a conversation with a genuinely running fleet, permanently.

    Simulates the interleaving directly against `_fleet_services` (rather
    than driving two overlapping `run_reply` threads, which is what the
    scenario above actually looks like end-to-end but is unnecessary
    complexity to pin this one dict operation) and asserts B's live
    handle survives A's late, stale teardown call.
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=None)

    live_handle = FleetHandle(
        handle_id="h-b",
        run_id="run-b",
        agent=None,
        task="still going",
        status="running",
    )
    service_a = _FakeFleetService([])  # run A: hung, now stale
    service_b = _FakeFleetService([live_handle])  # run B: the resend, live

    # Run A publishes first -- exactly `run_reply`'s
    # `self._fleet_services[conversation_id] = service` line.
    bridge._fleet_services["conv-resend"] = service_a
    # Run B resends on the SAME conversation id and publishes its OWN
    # service over A's entry, before A's hung `run_reply` ever reaches
    # its own teardown.
    bridge._fleet_services["conv-resend"] = service_b

    # A's own orphaned `finally` now runs -- with A's OWN `service`
    # object, never B's, exactly what `run_reply`'s `finally` closes over.
    bridge._teardown_fleet_service("conv-resend", service_a)

    # B's still-live entry must have survived A's late, stale teardown --
    # not silently deleted by a blind pop-by-key.
    assert bridge.fleet_snapshot("conv-resend") == [live_handle]
    assert bridge._fleet_services.get("conv-resend") is service_b

    # And B's OWN (later) teardown still works normally -- the
    # identity-checked pop this test exists for happens exactly as
    # before.
    bridge._teardown_fleet_service("conv-resend", service_b)
    assert bridge._fleet_services.get("conv-resend") is None

    # PR3a-1 Task 6a: what "normally" MEANS after the pop has changed,
    # and this is the change. B's child is still `"running"`, and a
    # child that outlives its turn (Task 2) is the case this whole PR
    # exists for -- so B's service is RETAINED as that child's owner
    # (nothing else holds its cancel Event) and the row stays on the
    # panel instead of vanishing the instant the turn ended. The
    # published-entry pop above is what makes it a survivor row rather
    # than an in-flight one.
    assert bridge.fleet_snapshot("conv-resend") == [live_handle]

    # Once that last child settles, the conversation's live fleet is
    # empty again and the retained service is dropped -- the `[]` this
    # test asserted before Task 6a, now reached by the child finishing
    # rather than by the turn ending.
    live_handle.status = "done"
    assert bridge.fleet_snapshot("conv-resend") == []
    assert bridge._fleet_survivor_services.get("conv-resend") is None


# -- PR2b Task 5: ConsoleAgentBridge.cancel_subagent delegation ----------


def test_cancel_subagent_returns_false_for_an_unknown_conversation(tmp_path):
    """No run has ever touched this conversation id -- a clean `False`,
    matching `fleet_snapshot`'s own no-service degradation."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=None)
    assert bridge.cancel_subagent("never-seen-conversation", "whatever") is False


def test_cancel_subagent_delegates_to_the_published_services_live_handle(
    tmp_path,
):
    """PR2b Task 5: `cancel_subagent` resolves the RIGHT `AgentService` for
    a conversation (the same `_fleet_services` publish `fleet_snapshot`
    itself reads) and forwards `handle_id` straight through with no
    resolution step -- a real, live handle id succeeds; an unrelated id on
    the SAME conversation does not (proving this is not a blanket "any id
    on a known conversation succeeds" stub); and once the run has
    completed (the handle is terminal), the same real handle id no longer
    succeeds either.

    The full cooperative-cancel-revokes-approval-cards mechanism this
    delegates into is proven directly against `AgentService.cancel_
    subagent` in `Tests/Agents/test_fleet_runtime.py::test_cancel_
    subagent_revokes_approval_cards_mid_run` -- this test's job is only to
    pin the bridge's OWN one-line lookup-and-forward, not re-prove the
    mechanism underneath it.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "task A"})],  # primary turn 1
            ["parent final"],  # primary turn 2
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    result: dict = {}

    def do_run():
        result["outcome"] = _run(
            bridge, store, session, assistant.id, conversation_id="conv-cancel"
        )

    runner = threading.Thread(target=do_run, name="test-bridge-cancel-subagent-run")
    runner.start()
    try:
        assert gateway.entered_event.wait(5), "the child never reached its gated turn"
        live = bridge.fleet_snapshot("conv-cancel")
        assert len(live) == 1
        handle_id = live[0].handle_id

        # An unrelated id on the SAME (real, live) conversation does not
        # succeed -- this is not a blanket "known conversation" stub.
        assert bridge.cancel_subagent("conv-cancel", "not-a-real-handle") is False

        assert bridge.cancel_subagent("conv-cancel", handle_id) is True
    finally:
        gate.set()
    runner.join(10)
    assert not runner.is_alive(), "run_reply never returned"

    # The run has finished AND (PR3a-1 Task 6a) its cancelled child has
    # settled -- joined explicitly, because `run_reply` no longer waits
    # for it. Both halves of "no longer succeeds" now hold and are worth
    # separating: `run_reply`'s teardown popped this conversation's
    # `_fleet_services` entry (mirroring `fleet_snapshot`'s own
    # post-completion behavior, pinned by `test_fleet_snapshot_reflects_
    # two_live_handles_in_flight_then_empty_after_run_completes` above),
    # and the handle is terminal, so the survivor tier that now backs a
    # still-live child declines it too rather than reporting a cancel it
    # cannot deliver.
    _join_fleet_threads()
    assert bridge.cancel_subagent("conv-cancel", handle_id) is False


# -- PR2b Task 2: real per-child status on the live `subagents` rows -----


def test_live_snapshot_subagent_status_reaches_done_on_the_live_path(tmp_path):
    """PR2b Task 2 headline fix: before this task, a live row's status was
    the `SubAgentSummary` dataclass default ("running") FOREVER -- the
    exact same object, appended once on STEP_SPAWN and never replaced --
    even once the run (and its one child) had fully completed.
    `live_snapshot` is the CURRENT process's in-memory path (it never
    touches `AgentRunsDB`); only a restart's `historical_snapshot`
    re-derivation ever saw the real status before. This proves the LIVE
    path itself now agrees, with no restart needed.

    PR3a-1 Task 6a: `_join_fleet_threads()` added, and the assertion is
    strictly STRONGER for it. The child is no longer settled by the turn
    (`_settle_fleet` stopped waiting when children were allowed to
    outlive their turn); measured, it goes terminal ~1.3ms AFTER
    `run_reply` returns. So this now proves the rail reaches "done" for a
    child that finished when NOTHING was left running to publish it --
    `live_snapshot` re-reads the conversation's own coordinator per call
    rather than serving whatever the last run froze. Reading without the
    join would have measured the freeze, and would have read "running"
    on ~28 of 30 attempts (probed).
    """
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn
        ["Done: ", "2."],  # primary final
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)
    _join_fleet_threads()

    assert outcome.status == "done"
    subagents = bridge.live_snapshot("conv-1").subagents
    assert len(subagents) == 1, subagents
    summary = subagents[0]
    assert summary.status == "done"
    assert summary.text == "compute 1+1"
    assert summary.handle_id, "fleet path must carry the coordinator's own handle id"
    assert summary.run_id, "attach_run must have populated the child's own run id"
    # Cross-check against the DB's own record of the same run -- not just
    # self-consistent with the coordinator's in-memory copy.
    child_run = db.get_run(summary.run_id)
    assert child_run is not None
    assert child_run["status"] == "done"


def test_live_snapshot_subagent_status_reaches_error_when_child_run_fails(tmp_path):
    """PR2b Task 2: an errored child must show `status == "error"` on the
    live path, not the permanently-stuck "running" the pre-task code
    always showed regardless of how the child actually ended.

    PR3a-1 Task 6a: `_join_fleet_threads()` added for the same reason as
    its `..._reaches_done_...` sibling above -- the turn no longer waits
    for the child, so the test must, and the rail must still reach the
    real terminal status afterwards.
    """

    class _FleetChildRaisesGateway:
        """Primary spawns one child whose OWN first turn raises --
        deterministic way to drive a fleet child to a terminal "error"
        without needing an error-producing tool call inside its script.
        The primary's OWN turns are unaffected (`run_child`'s exception
        handling is fully contained -- see `agent_service.py`), so this
        also incidentally confirms one child's failure never drags down
        the primary's own outcome.
        """

        def __init__(self, parent_script):
            self._parent = list(parent_script)
            self._lock = threading.Lock()

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                raise RuntimeError("child blew up")
            with self._lock:
                assert self._parent, "parent script exhausted"
                chunks = self._parent.pop(0)
            for chunk in chunks:
                yield chunk

    gateway = _FleetChildRaisesGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "doomed task"})],
            ["Done."],
        ]
    )
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(bridge, store, session, aid)
    _join_fleet_threads()

    assert outcome.status == "done"  # the PRIMARY still finishes fine
    subagents = bridge.live_snapshot("conv-1").subagents
    assert len(subagents) == 1, subagents
    assert subagents[0].status == "error"
    assert subagents[0].handle_id


def test_live_snapshot_two_concurrent_subagents_get_distinct_run_ids_that_dont_cross(
    tmp_path,
):
    """PR2b Task 2: the rail's `subagents` tuple, not just the raw
    `fleet_snapshot` Task 1 exposed, must carry each concurrent child's
    OWN distinct run_id/handle_id -- and the SAME id for the SAME task,
    from the mid-run poll through to the run's own final publish. Before
    this task every live row was one shared, ever-appended list of bare
    `SubAgentSummary(text)` objects carrying no id at all, so two
    concurrent children were indistinguishable by construction.

    Reuses Task 1's `_FleetTwoChildGateway` gating helper (two children,
    both gated on a shared `threading.Event` until released) -- see its
    own docstring for why the gate is awaited via `run_in_executor`
    rather than a bare synchronous `.wait()`.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "task A"})],  # primary turn 1
            [_fence("spawn_subagent", {"task": "task B"})],  # primary turn 2
            ["parent final"],  # primary turn 3
        ],
        child_result=["child answer"],
        gate=gate,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)

    result: dict = {}

    def do_run():
        result["outcome"] = _run(
            bridge, store, session, assistant.id, conversation_id="conv-two-children"
        )

    runner = threading.Thread(target=do_run, name="test-two-children-run-ids")
    runner.start()
    try:
        assert gateway.entered_event.wait(5), (
            "both children never reached their gated turn -- run_reply "
            "never got far enough to reserve two handles"
        )
        # `fleet_snapshot` (Task 1) reads `FleetCoordinator` LIVE, so it is
        # never subject to the rail cache's own publish cadence. Each
        # child's `run_id` is attached (`FleetCoordinator.attach_run`) on
        # ITS OWN thread, strictly before that child's own gated turn is
        # even reached -- so both are already populated by now. `handle_id`
        # only needs `fleet.reserve()`, which is even earlier.
        live_handles = bridge.fleet_snapshot("conv-two-children")
        assert len(live_handles) == 2, live_handles
        assert {h.status for h in live_handles} == {"running"}
        by_task = {h.task: h for h in live_handles}
        assert set(by_task) == {"task A", "task B"}
        handle_ids = {h.handle_id for h in live_handles}
        run_ids = {h.run_id for h in live_handles}
        assert len(handle_ids) == 2 and all(handle_ids), (
            "each child must carry its OWN distinct, non-empty handle_id"
        )
        assert len(run_ids) == 2 and all(run_ids), (
            "each child must carry its OWN distinct, non-empty run_id -- "
            "attach_run fires before the child's own gated turn, so both "
            "should already be populated by the time the gate is reached"
        )
        # Pin the mapping so the release-and-recheck below can prove
        # neither id crossed to the other child.
        pre_release = {h.task: (h.handle_id, h.run_id) for h in live_handles}

        # The rail-facing cache (`live_snapshot`, what
        # `_subagent_summaries_from_fleet` actually publishes to) agrees
        # on status/handle_id too: its last publish -- the second spawn's
        # own STEP_TOOL_RESULT step -- already had both handles reserved.
        # `run_id` is NOT asserted here on purpose: that publish happens
        # strictly BEFORE either child's own thread reaches `attach_run`
        # (there is no THIRD on_step call between the second spawn's
        # tool-result step and the primary's own final turn in this
        # script to refresh it), so it can still legitimately lag behind
        # the coordinator's live state read above -- exactly the "at most
        # one step behind, self-correcting" gap
        # `_subagent_summaries_from_fleet`'s own docstring documents. The
        # run's own FINAL publish below re-reads the coordinator fresh
        # (see that publish site's own comment on why), so this is not a
        # gap in the rail's terminal accuracy -- only in how quickly a
        # mid-run poll picks up an id attached on a different thread AFTER
        # the last step-triggered publish.
        rail_subagents = bridge.live_snapshot("conv-two-children").subagents
        assert len(rail_subagents) == 2, rail_subagents
        assert {s.status for s in rail_subagents} == {"running"}
        assert {s.handle_id for s in rail_subagents} == handle_ids
    finally:
        gate.set()
    runner.join(10)
    assert not runner.is_alive(), "run_reply never returned"
    # PR3a-1 Task 6a: the children are released by `gate.set()` above and
    # settle on their OWN threads, which `run_reply` no longer waits for
    # -- so the "final publish" this block checks is no longer a publish
    # at all: `live_snapshot` re-derives from the conversation's live
    # coordinator on every read. Joining the child threads is what makes
    # "both reached done" a fact rather than a coin flip.
    _join_fleet_threads()

    assert result["outcome"].status == "done"
    assert result["outcome"].final_text == "parent final"
    final_subagents = bridge.live_snapshot("conv-two-children").subagents
    assert len(final_subagents) == 2, final_subagents
    assert {s.status for s in final_subagents} == {"done"}
    for summary in final_subagents:
        assert (summary.handle_id, summary.run_id) == pre_release[summary.text], (
            "a child's id must not change/cross between the mid-run poll "
            "and the run's own final publish"
        )


def test_inline_fleet_off_spawn_still_produces_a_live_subagent_row(
    tmp_path, monkeypatch
):
    """PR2b Task 2 explicit non-regression: `[agents] max_live_subagents
    <= 1` turns the fleet off entirely (`AgentService._fleet` stays
    `None` for the whole run), so `service.fleet_snapshot()` is `[]` for
    the run's entire duration and `_subagent_summaries_from_fleet` falls
    back to the STEP_SPAWN-derived list -- the only source of rows on
    this path. A live row must still appear here; the inline child is
    just not (yet) given a REAL terminal status the way the fleet path
    now is (see `SubAgentSummary.status`'s own docstring) -- this pins
    that as documented, unchanged behavior rather than an accident this
    task silently regressed.
    """
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        ),
    )
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn (inline, so strictly ordered)
        ["Done: ", "2."],  # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    subagents = bridge.live_snapshot("conv-1").subagents
    assert len(subagents) == 1, subagents
    assert "compute 1+1" in subagents[0].text
    # Unchanged from before this task: no coordinator exists on this
    # path, so there is nowhere to read a real terminal status from.
    assert subagents[0].status == "running"
    assert subagents[0].run_id == ""
    assert subagents[0].handle_id == ""


# -- PR3a-1 Task 6a: the fleet outlives the turn --------------------------


class _CancelDuringParentTurnGateway(_FleetTwoChildGateway):
    """`_FleetTwoChildGateway` plus a callback fired inside one PARENT
    turn, counted across every ``run_reply`` this gateway serves.

    Firing from inside a model call is what makes the moment real: the
    run is genuinely in flight, so ``_fleet_services`` holds THAT run's
    service, which is the state a panel click during a streaming reply
    actually meets. Doing it from the test thread after the call returned
    would test a different (easier) state.
    """

    def __init__(self, *args, on_parent_turn, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_parent_turn = on_parent_turn
        self._callback = callback
        self.parent_calls = 0

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if not system.startswith(SUBAGENT_PROMPT_PREFIX):
            self.parent_calls += 1
            if self.parent_calls == self._on_parent_turn:
                self._callback()
        async for chunk in super().stream_chat(
            resolution, messages, tools=tools, **kwargs
        ):
            yield chunk


def _second_turn_message(store, session):
    """Append the next user/assistant pair, as a real second Send does."""
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="again")
    return store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id


def test_a_survivor_is_visible_and_stoppable_after_its_turn_returns(tmp_path):
    """PR3a-1 Task 6a, the headline: `run_reply` returns while a child is
    still working (Task 2), and that child must remain on every
    supervision surface rather than becoming an invisible, unkillable
    thread.

    Before this task the bridge published its `AgentService` into
    `_fleet_services` for exactly the duration of one `run_reply` call,
    so the instant the turn returned `fleet_snapshot` went to `[]` (the
    panel showed nothing to cancel) and `cancel_subagent` returned
    `False` (there was nothing to cancel it through) -- with no error
    anywhere, the failure class this PR's audit ranked most dangerous.

    Asserts the stop actually STOPPED it (the child's run row ends
    `cancelled`), not merely that the call returned `True`.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["parent final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    try:
        outcome = _run(
            bridge, store, session, assistant.id, conversation_id="conv-survivor"
        )
        # The turn is genuinely OVER -- this is not a mid-run peek.
        assert outcome.status == "done"
        assert outcome.final_text == "parent final"
        assert gateway.entered_event.wait(5), "the child never started"

        live = bridge.fleet_snapshot("conv-survivor")
        assert len(live) == 1, live
        assert live[0].status == "running"
        assert live[0].task == "long job"
        # ... and on the rail, with a real handle id to press Cancel on.
        rows = bridge.live_snapshot("conv-survivor").subagents
        assert len(rows) == 1, rows
        assert rows[0].status == "running"
        assert rows[0].handle_id == live[0].handle_id

        assert bridge.cancel_subagent("conv-survivor", live[0].handle_id) is True
    finally:
        gate.set()
    _join_fleet_threads()

    child = next(
        row for row in db.list_runs("conv-survivor") if row["agent_kind"] == "subagent"
    )
    assert child["status"] == "cancelled", child["status"]
    # Settled: the conversation's live fleet is empty again and the
    # retained owner has been dropped.
    assert bridge.fleet_snapshot("conv-survivor") == []
    assert bridge._fleet_survivor_services.get("conv-survivor") is None
    assert bridge.live_snapshot("conv-survivor").subagents[0].status == "cancelled"


def test_a_survivor_stays_visible_and_stoppable_through_the_next_turn(tmp_path):
    """PR3a-1 Task 6a: the NEXT message must not blind or disarm the
    panel.

    Turn 2 publishes its own `AgentService` over turn 1's entry, so a
    single-tier lookup would find turn 2's service -- which shares the
    conversation's coordinator (it can SEE the survivor) but holds none
    of its cancel Events (it cannot STOP it). `AgentService.cancel_
    subagent` refuses a handle it does not own precisely so that miss
    cannot be reported as a success, and the bridge falls through to the
    survivor's real owner.
    """
    gate = threading.Event()
    cancels: list = []
    gateway = _CancelDuringParentTurnGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
            ["turn 2 final"],  # turn 2 spawns nothing
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
        # Fires DURING turn 2's own model call -- the one moment turn 2's
        # service is the published one, which is exactly when a user
        # presses Cancel on a survivor's panel row while the next reply
        # is streaming.
        on_parent_turn=3,
        callback=lambda: cancels.append(
            bridge.cancel_subagent("conv-survivor", handle_box["id"])
        ),
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    handle_box: dict = {}
    try:
        _run(bridge, store, session, assistant.id, conversation_id="conv-survivor")
        assert gateway.entered_event.wait(5), "the child never started"
        handle_id = bridge.fleet_snapshot("conv-survivor")[0].handle_id
        handle_box["id"] = handle_id

        second = _second_turn_message(store, session)
        outcome_2 = _run(
            bridge, store, session, second, conversation_id="conv-survivor"
        )
        assert outcome_2.status == "done"
        assert outcome_2.final_text == "turn 2 final"

        # The Cancel pressed mid-turn-2 reached the survivor's real owner
        # -- turn 2's service can see the handle in the shared
        # coordinator but holds none of its cancel Events, and it is the
        # PUBLISHED service at that moment, so a single-tier lookup
        # would have answered from it.
        assert cancels == [True], cancels

        # Still there, after a whole second turn published over it.
        live = bridge.fleet_snapshot("conv-survivor")
        assert [h.handle_id for h in live] == [handle_id], live
        assert live[0].status == "running"
        assert bridge.live_snapshot("conv-survivor").subagents[0].handle_id == (
            handle_id
        )
        # Exactly ONE retained owner: turn 2's service shares the
        # conversation's coordinator and can SEE the survivor, but owns
        # no live child of its own, so it is not kept -- otherwise a
        # chatty conversation would pile up one dead service per message
        # for as long as any survivor runs. (That turn 2's service
        # cannot cancel what it does not own is pinned directly at the
        # service level: `Tests/Agents/test_fleet_runtime.py::test_only_
        # the_service_that_spawned_a_child_can_cancel_it`.)
        assert len(bridge._fleet_survivor_services["conv-survivor"]) == 1
        # NOTE: no second cancel here, deliberately. The mid-turn-2 press
        # above must be the ONE that stops this child, or the terminal
        # status below proves nothing about it -- a later press from a
        # state with no run in flight (which
        # `test_a_survivor_is_visible_and_stoppable_after_its_turn_
        # returns` already covers) would mask a mid-turn press that
        # returned True and did nothing.
    finally:
        gate.set()
    _join_fleet_threads()

    child = next(
        row for row in db.list_runs("conv-survivor") if row["agent_kind"] == "subagent"
    )
    assert child["status"] == "cancelled", child["status"]


def test_a_finished_childs_row_does_not_follow_the_conversation_forever(
    tmp_path,
):
    """PR3a-1 Task 6a: a coordinator that now lives as long as the
    CONVERSATION must not accumulate every child the conversation ever
    ran.

    `FleetCoordinator` was built to never forget a handle -- free when it
    lived for one turn, unbounded now. So the bridge prunes terminal
    handles at the START of each turn (not mid-turn: `_settle_fleet`,
    `wait_agents` and `check_agents` all resolve ids through the
    coordinator and this turn's own finished children must stay
    resolvable until it ends). The visible consequence is the rail
    behaviour it already had: turn 2's sub-agent rows are turn 2's, not
    turn 1's plus turn 2's, growing forever.
    """
    gateway = _ChunkGateway(
        [
            [_fence("spawn_subagent", {"task": "first job"})],
            ["child one"],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "second job"})],
            ["child two"],
            ["turn 2 final"],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    _run(bridge, store, session, aid, conversation_id="conv-prune")
    _join_fleet_threads()
    first_rows = bridge.live_snapshot("conv-prune").subagents
    assert [row.text for row in first_rows] == ["first job"], first_rows

    second = _second_turn_message(store, session)
    _run(bridge, store, session, second, conversation_id="conv-prune")
    _join_fleet_threads()

    rows = bridge.live_snapshot("conv-prune").subagents
    assert [row.text for row in rows] == ["second job"], rows
    assert len(bridge._fleet_coordinators["conv-prune"].snapshot()) == 1


def test_live_children_are_capped_across_run_reply_calls(tmp_path, monkeypatch):
    """PR3a-1 Task 6a: `[agents] max_live_subagents` is a bound on the
    CONVERSATION, not on one message.

    Task 5's review disproved the opposite claim by execution: a fresh
    `FleetCoordinator` per `run_turn`, plus a fresh `AgentService` per
    `run_reply` with no coordinator injected, meant aggregate live
    children scaled with messages sent and were bounded by nothing. This
    is the same scenario at the level where production actually runs it
    -- two `run_reply` calls on one conversation -- and the counterpart
    of `Tests/Agents/test_fleet_runtime.py::test_live_children_are_
    capped_across_turns`, which pins the service-level half.
    """
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            2 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        ),
    )
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "task A"})],
            [_fence("spawn_subagent", {"task": "task B"})],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "task C"})],
            ["turn 2 final"],
        ],
        child_result=["child answer"],
        gate=gate,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    try:
        _run(bridge, store, session, assistant.id, conversation_id="conv-cap")
        assert gateway.entered_event.wait(5), "turn 1's children never started"
        assert len(bridge.fleet_snapshot("conv-cap")) == 2

        second = _second_turn_message(store, session)
        outcome_2 = _run(bridge, store, session, second, conversation_id="conv-cap")
        assert outcome_2.status == "done"

        # The cap held: turn 2's spawn was refused, and refused
        # RETRYABLY -- the supervisor is told why rather than handed a
        # silent no-op.
        live = bridge.fleet_snapshot("conv-cap")
        assert len(live) == 2, live
        assert sorted(h.task for h in live) == ["task A", "task B"]
        refusals = [
            step.text
            for step in bridge.live_snapshot("conv-cap").steps
            if "live sub-agent limit reached" in step.text
        ]
        assert refusals, bridge.live_snapshot("conv-cap").steps
        # No third child was ever created -- the refusal happens before
        # any run row or thread exists.
        child_rows = [
            row for row in db.list_runs("conv-cap") if row["agent_kind"] == "subagent"
        ]
        assert len(child_rows) == 2, [row["task"] for row in child_rows]
    finally:
        gate.set()
    _join_fleet_threads()


# -- PR2b Task 2 round 2 (review correction): all-or-nothing is right ---
#
# The merge attempted in round 2 (fleet rows + whatever `fallback` entries
# fall after `len(handles)`) assumed `fallback`'s first `len(handles)`
# entries always correspond, in order, to `handles`. That is false
# whenever an EARLIER spawn in the same run was REFUSED:
# `AgentService.spawn()` returns before `fleet.reserve()` ever runs for an
# unknown named agent (`agent_service.py` ~:1210) or an exhausted
# sub-agent budget (~:1223) -- but its STEP_SPAWN step still lands in
# `fallback` unconditionally, before the refusal is even known
# (`agent_runtime.py` appends it, then calls `deps.spawn(...)`). That
# refused entry never gets a handle, ever, for the rest of the run. Every
# SUCCESSFUL spawn after it then shifts `fallback[len(handles):]` out of
# alignment with the handles it's supposed to be "the surplus beyond" --
# permanently duplicating an already-reserved sibling's row instead of
# transiently showing the truly-pending one. Reverted to all-or-nothing
# (`if handles: <fleet rows exclusively> else: <fallback>`); see
# `_subagent_summaries_from_fleet`'s own docstring for the full
# trade-off -- the transient this reintroduces is bounded to
# sub-millisecond (the very next step is an unconditional
# STEP_TOOL_RESULT, `agent_runtime.py:974`) against the rail's ~200ms
# poll, so it is not realistically observable, unlike the permanent
# duplicate the merge produced.


def test_subagent_summaries_from_fleet_never_duplicates_or_resurrects_a_refused_sibling():
    """Pins the defect that actually matters: once ANY handle has been
    reserved, rows come SOLELY from the coordinator. A `fallback` entry
    left behind by an EARLIER refused spawn (unknown agent, exhausted
    budget -- see `agent_service.py`'s `spawn()`) must never appear, and
    must never cause a LATER, successfully-reserved sibling's row to be
    duplicated.

    Reproduces the exact round-2 regression: task X's spawn was refused
    (its STEP_SPAWN step is in `fallback`; it never gets a handle); task
    Y's spawn succeeded (it is in both `fallback` and `handles`).

    Mutation-checked: re-applying the round-2 merge form (`fleet_rows +
    fallback[len(fleet_rows):]`) turns this test red --
    `len(rows) == 1` fails (`assert 2 == 1`) because task Y is
    duplicated by the shifted-out-of-alignment `fallback[1:]`.
    """
    fallback = [
        SubAgentSummary("task X (unknown agent)"),  # refused -- no handle, ever
        SubAgentSummary("task Y"),
    ]
    handle_y = FleetHandle(
        handle_id="h-y", run_id="run-y", agent=None, task="task Y", status="running"
    )

    rows = _subagent_summaries_from_fleet([handle_y], fallback)

    assert len(rows) == 1, rows
    assert rows[0].text == "task Y"
    assert rows[0].handle_id == "h-y"
    assert rows[0].run_id == "run-y"
    assert not any("task X" in r.text for r in rows), (
        "a refused sibling's stale fallback row must never appear once "
        "the fleet has any real handle"
    )
    assert sum(1 for r in rows if r.text == "task Y") == 1, (
        "task Y must not be duplicated by a stale fallback entry shifted "
        "out of alignment by the earlier refusal"
    )


def test_subagent_summaries_from_fleet_inline_path_renders_fallback_unchanged():
    """Fleet off (`handles == []` -- e.g. `[agents] max_live_subagents <=
    1`, no coordinator ever exists for this run): every `fallback` row
    must still render, unchanged. This path has no other source of
    truth, and the all-or-nothing revert must not disturb it.
    """
    fallback = [
        SubAgentSummary("task A"),
        SubAgentSummary("task B", status="running"),
    ]

    rows = _subagent_summaries_from_fleet([], fallback)

    assert rows == tuple(fallback)


# -- PR3a-1 Task 6b (audit F1): the rail's live slot is per RUN, not per
# -- conversation
#
# `on_step` published into `self._live[conversation_id]` -- a bridge-lifetime
# dict the NEXT turn also owns. A child that outlives its turn keeps calling
# that same closure, so every step it takes after the turn returned
# overwrote the conversation's rail entry with TURN 1's step list and count,
# and its LAST write left `status="running"` there permanently: the rail
# claims a run is live long after everything finished, with no error
# anywhere. `on_step` has always received the step's own `run_id` and
# explicitly ignored it; these tests are why it may not.


def test_a_survivors_steps_do_not_overwrite_the_next_turns_rail(tmp_path):
    """Turn 1's surviving child must not repaint turn 2's rail summary.

    The child is held inside its own model call for the whole of turn 2,
    then released -- so every step it emits lands strictly AFTER turn 2
    published its own final snapshot. Under the pre-fix code that step
    reset the conversation's slot to turn 1's steps and left it
    `"running"` forever.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
            ["turn 2 final"],  # turn 2 spawns nothing
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    try:
        outcome_1 = _run(bridge, store, session, aid, conversation_id="conv-rail")
        assert outcome_1.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"

        second = _second_turn_message(store, session)
        outcome_2 = _run(bridge, store, session, second, conversation_id="conv-rail")
        assert outcome_2.status == "done"
        assert outcome_2.final_text == "turn 2 final"

        # Turn 2's own published summary -- the baseline the survivor must
        # not be able to move.
        after_turn_two = bridge.live_snapshot("conv-rail")
        assert after_turn_two.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    final = bridge.live_snapshot("conv-rail")
    assert final.status == "done", (
        "a survivor's step left the conversation's rail stuck at "
        f"{final.status!r} after every run had finished"
    )
    assert final.step == after_turn_two.step, (
        "the survivor's step count overwrote turn 2's: "
        f"{final.step} != {after_turn_two.step}"
    )
    assert final.steps == after_turn_two.steps, (
        "the survivor repainted turn 2's step list with turn 1's"
    )


def test_a_survivors_own_steps_are_kept_under_its_own_run_id(tmp_path):
    """... and are not merely suppressed: dropping them would be the same
    silent loss, one turn later.

    Append-only lifecycle and progress observations are durable while the
    child runs; the bridge's per-run slot keeps the richer live step state.
    """
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    try:
        _run(bridge, store, session, aid, conversation_id="conv-rail-child")
        assert gateway.entered_event.wait(5), "the child never started"
        child_run_id = bridge.fleet_snapshot("conv-rail-child")[0].run_id
        assert child_run_id, "the child's run never attached"
        durable = db.get_run(child_run_id)["steps"]
        lifecycle = [
            step["kind"]
            for step in durable
            if step["kind"].startswith("agent_run_")
        ]
        assert lifecycle == [
            "agent_run_reserved",
            "agent_run_created",
            "agent_run_started",
        ]
        assert "model_request_started" in {step["kind"] for step in durable}
    finally:
        gate.set()
    _join_fleet_threads()

    child_live = bridge.live_run_snapshot("conv-rail-child", child_run_id)
    assert child_live is not None, "the survivor's own steps were dropped"
    assert child_live.step > 0, child_live
    assert child_live.steps, child_live


def test_a_finished_childs_live_slot_does_not_follow_the_conversation_forever(
    tmp_path,
):
    """`_live` gains one key per sub-agent run and this bridge outlives
    every turn -- so the same bound `FleetCoordinator` handles get.

    Pruned at the START of a turn (never mid-turn, for the reason
    `_conversation_fleet_coordinator` documents), keeping the current
    summary key plus any run still live: a survivor's steps must stay
    readable while it works, and a finished child's are in `AgentRunsDB`
    by then.
    """
    gateway = _ChunkGateway(
        [
            [_fence("spawn_subagent", {"task": "first job"})],
            ["child one"],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "second job"})],
            ["child two"],
            ["turn 2 final"],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    _run(bridge, store, session, aid, conversation_id="conv-live-prune")
    _join_fleet_threads()
    # Turn 1: its own summary slot plus its child's own slot.
    assert len(bridge._live["conv-live-prune"]) == 2, bridge._live["conv-live-prune"]

    second = _second_turn_message(store, session)
    _run(bridge, store, session, second, conversation_id="conv-live-prune")
    _join_fleet_threads()

    assert len(bridge._live["conv-live-prune"]) == 2, (
        f"turn 1's finished slots were never dropped: {bridge._live['conv-live-prune']}"
    )
    assert (
        bridge._live_primary_keys["conv-live-prune"] in bridge._live["conv-live-prune"]
    )


# -- PR3a-1 Task 6b (audit F5, second half): the navigate confirm's count
#
# `busy_fleet_session_count()` is the number the user is shown before
# "Leave" -- and the number the post-navigate toast reports as killed. It
# was (session with an active stream task) UNION (session with a pending
# approval round). A survivor is NEITHER, so the dialog said "0 runs will
# be killed" and then killed one. A dialog that lies to the user is worse
# than no dialog.


def test_busy_fleet_session_count_sees_a_session_whose_only_work_is_a_survivor(
    tmp_path,
):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    try:
        outcome = _run(bridge, store, session, assistant.id, conversation_id=session.id)
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"

        # Nothing else is busy: the turn is over, no approval is armed.
        assert controller.in_flight_run_count() == 0
        assert bridge.fleet_snapshot(session.id), "precondition: a live survivor"

        assert controller.busy_fleet_session_count() == 1, (
            "the confirm dialog would tell the user 0 runs will be killed, "
            "and then kill one"
        )
    finally:
        gate.set()
    _join_fleet_threads()

    # ... and it goes back to 0 once the survivor settles, so an idle
    # Console still navigates away with no dialog at all.
    assert controller.busy_fleet_session_count() == 0


def test_busy_fleet_session_count_ignores_a_terminal_child():
    """Only a STILL-RUNNING child counts.

    `fleet_snapshot` includes terminal handles while a run is in flight
    (`FleetCoordinator` never forgets one mid-turn), and a finished child
    is nothing teardown would kill -- counting it would inflate the number
    the confirm dialog shows in the opposite direction.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    class _FinishedFleetBridge:
        def fleet_snapshot(self, conversation_id):
            return [SimpleNamespace(status="done", handle_id="h1")]

    store = ConsoleChatStore()
    store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_FinishedFleetBridge()
    )
    assert controller.busy_fleet_session_count() == 0


def test_busy_fleet_session_count_degrades_when_the_bridge_raises():
    """Under-counting is the pre-PR3a-1 behaviour; a broken fleet read must
    never be the thing that blocks a navigation."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    class _RaisingFleetBridge:
        def fleet_snapshot(self, conversation_id):
            raise RuntimeError("boom")

    store = ConsoleChatStore()
    store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_RaisingFleetBridge()
    )
    assert controller.busy_fleet_session_count() == 0


def test_inline_child_live_slots_are_bounded_with_the_fleet_switched_off(
    tmp_path, monkeypatch
):
    """The kill-switch path emits child steps but has NO coordinator.

    So slot pruning cannot key off coordinator membership -- there is none
    to be a member of -- or `[agents] max_live_subagents <= 1` would leak
    one rail slot per inline sub-agent for the life of the process.
    """
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        ),
    )
    gateway = _ChunkGateway(
        [
            [_fence("spawn_subagent", {"task": "first job"})],
            ["child one"],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "second job"})],
            ["child two"],
            ["turn 2 final"],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    _run(bridge, store, session, aid, conversation_id="conv-inline-prune")
    assert bridge._fleet_coordinators.get("conv-inline-prune") is None
    assert len(bridge._live["conv-inline-prune"]) == 2

    second = _second_turn_message(store, session)
    _run(bridge, store, session, second, conversation_id="conv-inline-prune")

    assert len(bridge._live["conv-inline-prune"]) == 2, (
        "the inline path leaked a rail slot per child: "
        f"{bridge._live['conv-inline-prune']}"
    )


# --- PR3b Task 4: retention caps read beside max_live in the coordinator
# factory ([agents] retained_transcripts / retained_transcript_max_chars),
# applied to a NEW coordinator at construction and to an EXISTING one via
# set_retention_caps (the set_max_live shape). ---


def test_fleet_coordinator_factory_reads_the_retention_caps(tmp_path, monkeypatch):
    from Tests.Agents.conftest import pin_agent_settings
    from tldw_chatbook.Agents import agent_service as agent_service_module

    pin_agent_settings(
        monkeypatch,
        **{
            agent_service_module.MAX_LIVE_SUBAGENTS_KEY: 3,
            agent_service_module.RETAINED_TRANSCRIPTS_KEY: 2,
            agent_service_module.RETAINED_TRANSCRIPT_MAX_CHARS_KEY: 1234,
        },
    )
    bridge, _db, _store, _session, _aid = _bridge(tmp_path, [])
    coordinator = bridge._conversation_fleet_coordinator("conv-caps")
    assert coordinator is not None
    assert coordinator.retained_transcripts == 2
    assert coordinator.retained_transcript_max_chars == 1234


def test_fleet_coordinator_factory_resizes_retention_caps_in_place(
    tmp_path, monkeypatch
):
    from Tests.Agents.conftest import pin_agent_settings
    from tldw_chatbook.Agents import agent_service as agent_service_module

    pin_agent_settings(
        monkeypatch,
        **{
            agent_service_module.MAX_LIVE_SUBAGENTS_KEY: 3,
            agent_service_module.RETAINED_TRANSCRIPTS_KEY: 5,
            agent_service_module.RETAINED_TRANSCRIPT_MAX_CHARS_KEY: 200_000,
        },
    )
    bridge, _db, _store, _session, _aid = _bridge(tmp_path, [])
    first = bridge._conversation_fleet_coordinator("conv-resize")
    assert first is not None

    pin_agent_settings(
        monkeypatch,
        **{
            agent_service_module.RETAINED_TRANSCRIPTS_KEY: 1,
            agent_service_module.RETAINED_TRANSCRIPT_MAX_CHARS_KEY: 99,
        },
    )
    second = bridge._conversation_fleet_coordinator("conv-resize")
    # Re-sized IN PLACE, never replaced: replacing would drop the retained
    # transcripts along with every live handle (the set_max_live rule).
    assert second is first
    assert second.retained_transcripts == 1
    assert second.retained_transcript_max_chars == 99
