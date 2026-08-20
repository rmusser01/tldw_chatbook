"""Startup AGENTS.md delivery contracts for Console agent runs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunOutcome,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionSnapshot,
    InstructionSource,
    StartupInstructionCandidate,
)
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Chat import console_chat_controller as controller_mod
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ProjectInstructionBindingRecovery,
    build_project_instruction_dispatch_notice,
    commit_project_instruction_dispatch_decision,
    project_instruction_authority_is_current,
    resolve_project_instruction_binding,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    fingerprint_canonical_locator,
    project_instruction_notice_key,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding


SENTINEL = "AGENTS_AUTOMATIC_CHANNEL_SENTINEL_71f584"


def _candidate(
    tmp_path: Path, *, with_source: bool = True
) -> StartupInstructionCandidate:
    source = None
    if with_source:
        raw = SENTINEL.encode()
        source = InstructionSource(
            canonical_path=tmp_path / "AGENTS.md",
            relative_path="AGENTS.md",
            scope=".",
            kind="standard",
            body=SENTINEL,
            byte_count=len(raw),
            digest="d" * 64,
        )
    return StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=time.time_ns(),
        source=source,
        outcomes=(),
    )


class _ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, dict):
            message = item
        else:
            message = {"content": item}
        return {"choices": [{"message": message}]}


def _service(tmp_path: Path, replies, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = _ScriptedChat(replies)
    service = AgentService(
        AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        registry,
        chat_call=chat,
        **kwargs,
    )
    return service, chat


def _cfg(**changes) -> AgentConfig:
    values = {
        "model": "gpt-4o-mini",
        "system_prompt": "system",
        "allowed_tools": (),
        "native_tools": False,
        "response_reserve_tokens": 2048,
    }
    values.update(changes)
    return AgentConfig(**values)


def _sentinel_rows(call: dict) -> list[dict]:
    return [
        row
        for row in call["messages_payload"]
        if SENTINEL in str(row.get("content", ""))
    ]


def test_startup_rider_is_tagged_user_context_and_sent_once(tmp_path):
    seen = []
    service, chat = _service(
        tmp_path,
        ["done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda snapshot: (
            seen.append(snapshot) or "proceed"
        ),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(),
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(seen) == 1
    rows = _sentinel_rows(chat.calls[0])
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0][EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert "[Project instructions — untrusted repository context]" in rows[0]["content"]
    assert "AGENTS.md" in rows[0]["content"]


def test_fenced_tool_result_is_complete_before_separate_project_context(tmp_path):
    rows = [{"role": "user", "content": "Tool result — fs_read: complete"}]
    rider = agent_service.build_project_instruction_row(_candidate(tmp_path).source)
    combined = agent_service.append_project_instruction_rows(rows, [rider])
    assert combined[0] == rows[0]
    assert "Project instructions" not in combined[0]["content"]
    assert combined[1]["content"].startswith(
        "[Project instructions — untrusted repository context]"
    )


def test_no_root_still_confirms_before_first_provider_call(tmp_path):
    order = []

    def confirm(snapshot):
        order.append(("confirm", snapshot.startup_source))
        return "proceed"

    service, chat = _service(
        tmp_path,
        ["done"],
        startup_instruction_candidate=_candidate(tmp_path, with_source=False),
        confirm_project_instruction_dispatch=confirm,
    )
    original = chat.__call__

    def recorded(**kwargs):
        order.append(("provider", None))
        return original(**kwargs)

    service.chat_call = recorded
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(),
        api_endpoint="openai",
    )

    assert order == [("confirm", None), ("provider", None)]


def test_notice_key_reprompts_for_provider_or_endpoint_not_model():
    locator = "f" * 64
    first = project_instruction_notice_key(locator, "openai", "https://api.example/v1")
    assert first == project_instruction_notice_key(
        locator, "openai", "https://api.example/v1"
    )
    assert first != project_instruction_notice_key(
        locator, "anthropic", "https://api.example/v1"
    )
    assert first != project_instruction_notice_key(
        locator, "openai", "https://other.example/v1"
    )
    # Model identity is deliberately absent from the key contract.
    assert "model" not in project_instruction_notice_key.__annotations__


def test_notice_metadata_is_content_free_and_proceed_never_rereads(tmp_path):
    instruction_path = tmp_path / "AGENTS.md"
    instruction_path.write_text("NEW_BODY_MUST_NOT_BE_READ")
    seen = []
    service, chat = _service(
        tmp_path,
        ["done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda snapshot: (
            seen.append(
                agent_service.project_instruction_notice_metadata(
                    snapshot, destination_label="OpenAI (api.example)"
                )
            )
            or "proceed"
        ),
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(),
        api_endpoint="openai",
    )
    assert SENTINEL in str(chat.calls[0]["messages_payload"])
    assert "NEW_BODY_MUST_NOT_BE_READ" not in str(chat.calls[0]["messages_payload"])
    assert SENTINEL not in json.dumps(seen[0])
    assert seen[0]["relative_source"] == "AGENTS.md"
    assert seen[0]["scope"] == "."
    assert seen[0]["byte_count"] == len(SENTINEL.encode())


def test_token_omission_notice_keeps_content_free_source_metadata(
    monkeypatch, tmp_path
):
    seen = []
    snapshots = []
    service, _chat = _service(
        tmp_path,
        ["done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda snapshot: (
            snapshots.append(snapshot)
            or seen.append(
                agent_service.project_instruction_notice_metadata(
                    snapshot, destination_label="OpenAI"
                )
            )
            or "proceed"
        ),
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 1)
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(response_reserve_tokens=1),
        api_endpoint="openai",
    )
    assert seen[0]["relative_source"] == "AGENTS.md"
    assert seen[0]["scope"] == "."
    assert seen[0]["byte_count"] == len(SENTINEL.encode())
    assert "omitted_token_budget" in seen[0]["outcomes"]
    assert SENTINEL not in json.dumps(seen[0])
    assert snapshots[0].startup_source.body == SENTINEL
    assert snapshots[0].primary_delivery.source_digests == ()


def test_consent_notice_carries_owning_session_and_sanitized_exact_destination(
    tmp_path,
):
    service, _chat = _service(tmp_path, [])
    snapshot = service._freeze_startup_snapshot(
        _candidate(tmp_path),
        _cfg(),
        "openai",
        agent_service.ModelRequest(messages=(), tools=()),
    )
    notice = build_project_instruction_dispatch_notice(
        snapshot,
        session_id="session-owning-background-run",
        resolution=SimpleNamespace(
            provider="OpenAI",
            execution_key="openai",
            base_url="https://user:password@api.example/v1?token=secret#fragment",
        ),
    )
    assert notice.session_id == "session-owning-background-run"
    assert notice.destination_label == "OpenAI (https://api.example)"
    assert "password" not in repr(notice)
    assert "secret" not in repr(notice)
    assert SENTINEL not in repr(notice)


def test_cancel_and_disable_abort_before_provider_and_discard_snapshot(tmp_path):
    for decision in ("cancel", "disable"):
        service, chat = _service(
            tmp_path / decision,
            ["must not run"],
            startup_instruction_candidate=_candidate(tmp_path),
            confirm_project_instruction_dispatch=lambda _snapshot, d=decision: d,
        )
        _run_id, outcome = service.run_turn(
            conversation_id=decision,
            messages=[{"role": "user", "content": "question"}],
            config=_cfg(),
            api_endpoint="openai",
        )
        assert outcome.status == RUN_CANCELLED
        assert chat.calls == []


def test_missing_or_raising_consent_callback_fails_closed_without_provider(tmp_path):
    cases = (
        {},
        {
            "confirm_project_instruction_dispatch": lambda _snapshot: (
                _ for _ in ()
            ).throw(RuntimeError(SENTINEL))
        },
    )
    for index, kwargs in enumerate(cases):
        service, chat = _service(
            tmp_path / str(index),
            ["must not run"],
            startup_instruction_candidate=_candidate(tmp_path),
            **kwargs,
        )
        run_id, outcome = service.run_turn(
            conversation_id=str(index),
            messages=[{"role": "user", "content": "question"}],
            config=_cfg(),
            api_endpoint="openai",
        )
        assert outcome.status == RUN_CANCELLED
        assert chat.calls == []
        assert SENTINEL not in json.dumps(service.db.get_run(run_id), default=str)


def test_primary_and_child_each_receive_same_snapshot_once_per_request(tmp_path):
    spawn = {
        "content": None,
        "tool_calls": [
            {
                "id": "spawn-1",
                "type": "function",
                "function": {
                    "name": SPAWN_TOOL_NAME,
                    "arguments": json.dumps({"task": "inspect"}),
                },
            }
        ],
    }
    service, chat = _service(
        tmp_path,
        [spawn, "child done", "parent done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(
            native_tools=True,
            allowed_tools=(SPAWN_TOOL_NAME,),
        ),
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(chat.calls) == 3
    assert all(len(_sentinel_rows(call)) == 1 for call in chat.calls)


def test_child_chain_uses_its_own_exact_first_request_budget(monkeypatch, tmp_path):
    spawn = {
        "content": None,
        "tool_calls": [
            {
                "id": "spawn-1",
                "type": "function",
                "function": {
                    "name": SPAWN_TOOL_NAME,
                    "arguments": json.dumps({"task": "inspect"}),
                },
            }
        ],
    }
    service, chat = _service(
        tmp_path,
        [spawn, "child done", "parent done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        agent_service,
        "_count_model_messages",
        lambda messages, *_a: (
            95 if "sub-agent" in str(messages[0].get("content", "")).lower() else 10
        ),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(
            native_tools=True,
            allowed_tools=(SPAWN_TOOL_NAME,),
            response_reserve_tokens=10,
        ),
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(_sentinel_rows(chat.calls[0])) == 1
    assert len(_sentinel_rows(chat.calls[1])) == 0
    assert len(_sentinel_rows(chat.calls[2])) == 1


def test_primary_token_omission_is_delivery_local_when_child_admits(
    monkeypatch, tmp_path
):
    spawn = {
        "content": None,
        "tool_calls": [
            {
                "id": "spawn-1",
                "type": "function",
                "function": {
                    "name": SPAWN_TOOL_NAME,
                    "arguments": json.dumps({"task": "inspect"}),
                },
            }
        ],
    }
    snapshots = []
    service, chat = _service(
        tmp_path,
        [spawn, "child done", "parent done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda snapshot: (
            snapshots.append(snapshot) or "proceed"
        ),
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        agent_service,
        "_count_model_messages",
        lambda messages, *_a: (
            10 if "sub-agent" in str(messages[0].get("content", "")).lower() else 95
        ),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(
            native_tools=True,
            allowed_tools=(SPAWN_TOOL_NAME,),
            response_reserve_tokens=10,
        ),
        api_endpoint="openai",
    )

    snapshot = snapshots[0]
    assert outcome.status == RUN_DONE
    assert snapshot.global_outcomes == ()
    assert [item.code for item in snapshot.primary_delivery.outcomes] == [
        "omitted_token_budget"
    ]
    assert len(_sentinel_rows(chat.calls[0])) == 0
    assert len(_sentinel_rows(chat.calls[1])) == 1
    assert len(_sentinel_rows(chat.calls[2])) == 0


def test_exact_request_builder_is_the_payload_sent(tmp_path):
    service, chat = _service(tmp_path, ["done"])
    config = _cfg()
    expected = service._build_model_request(
        config,
        "openai",
        [],
        [{"role": "user", "content": "question"}],
        (),
        False,
    )
    service._make_call_model(config, "openai", [])(
        [{"role": "user", "content": "question"}], ()
    )
    assert chat.calls[0]["messages_payload"] == list(expected.messages)
    assert chat.calls[0].get("tools") == (list(expected.tools) or None)


def test_eviction_settings_are_frozen_before_consent(monkeypatch, tmp_path):
    class ActiveWriter:
        log_dir = None
        is_active = False

        def bind(self, _run_id):
            self.is_active = True

        def append(self, **_kwargs):
            return None

        def write_manifest(self, _manifest):
            return None

        def close(self):
            return None

    toggle = {"enabled": True, "min_rounds": 7}

    def setting(key, default):
        if key == agent_service.RUN_LOG_EVICT_ENABLED_KEY:
            return toggle["enabled"]
        if key == agent_service.RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY:
            return toggle["min_rounds"]
        return default

    seen = []
    real_bound = agent_service.bound_history_for_send

    def bound(messages, **kwargs):
        seen.append((kwargs["enabled"], kwargs["min_recent_rounds"]))
        return real_bound(messages, **kwargs)

    monkeypatch.setattr(agent_service, "_setting", setting)
    monkeypatch.setattr(agent_service, "bound_history_for_send", bound)
    service, _chat = _service(
        tmp_path,
        ["done"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda _snapshot: (
            toggle.update(enabled=False, min_rounds=1) or "proceed"
        ),
        run_log_writer=ActiveWriter(),
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(allowed_tools=("calculator",)),
        api_endpoint="openai",
    )
    assert seen
    # The frozen eviction shape keeps its configured minimum, but the first
    # request is deliberately log-neutral until the writer has survived a
    # request and is known active.
    assert set(seen) == {(False, 7)}


def test_safe_project_instruction_tokens_counts_tools_rows_and_reserve(
    monkeypatch, tmp_path
):
    service, _chat = _service(tmp_path, [])
    request = agent_service.ModelRequest(
        messages=({"role": "system", "content": "base"},),
        tools=({"type": "function", "function": {"name": "t"}},),
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 10)
    assert (
        service.safe_project_instruction_tokens(
            _cfg(response_reserve_tokens=30),
            "openai",
            request,
            [{"role": "user", "content": "wrapper"}],
        )
        == 40
    )


def test_invalid_model_limits_fail_safe_to_zero(monkeypatch, tmp_path):
    service, _chat = _service(tmp_path, [])
    request = agent_service.ModelRequest(messages=(), tools=())
    for invalid in (0, -1, None):
        monkeypatch.setattr(
            agent_service,
            "get_model_token_limit",
            lambda *_a, value=invalid, **_k: value,
        )
        assert (
            service.safe_project_instruction_tokens(_cfg(), "openai", request, []) == 0
        )


def test_automatic_body_is_absent_from_run_database_and_log(tmp_path):
    service, _chat = _service(
        tmp_path,
        ["answer"],
        startup_instruction_candidate=_candidate(tmp_path),
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    run_id, _outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "question"}],
        config=_cfg(),
        api_endpoint="openai",
    )
    assert SENTINEL not in json.dumps(service.db.get_run(run_id), default=str)
    log_dir = service.run_log_writer.log_dir
    if log_dir is not None:
        assert SENTINEL not in "".join(
            path.read_text(errors="replace")
            for path in log_dir.rglob("*")
            if path.is_file()
        )


def test_explicit_read_and_assistant_quote_remain_normally_persisted(tmp_path):
    (tmp_path / "AGENTS.md").write_text(SENTINEL)
    registry = ToolCatalogRegistry()
    registry.register_provider(
        LocalToolProvider(
            workspace_root=tmp_path,
            allow_write=False,
            resolve_state=lambda _hub: EffectiveToolState(
                state="allow", origin="tool_override"
            ),
        )
    )
    read_call = {
        "content": None,
        "tool_calls": [
            {
                "id": "read-1",
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "arguments": json.dumps({"path": "AGENTS.md"}),
                },
            }
        ],
    }
    chat = _ScriptedChat([read_call, f"Quoted: {SENTINEL}"])
    service = AgentService(
        AgentRunsDB(tmp_path / "explicit.db", client_id="test"),
        registry,
        chat_call=chat,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "Read and quote AGENTS.md"}],
        config=_cfg(native_tools=True, allowed_tools=("fs_read",)),
        api_endpoint="openai",
    )
    assert outcome.status == RUN_DONE
    assert SENTINEL in outcome.final_text
    assert SENTINEL in json.dumps(service.db.get_run(run_id), default=str)


class _BindingRegistry:
    def __init__(self, bindings):
        self.bindings = {binding.binding_id: binding for binding in bindings}

    def list_runtime_bindings(self, workspace_id):
        return tuple(
            binding
            for binding in self.bindings.values()
            if binding.workspace_id == workspace_id
        )

    def get_runtime_binding(self, binding_id):
        return self.bindings.get(binding_id)


def _binding(tmp_path, binding_id="b1", *, access="rw"):
    return WorkspaceRuntimeBinding(
        workspace_id="w1",
        binding_id=binding_id,
        binding_kind="local-filesystem",
        label=binding_id,
        locator=str(tmp_path),
        status="ready",
        metadata={"access": access},
    )


def _session(state):
    return SimpleNamespace(workspace_id="w1", project_instruction_state=state)


def test_sole_eligible_binding_auto_selects_and_captures_fingerprint(tmp_path):
    selection = resolve_project_instruction_binding(
        _session(ProjectInstructionControlState.new_session()),
        _BindingRegistry([_binding(tmp_path)]),
    )
    assert selection.binding.binding_id == "b1"
    assert selection.root == tmp_path.resolve()
    assert selection.allow_write is True
    assert selection.locator_fingerprint == fingerprint_canonical_locator(
        str(tmp_path.resolve())
    )


def test_binding_with_symlinked_ancestor_is_never_auto_selected(tmp_path):
    real_parent = tmp_path / "real"
    real_root = real_parent / "repo"
    real_root.mkdir(parents=True)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    binding = _binding(alias_parent / "repo")

    with pytest.raises(ProjectInstructionBindingRecovery, match="no_eligible_binding"):
        resolve_project_instruction_binding(
            _session(ProjectInstructionControlState.new_session()),
            _BindingRegistry([binding]),
        )


def test_noncanonical_binding_locator_is_never_auto_selected(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    binding = _binding(child / "..")

    with pytest.raises(ProjectInstructionBindingRecovery, match="no_eligible_binding"):
        resolve_project_instruction_binding(
            _session(ProjectInstructionControlState.new_session()),
            _BindingRegistry([binding]),
        )


def test_windows_reparse_component_is_never_eligible(monkeypatch, tmp_path):
    real_lstat = controller_mod.os.lstat

    def reparse_lstat(path):
        value = real_lstat(path)
        attributes = 0x400 if Path(path) == tmp_path else 0
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_file_attributes=attributes,
        )

    monkeypatch.setattr(controller_mod, "_WINDOWS", True, raising=False)
    monkeypatch.setattr(controller_mod, "_REPARSE_POINT", 0x400, raising=False)
    monkeypatch.setattr(controller_mod.os, "lstat", reparse_lstat)

    with pytest.raises(ProjectInstructionBindingRecovery, match="no_eligible_binding"):
        resolve_project_instruction_binding(
            _session(ProjectInstructionControlState.new_session()),
            _BindingRegistry([_binding(tmp_path)]),
        )


def test_zero_or_multiple_bindings_hold_for_recovery(tmp_path):
    session = _session(ProjectInstructionControlState.new_session())
    for bindings, code in (
        ([], "no_eligible_binding"),
        ([_binding(tmp_path, "a"), _binding(tmp_path, "b")], "choose_binding"),
    ):
        with pytest.raises(ProjectInstructionBindingRecovery, match=code):
            resolve_project_instruction_binding(session, _BindingRegistry(bindings))


def test_enabled_session_missing_or_failing_registry_holds_for_recovery(tmp_path):
    session = _session(ProjectInstructionControlState.new_session())

    class FailingRegistry:
        def list_runtime_bindings(self, _workspace_id):
            raise OSError("registry unavailable")

        def get_runtime_binding(self, _binding_id):
            raise KeyError("removed")

    with pytest.raises(ProjectInstructionBindingRecovery, match="binding_unavailable"):
        resolve_project_instruction_binding(session, None)
    with pytest.raises(ProjectInstructionBindingRecovery, match="binding_unavailable"):
        resolve_project_instruction_binding(session, FailingRegistry())

    selected = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="b1",
        working_folder_locator_fingerprint="f" * 64,
    )
    with pytest.raises(ProjectInstructionBindingRecovery, match="binding_unavailable"):
        resolve_project_instruction_binding(_session(selected), FailingRegistry())


def test_stale_consent_never_overwrites_changed_state_or_access(tmp_path):
    binding = _binding(tmp_path, access="rw")
    registry = _BindingRegistry([binding])
    store = SimpleNamespace()
    session = _session(ProjectInstructionControlState.new_session())
    session.id = "session-1"
    expected_selection = resolve_project_instruction_binding(session, registry)
    expected_state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id=binding.binding_id,
        working_folder_locator_fingerprint=expected_selection.locator_fingerprint,
    )
    session.project_instruction_state = expected_state
    writes = []
    store.sessions = lambda: (session,)
    store.set_session_project_instruction_state = lambda _session_id, state: (
        writes.append(state)
    )

    registry.bindings[binding.binding_id] = _binding(tmp_path, access="ro")
    assert not project_instruction_authority_is_current(
        store=store,
        session_id=session.id,
        registry=registry,
        expected_selection=expected_selection,
    )
    result = commit_project_instruction_dispatch_decision(
        store=store,
        session_id=session.id,
        registry=registry,
        expected_state=expected_state,
        expected_selection=expected_selection,
        notice_key="n" * 64,
        decision="proceed",
    )
    assert result == "cancel"
    assert writes == []

    session.project_instruction_state = ProjectInstructionControlState.legacy_disabled()
    result = commit_project_instruction_dispatch_decision(
        store=store,
        session_id=session.id,
        registry=registry,
        expected_state=expected_state,
        expected_selection=expected_selection,
        notice_key="n" * 64,
        decision="proceed",
    )
    assert result == "cancel"
    assert writes == []


@pytest.mark.asyncio
async def test_controller_notice_uses_owning_session_and_drift_cancels_bridge_send(
    tmp_path,
):
    (tmp_path / "AGENTS.md").write_text(SENTINEL)
    binding = _binding(tmp_path)
    registry = _BindingRegistry([binding])
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="w1")
    notices = []
    owning_loop_calls = []
    provider_transmissions = []
    main_thread = threading.get_ident()

    def call_from_thread(callback):
        owning_loop_calls.append(threading.get_ident())
        return callback()

    def confirm(notice):
        assert threading.get_ident() != main_thread
        notices.append(notice)
        store.set_session_project_instruction_state(
            session.id, ProjectInstructionControlState.legacy_disabled()
        )
        return "proceed"

    class Bridge:
        def run_reply(self, **kwargs):
            candidate = kwargs["startup_instruction_candidate"]
            delivery = InstructionChainDelivery(
                source_digests=(candidate.source.digest,),
                outcomes=candidate.outcomes,
            )
            snapshot = InstructionSnapshot(
                binding_id=candidate.binding_id,
                binding_root=candidate.binding_root,
                locator_fingerprint=candidate.locator_fingerprint,
                dispatch_started_wall_ns=candidate.dispatch_started_wall_ns,
                startup_source=candidate.source,
                global_outcomes=candidate.outcomes,
                primary_delivery=delivery,
                warning_codes=(),
            )
            decision = kwargs["confirm_project_instruction_dispatch"](snapshot)
            if decision == "proceed":
                provider_transmissions.append(True)
            return "run-1", RunOutcome(status=RUN_CANCELLED, steps=[], final_text="")

    class Gateway:
        async def resolve_for_send(self, _selection):
            return ConsoleProviderResolution(
                provider="OpenAI",
                base_url="https://user:password@api.example/v1?secret=yes",
                model="test-model",
                ready=True,
                readiness_key="openai",
                execution_key="openai",
                max_tokens=128,
            )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai",
        model="test-model",
        agent_bridge=Bridge(),
        agent_runtime_enabled=True,
        confirm_project_instruction_dispatch=confirm,
    )
    controller.app = SimpleNamespace(
        call_from_thread=call_from_thread,
        workspace_registry_service=registry,
    )

    result = await controller.submit_draft("question")

    assert result.accepted is True
    assert provider_transmissions == []
    assert len(owning_loop_calls) == 2
    assert notices[0].session_id == session.id
    assert notices[0].destination_label == "OpenAI (https://api.example)"
    assert SENTINEL not in repr(notices[0])


def test_removed_or_retargeted_binding_never_silently_retargets(tmp_path):
    original = tmp_path / "original"
    retarget = tmp_path / "retarget"
    original.mkdir()
    retarget.mkdir()
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="b1",
        working_folder_locator_fingerprint=fingerprint_canonical_locator(
            str(original.resolve())
        ),
    )
    with pytest.raises(ProjectInstructionBindingRecovery, match="binding_unavailable"):
        resolve_project_instruction_binding(_session(state), _BindingRegistry([]))
    with pytest.raises(ProjectInstructionBindingRecovery, match="binding_retargeted"):
        resolve_project_instruction_binding(
            _session(state), _BindingRegistry([_binding(retarget)])
        )


def test_disabled_session_does_not_consult_registry(tmp_path):
    class ExplodingRegistry:
        def list_runtime_bindings(self, _workspace_id):
            raise AssertionError("disabled sessions must not discover bindings")

    assert (
        resolve_project_instruction_binding(
            _session(ProjectInstructionControlState.legacy_disabled()),
            ExplodingRegistry(),
        )
        is None
    )
