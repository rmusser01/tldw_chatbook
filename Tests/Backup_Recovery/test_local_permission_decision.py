"""Native work and authority of one stock Console local tool invocation."""

import io
import json
import sys
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_permission_read_operation import _observe
from Tests.Chat import test_console_async_mcp_snapshot as fixtures
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Backup_Recovery import activation
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import permission_store
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

local_root = fixtures.local_root
catalog_store = fixtures.catalog_store
mcp_sources = fixtures.mcp_sources
snapshot_case = fixtures.snapshot_case
pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows permission reads"
)


@pytest.fixture
def local_decision(snapshot_case, tmp_path, request):
    source = snapshot_case.permissions
    source.set_global_default("ask")
    options = getattr(request, "param", {})
    profile_id = options.get("profile", "default")
    if profile_id != "default":
        source.ensure_profile(profile_id)
        source.set_global_default(options["state"], profile_id=profile_id)
    workspace = tmp_path / "decision-workspace"
    workspace.mkdir()
    (workspace / "evidence.txt").write_text("local decision", encoding="utf-8")
    provider, review = snapshot_case.controller._compose_local_provider(
        turn_context=SimpleNamespace(
            tool_configuration={"local_tools_enabled": True},
            tool_policy_profile_id=profile_id,
            persona_policy_rules=options.get("persona"),
        ),
        project_root=workspace,
        admitted_roots=options.get("roots", lambda _workspace: None)(workspace),
    )
    assert provider is not None
    source.load()
    return snapshot_case, provider, review


def _invoke(local_decision, *, args=None, before_event=None, retained=False):
    case, provider, _review = local_decision
    source = case.permissions
    counts, commands, phases, issued, descriptors = Counter(), [], [], [], set()
    command_frames = set()
    original_leases = set(storage._live_leases)

    def observe(frame, event, result):
        if before_event is not None:
            before_event(frame, event, result)
        code = frame.f_code
        if event == "call":
            if (
                code is permission_store._run_permission_read.__code__
                and frame.f_locals["source"] is source
            ):
                commands.append(frame.f_locals["captured"].method)
                command_frames.add(frame)
            if (
                command_frames
                and code is permission_store._PermissionPathFence.observe.__code__
                and frame.f_locals["self"] is source._permission_publication
            ):
                phases.append("published")
            if code is WorkspaceToolExecutor.execute.__code__:
                counts["executor"] += 1
        elif event == "return":
            if (
                command_frames
                and code is activation._ordinary_activation_observation.__code__
                and result is not None
            ):
                caller = frame.f_back
                if caller.f_code is raw._permission_read_boundary.__code__:
                    phases.append(
                        raw._states[caller.f_locals["operation"]].permission_read_stage
                    )
                else:
                    phases.append("pre-lock")
            if frame in command_frames:
                command_frames.remove(frame)
            if (
                code is storage.acquire_storage.__code__
                and result is not None
                and command_frames
            ):
                issued.append(result)
            if (
                code is raw.close_descriptor.__code__
                and frame.f_locals["fd"] in descriptors
            ):
                descriptors.remove(frame.f_locals["fd"])
                phases.append("payload retired")
        elif (
            command_frames
            and event == "c_call"
            and getattr(result, "__name__", None) == "read"
        ):
            handle = getattr(result, "__self__", None)
            if type(handle) is io.TextIOWrapper and any(
                state.source is source and any(handle is file for file in state.files)
                for state in raw._states.values()
            ):
                descriptors.add(handle.fileno())
                phases.append("payload read")

    with use_run_id("local-decision"), _observe(observe):
        result = provider.invoke_detailed(
            "local:fs_list", {"path": "."} if args is None else args
        )

    assert not command_frames
    if not retained:
        assert not descriptors
        assert not storage._pending_acquisitions
        assert set(storage._live_leases) == original_leases
        assert all(lease not in storage._live_leases for lease in issued)
        assert not any(state.source is source for state in raw._states.values())
    return result, commands, phases, counts, issued


def test_one_fresh_payload_for_stock_local_decision(local_decision, request):
    _case, provider, _review = local_decision
    provider.apply_batch_decisions("local-decision", {"fs_list": "approve_once"})
    result, commands, phases, counts, issued = _invoke(local_decision)

    request.node.user_properties.append(("permission_commands", json.dumps(commands)))
    request.node.user_properties.append(("permission_phases", json.dumps(phases)))
    assert result.result.ok and "evidence.txt" in str(result.result)
    assert result.final_gate == "allow" and result.approval_consumed
    assert counts["executor"] == 1
    per_command = [
        "pre-lock",
        "pre-lock",
        "setup",
        "metadata",
        "payload read",
        "payload retired",
        "final",
        "published",
    ]
    assert phases == per_command * len(commands), phases
    assert len(issued) == 2 * len(commands)
    assert commands == ["load"], commands


def test_post_card_permission_revocation_uses_fresh_payload(local_decision):
    case, provider, _review = local_decision
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook

    reviewed = []

    def answer(pending):
        reviewed.extend(row.llm_name for row in pending)
        case.permissions.set_global_default("deny")
        return {"fs_list": "approve_once"}

    review = build_local_review_hook(provider, answer)
    review([ToolCall("fs_list", {"path": "."})], "local-decision")
    assert reviewed == ["fs_list"]
    assert provider.stamped("local-decision", "fs_list") == "approve_once"
    result, commands, phases, counts, _issued = _invoke(local_decision)
    assert result.final_gate == "deny" and not result.dispatch_started
    assert not counts["executor"] and commands == ["load"]
    assert phases.count("published") == 1


@pytest.mark.parametrize(
    "change",
    [
        "custom_state",
        "mapping",
        "switch_helper",
        "substituted_policy",
        "substituted_profile",
        "gate_exempt",
    ],
)
def test_unqualified_input_keeps_original_switch_and_state(
    local_decision, monkeypatch, change
):
    _case, provider, _review = local_decision
    provider.apply_batch_decisions("local-decision", {"fs_list": "approve_once"})
    args = {"path": "."}
    if change == "custom_state":
        original = provider._resolve_state
        called = []

        def resolve(hub):
            called.append(hub.name)
            return original(hub)

        monkeypatch.setattr(provider, "_resolve_state", resolve)
    elif change == "mapping":

        class CustomDict(dict):
            pass

        args = CustomDict(args)
    elif change == "switch_helper":
        from tldw_chatbook.Agents import local_tool_authority

        monkeypatch.setattr(
            local_tool_authority, "kill_switch_engaged", lambda _provider: True
        )
    elif change == "substituted_policy":

        def replaced(*_args):
            pytest.fail("an unissued policy callback ran")

        monkeypatch.setattr(
            provider,
            "_console_local_policy",
            SimpleNamespace(read=replaced, project=replaced),
        )
    elif change == "substituted_profile":
        monkeypatch.setattr(
            provider,
            "_console_local_policy",
            replace(provider._console_local_policy, profile_id="unissued-profile"),
        )
    else:
        monkeypatch.setitem(
            provider._specs,
            "fs_list",
            replace(provider._specs["fs_list"], gate_exempt=True),
        )
    result, commands, _phases, counts, _issued = _invoke(local_decision, args=args)
    if change == "switch_helper":
        assert result.final_gate == "kill_switch" and not counts["executor"]
        assert not commands
    elif change == "gate_exempt":
        assert result.result.ok and not result.approval_consumed
        assert commands == ["get_kill_switch"]
    else:
        assert result.final_gate == "allow" and counts["executor"] == 1
        assert commands == ["get_kill_switch", "load"]
        assert result.result.ok is (change != "mapping")
    if change == "mapping":
        assert result.reason_code.value == "handler_raised"
    elif change == "custom_state":
        assert called == ["fs_list"]


@pytest.mark.parametrize(
    "drift", ["resolver", "promotion", "source_path", "gate_exempt"]
)
def test_late_projection_drift_refuses_without_second_read(
    local_decision, monkeypatch, drift
):
    case, provider, _review = local_decision
    provider.apply_batch_decisions("local-decision", {"fs_list": "approve_once"})
    changed = []

    def replacement(*_args, **_kwargs):
        raise AssertionError("selected decision cannot replay through a new callback")

    def inject(frame, event, _result):
        if event != "return" or changed:
            return
        if drift == "source_path":
            if (
                frame.f_code is raw._permission_read_boundary.__code__
                and not frame.f_locals["destination"]
            ):
                source = case.permissions
                changed.append(True)
                monkeypatch.setattr(
                    source, "path", source.path.with_name("foreign.json")
                )
        elif frame.f_code is permission_store._run_permission_read.__code__:
            changed.append(True)
            if drift == "gate_exempt":
                spec = provider._specs["fs_list"]
                monkeypatch.setitem(
                    provider._specs, "fs_list", replace(spec, gate_exempt=True)
                )
            else:
                name = "_resolve_state" if drift == "resolver" else "_invoke_promotion"
                monkeypatch.setattr(provider, name, replacement)

    result, commands, _phases, counts, _issued = _invoke(
        local_decision, before_event=inject
    )
    assert changed == [True]
    assert result.final_gate == "gate_error"
    assert not result.dispatch_started and not counts["executor"]
    assert commands == ["load"]


def test_switch_read_failure_keeps_one_distinct_state_read(local_decision):
    _case, provider, _review = local_decision
    provider.apply_batch_decisions("local-decision", {"fs_list": "approve_once"})
    failed = []

    raw_stat = raw.os.fstat

    def fault(fd):
        state = raw._states.get(getattr(raw._local, "operation", None))
        if (
            state is not None
            and state.permission_read_stage == "payload"
            and not failed
        ):
            failed.append(True)
            raise OSError("simulated ordinary payload open failure")
        return raw_stat(fd)

    def inject(frame, event, _result):
        if event == "return" and frame.f_code is raw._permission_read_boundary.__code__:
            if frame.f_locals["destination"] and not failed:
                raw.os.fstat = fault
        if failed and raw.os.fstat is fault:
            raw.os.fstat = raw_stat

    try:
        result, commands, phases, counts, _issued = _invoke(
            local_decision, before_event=inject
        )
    finally:
        raw.os.fstat = raw_stat
    assert failed == [True]
    assert result.result.ok and counts["executor"] == 1
    assert commands == ["load", "load"]
    assert phases.count("published") == 1


def test_uncertain_selected_close_refuses_without_second_read(local_decision):
    case, provider, _review = local_decision
    source = case.permissions
    provider.apply_batch_decisions("local-decision", {"fs_list": "approve_once"})
    original_close = raw.os.close
    injected, payload_fd = [], []
    before = set(storage._live_leases)

    def fail_close(fd):
        if payload_fd and fd == payload_fd[0]:
            state = raw._states[raw._local.operation]
            injected.append((state, fd))
            raise OSError("simulated uncertain payload close")
        return original_close(fd)

    def inject(frame, event, result):
        if event == "c_call" and getattr(result, "__name__", None) == "read":
            handle = getattr(result, "__self__", None)
            if type(handle) is io.TextIOWrapper and any(
                state.source is source and any(handle is file for file in state.files)
                for state in raw._states.values()
            ):
                payload_fd.append(handle.fileno())
                raw.os.close = fail_close

    try:
        result, commands, phases, counts, _issued = _invoke(
            local_decision, before_event=inject, retained=True
        )
        assert result.final_gate == "gate_error" and not counts["executor"]
        assert commands == ["load"] and "published" not in phases
        assert len(injected) == 1
        state, fd = injected[0]
        assert state.uncertain and not state.active and fd in state.descriptors
        assert set(storage._live_leases) - before == set(state.leases)
        assert source._permission_publication.uncertain
    finally:
        raw.os.close = original_close
        for operation, state in tuple(raw._states.items()):
            if state.source is source:
                for fd in tuple(state.descriptors):
                    original_close(fd)
                    state.descriptors.remove(fd)
                state.uncertain = False
                assert raw._retire(state)
                with storage._changed:
                    del raw._states[operation]
                    storage._raw_operations.discard(operation)
                    storage._changed.notify_all()


@pytest.mark.parametrize(
    "local_decision, stamp, expected",
    [
        ({"profile": "research", "state": "allow"}, None, "allow"),
        (
            {
                "profile": "research",
                "state": "allow",
                "persona": (
                    {
                        "rule_kind": "mcp_tool",
                        "rule_name": "fs_list",
                        "allowed": True,
                        "require_confirmation": True,
                    },
                ),
            },
            "deny",
            "deny",
        ),
    ],
    indirect=["local_decision"],
)
def test_named_profile_and_persona_floor(local_decision, stamp, expected):
    _case, provider, _review = local_decision
    if stamp is not None:
        provider.apply_batch_decisions("local-decision", {"fs_list": stamp})
    result, commands, _phases, counts, _issued = _invoke(local_decision)
    assert commands == ["load"]
    assert result.final_gate == expected
    assert bool(counts["executor"]) == (expected == "allow")


def _admitted_root(workspace):
    return (
        RunAdmittedWorkspaceRoot(
            workspace_id="test",
            binding_id="test",
            alias="fixture",
            root=workspace,
            locator_fingerprint="workspace",
            root_identity=((str(workspace), 0, 0, 0),),
            allow_write=True,
            guard=lambda _write: True,
        ),
    )


@pytest.mark.parametrize("local_decision", [{"roots": _admitted_root}], indirect=True)
def test_kill_switch_precedes_retired_alias(local_decision):
    case, provider, _review = local_decision
    provider._path_specs_by_alias.clear()
    case.permissions.set_kill_switch(True)
    killed, commands, _phases, counts, _issued = _invoke(local_decision)
    assert killed.final_gate == "kill_switch" and not counts["executor"]
    assert commands == ["load"]
    case.permissions.set_kill_switch(False)
    retired, commands, _phases, counts, _issued = _invoke(local_decision)
    assert retired.reason_code.value == "authority_unavailable"
    assert not counts["executor"] and commands == ["load"]
