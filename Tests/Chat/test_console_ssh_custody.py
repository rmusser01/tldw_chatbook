"""The original model operation covers real local-peer SSH request retirement."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest
import toml

from Tests.Chat.test_native_v2_command_events import until
from Tests.Chat.test_native_v2_mcp_events import (
    event_case as _event_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    hook_file as _hook_file,
)
from Tests.Chat.test_native_v2_mcp_events import (
    mcp_case as _mcp_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_console as _native_console,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_package as _native_package,
)
from Tests.Tools.test_remote_executor_ssh import FakeSsh
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

event_case, hook_file, mcp_case = _event_case, _hook_file, _mcp_case
native_console, native_package = _native_console, _native_package

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX platform required"
    ),
]


@pytest.fixture
async def ssh_console(mcp_case, hook_file, monkeypatch, tmp_path, request):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook import config
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    case, rig = mcp_case, mcp_case.rig
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools import workspace_file_roots

    monkeypatch.setattr(
        workspace_file_roots, "_default_registry_instance", rig.registry
    )
    rig.store.set_session_project_instruction_state(
        rig.session.id, ProjectInstructionControlState.legacy_disabled()
    )
    raw = toml.loads(hook_file.read_text())
    raw["console_ssh"] = {"session_worker": False, "bundle_cache": False}
    raw.setdefault("console", {})["local_tools_enabled"] = True
    hook_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    fake = FakeSsh(tmp_path, monkeypatch)
    manager = transport.SshMasterManager(ssh_bin=str(fake.bin), enabled=False)
    monkeypatch.setattr(transport, "_MASTER_MANAGER", manager)
    workspace = tmp_path / "remote-workspace"
    workspace.mkdir()
    (workspace / "alpha.txt").write_text("original remote body\n")
    binding = rig.registry.add_ssh_binding(
        rig.session.workspace_id,
        f"ssh://fake-host{workspace}",
        python_interpreter=sys.executable,
        ssh_bin=str(fake.bin),
        allow_write=getattr(request, "param", {}).get("write", False),
    )
    case.service.approve_for_session("local:__local__", "fs_read")
    case.service.approve_for_session("local:__local__", "fs_edit")
    rig.store.persistence.db.close()
    rig.store.persistence = ChatPersistenceService(
        CharactersRAGDB(tmp_path / "console-chat.sqlite", "ssh-custody"),
        workspace_registry=rig.registry,
    )

    state = SimpleNamespace(
        before_model=None,
        reads=1,
        operations=[],
        finishes=[],
        native_returns=[],
        edit=False,
    )

    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    reserve_tool = ExecutionOwner.reserve_tool

    def observe_operation(owner):
        operation = reserve_tool(owner)
        state.operations.append(operation)
        finish = operation.finish

        def observe_return():
            finish()
            from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

            state.native_returns.append(current_ssh_call())
            admission = _guard.context.get()
            state.finishes.append(
                (operation, admission, admission.live if admission else False)
            )

        operation.finish = observe_return
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", observe_operation)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            if state.before_model is not None:
                await state.before_model(len(self.payloads))
            if len(self.payloads) == 1:
                yield fence(
                    "load_tools",
                    {
                        "ids": ["local:fs_read", "local:fs_edit"]
                        if state.edit
                        else ["local:fs_read"]
                    },
                )
            elif len(self.payloads) <= state.reads + 1:
                yield fence(
                    "fs_read", {"path": "alpha.txt", "root_alias": binding.binding_id}
                )
            elif state.edit and len(self.payloads) == state.reads + 2:
                yield fence(
                    "fs_edit",
                    {
                        "path": "alpha.txt",
                        "root_alias": binding.binding_id,
                        "old_string": "original remote body",
                        "new_string": "edited remote body",
                    },
                )
            else:
                yield "read complete"

    gateway = Gateway()
    database = AgentRunsDB(tmp_path / "console-runs.sqlite", client_id="ssh-custody")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
        runtime_capacity=case.runtime.execution_capacity,
        get_hooks_v2=case.runtime.get_hooks_v2,
    )
    case.runtime.set_agent_bridge(bridge)
    rig.controller._agent_bridge = bridge
    rig.controller._agent_runtime_enabled = True
    rig.controller.provider_gateway = gateway
    state.case, state.rig, state.gateway = case, rig, gateway
    state.fake, state.manager, state.database = fake, manager, database
    state.config_file = hook_file
    state.root = tmp_path

    def submit():
        task = asyncio.create_task(
            rig.controller.submit_draft(
                "read the remote file", session_id=rig.session.id
            )
        )
        case.tasks.append(task)
        return task

    state.submit = submit
    try:
        yield state
    finally:
        database.close()


async def test_original_model_operation_waits_for_real_ssh_stderr_return(
    ssh_console, monkeypatch
):
    h = ssh_console
    release, entered = h.root / "release-io", h.root / "io-live"
    # The actual remote bundle exits; a real OS child retains its stderr pipe.
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        await asyncio.sleep(5)
        assert any(
            t.name.startswith("ssh-call-stderr-") and t.is_alive()
            for t in threading.enumerate()
        )
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1, (
            "original fs_read operation retired while actual SSH stderr IO was live"
        )
    finally:
        release.touch()
        await task
        await until(
            lambda: (
                not any(
                    t.name.startswith("ssh-call-stderr-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
        )
    assert len(h.gateway.payloads) == 3
    assert "original remote body" in str(h.gateway.payloads[2])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert h.finishes and all(live for _, _, live in h.finishes)


async def test_two_model_requests_return_while_original_shared_connection_stays_healthy(
    ssh_console, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Tools import remote_session_registry as sessions

    h = ssh_console
    raw = toml.loads(h.config_file.read_text())
    raw["console_ssh"]["session_worker"] = True
    h.config_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    registry = sessions.RemoteSessionRegistry()
    monkeypatch.setattr(sessions, "_REGISTRY", registry)
    entered, release = threading.Event(), threading.Event()
    h.reads = 2

    async def hold(index):
        if index == 4:
            entered.set()
            await asyncio.to_thread(release.wait, 10)

    h.before_model = hold
    task = h.submit()
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert len(registry._sessions) == 1
        worker = next(iter(registry._sessions.values()))
        assert worker.alive and worker._reader_thread.is_alive()
        assert worker._proc.poll() is None
        assert worker._native_acceptance.registry is registry
        assert len(h.fake.call_invocations()) == 1
        assert h.finishes and all(live for _, _, live in h.finishes)
        assert "original remote body" in str(h.gateway.payloads[2])
        assert "original remote body" in str(h.gateway.payloads[3])
    finally:
        release.set()
        await task
        registry.close_all()


async def test_never_started_destination_helper_retires_original_operation(ssh_console):
    h = ssh_console

    async def remove_peer(index):
        if index == 2:
            h.fake.bin.unlink()

    h.before_model = remove_peer
    assert (await h.submit()).accepted
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert not h.fake.call_invocations()
    assert "ssh -G could not re-resolve" in str(h.gateway.payloads[-1])
    assert h.finishes and all(live for _, _, live in h.finishes)


@pytest.mark.parametrize(
    "replacement", ["provider", "handler", "authority", "executor"]
)
async def test_changed_original_provider_callback_refuses_after_enqueue(
    ssh_console, monkeypatch, replacement
):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

    h, invoked = ssh_console, []
    start = threading.Thread.start

    def altered(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("replacement provider ran")

    def replace_before_start(thread):
        if thread.name == "tool-fs_read":
            from dataclasses import replace

            (service,) = h.rig.controller._agent_bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            if replacement == "provider":
                monkeypatch.setattr(LocalToolProvider, "_invoke_detailed", altered)
            elif replacement == "handler":
                spec = provider._path_specs_by_alias[authority.alias]["fs_read"]
                provider._path_specs_by_alias[authority.alias]["fs_read"] = replace(
                    spec, handler=altered
                )
            elif replacement == "authority":
                provider._admitted_roots[authority.alias] = replace(
                    authority, guard=lambda _: True
                )
            else:
                monkeypatch.setattr(
                    authority.workspace_executor._inner, "execute", altered
                )
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", replace_before_start)
    assert (await h.submit()).accepted
    assert not invoked and not h.fake.call_invocations()
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert "original SSH" in str(h.gateway.payloads[-1])


@pytest.mark.parametrize("shared", [False, True], ids=["one_shot", "session"])
@pytest.mark.parametrize(
    "replacement", ["manager", "_bundle_payload", "_remote_command"]
)
async def test_changed_receiving_ssh_source_refuses_after_enqueue(
    ssh_console, monkeypatch, shared, replacement
):
    from tldw_chatbook.Tools import remote_session_worker as session
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    h, invoked = ssh_console, []
    registry = enable_sessions(h, monkeypatch) if shared else None
    peer_entered = h.root / "replacement-peer-entered"
    replacement_peer = h.root / "replacement-ssh"
    replacement_peer.write_text(
        h.fake.bin.read_text().replace("\n", '\ntouch "$SSH_REPLACEMENT_ENTERED"\n', 1)
    )
    replacement_peer.chmod(0o700)
    monkeypatch.setenv("SSH_REPLACEMENT_ENTERED", str(peer_entered))
    start = threading.Thread.start
    original_builders = []

    def altered(*args, **kwargs):
        invoked.append(True)
        return original_builders[-1](*args, **kwargs)

    def replace_before_start(thread):
        if thread.name == "tool-fs_read":
            (service,) = h.rig.controller._agent_bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            if replacement == "manager":
                monkeypatch.setattr(
                    authority.workspace_executor._inner._ssh.transport,
                    "_manager",
                    transport.SshMasterManager(
                        ssh_bin=str(replacement_peer), enabled=False
                    ),
                )
            else:
                original_builders.append(
                    getattr(session if shared else transport, replacement)
                )
                monkeypatch.setattr(
                    session if shared else transport, replacement, altered
                )
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", replace_before_start)
    try:
        assert (await h.submit()).accepted
        assert not invoked and not peer_entered.exists()
        assert not h.fake.call_invocations()
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert "original SSH" in str(h.gateway.payloads[-1])
        assert h.finishes and all(live for _, _, live in h.finishes)
    finally:
        if registry is not None:
            registry.close_all()


def install_control_peer(h, monkeypatch, *, checks=True):
    """Actual Unix control server: check succeeds only after a socket exchange."""
    peer = h.root / "control-peer.py"
    peer.write_text("""import os, socket, subprocess, sys
from pathlib import Path
mode, *args = sys.argv[1:]
if mode == "serve":
    path = args[0]
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen()
    Path(path + ".pid").write_text(str(os.getpid()))
    print("READY", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            data = conn.recv(20)
            conn.sendall(b"OK")
        if data == b"exit": break
    server.close()
    Path(path).unlink()
else:
    path = next(x.split("=", 1)[1] for x in args if x.startswith("ControlPath=")).replace("%C", "peer-socket")
    if mode == "launch":
        proc = subprocess.Popen([sys.executable, __file__, "serve", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        assert proc.stdout.readline() == b"READY\\n"
        proc.stdout.close()
    else:
        if mode == "check" and os.environ.get("SSH_PEER_HIDE_CHECK"): sys.exit(1)
        try:
            with socket.socket(socket.AF_UNIX) as conn:
                conn.connect(path)
                conn.sendall(mode.encode())
                assert conn.recv(20) == b"OK"
        except OSError: sys.exit(1)
""")
    script = (
        h.fake.bin.read_text()
        .replace(
            "check) exit 1 ;;",
            'check) exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" check "$@" ;;',
        )
        .replace(
            "exit)  exit 0 ;;",
            'exit) exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" exit "$@" ;;',
        )
    )
    start = script.index('for arg in "$@"; do\n  if [ "$arg" = "-MNf" ]')
    end = script.index("# -- per-call:", start)
    script = (
        script[:start]
        + """for arg in "$@"; do
  if [ "$arg" = "-MNf" ]; then
    exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" launch "$@"
  fi
done

"""
        + script[end:]
    )
    h.fake.bin.write_text(script)
    monkeypatch.setenv("SSH_CONTROL_PYTHON", sys.executable)
    monkeypatch.setenv("SSH_CONTROL_PEER", str(peer))
    if not checks:
        monkeypatch.setenv("SSH_PEER_HIDE_CHECK", "1")
    h.manager._enabled = True
    h.manager._state_dir = h.fake.state_dir
    return peer


async def test_original_manager_accepts_verified_live_master_without_closing_it(
    ssh_console, monkeypatch
):
    import socket

    h = ssh_console
    install_control_peer(h, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    h.reads = 2

    async def hold(index):
        if index == 4:
            entered.set()
            await asyncio.to_thread(release.wait, 10)

    h.before_model = hold
    task = h.submit()
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        (acceptance,) = h.manager._native_acceptances.values()
        assert acceptance.manager is h.manager
        assert acceptance.launcher.returned and acceptance.launcher.accepted
        assert acceptance.launcher.process.poll() == 0
        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(str(acceptance.control_dir / "peer-socket"))
            conn.sendall(b"check")
            assert conn.recv(20) == b"OK"
        assert h.fake.count("-MNf") == 1
        assert h.fake.count("check") >= 2
        assert h.fake.count("exit") == 0
        assert "original remote body" in str(h.gateway.payloads[3])
        assert all(live for _, _, live in h.finishes)
    finally:
        release.set()
        await task
        h.manager.close_all()


async def test_destination_helper_waits_for_its_original_process_group(
    ssh_console, monkeypatch
):
    h = ssh_console
    entered, release = h.root / "helper-live", h.root / "helper-release"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'if [ "$arg" = "-G" ]; then',
            'if [ "$arg" = "-G" ]; then\n'
            '    (touch "$SSH_HELPER_ENTERED"; while [ ! -f "$SSH_HELPER_RELEASE" ]; do sleep .05; done) >/dev/null 2>&1 &',
        )
    )
    monkeypatch.setenv("SSH_HELPER_ENTERED", str(entered))
    monkeypatch.setenv("SSH_HELPER_RELEASE", str(release))
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists()
        await asyncio.sleep(0.2)
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not h.fake.call_invocations(), "request overtook original ssh -G group"
    finally:
        release.touch()
        await task
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert h.finishes and all(live for _, _, live in h.finishes)


@pytest.mark.parametrize(
    "mode",
    [
        "cleanup_false",
        "no_terminal",
        "ambiguous_master",
        "admission_one_shot",
        "admission_session",
    ],
)
async def test_unknown_peer_proof_keeps_original_model_worker_and_storage(
    ssh_console, monkeypatch, mode
):
    import json
    import os
    import subprocess
    from pathlib import Path

    h = ssh_console
    if os.environ.get("TASK8_SSH_UNPROVEN_MODE") != mode:
        receipt = h.root / "retention.json"
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_unknown_peer_proof_keeps_original_model_worker_and_storage[{mode}]",
                "--basetemp=" + str(h.root / "child"),
            ],
            env=dict(
                os.environ, TASK8_SSH_UNPROVEN_MODE=mode, TASK8_SSH_RECEIPT=str(receipt)
            ),
            capture_output=True,
            text=True,
            timeout=40,
        )
        (h.root / "retention-child.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(receipt.read_text()) == {
            "mode": mode,
            "held": True,
            "storage_live": True,
        }
        return

    if mode == "ambiguous_master":
        install_control_peer(h, monkeypatch, checks=False)
    elif mode == "admission_session":
        enable_sessions(h, monkeypatch)
        relay = h.root / "admission-session-relay.py"
        relay.write_text("""import json, os, struct, subprocess, sys, threading
child = subprocess.Popen(["sh", "-c", sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def upload():
    try:
        while data := os.read(0, 65536):
            child.stdin.write(data)
            child.stdin.flush()
    except BrokenPipeError: pass
    finally: child.stdin.close()
threading.Thread(target=upload, daemon=True).start()
while True:
    line = child.stdout.readline()
    if not line: sys.exit(child.wait())
    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
    if b"READY " in line: break
header, admitted = struct.Struct(">IIB"), {}
while raw := child.stdout.read(header.size):
    length, request, kind = header.unpack(raw)
    body = child.stdout.read(length)
    if kind == 16:
        frame = json.loads(body)
        if frame["outcome"] == "admitted": admitted[request] = body
        elif request in admitted:
            body = admitted[request]
            raw = header.pack(len(body), request, kind)
    sys.stdout.buffer.write(raw + body); sys.stdout.buffer.flush()
sys.exit(child.wait())
""")
        h.fake.bin.write_text(
            h.fake.bin.read_text().replace(
                'exec sh -c "$remote_cmd"',
                'exec "$SSH_RELAY_PYTHON" "$SSH_TERMINAL_RELAY" "$remote_cmd"',
            )
        )
        monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
        monkeypatch.setenv("SSH_TERMINAL_RELAY", str(relay))
    else:
        relay = h.root / "terminal-relay.py"
        relay.write_text("""import json, os, sys
admitted = None
for line in sys.stdin.buffer:
    start = line.find(b"{")
    if start < 0:
        sys.stdout.buffer.write(line)
        continue
    frame = json.loads(line[start:])
    if frame["outcome"] == "admitted": admitted = frame
    if frame["outcome"] != "admitted":
        if os.environ["TASK8_SSH_UNPROVEN_MODE"] == "no_terminal": continue
        if os.environ["TASK8_SSH_UNPROVEN_MODE"] == "admission_one_shot":
            if admitted is not None: frame = admitted
        else: frame["cleanup_proven"] = False
    sys.stdout.buffer.write(line[:start] + json.dumps(frame).encode() + b"\\n")
    sys.stdout.buffer.flush()
""")
        h.fake.bin.write_text(
            h.fake.bin.read_text().replace(
                'exec sh -c "$remote_cmd"',
                'sh -c "$remote_cmd" | "$SSH_RELAY_PYTHON" "$SSH_TERMINAL_RELAY"',
            )
        )
        monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
        monkeypatch.setenv("SSH_TERMINAL_RELAY", str(relay))

    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    waiting = []
    wait = threading.Event.wait

    def observe_retention(event, timeout=None):
        native = current_ssh_call()
        if native is not None and timeout is None and native.operation._stopping:
            waiting.append((native, _guard.context.get(), threading.current_thread()))
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe_retention)
    task = h.submit()
    await until(lambda: bool(waiting) or task.done())
    assert waiting, [
        (r.operation_id, r.local_returned, r.terminal.outcome, r.proven)
        for call in h.native_returns
        if call is not None
        for r in call.requests
        if r.terminal is not None
    ]
    native, storage, worker = waiting[0]
    assert worker.is_alive() and native.operation.owner._cleanup_unproven
    assert storage.live and not native.finished and not h.finishes
    assert ("db.agent_runs", h.database.db_path) in storage.sources
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
    assert native.requests and all(r.local_returned for r in native.requests)
    if mode == "cleanup_false":
        assert all(
            r.terminal is not None and r.terminal.cleanup_proven is False
            for r in native.requests
        )
    elif mode == "no_terminal":
        assert native.requests[0].terminal is None
    elif mode in {"admission_one_shot", "admission_session"}:
        assert [json.loads(r.wire)["operation"] for r in native.requests] == [
            "ping",
            "fs_read",
        ]
        assert native.requests[0].proven
        assert native.requests[1].terminal is None
    else:
        assert native.master_launches and not native.master_launches[0].proven
        assert not h.manager._native_acceptances
        # Actual test-owned infrastructure shutdown is not native proof/release.
        h.manager.close_all()
    Path(os.environ["TASK8_SSH_RECEIPT"]).write_text(
        json.dumps({"mode": mode, "held": True, "storage_live": True})
    )
    os._exit(
        0
    )  # End only this deliberately retained test process; never reset custody.


def enable_sessions(h, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Tools import remote_session_registry as sessions

    raw = toml.loads(h.config_file.read_text())
    raw["console_ssh"]["session_worker"] = True
    h.config_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    registry = sessions.RemoteSessionRegistry()
    monkeypatch.setattr(sessions, "_REGISTRY", registry)
    return registry


async def test_lost_cancel_keeps_actual_pending_terminal_until_reader_status(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    h = ssh_console
    registry = enable_sessions(h, monkeypatch)
    relay = h.root / "session-relay.py"
    entered, release = h.root / "terminal-waiting", h.root / "release-terminal"
    # Transparent transport peer: the original bundled session/child executes.
    # Delay only request 3's actual terminal/STATUS bytes (ping/read/read).
    relay.write_text("""import os, struct, subprocess, sys, threading, time
from pathlib import Path
child = subprocess.Popen(["sh", "-c", sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def upload():
    try:
        while data := os.read(0, 65536):
            child.stdin.write(data)
            child.stdin.flush()
    except BrokenPipeError: pass
    finally: child.stdin.close()
threading.Thread(target=upload, daemon=True).start()
while True:
    line = child.stdout.readline()
    if not line: sys.exit(child.wait())
    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
    if b"READY " in line: break
header = struct.Struct(">IIB")
while raw := child.stdout.read(header.size):
    length, request, kind = header.unpack(raw)
    body = child.stdout.read(length)
    if request == 3 and kind == 16 and b'"outcome":"admitted"' not in body and b'"outcome": "admitted"' not in body:
        Path(os.environ["SSH_RELAY_ENTERED"]).touch()
        while not Path(os.environ["SSH_RELAY_RELEASE"]).exists(): time.sleep(.01)
    sys.stdout.buffer.write(raw + body); sys.stdout.buffer.flush()
sys.exit(child.wait())
""")
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'exec "$SSH_RELAY_PYTHON" "$SSH_SESSION_RELAY" "$remote_cmd"',
        )
    )
    monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
    monkeypatch.setenv("SSH_SESSION_RELAY", str(relay))
    monkeypatch.setenv("SSH_RELAY_ENTERED", str(entered))
    monkeypatch.setenv("SSH_RELAY_RELEASE", str(release))
    h.reads = 2

    # Configure the actual original executor before its first model enqueue.
    async def budget(index):
        if index == 2:
            bridge = h.rig.controller._agent_bridge
            (service,) = bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            executor = authority.workspace_executor._inner
            executor._budget_seconds = 1
            executor._ssh.transport._grace_seconds = 0.1

    h.before_model = budget
    waiting = []
    wait = threading.Event.wait

    def observe_lost_cancel(event, timeout=None):
        native = current_ssh_call()
        if (
            native is not None
            and timeout is None
            and worker is not None
            and any(event is pending.done for pending in worker._pending.values())
        ):
            waiting.append(native)
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe_lost_cancel)
    task = h.submit()
    worker = None
    locked = False
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), [p[-2:] for p in h.gateway.payloads]
        (worker,) = registry._sessions.values()
        locked = worker._write_lock.acquire(blocking=False)
        assert locked
        await until(lambda: bool(waiting))
        assert 3 in worker._pending, (
            "lost CANCEL dropped the original request observation"
        )
        assert worker._pending[3].status is None
        assert waiting[0].request.issued
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        if locked:
            worker._write_lock.release()
        release.touch()
        await task
        registry.close_all()
    assert worker._next_id == 4, "a possibly issued request was retried"
    assert len(h.fake.call_invocations()) == 1
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert all(live for _, _, live in h.finishes)


async def test_unpublished_session_keeps_original_io_until_close_returns(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    h = ssh_console
    registry = enable_sessions(h, monkeypatch)
    entered, release = h.root / "unpublished-io", h.root / "release-unpublished"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    unpublished = []
    joining = threading.Event()
    start, join = threading.Thread.start, threading.Thread.join

    def observe_original_join(thread, timeout=None):
        if (
            timeout is None
            and thread.name == "ssh-session-stderr"
            and current_ssh_call() is not None
        ):
            joining.set()
        return join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "join", observe_original_join)

    def close_before_publication(thread):
        native = current_ssh_call()
        if thread.name == "ssh-session-reader" and native is not None:
            attempt = native.session_starts[-1]
            unpublished.append(attempt)
            registry.close_key(attempt.key[0])
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", close_before_publication)
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists() and unpublished
        await until(joining.is_set)
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not unpublished[0].proven
        assert unpublished[0].worker._native_acceptance is None
        assert unpublished[0].worker._stderr_thread.is_alive()
        assert not registry._sessions
    finally:
        release.touch()
        await task
        registry.close_all()
    assert unpublished[0].proven
    assert unpublished[0].worker._proc.poll() is not None
    assert not unpublished[0].worker._stderr_thread.is_alive()
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert all(live for _, _, live in h.finishes)


@pytest.mark.parametrize("ssh_console", [{"write": True}], indirect=True)
async def test_original_edit_operation_covers_actual_cas_read_before_write(ssh_console):
    import json

    h = ssh_console
    h.edit = True
    assert (await h.submit()).accepted
    assert (
        h.root / "remote-workspace" / "alpha.txt"
    ).read_text() == "edited remote body\n"
    edit = next(
        call for call in h.native_returns if call is not None and call.name == "fs_edit"
    )
    assert [json.loads(request.wire)["operation"] for request in edit.requests] == [
        "fs_read",
        "fs_edit",
    ]
    assert all(request.proven for request in edit.requests)
    assert (
        edit.finished and h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    )
    assert all(live for _, _, live in h.finishes)
