"""Real selected-profile admission with native process-local Console scratch."""

import json
import os
import pwd
import sys
import tempfile
from functools import partial
from pathlib import Path


def native_writer_control(result_path, mode):
    from Tests import network_guard, real_profile_guard

    home = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    assert home == {"HOME": pwd.getpwuid(os.getuid()).pw_dir, "USERPROFILE": None}
    real_profile_guard.install()
    network_guard.install()
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    root = Path(
        tempfile.mkdtemp(prefix="native-scratch-", dir=os.environ["TMPDIR"])
    ).resolve()
    profile = root / "profile"
    profile.mkdir(mode=0o700)
    (profile / "data").mkdir(mode=0o700)
    selector = profile / "config.toml"
    selector.write_text(
        f'[general]\nusers_name="scratch"\n[paths]\ndata_dir="{profile / "data"}"\n[database]\nworkspaces_db_path="{profile / "workspaces.db"}"\n[model_catalog]\nauto_refresh=false\n'
    )
    selector.chmod(0o600)
    os.environ["TLDW_CONFIG_PATH"] = str(selector)
    from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths

    bootstrap.default_bootstrap_root = lambda: root / "bootstrap"
    bootstrap.default_config_path = lambda: selector
    profile_paths.default_config_path = lambda: selector
    profile_paths.default_base_data_dir = lambda: profile / "data"
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    if mode in {"bound", "controls"}:
        authority = admission_authority(root / "bootstrap")
        authority.register("scratch.profile", (profile,))
        bind_profile(
            root / "bootstrap", selector, ("scratch.profile",), authority.control_root
        )
    assert mode in {"bound", "ordinary", "controls"}
    from tldw_chatbook import config
    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.run_log import MANIFEST_NAME, RunLogWriter
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    assert config.get_user_data_dir().is_relative_to(profile)
    selected_before = selector.read_bytes()
    runs_path = profile / "agent_runs.db"
    manager = ConsoleScratchSpaceManager(temp_parent=root)
    snapshot = manager.snapshot("native-scratch-control")
    writer = RunLogWriter(
        root=snapshot.root, access_scope=partial(manager.lease, snapshot)
    )
    report = {"mode": mode, "root": str(root), "entry_home": home}
    database = lease = None
    try:
        lease = ConsoleExecutionOwner.acquire(runs_path)
        database = AgentRunsDB(
            runs_path, "native-scratch-control", execution_owner=lease
        )
        service = AgentService(database, ToolCatalogRegistry(), run_log_writer=writer)
        assert activation._explicit_writer_root(service) == snapshot.root
        with activation.execution(
            service, sources=(("agents.history", snapshot.root),)
        ):
            state = activation._guard.context.get()
            current = state.leases[selector]
            _bootstrap, names = current.execution_context(selector)
            report["names"] = names
            assert names == (
                ("bootstrap.unbound",) if mode == "ordinary" else ("scratch.profile",)
            )
            writer.bind("native-scratch-run")
            assert writer.is_active and writer.log_dir.is_relative_to(snapshot.root)
            assert (
                writer.append(
                    run_id="native-scratch-run",
                    kind="primary",
                    type="model",
                    content="private native log",
                )
                == 1
            )
            writer.write_manifest({"scope": "native-control"})
            if mode != "controls":
                writer.close()
            assert (writer.log_dir / MANIFEST_NAME).is_file()
            assert (
                b"private native log"
                in next(writer.log_dir.glob("logs.*.txt")).read_bytes()
            )
        if mode == "controls":
            _native_controls(service, manager, snapshot, writer, report)
            with activation.execution(service):
                writer.close()
        report.update(completed=True, names=names)
    finally:
        if database is not None:
            database.close()
        if lease is not None:
            lease.close()
        report["scratch_retired"] = manager.dispose()
        report.update(
            selector_unchanged=selector.read_bytes() == selected_before,
            profile_guard=real_profile_guard.take_violations(),
            network_guard=network_guard.blocked_attempts(),
            exit_home={key: os.environ.get(key) for key in home},
        )
        result_path.write_text(json.dumps(report, indent=2))
    assert report["scratch_retired"] and not snapshot.root.exists()
    assert report["selector_unchanged"] and report["entry_home"] == report["exit_home"]
    assert not report["profile_guard"] and not report["network_guard"]


def _native_controls(service, manager, snapshot, writer, report):
    """Real native worker admission; fixed callback/custody negative controls."""
    import asyncio
    import contextvars
    import signal
    import threading
    import time
    from contextlib import nullcontext
    from dataclasses import replace

    import pytest

    from tldw_chatbook.Agents import activation, scratch_admission
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import AgentService, _call_with_timeout
    from tldw_chatbook.Agents.run_log import RunLogWriter
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_scratch_space import (
        ConsoleScratchSpaceManager,
        ConsoleScratchSpaceUnavailable,
    )

    checks = report["controls"] = []

    def refused(callback):
        with pytest.raises(
            activation.AgentActivationRequired, match="^agent_activation_required$"
        ):
            callback()

    def enter(candidate):
        with activation.execution(candidate):
            raise AssertionError("refused entry yielded")

    for label, access in (
        ("custom_access", lambda: nullcontext()),
        ("copied_snapshot", partial(manager.lease, replace(snapshot))),
    ):
        other = RunLogWriter(root=snapshot.root, access_scope=access)
        other_service = AgentService(
            service.db, ToolCatalogRegistry(), run_log_writer=other
        )
        # Harmless service entry has no scratch source until bind. The actual
        # writer effect must still take the unchanged ordinary storage path.
        before_files = {
            path.relative_to(snapshot.root): path.read_bytes()
            for path in snapshot.root.rglob("*")
            if path.is_file()
        }
        with activation.execution(other_service):
            other.bind("refused-custom-writer")
            assert other._bind_attempted and not other.is_active
            assert (
                other.append(
                    run_id="refused-custom-writer",
                    kind="primary",
                    type="model",
                    content="must not persist",
                )
                is None
            )
        assert other.log_dir is None
        assert before_files == {
            path.relative_to(snapshot.root): path.read_bytes()
            for path in snapshot.root.rglob("*")
            if path.is_file()
        }
        checks.append(label)

    # An already active outer lease must not mask a modified contextmanager cell.
    wrapper = ConsoleScratchSpaceManager.lease
    cell = next(
        cell
        for cell in wrapper.__closure__
        if cell.cell_contents is wrapper.__wrapped__
    )
    original = cell.cell_contents
    with manager.lease(snapshot):
        try:
            cell.cell_contents = lambda *_args, **_kwargs: nullcontext(snapshot.root)
            refused(lambda: enter(service))
        finally:
            cell.cell_contents = original
    checks.append("mutated_contextmanager_closure")
    kw = RunLogWriter._write_bytes.__kwdefaults__
    assert kw and "sync" in kw
    old = kw["sync"]
    try:
        kw["sync"] = not old
        refused(lambda: enter(service))
    finally:
        kw["sync"] = old
    checks.append("inplace_kwdefaults")

    errors = []
    with activation.execution(service):
        with (
            pytest.raises(activation.AgentActivationRequired),
            scratch_admission.writer_storage(
                writer, snapshot.root / "not-a-native-log"
            ),
        ):
            raise AssertionError("arbitrary scratch path admitted")
        copied = contextvars.copy_context()

        def foreign():
            try:
                enter(service)
            except BaseException as error:  # noqa: BLE001 -- preserve the actual foreign-thread failure
                errors.append(error)

        thread = threading.Thread(target=lambda: copied.run(foreign))
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
    assert len(errors) == 1 and type(errors[0]) is activation.AgentActivationRequired
    checks.extend(("non_log_path", "copied_context_thread"))

    # Actual native tool timeout worker: cancel only the waiter, then prove the
    # entered worker owns its own real leases and can finish after parent exit.
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    threads, worker_frames = [], []

    def effect():
        threads.append(threading.current_thread())
        worker_frames.append(scratch_admission._current.get())
        entered.set()
        assert release.wait(8)
        with activation.execution(service):
            assert (
                writer.append(
                    run_id="native-scratch-run",
                    kind="primary",
                    type="model",
                    content="after cancelled waiter",
                )
                is not None
            )
        finished.set()
        return ToolResult(ok=True, content="done")

    try:
        with activation.execution(service):
            parent_frame = scratch_admission._current.get()
            result = _call_with_timeout(
                effect,
                5,
                "scratch-control",
                should_cancel=entered.is_set,
                execution_owner=service,
            )
            assert (
                entered.is_set()
                and not result.ok
                and result.dispatch_state == "uncertain"
            )
        assert not parent_frame.live and worker_frames[0] is not parent_frame
        assert worker_frames[0].live
        with manager._condition:
            assert manager._matching_record_locked(snapshot).leases > 0
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
    assert finished.is_set() and not worker_frames[0].live
    checks.append("native_tool_cancel_independent_worker")

    async def async_controls():
        entered, release = asyncio.Event(), asyncio.Event()
        frames = []
        with activation.execution(service):
            parent = scratch_admission._current.get()

            async def copied_task():
                refused(lambda: enter(service))

            await asyncio.create_task(copied_task())

            @activation.async_worker_guard(service)
            async def actual_worker():
                frames.append(scratch_admission._current.get())
                entered.set()
                await release.wait()
                with activation.execution(service):
                    assert (
                        writer.append(
                            run_id="native-scratch-run",
                            kind="primary",
                            type="model",
                            content="after async parent exit",
                        )
                        is not None
                    )

            task = asyncio.create_task(actual_worker())
            await asyncio.wait_for(entered.wait(), 5)
        try:
            assert not parent.live and frames[0] is not parent and frames[0].live
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        assert not frames[0].live

    asyncio.run(async_controls())
    checks.extend(("copied_context_task", "native_async_independent_worker"))

    # Fork while another thread owns the actual manager lock. Every child
    # lifecycle must refuse BEFORE that inherited lock, without parent deletion.
    held, unlock = threading.Event(), threading.Event()

    def hold_lock():
        with manager._condition:
            held.set()
            assert unlock.wait(10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(5)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            outcomes = []
            calls = (
                lambda: manager.snapshot("fork-child"),
                lambda: manager.lease(snapshot).__enter__(),
                lambda: manager.is_live(snapshot),
                lambda: manager.close("native-scratch-control"),
                manager.tombstone_all,
                lambda: manager.wait_for_cleanup(0.01),
                manager.dispose,
            )
            for call in calls:
                try:
                    call()
                except ConsoleScratchSpaceUnavailable:
                    outcomes.append("refused")
                else:
                    outcomes.append("unexpected")
            try:
                enter(service)
            except activation.AgentActivationRequired:
                outcomes.append("agent_refused")
            os.write(write_fd, json.dumps(outcomes).encode())
            os._exit(0)
        except BaseException as error:  # noqa: BLE001 -- child must report then exit, never run parent teardown
            os.write(
                write_fd,
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "completed_calls": outcomes,
                    }
                ).encode(),
            )
            os._exit(3)
    os.close(write_fd)
    status = None
    try:
        until = time.monotonic() + 5
        while time.monotonic() < until:
            observed, code = os.waitpid(child, os.WNOHANG)
            if observed:
                status = code
                break
            time.sleep(0.02)
        if status is None:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            raise AssertionError("fork child blocked on inherited manager lock")
        outcomes = json.loads(os.read(read_fd, 4096))
        report["fork_child"] = {"status": status, "outcomes": outcomes}
        assert status == 0, outcomes
        assert outcomes == ["refused"] * 7 + ["agent_refused"], outcomes
    finally:
        os.close(read_fd)
        unlock.set()
        holder.join(5)
        assert not holder.is_alive()
    assert manager.is_live(snapshot) and snapshot.root.exists()
    checks.append("fork_pid_before_lock_and_no_parent_cleanup")

    stale = manager.snapshot("stale-control")
    stale_writer = RunLogWriter(
        root=stale.root, access_scope=partial(manager.lease, stale)
    )
    stale_service = AgentService(
        service.db, ToolCatalogRegistry(), run_log_writer=stale_writer
    )
    manager.close("stale-control")
    refused(lambda: enter(stale_service))
    checks.append("tombstoned_snapshot")


if __name__ == "__main__":
    native_writer_control(Path(sys.argv[1]), sys.argv[2])
