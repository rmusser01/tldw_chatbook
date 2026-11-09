"""Real selected Git wrappers cannot retire their original native custody early."""

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Workspaces.test_native_change_review_initialization import (  # noqa: F401
    native_initializer,
    wait_until,
)
from Tests.Workspaces.test_native_change_review_ownership import (  # noqa: F401
    native_review,
    register,
)
from tldw_chatbook.Workspaces.change_review_ownership import _SnapshotTicket
from tldw_chatbook.Workspaces.change_tracking import ShadowRepo
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def selected_wrapper(tmp_path, monkeypatch, request):
    if os.name != "posix":
        pytest.skip("native process-group custody is qualified on POSIX")
    launch = getattr(request, "param", "version")
    command = "add" if launch.startswith("final-") else "init"
    launch = launch.removeprefix("final-")
    real_git = shutil.which("git")
    assert real_git
    directory = tmp_path / "selected-wrapper"
    directory.mkdir()
    entered, release, returned = (
        directory / name for name in ("entered", "release", "returned")
    )
    armed = directory / "armed"
    armed.touch()
    child = directory / "child.py"
    child.write_text(
        "import os,time\nfrom pathlib import Path\n"
        f"Path({str(entered)!r}).write_text(str(os.getpid()))\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
    )
    wrapper = directory / "git"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,subprocess,sys,time\nfrom pathlib import Path\n"
        f"directory=Path({str(directory)!r})\nargs=sys.argv[1:]\n"
        f"selected=({launch!r} == 'version' and args == ['--version']) or "
        f"({launch!r} == 'config' and '--name-only' in args) or "
        f"({launch!r} == 'command' and {command!r} in args)\n"
        "held=False\n"
        "if selected and (directory/'armed').exists():\n"
        " try:\n  (directory/'once').mkdir()\n  held=True\n"
        " except FileExistsError: pass\n"
        "if held:\n"
        f" subprocess.Popen([{sys.executable!r},{str(child)!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        " while not (directory/'entered').exists(): time.sleep(.01)\n"
        f"result=subprocess.run([{real_git!r},*args],capture_output=True)\n"
        "sys.stdout.buffer.write(result.stdout)\nsys.stderr.buffer.write(result.stderr)\n"
        "if held: (directory/'returned').write_text(str(os.getpid()))\n"
        "sys.exit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    env = ShadowRepo._env
    monkeypatch.setattr(
        ShadowRepo,
        "_env",
        lambda repo: {
            **env(repo),
            "HOME": str(directory),
            "XDG_CONFIG_HOME": str(directory),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    state = SimpleNamespace(**locals())
    try:
        yield state
    finally:
        release.touch()
        if entered.exists():
            pid = int(entered.read_text())
            wait_until(lambda: not process_exists(pid))


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize(
    "selected_wrapper", ["version", "config", "command"], indirect=True
)
@pytest.mark.parametrize("producer", ["initializer", "snapshot"])
def test_managed_descendant_keeps_original_worker_and_opaque_claim(
    selected_wrapper, request, monkeypatch, producer
):
    """Removing tree drain lets the real host return while its child is alive."""
    wrapper = selected_wrapper
    host = request.getfixturevalue(
        "native_initializer" if producer == "initializer" else "native_review"
    )
    tickets = []
    original = _SnapshotTicket.__init__

    def observe(ticket, *args, **kwargs):
        original(ticket, *args, **kwargs)
        tickets.append(ticket)

    monkeypatch.setattr(_SnapshotTicket, "__init__", observe)
    reservation = None
    try:
        if producer == "initializer":
            host.registry.set_change_review_enabled("init-workspace", True)
            host.service.admit_turn("init-workspace")
        else:
            reservation = register(host)
        wait_until(wrapper.entered.exists)
        wait_until(wrapper.returned.exists)
        wait_until(lambda: not process_exists(int(wrapper.returned.read_text())))
        ticket = tickets[-1]
        assert not ticket.retired.wait(3), (
            "direct-child exit retired a live managed descendant"
        )
        assert ticket.owner.is_admitted
        assert host.runtime.execution_capacity.snapshot().tool_workers == 1
        assert process_exists(int(wrapper.entered.read_text()))
        with host.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
    finally:
        wrapper.release.touch()
        if producer == "initializer":
            wait_until(lambda: host.service._queue.unfinished_tasks == 0)
        elif reservation is not None:
            assert reservation.await_baseline(10)
            host.coordinator.cancel(reservation)
    wait_until(lambda: not host.runtime.execution_capacity.snapshot().executions)


@pytest.mark.parametrize("producer", ["initializer", "snapshot"])
def test_root_drain_waits_for_operation_finish_return(request, monkeypatch, producer):
    """Removing the ticket's final join exposes positive drain inside finish()."""
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation

    host = request.getfixturevalue(
        "native_initializer" if producer == "initializer" else "native_review"
    )
    entered, release = threading.Event(), threading.Event()
    tickets = []
    original_init, original_finish = _SnapshotTicket.__init__, OwnedOperation.finish

    def observe(ticket, *args, **kwargs):
        original_init(ticket, *args, **kwargs)
        tickets.append(ticket)

    def held_finish(operation):
        original_finish(operation)
        if tickets and operation is tickets[0].operation:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(_SnapshotTicket, "__init__", observe)
    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    reservation = None
    try:
        if producer == "initializer":
            host.registry.set_change_review_enabled("init-workspace", True)
            host.service.admit_turn("init-workspace")
        else:
            reservation = register(host)
        assert entered.wait(10)
        ticket = tickets[0]
        assert ticket.owner.is_admitted, (
            "root drained before original operation.finish returned"
        )
        with host.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        assert not ticket.retired.is_set()
    finally:
        release.set()
        if producer == "initializer":
            wait_until(lambda: host.service._queue.unfinished_tasks == 0)
        elif reservation is not None:
            assert reservation.await_baseline(10)
            host.coordinator.cancel(reservation)


@pytest.mark.parametrize("producer", ["initializer", "snapshot"])
def test_unqualified_platform_keeps_ordinary_change_review(
    request, monkeypatch, producer
):
    """Unqualified attachment must not advertise native ownership or refuse ordinary Git."""
    from tldw_chatbook.Utils import platform_files
    from tldw_chatbook.Workspaces.change_review_consent import RootReadinessState

    monkeypatch.setattr(
        platform_files, "supports_native_process_groups", lambda: False
    )
    host = request.getfixturevalue(
        "native_initializer" if producer == "initializer" else "native_review"
    )
    if producer == "initializer":
        host.registry.set_change_review_enabled("init-workspace", True)
        host.service.admit_turn("init-workspace")
        wait_until(lambda: host.service._queue.unfinished_tasks == 0)
        assert (
            host.service.status("init-workspace").roots[0].state
            is RootReadinessState.READY
        )
        assert host.service._native is None
        assert not host.service._native_expected
    else:
        reservation = register(host)
        assert reservation.await_baseline(10)
        host.coordinator.cancel(reservation)
        assert host.coordinator._native is None
    assert host.runtime._execution_capacity is None


@pytest.mark.parametrize("selected_wrapper", ["version"], indirect=True)
@pytest.mark.asyncio
async def test_timeout_and_cancelled_disposal_keep_original_storage_until_late_proof(
    selected_wrapper, request, monkeypatch
):
    """A failed stop cannot release scopes while the original controller can still drain."""
    import asyncio

    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Notes.git_process_containment import ProcessTreeController
    from tldw_chatbook.Workspaces import change_tracking

    wrapper = selected_wrapper
    host = request.getfixturevalue("native_initializer")
    tickets = []
    entered = threading.Event()
    original_init, original_wait = _SnapshotTicket.__init__, ProcessTreeController.wait

    def observe(ticket, *args, **kwargs):
        original_init(ticket, *args, **kwargs)
        tickets.append(ticket)

    def unavailable_stop(controller, tree):
        raise OSError("controlled stop unavailable")

    async def wait(controller, tree, *, timeout):
        if timeout == float("inf"):
            entered.set()
        return await original_wait(controller, tree, timeout=timeout)

    monkeypatch.setattr(_SnapshotTicket, "__init__", observe)
    monkeypatch.setattr(ProcessTreeController, "terminate", unavailable_stop)
    monkeypatch.setattr(ProcessTreeController, "kill", unavailable_stop)
    monkeypatch.setattr(ProcessTreeController, "wait", wait)
    monkeypatch.setattr(change_tracking, "_GIT_TIMEOUT_SECONDS", 0.2)
    with storage_admission._lock:
        before = set(storage_admission._live_leases)
    host.registry.set_change_review_enabled("init-workspace", True)
    host.service.admit_turn("init-workspace")
    disposal = None
    try:
        assert entered.wait(5)
        ticket = tickets[-1]
        assert ticket.operation._stopping and not ticket.owner._cleanup_unproven
        with storage_admission._lock:
            retained = storage_admission._live_leases - before
        assert len(retained) >= 3
        disposal = asyncio.create_task(host.runtime.dispose(timeout_seconds=0))
        for _ in range(2):
            await asyncio.sleep(0.05)
            disposal.cancel()
        await asyncio.sleep(0.05)
        assert not disposal.done()
        assert ticket.owner.is_admitted and not ticket.retired.is_set()
        assert process_exists(int(wrapper.entered.read_text()))
        with storage_admission._lock:
            assert retained <= storage_admission._live_leases
    finally:
        wrapper.release.touch()
        if disposal is not None:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(disposal, 10)
    assert ticket.retired.is_set()
    assert not ticket.owner.is_admitted
    with storage_admission._lock:
        assert retained.isdisjoint(storage_admission._live_leases)


def test_unavailable_cleanup_proof_retains_original_worker_in_isolated_process(
    request, monkeypatch, tmp_path
):
    """Irrecoverable proof loss must hold custody; this probe deliberately never drains."""
    import subprocess

    receipt = os.environ.get("TLDW_GIT_CUSTODY_UNPROVEN_PROBE")
    if receipt is None:
        result_path = tmp_path / "unknown.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                __file__
                + "::test_unavailable_cleanup_proof_retains_original_worker_in_isolated_process",
            ],
            env={**os.environ, "TLDW_GIT_CUSTODY_UNPROVEN_PROBE": str(result_path)},
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            result_path.read_text()
            if result_path.exists()
            else result.stdout.decode(errors="replace")
            + result.stderr.decode(errors="replace")
        )
        evidence = json.loads(result_path.read_text())
        assert evidence["retained"] and not evidence["native_cleanup_proven"]
        assert not process_exists(evidence["child_pid"])
        assert not process_exists(evidence["probe_pid"])
        print("Unknown cleanup observation:", json.dumps(evidence, sort_keys=True))
        return

    try:
        from tldw_chatbook.Backup_Recovery import storage_admission
        from tldw_chatbook.Notes.git_process_containment import ProcessTreeController
        from tldw_chatbook.Workspaces import change_tracking

        wrapper = request.getfixturevalue("selected_wrapper")
        host = request.getfixturevalue("native_initializer")
        tickets = []
        original = _SnapshotTicket.__init__

        def observe(ticket, *args, **kwargs):
            original(ticket, *args, **kwargs)
            tickets.append(ticket)

        async def unavailable(controller, tree, *, timeout):
            raise OSError("controlled observation unavailable")

        monkeypatch.setattr(_SnapshotTicket, "__init__", observe)
        monkeypatch.setattr(ProcessTreeController, "wait", unavailable)
        monkeypatch.setattr(change_tracking, "_GIT_TIMEOUT_SECONDS", 5)
        with storage_admission._lock:
            before = set(storage_admission._live_leases)
        host.registry.set_change_review_enabled("init-workspace", True)
        host.service.admit_turn("init-workspace")
        wait_until(lambda: tickets and tickets[-1].owner._cleanup_unproven)
        ticket = tickets[-1]
        wrapper.release.touch()
        child_pid = int(wrapper.entered.read_text())
        wait_until(lambda: not process_exists(child_pid))
        host.service.shutdown(timeout=0)
        assert ticket.operation._stopping and ticket.owner.is_admitted
        assert not ticket.retired.is_set() and host.service._queue.unfinished_tasks == 1
        with storage_admission._lock:
            assert len(storage_admission._live_leases - before) >= 3
        with host.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        Path(receipt).write_text(
            json.dumps(
                {
                    "retained": True,
                    "native_cleanup_proven": False,
                    "child_pid": child_pid,
                    "probe_pid": os.getpid(),
                }
            )
        )
        # This process owns the intentionally nonreturning daemon and its private DBs.
        # Its managed child is gone; process exit is NOT reported as native drain.
        os._exit(0)
    except BaseException:  # noqa: BLE001 -- the isolated nonreturning probe must report and exit
        import traceback

        failure = traceback.format_exc()
        Path(receipt).write_text(failure)
        if "wrapper" in locals():
            wrapper.release.touch()
        os.write(2, failure.encode())
        os._exit(1)


@pytest.mark.parametrize(
    "selected_wrapper",
    ["final-version", "final-config", "final-command"],
    indirect=True,
)
def test_final_snapshot_child_and_original_publisher_both_hold_custody(
    selected_wrapper, request, monkeypatch
):
    """Final publication must join its own Git tree and original host return."""
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    wrapper = selected_wrapper
    wrapper.armed.unlink()
    host = request.getfixturevalue("native_review")
    reservation = register(host)
    assert reservation.await_baseline(10)
    wait_until(lambda: reservation.baseline_completion.value != "pending")
    wait_until(lambda: not host.runtime.execution_capacity.snapshot().executions)
    before = reservation._handle.baselines[str(host.root)]
    filename = "café\nimage.bin"
    changed_bytes = b"\x00after\xfe"
    (host.root / filename).write_bytes(changed_bytes)
    (host.root / "a.txt").write_text("after\n")
    run_id = host.bridge._db.create_run(conversation_id="capture", agent_kind="primary")
    tickets, published_owners = [], []
    entered, release = threading.Event(), threading.Event()
    original_init, publish = _SnapshotTicket.__init__, host.coordinator._publish

    def observe(ticket, *args, **kwargs):
        original_init(ticket, *args, **kwargs)
        tickets.append(ticket)

    def held_publish(item):
        published_owners.append(current_execution_owner())
        entered.set()
        assert release.wait(10)
        return publish(item)

    monkeypatch.setattr(_SnapshotTicket, "__init__", observe)
    monkeypatch.setattr(host.coordinator, "_publish", held_publish)
    wrapper.armed.touch()
    host.coordinator.finalize(reservation, run_id=run_id, kind="turn")
    try:
        wait_until(wrapper.entered.exists)
        wait_until(wrapper.returned.exists)
        wait_until(lambda: not process_exists(int(wrapper.returned.read_text())))
        ticket = tickets[-1]
        assert not entered.wait(0.2)
        assert ticket.owner.is_admitted and not ticket.retired.is_set()
        with host.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        wrapper.release.touch()
        assert entered.wait(10)
        assert published_owners == [ticket.owner]
        assert ticket.owner.is_admitted and not ticket.retired.is_set()
        assert not process_exists(int(wrapper.entered.read_text()))
    finally:
        wrapper.release.touch()
        release.set()
    wait_until(ticket.retired.is_set)
    assert reservation.completion.value == "published"
    repo = host.tracker.service.repo_for_root(host.root)
    assert repo.file_bytes(before, "a.txt") == b"before\n"
    end = reservation._handle.end_shas[str(host.root)]
    assert repo.file_bytes(end, filename) == changed_bytes
    assert host.bridge._db.change_snapshots_for_run(run_id)
    assert not ticket.owner.is_admitted
