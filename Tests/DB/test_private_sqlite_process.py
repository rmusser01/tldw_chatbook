"""Owned helper admission, bounded transport and captured-child cleanup."""

# ruff: noqa: SIM117 -- keep resource ownership and expected-failure scopes explicit

import asyncio
import importlib
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier, Event

import pytest


def api():
    return importlib.import_module("tldw_chatbook.DB.private_sqlite_process")


def deadline(seconds=2):
    return api().OperationDeadline(time.monotonic() + seconds)


def test_retained_saturation_leaves_transient_envelope_and_atomic_refusal():
    process = api()
    admission = process.HelperAdmission()
    with ExitStack() as stack:
        for _ in range(4):
            stack.enter_context(
                admission.reserve(transient=0, retained=1, deadline=deadline())
            )
        stack.enter_context(
            admission.reserve(transient=2, retained=0, deadline=deadline())
        )
        with pytest.raises(process.HelperTimeoutError):
            admission.reserve(transient=1, retained=1, deadline=deadline(0))
        # Failed pair acquisition must not have consumed its transient half.
        stack.enter_context(
            admission.reserve(transient=2, retained=0, deadline=deadline())
        )


@pytest.mark.parametrize("transient,retained,workers", [(2, 0, 2), (1, 1, 4)])
def test_whole_operation_envelopes_admit_concurrently(transient, retained, workers):
    process = api()
    admission = process.HelperAdmission()
    held = Barrier(workers + 1)
    release = Barrier(workers + 1)

    def own():
        with admission.reserve(
            transient=transient, retained=retained, deadline=deadline()
        ):
            held.wait(3)
            release.wait(3)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(own) for _ in range(workers)]
        held.wait(3)
        try:
            with pytest.raises(process.HelperTimeoutError):
                admission.reserve(transient=1, retained=0, deadline=deadline(0))
        finally:
            release.wait(3)
        for future in futures:
            future.result(3)
    with admission.reserve(transient=4, retained=4, deadline=deadline()):
        pass


@pytest.mark.parametrize(
    "transient,retained", [(5, 0), (0, 5), (-1, 0), (True, 0), (0, 0)]
)
def test_invalid_envelopes_refuse_without_waiting(transient, retained):
    with pytest.raises(api().HelperUnavailableError):
        api().HelperAdmission().reserve(
            transient=transient, retained=retained, deadline=deadline()
        )


def test_expired_deadline_refuses_even_available_capacity():
    with pytest.raises(api().HelperTimeoutError):
        api().HelperAdmission().reserve(transient=1, retained=0, deadline=deadline(0))


def request(path):
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    return codec.PrepareRequest(str(path), True, True, False)


def test_actual_preparation_and_nested_borrowing_reap(tmp_path):
    process = api()
    admission = process.HelperAdmission()
    with admission.reserve(transient=2, retained=0, deadline=deadline()) as reservation:
        with reservation.operation_scope():
            with reservation.operation_scope():
                first = process.HelperLease.start(
                    request(tmp_path / "first"),
                    operation="prepare",
                    reservation=reservation,
                    deadline=deadline(),
                )
            second = process.HelperLease.start(
                request(tmp_path / "second"),
                operation="prepare",
                reservation=reservation,
                deadline=deadline(),
            )
            response = first.initial_response
            response["result"]["artifacts"][0] = "absent"
            assert first.initial_result.artifacts[0] == "created_private"
            assert (
                first.initial_result.main_identity.ino
                == (tmp_path / "first").stat().st_ino
            )
            with pytest.raises(process.HelperUnavailableError):
                process.HelperLease.start(
                    request(tmp_path / "excess"),
                    operation="prepare",
                    reservation=reservation,
                    deadline=deadline(),
                )
            with pytest.raises(process.HelperUnavailableError):
                admission.reserve(transient=1, retained=0, deadline=deadline())
            assert not (tmp_path / "excess").exists()
            first.close()
            second.close()
    assert first.cleanup_state == second.cleanup_state == "reaped"
    with admission.reserve(transient=4, retained=4, deadline=deadline()):
        pass


def test_reservation_exit_closes_its_child(tmp_path):
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        lease = process.HelperLease.start(
            request(tmp_path / "db"),
            operation="prepare",
            reservation=owner,
            deadline=deadline(),
        )
    assert lease.cleanup_state == "reaped"


def test_source_pin_rechecks_only_original_identity(tmp_path):
    process = api()
    path = tmp_path / "db"
    path.touch(mode=0o600)
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        lease = process.HelperLease.start(
            codec.PrepareRequest(str(path), False, False, True),
            operation="pin_source",
            reservation=owner,
            deadline=deadline(),
        )
        assert lease.request("recheck_source", deadline=deadline())["status"] == "ok"
        with pytest.raises(process.HelperProtocolError):
            lease.request("prepare", deadline=deadline())
        path.rename(tmp_path / "old")
        path.touch(mode=0o600)
        assert (
            lease.request("recheck_source", deadline=deadline())["status"]
            == "private_path_error"
        )


def test_private_failure_is_closed_and_has_no_success_result(tmp_path):
    process = api()
    path = tmp_path / "link"
    path.symlink_to(tmp_path / "absent")
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        lease = process.HelperLease.start(
            request(path), operation="prepare", reservation=owner, deadline=deadline()
        )
        assert lease.initial_response["status"] == "private_path_error"
        assert str(path) not in str(lease.initial_response)
        with pytest.raises(process.HelperUnavailableError):
            _ = lease.initial_result


@pytest.fixture
def fault_child(monkeypatch):
    """Replace only process creation; actual IPC, deadlines and cleanup run."""
    real_popen = subprocess.Popen
    children = []

    def install(code):
        def launch(*args, **kwargs):
            child = real_popen([sys.executable, "-I", "-S", "-c", code], **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr(api().subprocess, "Popen", launch)
        return children

    yield install
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize(
    "code,error",
    [
        ("import os; os._exit(7)", "HelperUnavailableError"),
        ("import os; os.write(1,b'\\x00'); os._exit(0)", "HelperProtocolError"),
        (
            "import os,struct; os.write(1,struct.pack('!I',65537)); os._exit(0)",
            "HelperProtocolError",
        ),
        (
            "import os,struct; os.write(1,struct.pack('!I',2)+b'{}'); os._exit(0)",
            "HelperProtocolError",
        ),
        (
            "import os,struct; os.write(1,struct.pack('!I',20)+b'{}'); os._exit(0)",
            "HelperProtocolError",
        ),
        ("import time; time.sleep(30)", "HelperTimeoutError"),
        ("import os,time; os.write(1,b'\\x00'); time.sleep(30)", "HelperTimeoutError"),
    ],
)
def test_real_fault_children_fail_closed_and_reap(tmp_path, fault_child, code, error):
    process = api()
    children = fault_child(code)
    admission = process.HelperAdmission()
    started = time.monotonic()
    with admission.reserve(transient=1, retained=0, deadline=deadline()) as owner:
        with pytest.raises(getattr(process, error)) as caught:
            process.HelperLease.start(
                request(tmp_path / "secret"),
                operation="prepare",
                reservation=owner,
                deadline=deadline(0.15),
            )
        assert str(tmp_path) not in str(caught.value)
    assert time.monotonic() - started < 3.8
    assert len(children) == 1 and children[0].poll() is not None
    with admission.reserve(transient=4, retained=4, deadline=deadline()):
        pass


def test_oversized_request_refuses_before_launch(tmp_path, fault_child):
    children = fault_child("raise SystemExit(99)")
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        with pytest.raises(process.HelperProtocolError):
            process.HelperLease.start(
                request("x" * 65536),
                operation="prepare",
                reservation=owner,
                deadline=deadline(),
            )
    assert not children


def test_terminal_owner_keeps_one_retained_charge_after_context_exit():
    process = api()
    admission = process.HelperAdmission()
    with admission.reserve(transient=1, retained=1, deadline=deadline()) as owner:
        owner.retain_terminal_owner()
        owner.retain_terminal_owner()  # Same owner, never an additional permit.
    with admission.reserve(transient=4, retained=3, deadline=deadline()):
        with pytest.raises(process.HelperTimeoutError):
            admission.reserve(transient=0, retained=1, deadline=deadline(0))
    with pytest.raises(process.HelperUnavailableError):
        owner.retain_terminal_owner()


def test_terminal_owner_cannot_mint_retained_capacity():
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        with pytest.raises(process.HelperUnavailableError):
            owner.retain_terminal_owner()


def leaf_bootstrap():
    package = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    return f"""
import os, sys, time
from types import ModuleType
from pathlib import Path
package = Path({str(package)!r})
for name, directory in (("tldw_chatbook",package),("tldw_chatbook.DB",package/"DB"),("tldw_chatbook.Utils",package/"Utils")):
    namespace = ModuleType(name)
    namespace.__path__ = [str(directory)]
    sys.modules[name] = namespace
"""


def test_actual_child_output_backpressure_has_finite_deadline():
    # A blocking write after select would strand this child until killed.
    code = (
        leaf_bootstrap()
        + """
from tldw_chatbook.DB.private_sqlite_helper import _PrivatePipe
pipe = _PrivatePipe()
pipe.deadline = time.monotonic() + 0.2
try:
    pipe.write(b'x' * (4 * 1024 * 1024))
except TimeoutError:
    raise SystemExit(0)
raise SystemExit(9)
"""
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert child.wait(timeout=2) == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        child.stdin.close()
        child.stdout.close()


def test_shared_operation_budget_covers_admission_and_launch(
    tmp_path, monkeypatch, fault_child
):
    process = api()
    children = fault_child("import time; time.sleep(30)")
    clock = [100.0]
    monkeypatch.setattr(process.time, "monotonic", lambda: clock[0])
    operation = process.OperationDeadline(101.0)
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=operation
    ) as owner:
        clock[0] = 101.1
        with pytest.raises(process.HelperTimeoutError):
            process.HelperLease.start(
                request(tmp_path / "late"),
                operation="prepare",
                reservation=owner,
                deadline=process.OperationDeadline(200.0),
            )
    assert not children and not (tmp_path / "late").exists()


@pytest.mark.parametrize("expires_at", [130.0, None])
def test_late_sequential_preparation_uses_operation_not_admission_phase_budget(
    tmp_path, monkeypatch, expires_at
):
    process = api()
    real_clock = time.monotonic
    clock = [100.0]
    with monkeypatch.context() as patch:
        patch.setattr(process.time, "monotonic", lambda: clock[0])
        budget = process.OperationDeadline(expires_at)
        owner = process.HelperAdmission().reserve(
            transient=1, retained=0, deadline=budget
        )
        clock[0] = 110.0
        # Map the clock to real time for subprocess waits, retaining elapsed10s.
        shift = 110.0 - real_clock()
        patch.setattr(process.time, "monotonic", lambda: real_clock() + shift)
        with owner:
            lease = process.HelperLease.start(
                request(tmp_path / "late"),
                operation="prepare",
                reservation=owner,
                deadline=budget,
            )
            assert lease.initial_response["status"] == "ok"


def test_partial_events_do_not_reset_five_second_cap(
    tmp_path, fault_child, monkeypatch
):
    process = api()
    fault_child("import os,time; os.write(1,b'\\x00'); time.sleep(30)")
    real_clock = time.monotonic
    selector_type = process.selectors.DefaultSelector
    waits = []
    offset = [0.0]
    monkeypatch.setattr(process.time, "monotonic", lambda: real_clock() + offset[0])

    class PartialSelector(selector_type):
        def select(self, timeout=None):
            waits.append(timeout)
            result = super().select(min(timeout, 0.25))
            offset[0] += 3.0
            return result

    monkeypatch.setattr(process.selectors, "DefaultSelector", PartialSelector)
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline(30)
    ) as owner:
        with pytest.raises(process.HelperTimeoutError):
            process.HelperLease.start(
                request(tmp_path / "db"),
                operation="prepare",
                reservation=owner,
                deadline=deadline(30),
            )
    assert waits[0] <= 5 and waits[1] < 2.1


@pytest.mark.parametrize(
    "code",
    [
        "import time; time.sleep(30)",
        "import os,time; os.write(1,b'\\x00'); time.sleep(30)",
    ],
)
def test_stalled_large_input_is_deadline_bounded(tmp_path, fault_child, code):
    process = api()
    children = fault_child(code)
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        with pytest.raises(process.HelperTimeoutError):
            process.HelperLease.start(
                request("x" * 64000),
                operation="prepare",
                reservation=owner,
                deadline=deadline(0.1),
            )
    assert children[0].poll() is not None


def test_failed_reaping_stays_owned_and_charged_until_retry(tmp_path, monkeypatch):
    process = api()
    admission = process.HelperAdmission()
    owner = admission.reserve(transient=1, retained=0, deadline=deadline())
    lease = process.HelperLease.start(
        request(tmp_path / "db"),
        operation="prepare",
        reservation=owner,
        deadline=deadline(),
    )
    # OS wait failure cannot be induced portably. Keep the actual captured child;
    # refuse its OS operations only, then restore them for deterministic cleanup.
    with monkeypatch.context() as patch:
        patch.setattr(lease._child, "poll", lambda: None)
        patch.setattr(
            lease._child,
            "wait",
            lambda **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired("owned", 0)
            ),
        )
        patch.setattr(lease._child, "terminate", lambda: None)
        patch.setattr(lease._child, "kill", lambda: None)
        with pytest.raises(process.HelperCleanupError):
            owner.__exit__(None, None, None)
        assert lease.cleanup_state == "still_owned"
        with admission.reserve(transient=3, retained=4, deadline=deadline()):
            with pytest.raises(process.HelperTimeoutError):
                admission.reserve(transient=1, retained=0, deadline=deadline(0))
    lease.close()
    assert lease.cleanup_state == "reaped"
    with admission.reserve(transient=4, retained=4, deadline=deadline()):
        pass


@pytest.mark.parametrize(
    "control", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_cleanup_failure_does_not_mask_control_flow(tmp_path, monkeypatch, control):
    process = api()
    owner = process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    )
    lease = process.HelperLease.start(
        request(tmp_path / "db"),
        operation="prepare",
        reservation=owner,
        deadline=deadline(),
    )
    original = control()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                lease,
                "close",
                lambda: (_ for _ in ()).throw(process.HelperCleanupError()),
            )
            with pytest.raises(control) as caught, owner:
                raise original
            assert caught.value is original
            assert original.__notes__ == ["private_sqlite_helper_cleanup_failed"]
    finally:
        lease.close()


def test_fork_refuses_inherited_owners_and_pipes_without_harming_parent(tmp_path):
    process = api()
    admission = process.HelperAdmission()
    with admission.reserve(transient=1, retained=0, deadline=deadline()) as owner:
        path = tmp_path / "db"
        path.touch(mode=0o600)
        codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
        lease = process.HelperLease.start(
            codec.PrepareRequest(str(path), False, False, True),
            operation="pin_source",
            reservation=owner,
            deadline=deadline(),
        )
        pid = os.fork()
        if pid == 0:
            try:
                for action in (
                    lambda: admission.reserve(
                        transient=1, retained=0, deadline=deadline()
                    ),
                    lambda: owner.__enter__(),
                    lambda: lease.request("recheck_source", deadline=deadline()),
                    lease.close,
                ):
                    try:
                        action()
                    except process.HelperUnavailableError:
                        continue
                    os._exit(8)
                os._exit(0)
            except BaseException:  # noqa: BLE001 - fork child must not run pytest finalizers
                os._exit(9)
        assert os.waitpid(pid, 0)[1] == 0
        assert lease.request("recheck_source", deadline=deadline())["status"] == "ok"


def test_normal_and_early_failure_cycles_do_not_grow_children_or_fds(tmp_path):
    process = api()
    admission = process.HelperAdmission()
    fd_root = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
    before = len(os.listdir(fd_root))
    children = []
    for index in range(20):
        path = tmp_path / str(index)
        if index % 2:
            path.symlink_to(tmp_path / "absent")
        with admission.reserve(transient=1, retained=0, deadline=deadline()) as owner:
            lease = process.HelperLease.start(
                request(path),
                operation="prepare",
                reservation=owner,
                deadline=deadline(),
            )
            children.append(lease._child)
        assert lease.cleanup_state == "reaped"
    assert all(child.poll() is not None for child in children)
    assert len(os.listdir(fd_root)) == before


@pytest.mark.parametrize("inherit_write_end", [False, True])
def test_parent_exit_ends_actual_helper_even_with_inherited_pipe(
    tmp_path, inherit_write_end
):
    witness = """
import os,select,sys
ready,_,_=select.select([int(sys.argv[1])],[],[],2.5)
if ready and os.read(int(sys.argv[1]),1)==b'':
    print('helper_eof',flush=True)
else:
    print('helper_still_alive',flush=True)
"""
    code = (
        leaf_bootstrap()
        + f"""
import subprocess
from tldw_chatbook.DB.private_sqlite_process import HelperAdmission, HelperLease, OperationDeadline
from tldw_chatbook.DB.private_sqlite_protocol import PrepareRequest
owner = HelperAdmission().reserve(transient=1,retained=0,deadline=OperationDeadline(None))
lease = HelperLease.start(PrepareRequest({str(tmp_path / "db")!r},True,True,False),operation='prepare',reservation=owner,deadline=OperationDeadline(None))
read_fd = lease._child.stdout.fileno()
write_fd = lease._child.stdin.fileno()
subprocess.Popen([sys.executable,'-I','-S','-c',{witness!r},str(read_fd)],pass_fds=(read_fd,write_fd) if {inherit_write_end!r} else (read_fd,),stdin=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print('parent_ready',flush=True)
os._exit(0)
"""
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=4)
        assert child.returncode == 0 and not stderr
        assert stdout.splitlines() == [b"parent_ready", b"helper_eof"]
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


def test_force_cleanup_signals_only_captured_child(tmp_path, fault_child):
    sibling = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        children = fault_child(
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        )
        process = api()
        with process.HelperAdmission().reserve(
            transient=1, retained=0, deadline=deadline()
        ) as owner:
            with pytest.raises(process.HelperTimeoutError):
                process.HelperLease.start(
                    request(tmp_path / "db"),
                    operation="prepare",
                    reservation=owner,
                    deadline=deadline(0.15),
                )
        assert children[0].returncode == -signal.SIGKILL
        assert sibling.poll() is None
    finally:
        sibling.kill()
        sibling.wait(timeout=3)


def test_retained_handoff_releases_transient_capacity_but_keeps_live_child(
    tmp_path, monkeypatch
):
    process = api()
    # Task4 adds the fixed TTS initializer. Exercise its resource classification
    # with an actual pinned helper, without adding a test operation to production.
    monkeypatch.setitem(process._LEASE_KINDS, "pin_source", "retained")
    admission = process.HelperAdmission()
    path = tmp_path / "db"
    path.touch(mode=0o600)
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    with admission.reserve(transient=1, retained=1, deadline=deadline()) as owner:
        lease = process.HelperLease.start(
            codec.PrepareRequest(str(path), False, False, True),
            operation="pin_source",
            reservation=owner,
            deadline=deadline(),
        )
        owner.handoff_retained(lease)
    try:
        with admission.reserve(transient=4, retained=3, deadline=deadline()):
            assert (
                lease.request("recheck_source", deadline=deadline())["status"] == "ok"
            )
            with pytest.raises(process.HelperTimeoutError):
                admission.reserve(transient=0, retained=1, deadline=deadline(0))
    finally:
        lease.close()
    with admission.reserve(transient=4, retained=4, deadline=deadline()):
        pass


def test_transient_helper_cannot_be_handed_off(tmp_path):
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline()
    ) as owner:
        lease = process.HelperLease.start(
            request(tmp_path / "db"),
            operation="prepare",
            reservation=owner,
            deadline=deadline(),
        )
        with pytest.raises(process.HelperUnavailableError):
            owner.handoff_retained(lease)


@pytest.mark.parametrize("operation", [None, [], "tts_exact_current", "recheck_source"])
def test_invalid_launch_operation_is_source_free(tmp_path, operation):
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        with pytest.raises(process.HelperProtocolError):
            process.HelperLease.start(
                request(tmp_path / "db"),
                operation=operation,
                reservation=owner,
                deadline=deadline(),
            )


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), "secret"])
def test_invalid_deadline_refuses_source_free(expires_at):
    with pytest.raises(api().HelperUnavailableError):
        api().OperationDeadline(expires_at)


@pytest.mark.parametrize("terminal", [False, True])
def test_retained_handoff_exception_and_terminal_reap_keep_correct_charge(
    tmp_path, monkeypatch, terminal
):
    process = api()
    monkeypatch.setitem(process._LEASE_KINDS, "pin_source", "retained")
    admission = process.HelperAdmission()
    path = tmp_path / "db"
    path.touch(mode=0o600)
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    owner = admission.reserve(transient=1, retained=1, deadline=deadline())
    lease = process.HelperLease.start(
        codec.PrepareRequest(str(path), False, False, True),
        operation="pin_source",
        reservation=owner,
        deadline=deadline(),
    )
    owner.handoff_retained(lease)
    if terminal:
        owner.__exit__(None, None, None)
        lease._child.kill()
        # Retain the exact owner before independently reaping lost live proof.
        lease.retain_terminal_owner()
        lease.close()
        assert lease.cleanup_state == "terminal_retained"
        with admission.reserve(transient=4, retained=3, deadline=deadline()):
            with pytest.raises(process.HelperTimeoutError):
                admission.reserve(transient=0, retained=1, deadline=deadline(0))
    else:
        original = KeyboardInterrupt()
        owner.__exit__(KeyboardInterrupt, original, None)
        assert lease.cleanup_state == "reaped"
        with admission.reserve(transient=4, retained=4, deadline=deadline()):
            pass


@pytest.mark.asyncio
async def test_cancelled_async_waiter_does_not_abandon_shielded_owner(tmp_path):
    process = api()
    ready = Event()
    release = Event()
    pids = []

    def worker():
        with process.HelperAdmission().reserve(
            transient=1, retained=0, deadline=deadline()
        ) as owner:
            lease = process.HelperLease.start(
                request(tmp_path / "db"),
                operation="prepare",
                reservation=owner,
                deadline=deadline(),
            )
            pids.append(lease._child.pid)
            ready.set()
            assert release.wait(3)
        return lease.cleanup_state

    owned = asyncio.create_task(asyncio.to_thread(worker))

    async def wait_for_owner():
        return await asyncio.shield(owned)

    waiter = asyncio.create_task(wait_for_owner())
    try:
        assert await asyncio.to_thread(ready.wait, 3)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not owned.done()
        os.kill(pids[0], 0)
    finally:
        release.set()
    assert await asyncio.wait_for(owned, 3) == "reaped"


def test_unsolicited_extra_reply_is_rejected(tmp_path, fault_child):
    # Closed metadata is still invalid when two replies answer one request.
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    path = tmp_path / "db"
    path.touch(mode=0o600)
    identity = codec.FileIdentity.from_stat(path.stat()).to_payload()
    frame = codec.encode_frame(
        {
            "version": 1,
            "operation": "prepare",
            "status": "ok",
            "result": {
                "main_identity": identity,
                "artifacts": ["already_private", "absent", "absent", "absent"],
            },
        }
    )
    fault_child(f"import os,time; os.write(1,{frame!r}*2); time.sleep(30)")
    process = api()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=deadline()
    ) as owner:
        with pytest.raises(process.HelperProtocolError):
            process.HelperLease.start(
                request(path),
                operation="prepare",
                reservation=owner,
                deadline=deadline(),
            )


def test_transient_recheck_cannot_extend_enclosing_operation_budget(
    tmp_path, monkeypatch
):
    process = api()
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    path = tmp_path / "db"
    path.touch(mode=0o600)
    operation = deadline()
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=operation
    ) as owner:
        lease = process.HelperLease.start(
            codec.PrepareRequest(str(path), False, False, True),
            operation="pin_source",
            reservation=owner,
            deadline=operation,
        )
        with monkeypatch.context() as patch:
            patch.setattr(process.time, "monotonic", lambda: operation.expires_at + 1)
            with pytest.raises(process.HelperTimeoutError):
                lease.request(
                    "recheck_source", deadline=process.OperationDeadline(None)
                )
