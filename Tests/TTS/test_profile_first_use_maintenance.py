"""Cold ordinary authority initialization over actual private marker/native work."""

import pytest
from Tests.TTS.test_profile_repository_maintenance import _run_private_child


def _first_use_child(root, mode):
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Backup_Recovery import control_records as records
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    marker, release = threading.Event(), threading.Event()
    original = records._write
    selected = root / "selected"
    selected.write_bytes(b"ordinary selected bytes")
    recursive = []

    class Control(BaseException):
        pass

    def write(authority, name, data):
        original(authority, name, data)
        if name == "unbound-owner":
            if mode == "recursive":
                try:
                    storage.acquire_storage(selected)
                except storage.bootstrap.RecoveryRequired as error:
                    recursive.append(str(error))
            marker.set()
            assert release.wait(4)
            if mode == "failure":
                raise OSError("actual marker written; registration not started")
            if mode == "control":
                raise Control("actual marker written; registration interrupted")

    records._write = write
    if mode == "unqualified":
        storage.qualified_for = lambda *args: (False, "native unavailable")
        with storage.acquire_storage(selected) as lease:
            assert lease._key is None
        assert not marker.is_set() and not storage._pending_acquisitions
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(storage.acquire_storage, selected)
        assert marker.wait(3)
        follower = pool.submit(storage.acquire_storage, selected)
        deadline = time.monotonic() + 2
        while len(storage._pending_acquisitions) != 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(storage._pending_acquisitions) == 2
        assert not follower.done()
        pause = None
        try:
            if mode == "pause":
                pause = storage._begin_local_pause()
                with pytest.raises(
                    storage.bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    follower.result(timeout=2)
                assert not pause.drain(time.monotonic() + 0.02)
            if mode == "fork":
                pid = os.fork()
                if pid == 0:
                    try:
                        storage.acquire_storage(selected)
                    except storage.bootstrap.RecoveryRequired as error:
                        os._exit(
                            0 if str(error) == "forked_owner_restart_required" else 4
                        )
                    os._exit(5)
                _, status = os.waitpid(pid, 0)
                assert os.waitstatus_to_exitcode(status) == 0
        finally:
            release.set()
        if mode in {"failure", "control"}:
            expected = (
                Control if mode == "control" else storage.bootstrap.RecoveryRequired
            )
            with pytest.raises(expected):
                leader.result(timeout=3)
            with pytest.raises(
                storage.bootstrap.RecoveryRequired, match="recovery_scope_uncertain"
            ):
                follower.result(timeout=3)
            # No live initializer remains to justify waiting or repair.
            with pytest.raises(
                storage.bootstrap.RecoveryRequired, match="recovery_scope_uncertain"
            ):
                storage.acquire_storage(selected)
        elif mode == "pause":
            with pytest.raises(
                storage.bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                leader.result(timeout=3)
            assert pause.drain(time.monotonic() + 0.2)
            pause.resume()
            with storage.acquire_storage(selected):
                pass
        else:
            first, second = leader.result(timeout=3), follower.result(timeout=3)
            assert first is not second and first._key == second._key
            first.close()
            assert second in storage._live_leases
            second.close()
        if mode == "recursive":
            assert recursive == ["recursive_authority_initialization"]
    assert not storage._pending_acquisitions
    assert not storage._live_leases
    assert selected.read_bytes() == b"ordinary selected bytes"


@pytest.mark.parametrize(
    "mode",
    ["normal", "pause", "failure", "control", "recursive", "unqualified", "fork"],
)
def test_cold_first_authority_initialization_is_coordinated_without_repair(
    tmp_path, mode
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_first_use_maintenance import _first_use_child
_first_use_child(Path(sys.argv[1]), sys.argv[2])
""",
        mode,
    )
