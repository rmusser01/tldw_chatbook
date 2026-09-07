from __future__ import annotations

import importlib
import multiprocessing
from pathlib import Path
from typing import Any


def _coordinator_module():
    return importlib.import_module("tldw_chatbook.Notes.notes_sync_coordinator")


def _no_private_conflict(_root: Path) -> None:
    """Isolate the OS-lock race from spawned config bootstrap I/O."""

    return None


def _race_for_root(
    root: str,
    lock_directory: str,
    mutation_log: str,
    start: Any,
    ready: Any,
    release_owner: Any,
    results: Any,
) -> None:
    module = _coordinator_module()
    coordinator = module.NotesSyncRootCoordinator(Path(lock_directory))
    ready.set()
    if not start.wait(10.0):
        results.put("start_timeout")
        return
    admission = coordinator.try_acquire(
        Path(root),
        private_conflict=_no_private_conflict,
    )
    results.put((admission.state.value, admission.reason_code))
    try:
        admission.require_authority("write")
    except module.RootAuthorityError:
        pass
    else:
        Path(mutation_log).write_text("owner\n", encoding="utf-8")
    if admission.state is module.RootAdmissionState.OWNER:
        release_owner.wait(10.0)
        coordinator.release(admission.lease)


def _hold_root_until_killed(
    root: str,
    lock_directory: str,
    acquired: Any,
) -> None:
    module = _coordinator_module()
    coordinator = module.NotesSyncRootCoordinator(Path(lock_directory))
    admission = coordinator.try_acquire(
        Path(root),
        private_conflict=_no_private_conflict,
    )
    if admission.state is module.RootAdmissionState.OWNER:
        acquired.set()
        multiprocessing.Event().wait(30.0)


def _stop_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.kill()
        process.join(5.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)


def test_two_spawned_processes_racing_one_root_have_exactly_one_owner(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    mutation_log = tmp_path / "mutations.txt"
    start = context.Event()
    release_owner = context.Event()
    ready = (context.Event(), context.Event())
    results = context.Queue()
    processes = [
        context.Process(
            target=_race_for_root,
            args=(
                str(root),
                str(lock_directory),
                str(mutation_log),
                start,
                ready[index],
                release_owner,
                results,
            ),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    try:
        assert all(signal.wait(10.0) for signal in ready)
        start.set()
        outcomes = [results.get(timeout=10.0) for _ in processes]
        states = sorted(state for state, _reason in outcomes)
        assert states == sorted(
            [
                module.RootAdmissionState.OWNER.value,
                module.RootAdmissionState.PASSIVE.value,
            ]
        ), outcomes
        assert mutation_log.read_text(encoding="utf-8") == "owner\n"
    finally:
        release_owner.set()
        for process in processes:
            process.join(5.0)
            _stop_process(process)
        results.close()


def test_forced_process_death_releases_os_lock(tmp_path: Path) -> None:
    module = _coordinator_module()
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    acquired = context.Event()
    process = context.Process(
        target=_hold_root_until_killed,
        args=(str(root), str(lock_directory), acquired),
    )
    process.start()
    try:
        assert acquired.wait(10.0), (
            f"child did not acquire lock; exit={process.exitcode}"
        )
        process.kill()
        process.join(10.0)
        assert not process.is_alive()

        coordinator = module.NotesSyncRootCoordinator(lock_directory)
        admission = coordinator.try_acquire(
            root,
            private_conflict=_no_private_conflict,
        )
        assert admission.state is module.RootAdmissionState.OWNER
        coordinator.release(admission.lease)
    finally:
        _stop_process(process)
