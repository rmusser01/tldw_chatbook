"""Reversible File Notes fences preserve real file publication and Git ownership."""

import asyncio
import os
import sys
import threading
import time

import pytest

from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitStatusAdmissionError,
)
from tldw_chatbook.Notes.file_notes_service import FileNotesService
from tldw_chatbook.Notes.file_notes_session_owner import FileNotesSessionOwner


@pytest.fixture
def notes(tmp_path):
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path)
    service = FileNotesService(
        tmp_path, None, session_owner=owner, session_binding=binding
    )
    return owner, binding, service


@pytest.mark.asyncio
async def test_pause_fences_real_file_source_and_preserves_session(notes, tmp_path):
    owner, binding, service = notes
    assert service.create_file("before.md", "durable").succeeded
    git = FileNotesGitService(owner)
    owner.attach_git_service(git)
    before = owner.snapshot(binding)
    owner._maintenance_close_admission()
    with pytest.raises(RuntimeError, match="maintenance"):
        service.create_file("blocked.md", "do not publish")
    assert not (tmp_path / "blocked.md").exists()
    assert owner.try_acquire_transition(binding, "path") is None
    assert owner.admit_mutation(binding).lease is None
    assert owner.admit_status(binding).lease is None
    assert await owner._maintenance_drain(time.monotonic() + 1)
    assert owner.snapshot(binding) == before
    assert owner.attached_git_service() is git
    owner._maintenance_resume()
    assert service.create_file("after.md", "resumed").succeeded
    assert (tmp_path / "before.md").read_text() == "durable"
    await owner.shutdown_async()


@pytest.mark.asyncio
async def test_admitted_disk_publication_finishes_without_cancel(
    notes, tmp_path, monkeypatch
):
    owner, binding, service = notes
    entered, release = threading.Event(), threading.Event()
    original = service._finish_published_file

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_finish_published_file", gated)
    worker = asyncio.create_task(
        asyncio.to_thread(service.create_file, "note.md", "kept")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        owner._maintenance_close_admission()
        assert not await owner._maintenance_drain(time.monotonic())
        waiter = asyncio.create_task(owner._maintenance_drain(time.monotonic() + 2))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not worker.done()
        release.set()
        assert (await worker).succeeded
        assert await owner._maintenance_drain(time.monotonic() + 1)
        assert (tmp_path / "note.md").read_text() == "kept"
        assert owner.snapshot(binding).changes[0].change.relative_path == "note.md"
    finally:
        release.set()
        await worker
        owner._maintenance_resume()


@pytest.mark.asyncio
async def test_existing_root_and_mutation_leases_block_drain(notes):
    owner, binding, _ = notes
    lease = owner.admit_mutation(binding).lease
    owner._maintenance_close_admission()
    assert not await owner._maintenance_drain(time.monotonic())
    lease.release()
    assert await owner._maintenance_drain(time.monotonic() + 1)
    owner._maintenance_resume()
    stable = owner.acquire_stable_root(None)
    owner._maintenance_close_admission()
    try:
        assert not await owner._maintenance_drain(time.monotonic())
    finally:
        stable.release()
    assert await owner._maintenance_drain(time.monotonic() + 1)
    owner._maintenance_resume()


@pytest.mark.asyncio
async def test_real_git_child_is_not_cancelled_to_drain(notes, tmp_path):
    owner, _, _ = notes
    runner = AsyncGitProcessRunner()
    git = FileNotesGitService(owner, runner=runner)
    owner.attach_git_service(git)
    started, release = tmp_path / "started", tmp_path / "release"
    code = (
        "import pathlib,time,sys; pathlib.Path(sys.argv[1]).touch(); "
        "\nwhile not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)"
    )
    child = asyncio.create_task(
        runner.run(
            [sys.executable, "-c", code, str(started), str(release)],
            cwd=str(tmp_path),
            environment=dict(os.environ),
            timeout=3,
        )
    )
    try:
        for _ in range(200):
            if started.exists():
                break
            await asyncio.sleep(0.005)
        assert started.exists()
        owner._maintenance_close_admission()
        assert not await owner._maintenance_drain(time.monotonic())
        assert not child.done()
        release.touch()
        assert (await child).returncode == 0
        assert await owner._maintenance_drain(time.monotonic() + 1)
        owner._maintenance_resume()
        assert (await git.discover(owner.current_binding())).state == "not_repository"
    finally:
        release.touch()
        await child
        await owner.shutdown_async()


@pytest.mark.asyncio
async def test_terminal_shutdown_cannot_resume(notes):
    owner, _, _ = notes
    with pytest.raises(RuntimeError, match="maintenance"):
        await owner._maintenance_drain(time.monotonic())
    owner._maintenance_close_admission()
    owner.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        owner._maintenance_resume()


@pytest.mark.asyncio
async def test_pending_replica_move_blocks_drain_without_losing_bytes(
    notes, tmp_path, monkeypatch
):
    owner, binding, service = notes
    assert service.create_file("old.md", "pending recovery bytes").succeeded
    original = service._load_file

    def failed_refresh(_path):
        raise OSError("injected move refresh failure")

    with monkeypatch.context() as patch:
        patch.setattr(service, "_load_file", failed_refresh)
        assert service.move_file("old.md", "new.md").succeeded
    before = owner.snapshot(binding)
    owner._maintenance_close_admission()
    assert not await owner._maintenance_drain(time.monotonic())
    assert (tmp_path / "new.md").read_text() == "pending recovery bytes"
    assert service._pending_replica_moves == {"old.md": "new.md"}
    owner._maintenance_resume()
    assert owner.snapshot(binding) == before
    assert original("new.md").body == "pending recovery bytes"


@pytest.mark.asyncio
async def test_quarantine_survives_refused_drain_and_resume(notes):
    from Tests.Notes.test_file_notes_session_owner import _publish_uncertain_commit

    owner, binding, _ = notes
    _, _, capability = _publish_uncertain_commit(owner, binding)
    before = owner.snapshot(binding)
    owner._maintenance_close_admission()
    assert not await owner._maintenance_drain(time.monotonic())
    assert owner.admit_commit_recovery(binding, capability).lease is None
    assert owner.snapshot(binding) == before
    owner._maintenance_resume()
    assert owner.snapshot(binding) == before
    recovery = owner.admit_commit_recovery(binding, capability)
    assert recovery.lease is not None
    recovery.lease.release()


@pytest.mark.asyncio
async def test_direct_git_query_is_fenced_and_existing_query_settles(
    notes, monkeypatch
):
    owner, binding, _ = notes
    git = FileNotesGitService(owner)
    owner.attach_git_service(git)
    entered, release = asyncio.Event(), asyncio.Event()
    original = git._run_discovery

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(git, "_run_discovery", gated)
    query = asyncio.create_task(git.discover(binding))
    await entered.wait()
    try:
        owner._maintenance_close_admission()
        assert not await owner._maintenance_drain(time.monotonic())
        with pytest.raises(RuntimeError, match="maintenance"):
            await git.discover(binding)
        with pytest.raises(GitStatusAdmissionError, match="maintenance"):
            git.start_status(binding, ())
        release.set()
        assert (await query).state == "not_repository"
        assert await owner._maintenance_drain(time.monotonic() + 1)
        assert owner.attached_git_service() is git
    finally:
        release.set()
        await query
        owner._maintenance_resume()
        await owner.shutdown_async()


@pytest.mark.asyncio
async def test_admitted_commit_can_publish_uncertainty_while_paused(notes):
    from Tests.Notes.test_file_notes_session_owner import (
        _capture_commit_authority,
        _prepare_commit_authority,
    )
    from tldw_chatbook.Notes.file_notes_git_commit import CommitRecoveryProjection
    from tldw_chatbook.Notes.file_notes_session_owner import CommitPublication

    owner, binding, _ = notes
    repository, _, sequences, _ = _prepare_commit_authority(owner, binding)
    lease, capture = _capture_commit_authority(owner, binding, repository, sequences)
    owner._maintenance_close_admission()
    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            state="uncertain",
            recovery_projection=CommitRecoveryProjection(
                message="Retained exact recovery", can_check_again=True
            ),
        ),
    )
    lease.release()
    assert publication.published
    assert owner.snapshot(binding).commit_recovery is not None
    assert not await owner._maintenance_drain(time.monotonic())
    owner._maintenance_resume()
    assert owner.snapshot(binding).commit_recovery.message == "Retained exact recovery"
