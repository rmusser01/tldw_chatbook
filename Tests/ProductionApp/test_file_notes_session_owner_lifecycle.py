"""Production-app lifecycle proof for the File Notes process owner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from textual.app import App

from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
)
from tldw_chatbook.Notes.file_notes_session_owner import FileNotesSessionOwner
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)
from Tests.Notes.test_file_notes_git_service import (
    _StubbornProcess,
    _change,
    _entry,
    _ownership,
    _repository_at,
    _single_group,
)
from Tests.UI.test_screen_navigation import _build_test_app


@dataclass
class _OwnerProbe:
    events: list[str]
    shutdown_calls: int = 0

    async def shutdown_async(self) -> None:
        self.shutdown_calls += 1
        self.events.append("git-owner-settled")


@dataclass
class _ReplicaWorkspaceProbe:
    events: list[str]
    shutdown_calls: int = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.events.append("replica-closed")


class _FailingOwnerProbe:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def shutdown_async(self) -> None:
        self.events.append("git-owner-failed")
        raise RuntimeError("forced owner shutdown failure")


class _BlockingOwnerProbe:
    def __init__(
        self,
        events: list[str],
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.events = events
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def shutdown_async(self) -> None:
        self.events.append("git-owner-started")
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            self.events.append("git-owner-failed")
            raise self.failure
        self.events.append("git-owner-settled")


async def _wait_for_library(app, pilot) -> LibraryScreen:
    for _ in range(300):
        if isinstance(app.screen, LibraryScreen):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("production TldwCli did not mount LibraryScreen")


@pytest.mark.asyncio
async def test_file_notes_owner_settles_before_mounted_library_replica() -> None:
    """The private Textual shutdown hook must run before screen teardown."""
    events: list[str] = []
    owner = _OwnerProbe(events)
    workspace = _ReplicaWorkspaceProbe(events)
    app = _build_test_app(configured_default="library")
    app.app_config["_first_run"] = False
    app.file_notes_session_owner = owner

    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _wait_for_library(app, pilot)
        screen._library_file_notes_workspace = workspace

    assert owner.shutdown_calls == 1
    assert workspace.shutdown_calls == 1
    assert events.index("git-owner-settled") < events.index("replica-closed")


@pytest.mark.asyncio
async def test_file_notes_owner_settles_when_library_never_mounted() -> None:
    """App ownership is independent of whether a Library screen existed."""
    events: list[str] = []
    owner = _OwnerProbe(events)
    app = _build_test_app(configured_default="chat")
    app.app_config["_first_run"] = False
    app.file_notes_session_owner = owner

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, LibraryScreen)

    assert owner.shutdown_calls == 1
    assert events == ["git-owner-settled"]


@pytest.mark.asyncio
async def test_file_notes_owner_failure_still_runs_textual_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner failure is preserved only after superclass shutdown is attempted."""
    events: list[str] = []
    app = _build_test_app(configured_default="chat")
    app.file_notes_session_owner = _FailingOwnerProbe(events)

    async def shutdown_textual(_app) -> None:
        events.append("textual-shutdown")

    monkeypatch.setattr(App, "_shutdown", shutdown_textual)

    with pytest.raises(RuntimeError, match="forced owner shutdown failure"):
        await app._shutdown()

    assert events == ["git-owner-failed", "textual-shutdown"]


@pytest.mark.asyncio
async def test_app_shutdown_cancellation_waits_for_owner_before_textual_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation is delayed until the retained owner has settled."""
    events: list[str] = []
    owner = _BlockingOwnerProbe(events)
    app = _build_test_app(configured_default="chat")
    app.file_notes_session_owner = owner

    async def shutdown_textual(_app) -> None:
        assert events[-1] == "git-owner-settled"
        events.extend(("textual-shutdown", "screen-closed", "replica-closed"))

    monkeypatch.setattr(App, "_shutdown", shutdown_textual)
    shutdown = asyncio.create_task(app._shutdown())
    await owner.started.wait()

    shutdown.cancel("first shutdown cancellation")
    await asyncio.sleep(0)
    shutdown.cancel("second shutdown cancellation")
    await asyncio.sleep(0)
    events_before_release = tuple(events)
    shutdown_done_before_release = shutdown.done()

    owner.release.set()
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await shutdown

    assert events_before_release == ("git-owner-started",)
    assert not shutdown_done_before_release
    assert cancellation.value.args == ("first shutdown cancellation",)
    assert events == [
        "git-owner-started",
        "git-owner-settled",
        "textual-shutdown",
        "screen-closed",
        "replica-closed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["owner", "textual"])
async def test_app_shutdown_cancellation_preserves_non_cancellation_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    """Owner and superclass failures remain primary over caller cancellation."""
    events: list[str] = []
    owner = _BlockingOwnerProbe(
        events,
        failure=(
            RuntimeError("forced owner failure")
            if failure_phase == "owner"
            else None
        ),
    )
    app = _build_test_app(configured_default="chat")
    app.file_notes_session_owner = owner

    async def shutdown_textual(_app) -> None:
        events.append("textual-shutdown")
        if failure_phase == "textual":
            raise ValueError("forced Textual failure")

    monkeypatch.setattr(App, "_shutdown", shutdown_textual)
    shutdown = asyncio.create_task(app._shutdown())
    await owner.started.wait()
    shutdown.cancel("shutdown cancellation")
    await asyncio.sleep(0)
    owner.release.set()

    expected_error = RuntimeError if failure_phase == "owner" else ValueError
    with pytest.raises(expected_error) as failure:
        await shutdown

    terminal_event = (
        "git-owner-failed" if failure_phase == "owner" else "git-owner-settled"
    )
    assert events.index(terminal_event) < events.index("textual-shutdown")
    assert any(
        "cancellation" in note.lower()
        for note in getattr(failure.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_app_shutdown_settles_retained_child_after_forced_workspace_unmount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The app owner, never the workspace, settles one uncertain Git child."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    child = _StubbornProcess()
    subprocess_calls = 0

    async def create_subprocess_exec(*_argv, **_kwargs):
        nonlocal subprocess_calls
        subprocess_calls += 1
        return child

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        create_subprocess_exec,
    )
    unlinked_index_locks: list[Path] = []
    real_unlink = Path.unlink

    def unlink(path: Path, *args, **kwargs) -> None:
        if path.name == "index.lock":
            unlinked_index_locks.append(path)
            return
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)

    owner = FileNotesSessionOwner()
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    owner.attach_git_service(service)
    binding = owner.select_root(root)
    repository = _repository_at(root)
    group = _single_group("note.md")
    assert owner.record_change(binding, _change(1, "modified", "note.md").change)
    assert owner.publish_trust(binding, repository)
    assert owner.publish_ownership(
        binding,
        {1: _ownership(group, {"note.md": _entry("note.md")})},
    )

    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=None,
        session_owner=owner,
    )
    app = _build_test_app(configured_default="library")
    app.app_config["_first_run"] = False
    app.file_notes_session_owner = owner
    status_waiter = None

    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _wait_for_library(app, pilot)
        screen._library_file_notes_workspace = workspace
        status_waiter = service.start_status(
            binding,
            (_change(1, "modified", "note.md"),),
        )
        await child.communicate_started.wait()

        await workspace.shutdown()
        assert subprocess_calls == 1
        assert child.terminate_calls == 0
        assert child.kill_calls == 0

    assert status_waiter is not None
    status = await status_waiter
    assert status.state in {"stale", "unavailable", "error"}
    assert subprocess_calls == 1
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_calls == 2
    assert not owner.snapshot(binding).staging_ownership
    assert unlinked_index_locks == []
