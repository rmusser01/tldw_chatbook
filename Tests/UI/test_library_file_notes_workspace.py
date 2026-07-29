"""Focused mounted tests for Library File Notes."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, TextArea, Tree

# Avoid importing the unrelated optional MLX stack during focused UI tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module  # noqa: E402
from tldw_chatbook.config import ConfigMutationResult  # noqa: E402
from tldw_chatbook.Library.library_shell_state import (  # noqa: E402
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_session_owner import (  # noqa: E402
    FileNotesSessionOwner,
)
from tldw_chatbook.Notes.file_notes_service import (  # noqa: E402
    FileNotesService,
    OperationResult,
    ReconcileResult,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen  # noqa: E402
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    LibraryFileNotesWorkspace,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.app_factory import _build_test_app  # noqa: E402


class _WorkspaceHarness(App[None]):
    """Mount one retained workspace without the rest of Library."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        yield self.workspace


class _TwoWorkspaceHarness(App[None]):
    """Mount two workspaces that share one process owner."""

    def __init__(
        self,
        first: LibraryFileNotesWorkspace,
        second: LibraryFileNotesWorkspace,
    ) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def compose(self) -> ComposeResult:
        with Vertical(id="first-workspace-host"):
            yield self.first
        with Vertical(id="second-workspace-host"):
            yield self.second


class _DynamicWorkspaceHarness(App[None]):
    """Mount a second workspace after the first is already running."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        with Vertical(id="primary-workspace-host"):
            yield self.workspace
        yield Vertical(id="dynamic-workspace-host")


def test_workspace_transition_admission_is_exact_binding_and_idempotent(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path / "notes",
        replica=None,
        session_owner=owner,
    )
    workspace._session_binding = binding

    release = workspace.acquire_transition("source")
    assert callable(release)
    assert owner.try_acquire_mutation(binding) is None
    release()
    release()

    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    assert workspace.acquire_transition("screen") is False
    mutation.release()


def test_reconcile_tolerates_projection_disappearing_during_unmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = LibraryFileNotesWorkspace(root=None, replica=None)

    def missing_root_surface(*, offline: bool | None = None) -> None:
        del offline
        raise NoMatches("root surface was removed during unmount")

    monkeypatch.setattr(workspace, "_update_root_surface", missing_root_surface)

    applied = workspace._apply_reconcile(
        ReconcileResult(status="ok"),
        ("deleted.md",),
    )

    assert applied is False
    assert workspace.entries == {}
    assert workspace._deleted_paths == ("deleted.md",)


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    message: str,
    *,
    attempts: int = 150,
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(message)


def _static_text(workspace: LibraryFileNotesWorkspace, selector: str) -> str:
    widget = workspace.query_one(selector)
    renderable = widget.label if isinstance(widget, Button) else widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _tree_labels(tree: Tree) -> list[str]:
    labels: list[str] = []

    def visit(node) -> None:
        label = getattr(node.label, "plain", str(node.label))
        labels.append(label)
        for child in node.children:
            visit(child)

    visit(tree.root)
    return labels


def _replace_editor_text(editor: TextArea, text: str) -> None:
    editor.select_all()
    editor.replace(text, editor.selection.start, editor.selection.end)


def _delayed_call(call):
    started = threading.Event()
    release = threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        release.wait(5)
        return call(*args, **kwargs)

    return delayed, started, release


def _event_loop_heartbeat(
    event_loop: asyncio.AbstractEventLoop,
    blocked: threading.Event,
    *release_on_failure: threading.Event,
) -> tuple[threading.Thread, threading.Event, list[bool]]:
    checked = threading.Event()
    observations: list[bool] = []

    def check() -> None:
        heartbeat_ran = threading.Event()
        if blocked.wait(timeout=5):
            event_loop.call_soon_threadsafe(heartbeat_ran.set)
            observations.append(heartbeat_ran.wait(timeout=1))
        else:
            observations.append(False)
        checked.set()
        if not observations[-1]:
            for release in release_on_failure:
                release.set()

    return threading.Thread(target=check, daemon=True), checked, observations


def _root_transition_workspace(tmp_path: Path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    return old_root, new_root, owner, replica, workspace


@pytest.mark.asyncio
async def test_empty_offline_and_persisted_root_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica = FileNotesReplica(":memory:")
    empty = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(empty).run_test() as pilot:
        await pilot.pause()
        assert empty.query_one("#file-notes-choose-root", Button).display
        assert (
            _static_text(empty, "#file-notes-root-status") == "Choose a notes folder."
        )

        root = tmp_path / "chosen"
        root.mkdir()
        saved: list[tuple[str, str, str]] = []

        def save_root_mutation(
            section_values: dict[str, dict[str, str]],
        ) -> ConfigMutationResult:
            saved.append(
                (
                    "file_notes",
                    "root",
                    section_values["file_notes"]["root"],
                )
            )
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            workspace_module,
            "apply_settings_mutation_to_cli_config",
            save_root_mutation,
            raising=False,
        )
        assert await empty.set_root(root)
        assert saved == [("file_notes", "root", str(root.resolve()))]
    replica.close()

    missing_root = tmp_path / "missing"
    offline_replica = FileNotesReplica(":memory:")
    offline = LibraryFileNotesWorkspace(root=missing_root, replica=offline_replica)
    async with _WorkspaceHarness(offline).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: offline.initialized,
            "offline root scan did not finish",
        )
        assert not missing_root.exists()
        assert "Offline" in _static_text(offline, "#file-notes-root-status")
    offline_replica.close()

    persisted_root = tmp_path / "persisted"
    persisted_root.mkdir()
    (persisted_root / "kept.md").write_text("persisted body", encoding="utf-8")
    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(persisted_root) if (section, key) == ("file_notes", "root") else default
        ),
    )
    persisted_replica = FileNotesReplica(":memory:")
    persisted = LibraryFileNotesWorkspace(replica=persisted_replica)
    async with _WorkspaceHarness(persisted).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: persisted.initialized and "kept.md" in persisted.entries,
            "persisted root was not scanned",
        )
        assert persisted.root == persisted_root.resolve()
        assert "kept.md" in _tree_labels(persisted.query_one("#file-notes-tree", Tree))
    persisted_replica.close()


@pytest.mark.asyncio
async def test_root_transition_retains_and_freezes_old_document_until_scan_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    old_root.mkdir()
    (old_root / "old.md").write_text("old body", encoding="utf-8")
    new_root = tmp_path / "new"
    new_root.mkdir()
    (new_root / "new.md").write_text("new body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )
    original_scan = FileNotesService.scan
    scan_started = threading.Event()
    release_scan = threading.Event()

    def delayed_scan(service):
        if service.root == new_root.resolve():
            scan_started.set()
            release_scan.wait(5)
        return original_scan(service)

    monkeypatch.setattr(FileNotesService, "scan", delayed_scan)
    async with _WorkspaceHarness(workspace).run_test(size=(110, 36)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("old.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "saved before root change")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "root-change draft did not become dirty",
        )
        transition = asyncio.create_task(
            workspace.set_root(new_root, persist=False)
        )
        await _wait_until(
            pilot,
            scan_started.is_set,
            "candidate root scan did not start",
        )
        assert workspace.root == old_root.resolve()
        assert workspace.current_path == "old.md"
        assert editor.text == "saved before root change"
        assert editor.read_only
        assert workspace.query_one("#file-notes-new", Button).disabled
        assert workspace.query_one("#file-notes-search", Input).disabled
        assert (old_root / "old.md").read_text(encoding="utf-8") == (
            "saved before root change"
        )

        release_scan.set()
        assert await transition
        assert workspace.root == new_root.resolve()
        assert workspace.current_path == ""
        assert editor.text == ""
        assert "new.md" in workspace.entries
    release_scan.set()
    replica.close()


@pytest.mark.asyncio
async def test_root_transition_rebinds_after_owned_replica_reopens(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica_path=tmp_path / "owned.sqlite",
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "initial scan did not finish",
        )
        old_service = workspace._service
        assert old_service is not None
        old_service.close()
        workspace._replica = None
        workspace._service = None

        assert await workspace.set_root(new_root, persist=False)
        service = workspace._service
        assert service is not None
        assert service.create_file("new.md", "new").status == "ok"
        workspace._refresh_session_changes()
        assert (
            _static_text(workspace, "#file-notes-session-changes")
            == "Session Git (1)"
        )

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_delayed_old_workspace_cannot_replace_current_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    new_root = (tmp_path / "new").resolve()
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    old_replica = FileNotesReplica(":memory:")
    new_replica = FileNotesReplica(":memory:")
    old_canonical_started = threading.Event()
    release_old_canonical = threading.Event()
    real_canonical_root = LibraryFileNotesWorkspace._canonical_root

    def delayed_canonical_root(value: object) -> Path | None:
        canonical = real_canonical_root(value)
        if canonical == old_root and not old_canonical_started.is_set():
            old_canonical_started.set()
            assert release_old_canonical.wait(timeout=5)
        return canonical

    monkeypatch.setattr(
        LibraryFileNotesWorkspace,
        "_canonical_root",
        staticmethod(delayed_canonical_root),
    )
    old_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=old_replica,
        session_owner=owner,
        poll_interval=10,
    )
    new_workspace = LibraryFileNotesWorkspace(
        root=new_root,
        replica=new_replica,
        session_owner=owner,
        poll_interval=10,
    )
    old_workspace._active = True
    old_initialization = asyncio.create_task(old_workspace._initialize())

    try:
        assert await asyncio.to_thread(old_canonical_started.wait, 1)
        async with _WorkspaceHarness(new_workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: new_workspace.initialized,
                "current workspace did not initialize",
            )
            current_service = new_workspace._service
            current_binding = new_workspace._session_binding
            assert current_service is not None
            assert current_binding is not None
            assert current_service.create_file("before.md", "before").status == "ok"

            release_old_canonical.set()
            await old_initialization

            assert current_service.create_file("after.md", "after").status == "ok"
            assert [
                item.change.relative_path
                for item in owner.snapshot(current_binding).changes
            ] == ["before.md", "after.md"]
            assert current_service.session_changes == tuple(
                item.change for item in owner.snapshot(current_binding).changes
            )
    finally:
        release_old_canonical.set()
        if not old_initialization.done():
            await old_initialization
        old_workspace._active = False
        await old_workspace.shutdown()
        await new_workspace.shutdown()
        owner.shutdown()
        old_replica.close()
        new_replica.close()


@pytest.mark.asyncio
async def test_overlapping_root_persistence_only_winner_updates_config_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    slow_root = (tmp_path / "slow").resolve()
    winner_root = (tmp_path / "winner").resolve()
    old_root.mkdir()
    slow_root.mkdir()
    winner_root.mkdir()
    owner = FileNotesSessionOwner()
    slow_replica = FileNotesReplica(":memory:")
    winner_replica = FileNotesReplica(":memory:")
    slow_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=slow_replica,
        session_owner=owner,
        poll_interval=10,
    )
    winner_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=winner_replica,
        session_owner=owner,
        poll_interval=10,
    )
    slow_scan_started = threading.Event()
    release_slow_scan = threading.Event()
    real_scan = FileNotesService.scan
    persisted_roots: list[str] = []

    def delayed_scan(service: FileNotesService):
        if service.root == slow_root:
            slow_scan_started.set()
            assert release_slow_scan.wait(timeout=5)
        return real_scan(service)

    def persist_mutation(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persisted_roots.append(section_values["file_notes"]["root"])
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(FileNotesService, "scan", delayed_scan)
    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist_mutation,
        raising=False,
    )
    slow_transition: asyncio.Task[bool] | None = None
    try:
        async with _TwoWorkspaceHarness(
            slow_workspace,
            winner_workspace,
        ).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: slow_workspace.initialized and winner_workspace.initialized,
                "shared-root workspaces did not initialize",
            )
            old_service = slow_workspace._service
            old_binding = slow_workspace._session_binding
            assert old_service is not None
            assert old_binding is not None
            assert winner_workspace._session_binding == old_binding

            slow_transition = asyncio.create_task(
                slow_workspace.set_root(slow_root)
            )
            await _wait_until(
                pilot,
                slow_scan_started.is_set,
                "slow candidate scan did not start",
            )
            assert await winner_workspace.set_root(winner_root)
            winner_binding = winner_workspace._session_binding
            assert winner_binding is not None

            release_slow_scan.set()
            assert not await slow_transition

            assert persisted_roots == [str(winner_root)]
            assert owner.current_binding() == winner_binding
            assert winner_workspace.root == winner_root
            assert slow_workspace.root == old_root
            assert slow_workspace._service is old_service
            assert slow_workspace._session_binding == old_binding
    finally:
        release_slow_scan.set()
        if slow_transition is not None and not slow_transition.done():
            await slow_transition
        await slow_workspace.shutdown()
        await winner_workspace.shutdown()
        owner.shutdown()
        slow_replica.close()
        winner_replica.close()


@pytest.mark.parametrize("selection_timing", ("during", "after"))
@pytest.mark.asyncio
async def test_fresh_shared_workspace_follows_committed_owner_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_timing: str,
) -> None:
    old_root = (tmp_path / "old").resolve()
    winner_root = (tmp_path / "winner").resolve()
    old_root.mkdir()
    winner_root.mkdir()
    (winner_root / "winner.md").write_text("winner", encoding="utf-8")
    owner = FileNotesSessionOwner()
    winner_replica = FileNotesReplica(":memory:")
    fresh_replica = FileNotesReplica(":memory:")
    winner = LibraryFileNotesWorkspace(
        root=old_root,
        replica=winner_replica,
        session_owner=owner,
        poll_interval=10,
    )
    persisted_root = [str(old_root)]
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    config_read = threading.Event()
    allow_config_return = threading.Event()
    event_loop = asyncio.get_running_loop()
    fresh: LibraryFileNotesWorkspace | None = None

    def get_setting(
        section: str,
        key: str | None = None,
        default: object = None,
    ) -> object:
        if (section, key) == ("file_notes", "root"):
            configured = persisted_root[0]
            config_read.set()
            if selection_timing == "after":
                assert allow_config_return.wait(timeout=5)
            return configured
        return default

    def persist(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persisted_root[0] = section_values["file_notes"]["root"]
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(workspace_module, "get_cli_setting", get_setting)
    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    heartbeat_thread, heartbeat_checked, heartbeat_while_waiting = (
        _event_loop_heartbeat(
            event_loop,
            config_read,
            release_persistence,
            allow_config_return,
        )
    )
    transition: asyncio.Task[bool] | None = None
    heartbeat_started = False
    try:
        harness = _DynamicWorkspaceHarness(winner)
        async with harness.run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: winner.initialized,
                "winner workspace did not initialize",
            )
            old_binding = owner.current_binding()
            assert old_binding is not None
            fresh = LibraryFileNotesWorkspace(
                replica=fresh_replica,
                session_owner=owner,
                poll_interval=10,
            )
            assert fresh._initial_session_binding == old_binding
            transition = asyncio.create_task(winner.set_root(winner_root))
            await _wait_until(
                pilot,
                persistence_started.is_set,
                "winner persistence did not start",
            )

            heartbeat_thread.start()
            heartbeat_started = True
            await harness.query_one(
                "#dynamic-workspace-host",
                Vertical,
            ).mount(fresh)
            await _wait_until(
                pilot,
                config_read.is_set,
                "fresh workspace did not read configured root",
            )
            if selection_timing == "during":
                await pilot.pause()
                assert not fresh.initialized
            release_persistence.set()
            assert await transition
            allow_config_return.set()
            await _wait_until(
                pilot,
                heartbeat_checked.is_set,
                "event-loop heartbeat was not checked",
            )
            assert heartbeat_while_waiting == [True]
            await _wait_until(
                pilot,
                lambda: fresh.initialized and fresh._service is not None,
                "fresh workspace did not initialize after root commit",
            )

            binding = owner.current_binding()
            assert binding is not None
            assert fresh.root == winner_root
            assert fresh._session_binding == binding
            assert fresh._service is not None
            assert fresh._service.root == winner_root
            assert set(fresh.entries) == {"winner.md"}
    finally:
        release_persistence.set()
        allow_config_return.set()
        if transition is not None and not transition.done():
            assert await transition
        if heartbeat_started:
            heartbeat_thread.join(timeout=1)
        await winner.shutdown()
        if fresh is not None:
            await fresh.shutdown()
        owner.shutdown()
        winner_replica.close()
        fresh_replica.close()


@pytest.mark.asyncio
async def test_bound_injected_owner_overrides_unrelated_explicit_seed(
    tmp_path: Path,
) -> None:
    owner_root = (tmp_path / "owner").resolve()
    unrelated_root = (tmp_path / "unrelated").resolve()
    owner_root.mkdir()
    unrelated_root.mkdir()
    (owner_root / "owner.md").write_text("owner", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(owner_root)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=unrelated_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized and workspace._service is not None,
                "bound-owner workspace did not initialize",
            )
            assert owner.current_binding() == binding
            assert workspace.root == owner_root
            assert workspace._session_binding == binding
            assert set(workspace.entries) == {"owner.md"}
    finally:
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_failed_root_persistence_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = (
        _root_transition_workspace(tmp_path)
    )

    def fail_persistence(*_args: object, **_kwargs: object) -> None:
        raise OSError("forced persistence failure")

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        fail_persistence,
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        with pytest.raises(OSError, match="forced persistence failure"):
            await workspace.set_root(new_root)

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path
            for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_before_replace_root_failure_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = (
        _root_transition_workspace(tmp_path)
    )

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: ConfigMutationResult(
            False,
            False,
            "before_replace",
        ),
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        assert not await workspace.set_root(new_root)

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path
            for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cache_reload_failure_adopts_persisted_root_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = (
        _root_transition_workspace(tmp_path)
    )

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: ConfigMutationResult(
            True,
            False,
            "cache_reload",
        ),
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        assert await workspace.set_root(new_root)

        new_binding = workspace._session_binding
        assert new_binding is not None
        assert new_binding != old_binding
        assert workspace.root == new_root.resolve()
        assert workspace._service is not old_service
        assert owner.current_binding() == new_binding
        assert "cache reload" in workspace._runtime_warning.lower()

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cancelled_root_persistence_settles_and_adopts_written_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = (
        _root_transition_workspace(tmp_path)
    )
    (old_root / "open.md").write_text("old root", encoding="utf-8")
    (old_root / "deleted.md").write_text("old tombstone", encoding="utf-8")
    (new_root / "new.md").write_text("new root", encoding="utf-8")
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    persistence_finished = threading.Event()
    persisted_roots: list[str] = []
    event_loop = asyncio.get_running_loop()

    def persist(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persisted_roots.append(section_values["file_notes"]["root"])
        persistence_finished.set()
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    heartbeat_thread, heartbeat_checked, heartbeat_during_persistence = (
        _event_loop_heartbeat(
            event_loop,
            persistence_started,
            release_persistence,
        )
    )
    transition: asyncio.Task[bool] | None = None
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "old workspace did not initialize",
            )
            old_service = workspace._service
            assert old_service is not None
            assert old_service.delete_file("deleted.md").status == "ok"
            assert await workspace.refresh_files()
            assert set(workspace.entries) == {"open.md"}
            assert workspace._deleted_paths == ("deleted.md",)
            assert await workspace.open_path("open.md")
            assert workspace.current_document is not None

            heartbeat_thread.start()
            transition = asyncio.create_task(workspace.set_root(new_root))
            await _wait_until(
                pilot,
                heartbeat_checked.is_set,
                "event-loop heartbeat was not checked",
            )
            assert heartbeat_during_persistence == [True]

            transition.cancel()
            await pilot.pause()
            assert not transition.done()
            assert not persistence_finished.is_set()

            release_persistence.set()
            with pytest.raises(asyncio.CancelledError):
                await transition

            binding = workspace._session_binding
            assert persistence_finished.is_set()
            assert persisted_roots == [str(new_root.resolve())]
            assert binding is not None
            assert workspace.root == new_root.resolve()
            assert workspace._service is not None
            assert workspace._service.root == new_root.resolve()
            assert owner.current_binding() == binding
            assert workspace.current_document is None
            assert workspace.current_path == ""
            assert set(workspace.entries) == {"new.md"}
            assert workspace._deleted_paths == ()
            assert workspace._root_offline is False
            tree_labels = _tree_labels(
                workspace.query_one("#file-notes-tree", Tree)
            )
            assert "new.md" in tree_labels
            assert "open.md" not in tree_labels
            assert "deleted.md" not in tree_labels
    finally:
        release_persistence.set()
        if transition is not None and not transition.done():
            transition.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transition
        heartbeat_thread.join(timeout=1)
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_injected_owner_shutdown_waits_for_root_commit_before_replica_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    new_root = (tmp_path / "new").resolve()
    old_root.mkdir()
    new_root.mkdir()
    (new_root / "new.md").write_text("new root", encoding="utf-8")
    owner = FileNotesSessionOwner()
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica_path=tmp_path / "owned.sqlite",
        session_owner=owner,
        poll_interval=10,
    )
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    persistence_finished = threading.Event()
    owner_wait_entered = threading.Event()
    replica_closed = threading.Event()
    close_observations: list[tuple[bool, Path | None, object]] = []
    owned_replica: FileNotesReplica | None = None
    real_wait = FileNotesSessionOwner.wait_for_root_commit
    real_close = FileNotesReplica.close

    def persist(
        _section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persistence_finished.set()
        return ConfigMutationResult(True, True, None)

    def observed_wait(session_owner: FileNotesSessionOwner) -> None:
        if session_owner is owner:
            owner_wait_entered.set()
        real_wait(session_owner)

    def observed_close(replica: FileNotesReplica) -> None:
        if replica is owned_replica:
            service = workspace._service
            close_observations.append(
                (
                    persistence_finished.is_set(),
                    None if service is None else service.root,
                    owner.current_binding(),
                )
            )
            replica_closed.set()
        real_close(replica)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    monkeypatch.setattr(
        FileNotesSessionOwner,
        "wait_for_root_commit",
        observed_wait,
    )
    monkeypatch.setattr(FileNotesReplica, "close", observed_close)
    transition: asyncio.Task[bool] | None = None
    shutdown_task: asyncio.Task[None] | None = None
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "owned-replica workspace did not initialize",
            )
            owned_replica = workspace._replica
            assert owned_replica is not None

            transition = asyncio.create_task(workspace.set_root(new_root))
            await _wait_until(
                pilot,
                persistence_started.is_set,
                "root persistence did not start",
            )
            shutdown_task = asyncio.create_task(workspace.shutdown())
            await _wait_until(
                pilot,
                lambda: owner_wait_entered.is_set() or replica_closed.is_set(),
                "shutdown neither waited nor closed the replica",
            )

            transition.cancel()
            await pilot.pause()
            assert not transition.done()
            assert owner_wait_entered.is_set()
            assert not replica_closed.is_set()
            assert not shutdown_task.done()
            assert workspace._replica is owned_replica

            release_persistence.set()
            with pytest.raises(asyncio.CancelledError):
                await transition
            await shutdown_task

            binding = owner.current_binding()
            assert persistence_finished.is_set()
            assert replica_closed.is_set()
            assert close_observations == [(True, new_root, binding)]
            assert binding is not None
            assert binding.root_key == str(new_root)
            assert workspace._replica is None
            assert workspace._service is None
            await pilot.pause()
            assert workspace._replica is None
            assert workspace._service is None

            status = owner.try_acquire_status(binding)
            assert status is not None
            status.release()
    finally:
        release_persistence.set()
        if transition is not None and not transition.done():
            transition.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transition
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        await workspace.shutdown()
        owner.shutdown()


@pytest.mark.asyncio
async def test_stale_candidate_scan_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    candidate_scan_started = threading.Event()
    release_candidate_scan = threading.Event()
    real_scan = FileNotesService.scan

    def delayed_candidate_scan(service: FileNotesService):
        if service.root == new_root.resolve():
            candidate_scan_started.set()
            assert release_candidate_scan.wait(timeout=5)
        return real_scan(service)

    monkeypatch.setattr(FileNotesService, "scan", delayed_candidate_scan)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        transition = asyncio.create_task(
            workspace.set_root(new_root, persist=False)
        )
        await _wait_until(
            pilot,
            candidate_scan_started.is_set,
            "candidate scan did not start",
        )
        workspace.on_unmount()
        release_candidate_scan.set()
        assert not await transition

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path
            for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    release_candidate_scan.set()
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_tree_search_open_dirty_and_autosave_keep_one_editor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "alpha.md").write_text(
        "needle in this body\n",
        encoding="utf-8",
    )
    (root / "beta.txt").write_text("other body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=0.08,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(110, 36)) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized and len(workspace.entries) == 2,
            "initial tree did not load",
        )
        tree = workspace.query_one("#file-notes-tree", Tree)
        assert {"folder", "alpha.md", "beta.txt"}.issubset(_tree_labels(tree))

        search = workspace.query_one("#file-notes-search", Input)
        search.value = "needle"
        await _wait_until(
            pilot,
            lambda: workspace.query_one("#file-notes-search-results", Tree).display,
            "search results did not replace the tree",
        )
        assert not tree.display
        results = workspace.query_one("#file-notes-search-results", Tree)
        assert "folder/alpha.md" in _tree_labels(results)
        editor = workspace.query_one("#file-notes-editor", TextArea)
        match = next(
            node
            for node in results.root.children
            if node.data == ("file", "folder/alpha.md")
        )
        results.select_node(match)
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "folder/alpha.md",
            "selecting the visible search result did not open its file",
        )
        assert editor.text == "needle in this body\n"

        search.value = ""
        await _wait_until(pilot, lambda: tree.display, "tree did not return")

        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        assert workspace.save_state == "saved"

        _replace_editor_text(editor, "changed body")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "body edit did not become dirty",
        )
        assert not workspace.leave_allowed
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "saved",
            "debounced autosave did not complete",
        )
        assert (root / "folder" / "alpha.md").read_text(encoding="utf-8") == (
            "changed body\n"
        )
        assert workspace.query_one("#file-notes-editor", TextArea) is editor
    await workspace.shutdown()
    assert replica.list_deleted(str(root.resolve())) == []
    replica.close()


@pytest.mark.asyncio
async def test_create_move_delete_protect_and_restore_use_real_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "start.md").write_text("start", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("start.md")

        editor = workspace.query_one("#file-notes-editor", TextArea)
        service = workspace._service
        assert service is not None
        delayed_create, create_started, release_create = _delayed_call(
            service.create_file
        )
        monkeypatch.setattr(service, "create_file", delayed_create)
        path_input = workspace.query_one("#file-notes-path", Input)
        path_input.value = "created.md"
        create_button = workspace.query_one("#file-notes-new", Button)
        creating = asyncio.create_task(workspace._new_file(Button.Pressed(create_button)))
        await _wait_until(pilot, create_started.is_set, "new file did not start")
        editor.focus()
        await pilot.press("x")
        state_during_create = (editor.read_only, workspace.leave_allowed, editor.text)
        release_create.set()
        await creating
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "created.md",
            "new file did not open",
        )
        assert state_during_create == (True, False, "start")
        assert (root / "created.md").exists()

        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and workspace.current_document.protected
            ),
            "protect did not apply",
        )
        assert str(workspace.query_one("#file-notes-protect", Button).label) == (
            "Unprotect"
        )
        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and not workspace.current_document.protected
            ),
            "unprotect did not apply",
        )

        path_input.value = "moved.md"
        workspace.query_one("#file-notes-move", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "moved.md",
            "move did not open destination",
        )
        assert not (root / "created.md").exists()
        assert (root / "moved.md").exists()
        assert replica.list_deleted(str(root.resolve())) == []
        assert "Recently deleted" not in _tree_labels(
            workspace.query_one("#file-notes-tree", Tree)
        )

        workspace.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                str(workspace.query_one("#file-notes-delete", Button).label)
                == "Confirm delete"
            ),
            "delete confirmation did not arm",
        )
        assert (root / "moved.md").exists()
        assert str(workspace.query_one("#file-notes-delete", Button).label) == (
            "Confirm delete"
        )
        workspace.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: not (root / "moved.md").exists(),
            "confirmed delete did not remove the file",
        )
        assert "Recently deleted" in _tree_labels(
            workspace.query_one("#file-notes-tree", Tree)
        )

        workspace.query_one("#file-notes-restore", Button).press()
        await _wait_until(
            pilot,
            lambda: (root / "moved.md").exists(),
            "restore did not recreate exact file",
        )
        assert (
            _static_text(workspace, "#file-notes-session-changes")
            == "Session Git (1)"
        )
    replica.close()


@pytest.mark.asyncio
async def test_injected_owner_retains_same_root_log_across_workspaces(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    first = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )

    async with _WorkspaceHarness(first).run_test() as pilot:
        await _wait_until(pilot, lambda: first.initialized, "first scan did not finish")
        service = first._service
        assert service is not None
        assert service.create_file("retained.md", "retained").status == "ok"
        first._refresh_session_changes()
        assert (
            _static_text(first, "#file-notes-session-changes") == "Session Git (1)"
        )
    await first.shutdown()

    binding = owner.select_root(root)
    status = owner.try_acquire_status(binding)
    assert status is not None
    status.release()

    second = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    async with _WorkspaceHarness(second).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: second.initialized,
            "second scan did not finish",
        )
        assert (
            _static_text(second, "#file-notes-session-changes") == "Session Git (1)"
        )
    await second.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_direct_workspace_shuts_down_only_its_private_session_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "workspace scan did not finish",
        )
        owner = workspace._session_owner
        binding = owner.select_root(root)

    await workspace.shutdown()
    await workspace.shutdown()

    assert owner.try_acquire_status(binding) is None
    replica.close()


@pytest.mark.parametrize(
    ("button_id", "handler_name", "service_method", "action"),
    (
        ("#file-notes-new", "_new_file", "create_file", "Create"),
        ("#file-notes-move", "_move_file", "move_file", "Move"),
        ("#file-notes-restore", "_restore_file", "restore_file", "Restore"),
        ("#file-notes-save-copy", "_save_copy", "save_copy", "Save Copy"),
    ),
)
@pytest.mark.asyncio
async def test_raw_path_actions_validate_input_before_service_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    button_id: str,
    handler_name: str,
    service_method: str,
    action: str,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "start.md").write_text("start", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("start.md")
        service = workspace._service
        assert service is not None
        validation_calls: list[tuple[str, int, bool]] = []
        calls: list[tuple[object, ...]] = []
        raw_path = (" " * 4097) + r"nested/double..dots\<script>note.md"
        destination = raw_path.strip()
        real_validate_text_input = workspace_module.validate_text_input

        def capture_path_validation(
            text: str,
            max_length: int = 10000,
            allow_html: bool = False,
        ) -> bool:
            validation_calls.append((text, max_length, allow_html))
            return real_validate_text_input(
                text,
                max_length=max_length,
                allow_html=allow_html,
            )

        def capture_call(*args: object) -> OperationResult:
            calls.append(args)
            return OperationResult(
                status="error",
                relative_path=destination,
                message="service should not receive invalid input",
            )

        monkeypatch.setattr(
            workspace_module,
            "validate_text_input",
            capture_path_validation,
        )
        monkeypatch.setattr(service, service_method, capture_call)
        workspace.query_one("#file-notes-path", Input).value = raw_path
        button = workspace.query_one(button_id, Button)
        await getattr(workspace, handler_name)(Button.Pressed(button))

        assert validation_calls == [(raw_path, 4096, True)]
        assert calls == []
        assert _static_text(
            workspace,
            "#file-notes-action-status",
        ) == f"{action} failed: unsupported path text."
        assert workspace.current_path == "start.md"
    replica.close()


@pytest.mark.asyncio
async def test_conflict_reload_save_copy_and_leave_guards_preserve_draft(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nold\r\nbody\r\n")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        first_session = workspace.session_key
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "dirty reload guard")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "dirty Reload setup did not arm",
        )
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.save_state == "saved"
                and editor.text == "dirty reload guard\n"
            ),
            "Reload discarded a merely dirty draft instead of flushing it",
        )
        assert source.read_bytes().endswith(b"dirty reload guard\r\n")

        _replace_editor_text(editor, "kept\ndraft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )

        source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: External\r\n---\r\nexternal\r\n")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        assert editor.text == "kept\ndraft"
        assert not await workspace.flush_pending_work()

        workspace.query_one("#file-notes-path", Input).value = "copy.md"
        workspace.query_one("#file-notes-save-copy", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "copy.md",
            "save copy did not open the safe copy",
        )
        assert (root / "copy.md").read_bytes() == (
            b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nkept\r\ndraft\r\n"
        )

        assert await workspace.open_path("source.md")
        _replace_editor_text(editor, "another draft")
        await pilot.pause()
        source.write_bytes(
            b"\xef\xbb\xbf---\r\ntitle: External 2\r\n---\r\nreload me\r\n"
        )
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.save_state == "saved"
                and workspace.query_one("#file-notes-editor", TextArea).text
                == "reload me\n"
            ),
            "reload did not resolve the conflict",
        )
        assert workspace.session_key != first_session
        assert workspace.query_one("#file-notes-editor", TextArea) is editor

        _replace_editor_text(editor, "flush me")
        await pilot.pause()
        assert await workspace.flush_pending_work()
        assert source.read_bytes().endswith(b"flush me\r\n")

        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and workspace.current_document.protected
            ),
            "protected error setup did not finish",
        )
        replica.close()
        _replace_editor_text(editor, "surviving error draft")
        await pilot.pause()
        assert not await workspace.flush_pending_work()
        assert workspace.save_state == "error"
        assert editor.text == "surviving error draft"
    replica.close()


@pytest.mark.asyncio
async def test_recently_deleted_survives_a_second_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    original = b"recover me exactly\r\n"
    (root / "recover.md").write_bytes(original)
    replica_path = tmp_path / "file_notes.sqlite"

    first_replica = FileNotesReplica(replica_path)
    first = LibraryFileNotesWorkspace(
        root=root,
        replica=first_replica,
        poll_interval=10,
    )
    async with _WorkspaceHarness(first).run_test() as pilot:
        await _wait_until(pilot, lambda: first.initialized, "first scan did not finish")
        assert await first.open_path("recover.md")
        first.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                str(first.query_one("#file-notes-delete", Button).label)
                == "Confirm delete"
            ),
            "delete confirmation did not arm",
        )
        first.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: not (root / "recover.md").exists(),
            "delete did not finish",
        )
    first_replica.close()

    second_replica = FileNotesReplica(replica_path)
    second = LibraryFileNotesWorkspace(
        root=root,
        replica=second_replica,
        poll_interval=10,
    )
    async with _WorkspaceHarness(second).run_test() as pilot:
        await _wait_until(
            pilot, lambda: second.initialized, "second scan did not finish"
        )
        assert "Recently deleted" in _tree_labels(
            second.query_one("#file-notes-tree", Tree)
        )
        assert second.select_deleted("recover.md")
        second.query_one("#file-notes-restore", Button).press()
        await _wait_until(
            pilot,
            lambda: (root / "recover.md").exists(),
            "second workspace could not restore tombstone",
        )
        assert (root / "recover.md").read_bytes() == original
    second_replica.close()


@pytest.mark.asyncio
async def test_poll_and_narrow_navigation_retain_the_text_area(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("first", encoding="utf-8")
    (root / "delete.md").write_text("gone soon", encoding="utf-8")
    (root / "folder").mkdir()
    (root / "folder" / "nested.md").write_text("nested", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=0.05,
        autosave_delay=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(64, 28)) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.narrow,
            "workspace did not choose narrow mode from its width",
        )
        editor = workspace.query_one("#file-notes-editor", TextArea)
        assert workspace.navigator_visible
        assert not workspace.editor_visible
        tree = workspace.query_one("#file-notes-tree", Tree)
        folder = next(
            node
            for node in tree.root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        folder.expand()

        assert await workspace.open_path("open.md")
        assert workspace.editor_visible
        assert not workspace.navigator_visible
        assert workspace.query_one("#file-notes-editor", TextArea) is editor

        service = workspace._service
        assert service is not None
        delayed_open, reload_started, release_reload = _delayed_call(
            service.open_file
        )
        monkeypatch.setattr(service, "open_file", delayed_open)
        (root / "open.md").write_text("external", encoding="utf-8")
        (root / "created.md").write_text("new", encoding="utf-8")
        (root / "delete.md").unlink()
        await _wait_until(
            pilot,
            reload_started.is_set,
            "external reload did not start",
        )
        _replace_editor_text(editor, "draft during reload")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "reload-window edit did not become dirty",
        )
        release_reload.set()
        await _wait_until(
            pilot,
            lambda: (
                set(workspace.entries) == {"created.md", "folder/nested.md", "open.md"}
                and workspace.save_state == "conflict"
            ),
            "poll did not reconcile external create/modify/delete",
        )
        assert editor.text == "draft during reload"
        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        refreshed_folder = next(
            node
            for node in workspace.query_one("#file-notes-tree", Tree).root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        assert refreshed_folder.is_expanded
        await pilot.pause(0.15)
        active = [
            worker
            for worker in workspace.workers
            if worker.node is workspace and not worker.is_finished
        ]
        assert len(active) <= 1

        workspace.query_one("#file-notes-back", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.navigator_visible and not workspace.editor_visible,
            "Back did not return to the retained navigator",
        )
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (160, 45)])
async def test_library_notes_source_choices_render_and_switch_by_keyboard(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
        poll_interval=10,
        autosave_delay=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(
        app,
        file_notes_workspace_factory=lambda: workspace,
    )

    async with LibraryHarness(app, screen=screen).run_test(size=size) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-strip")),
            "Notes source strip did not compose",
        )

        strip = screen.query_one("#library-notes-source-strip")
        separator = screen.query_one("#library-notes-source-separator")
        database = screen.query_one("#library-notes-source-database", Button)
        files = screen.query_one("#library-notes-source-files", Button)
        await _wait_until(
            pilot,
            lambda: (
                separator.region.width == 1
                and database.region.width > 0
                and files.region.width > 0
            ),
            "Notes source choices did not receive visible geometry",
        )
        assert strip.content_region.contains_region(database.region)
        assert strip.content_region.contains_region(separator.region)
        assert strip.content_region.contains_region(files.region)
        assert separator.region.width == 1
        assert str(database.label) == "Database (selected)"
        assert database.has_class("-selected")
        assert not database.disabled
        assert database.can_focus
        assert str(files.label) == "Files"
        assert not files.disabled
        assert files.can_focus

        for _ in range(60):
            if database.has_focus:
                break
            await pilot.press("tab")
        assert database.has_focus
        assert strip.content_region.contains_region(database.region)
        await pilot.press("tab")
        await _wait_until(
            pilot,
            lambda: files.has_focus,
            "Tab did not move from Database to the visible Files source",
        )
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "files"
                and workspace.initialized
                and bool(screen.query("#library-file-notes-workspace"))
            ),
            "Files source did not open from the keyboard",
        )

        strip = screen.query_one("#library-notes-source-strip")
        database = screen.query_one("#library-notes-source-database", Button)
        files = screen.query_one("#library-notes-source-files", Button)
        assert strip.content_region.contains_region(database.region)
        assert strip.content_region.contains_region(files.region)
        assert str(database.label) == "Database"
        assert not database.disabled
        assert str(files.label) == "Files (selected)"
        assert files.has_class("-selected")
        assert not files.disabled

        for _ in range(60):
            if files.has_focus:
                break
            await pilot.press("tab")
        assert files.has_focus
        assert strip.content_region.contains_region(files.region)
        await pilot.press("shift+tab")
        await _wait_until(
            pilot,
            lambda: database.has_focus,
            "Shift+Tab did not move from Files to the visible Database source",
        )
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "database"
                and bool(screen.query("#library-notes-canvas"))
            ),
            "Database source did not reopen from the keyboard",
        )
        assert (
            str(
                screen.query_one(
                    "#library-notes-source-database",
                    Button,
                ).label
            )
            == "Database (selected)"
        )

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_library_database_files_switch_retains_workspace_and_database_canvas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    (root / "other.md").write_text("other file", encoding="utf-8")
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
        poll_interval=10,
        autosave_delay=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(
        app,
        file_notes_workspace_factory=lambda: workspace,
    )
    host = LibraryHarness(app, screen=screen)
    save_started = threading.Event()
    release_save = threading.Event()
    open_started = threading.Event()
    release_open = threading.Event()
    detail_started = threading.Event()
    release_detail = threading.Event()

    def delayed_detail(**_kwargs):
        detail_started.set()
        release_detail.wait(5)
        return {
            "id": "db-note-1",
            "title": "Database note",
            "content": "body",
            "version": 1,
        }

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-canvas")),
            "Database Notes canvas did not compose",
        )
        assert screen.query_one("#library-rail")
        assert screen.query_one("#library-notes-source-strip")
        assert screen._library_file_notes_workspace is None

        app.notes_scope_service.get_note_detail = delayed_detail
        screen._selected_note_id = "db-note-1"
        screen._library_notes_view = "editor"
        detail_task = asyncio.create_task(
            screen._refresh_library_note_detail("db-note-1")
        )
        await _wait_until(
            pilot,
            detail_started.is_set,
            "Database detail fetch did not start",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "files",
            "Files source handler did not run",
        )
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-file-notes-workspace")),
            "Files workspace did not replace the Database rail/canvas",
        )
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace._replica is not None,
            "Files workspace did not initialize",
        )
        retained = screen.query_one(
            "#library-file-notes-workspace",
            LibraryFileNotesWorkspace,
        )
        editor = retained.query_one("#file-notes-editor", TextArea)
        owned_replica = retained._replica
        assert owned_replica is not None
        assert retained is workspace
        assert not screen.query("#library-rail")

        release_detail.set()
        await detail_task
        assert screen._library_note_detail is None
        screen._library_notes_view = "list"
        screen._selected_note_id = None

        assert await retained.open_path("library.md")
        service = retained._service
        assert service is not None
        original_open = service.open_file

        def delayed_open(relative_path):
            if relative_path == "other.md":
                open_started.set()
                release_open.wait(5)
            return original_open(relative_path)

        monkeypatch.setattr(service, "open_file", delayed_open)
        opening = asyncio.create_task(retained.open_path("other.md"))
        await _wait_until(pilot, open_started.is_set, "slow open did not start")
        before_open = editor.text
        editor.focus()
        await pilot.press("x")
        competing_open = await retained.open_path("library.md")
        frozen_during_open = editor.read_only
        text_during_open = editor.text
        release_open.set()
        assert await opening
        monkeypatch.setattr(service, "open_file", original_open)
        assert await retained.open_path("library.md")

        screen._apply_local_source_snapshot(
            {
                "notes": ({"title": "Updated DB note", "id": "db-note-2"},),
                "media": (),
                "conversations": (),
            },
            {"notes": 1, "media": 0, "conversations": 0},
            {"notes": True, "media": True, "conversations": True},
        )
        await pilot.pause()
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert retained.query_one("#file-notes-editor", TextArea) is editor

        _replace_editor_text(editor, "draft")
        await pilot.pause()
        (root / "library.md").write_text("external", encoding="utf-8")
        await retained.refresh_files()
        assert retained.save_state == "conflict"

        assert not await retained.open_path("other.md")
        assert retained.current_path == "library.md"
        assert editor.text == "draft"

        assert not await retained.set_root(replacement_root, persist=False)
        assert retained.root == root.resolve()
        assert editor.text == "draft"

        assert not await screen.flush_pending_work()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-file-notes-workspace") is retained

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert editor.text == "draft"

        screen.query_one("#library-notes-source-database", Button).press()
        await pilot.pause()
        assert screen._library_notes_source == "files"
        assert screen.query_one("#library-file-notes-workspace") is retained

        retained.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: retained.save_state == "saved",
            "reload did not clear the source-switch veto",
        )
        _replace_editor_text(editor, "saved before hiding")
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty",
            "pre-remount edit did not become dirty",
        )
        assert await retained.flush_pending_work()
        assert (
            _static_text(retained, "#file-notes-session-changes")
            == "Session Git (1)"
        )
        screen.query_one("#library-notes-source-database", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-canvas")),
            "Database Notes did not return",
        )
        assert screen.query_one("#library-rail")
        assert screen._local_source_records["notes"][0]["title"] == "Updated DB note"

        (root / "library.md").write_text("changed while hidden", encoding="utf-8")
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-file-notes-workspace")),
            "retained Files workspace did not remount",
        )
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert retained.query_one("#file-notes-editor", TextArea) is editor
        await _wait_until(
            pilot,
            lambda: editor.text == "changed while hidden",
            "remount did not reconcile the retained open file",
        )
        assert (
            _static_text(retained, "#file-notes-session-changes")
            == "Session Git (1)"
        )

        original_finish = service._finish_published_file

        def delayed_finish(*args, **kwargs):
            save_started.set()
            release_save.wait(5)
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(service, "_finish_published_file", delayed_finish)
        _replace_editor_text(editor, "draft across forced remount")
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty",
            "forced-remount draft did not become dirty",
        )
        retained._start_autosave()
        await _wait_until(
            pilot,
            lambda: save_started.is_set() and retained.save_state == "saving",
            "forced-remount save did not start",
        )
        session_key = retained._session_key

        screen.refresh(recompose=True)
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty"
            and retained._active
            and bool(screen.query("#library-file-notes-workspace")),
            "Files workspace did not remount during save",
        )
        timer_before_release = retained._autosave_timer
        release_save.set()
        await _wait_until(
            pilot,
            lambda: retained.save_state == "saved",
            "published remount draft was not adopted",
        )
        assert timer_before_release is None
        assert (root / "library.md").read_text(encoding="utf-8") == (
            "draft across forced remount"
        )
        assert editor.text == "draft across forced remount"
        assert retained._session_key == session_key
        assert retained.query_one("#file-notes-editor", TextArea) is editor
        assert not competing_open
        assert frozen_during_open
        assert text_during_open == before_open
    assert workspace._shutdown
    await workspace.shutdown()
    with pytest.raises(sqlite3.ProgrammingError):
        owned_replica.list_deleted(str(root.resolve()))
