"""Mounted behavior tests for the File Notes Session Git navigator."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Vertical
from textual.widgets import Button, Input, Label, ListView, Static, TextArea, Tree

sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from tldw_chatbook.Notes.file_notes_git_service import (  # noqa: E402
    FileNotesGitService,
    DiscoveryResult,
    GitActionResult,
    GitCommandResult,
    GitMutationAdmissionError,
    coalesce_session_changes,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_session_owner import (  # noqa: E402
    FileSystemIdentity,
    FileNotesSessionOwner,
    HeadIdentity,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionChange,
    SessionChangeAction,
    SessionChangeGroup,
    SessionGitRow,
    SessionGitRowState,
    SessionGitStageAction,
    SessionGitStatus,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (  # noqa: E402
    LibraryFileNotesGitPanel,
    SessionGitTrustDialog,
    _middle_elide_cells,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    LibraryFileNotesWorkspace,
)


class _PanelHarness(App[None]):
    """Mount one panel and record its typed presentation messages."""

    def __init__(self, panel: LibraryFileNotesGitPanel) -> None:
        super().__init__()
        self.panel = panel
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield self.panel

    def on_library_file_notes_git_panel_back_requested(
        self,
        message: LibraryFileNotesGitPanel.BackRequested,
    ) -> None:
        self.messages.append(message)

    def on_library_file_notes_git_panel_refresh_requested(
        self,
        message: LibraryFileNotesGitPanel.RefreshRequested,
    ) -> None:
        self.messages.append(message)

    def on_library_file_notes_git_panel_trust_requested(
        self,
        message: LibraryFileNotesGitPanel.TrustRequested,
    ) -> None:
        self.messages.append(message)

    def on_library_file_notes_git_panel_stage_requested(
        self,
        message: LibraryFileNotesGitPanel.StageRequested,
    ) -> None:
        self.messages.append(message)

    def on_library_file_notes_git_panel_unstage_requested(
        self,
        message: LibraryFileNotesGitPanel.UnstageRequested,
    ) -> None:
        self.messages.append(message)


class _DialogHarness(App[None]):
    """Open a Session Git trust dialog at mount."""

    def __init__(self, dialog: SessionGitTrustDialog) -> None:
        super().__init__()
        self.dialog = dialog
        self.result: bool | None = None

    def on_mount(self) -> None:
        self.push_screen(self.dialog, callback=self._remember)

    def _remember(self, result: bool | None) -> None:
        self.result = result


class _WorkspaceHarness(App[None]):
    """Mount one real File Notes workspace."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        yield self.workspace


class _RemountWorkspaceHarness(App[None]):
    """Mount one retained workspace beneath a removable host."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        with Vertical(id="remount-workspace-host"):
            yield self.workspace


class _FakeGitService:
    """Real-signature retained-task fake for the workspace presentation seam."""

    def __init__(
        self,
        owner: FileNotesSessionOwner,
        rows: tuple[SessionGitRow, ...],
    ) -> None:
        self.owner = owner
        self.repository = _repository()
        self.head = HeadIdentity.attached("feature/session-git", "a" * 40)
        self.rows = rows
        self.discovery_calls: list[SessionBinding] = []
        self.revalidate_calls: list[tuple[SessionBinding, RepositoryIdentity]] = []
        self.status_calls: list[tuple[SequencedSessionChange, ...]] = []
        self.stage_calls: list[tuple[int, ...]] = []
        self.unstage_calls: list[tuple[int, ...]] = []
        self.revalidate_result = True
        self.discovery_result: DiscoveryResult | None = None
        self.status_release: asyncio.Event | None = None
        self.action_release: asyncio.Event | None = None
        self._status_binding: SessionBinding | None = None
        self._status_task: asyncio.Task[SessionGitStatus] | None = None

    async def discover(self, binding: SessionBinding) -> DiscoveryResult:
        self.discovery_calls.append(binding)
        return self.discovery_result or DiscoveryResult(
            "ready",
            repository=self.repository,
            head=self.head,
        )

    async def revalidate_repository(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        self.revalidate_calls.append((binding, repository))
        return self.revalidate_result

    def retained_status(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[SessionGitStatus] | None:
        """Expose active service work without starting or transferring it."""
        task = self._status_task
        if binding != self._status_binding or task is None or task.done():
            return None
        return task

    def start_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
    ) -> asyncio.Task[SessionGitStatus]:
        self.status_calls.append(tuple(changes))
        release = self.status_release
        repository = self.repository
        head = self.head
        rows = self.rows

        async def finish() -> SessionGitStatus:
            if release is not None:
                await release.wait()
            generation = self.owner.next_status_generation(binding)
            assert generation is not None
            status = SessionGitStatus(
                binding_generation=binding.generation,
                status_generation=generation,
                state="ready",
                rows=rows,
                repository=repository,
                head=head,
            )
            self.owner.publish_status(binding, status)
            return status

        task = asyncio.create_task(finish())
        self._status_binding = binding
        self._status_task = task
        return task

    def start_stage(
        self,
        binding: SessionBinding,
        group_ids: tuple[int, ...],
    ) -> asyncio.Task[GitActionResult]:
        return self._start_action("stage", binding, group_ids)

    def start_unstage(
        self,
        binding: SessionBinding,
        group_ids: tuple[int, ...],
    ) -> asyncio.Task[GitActionResult]:
        return self._start_action("unstage", binding, group_ids)

    def _start_action(
        self,
        action: str,
        binding: SessionBinding,
        group_ids: tuple[int, ...],
    ) -> asyncio.Task[GitActionResult]:
        admission = self.owner.admit_mutation(binding)
        if admission.lease is None:
            raise GitMutationAdmissionError(
                admission.reason or "mutation_active",
                "mutation refused",
            )
        requested = tuple(group_ids)
        if action == "stage":
            self.stage_calls.append(requested)
        else:
            self.unstage_calls.append(requested)

        async def finish() -> GitActionResult:
            try:
                if self.action_release is not None:
                    await self.action_release.wait()
                self.owner.clear_status(binding)
                return GitActionResult(
                    action,  # type: ignore[arg-type]
                    "success",
                    requested,
                    staged_group_ids=requested if action == "stage" else (),
                    unstaged_group_ids=requested if action == "unstage" else (),
                )
            finally:
                assert admission.lease is not None
                admission.lease.release()

        return asyncio.create_task(finish())

    def shutdown(self) -> None:
        return


class _PathspecRecordingRunner:
    """Serve deterministic Git reads and retain real service command argv."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str | bytes, ...]] = []

    async def run(
        self,
        argv: Sequence[str | bytes],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        del cwd, environment, stdin, timeout
        command = tuple(argv)
        self.calls.append(command)
        text = tuple(os.fsdecode(argument) for argument in command)
        if "--is-inside-work-tree" in text:
            return GitCommandResult(0, b"true\n", b"")
        if "--show-toplevel" in text:
            return GitCommandResult(0, os.fsencode(self.root) + b"\n", b"")
        if "--absolute-git-dir" in text or "--git-common-dir" in text:
            return GitCommandResult(
                0,
                os.fsencode(self.root / ".git") + b"\n",
                b"",
            )
        if "symbolic-ref" in text:
            return GitCommandResult(0, b"refs/heads/main\n", b"")
        if "rev-parse" in text:
            return GitCommandResult(0, (b"a" * 40) + b"\n", b"")
        if "config" in text:
            return GitCommandResult(1, b"", b"")
        if "ls-files" in text:
            return GitCommandResult(0, b"", b"")
        if "status" in text:
            boundary = command.index("--")
            payload = b"".join(
                b"? " + os.fsencode(path) + b"\0"
                for path in command[boundary + 1 :]
            )
            return GitCommandResult(0, payload, b"")
        raise AssertionError(f"Unexpected Git command: {text!r}")

    def shutdown(self) -> None:
        return


def _repository(
    worktree_root: str = "/canonical/repository",
    *,
    identity: FileSystemIdentity | None = None,
) -> RepositoryIdentity:
    identity = identity or FileSystemIdentity(1, 2)
    return RepositoryIdentity(
        worktree_root=worktree_root,
        git_dir=f"{worktree_root}/.git",
        git_common_dir=f"{worktree_root}/.git",
        worktree_identity=identity,
        git_dir_identity=identity,
        git_common_dir_identity=identity,
    )


def _row(
    state: SessionGitRowState,
    *,
    group_id: int = 1,
    stage_action: SessionGitStageAction | None = None,
    unstage_eligible: bool = False,
    disabled_reason: str | None = None,
    latest_action: SessionChangeAction = "modified",
    source_path: str | None = None,
    destination_path: str | None = None,
) -> SessionGitRow:
    source = source_path or f"note-{group_id}.md"
    current = destination_path or source
    return SessionGitRow(
        SessionChangeGroup(
            group_id=group_id,
            endpoints=(
                (source,) if destination_path is None else (source, destination_path)
            ),
            source_path=source,
            destination_path=destination_path,
            current_path=current,
            latest_action=latest_action,
            latest_sequence=group_id,
            move_edges=(
                () if destination_path is None else ((source, destination_path),)
            ),
        ),
        state,
        stage_action=stage_action,
        unstage_eligible=unstage_eligible,
        disabled_reason=disabled_reason,
    )


def _status(
    *rows: SessionGitRow,
    state: str = "ready",
    message: str | None = None,
) -> SessionGitStatus:
    return SessionGitStatus(
        binding_generation=1,
        status_generation=1,
        state=state,  # type: ignore[arg-type]
        rows=rows,
        repository=_repository(),
        head=HeadIdentity.attached("feature/session-git", "a" * 40),
        message=message,
    )


def _workspace_fixture(
    tmp_path: Path,
    *,
    trusted: bool = True,
) -> tuple[
    Path,
    FileNotesSessionOwner,
    SessionBinding,
    FileNotesReplica,
    _FakeGitService,
    LibraryFileNotesWorkspace,
]:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "one.md").write_text("needle one", encoding="utf-8")
    (root / "two.md").write_text("needle two", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("modified", "folder/one.md"))
    assert owner.record_change(binding, SessionChange("modified", "two.md"))
    rows = (
        _row("unstaged", group_id=1, stage_action="stage"),
        _row("unstaged", group_id=2, stage_action="stage"),
    )
    git_service = _FakeGitService(owner, rows)
    owner.attach_git_service(git_service)
    if trusted:
        assert owner.publish_trust(binding, git_service.repository)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
        autosave_delay=10,
    )
    return root, owner, binding, replica, git_service, workspace


def _text(widget: Static | Label) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _assert_git_mutations_disabled(
    workspace: LibraryFileNotesWorkspace,
) -> None:
    for selector in (
        "#file-notes-git-stage-selected",
        "#file-notes-git-stage-all",
    ):
        assert workspace.query_one(selector, Button).disabled


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    message: str,
    *,
    attempts: int = 80,
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(message)


async def _assert_visible_panel_buttons_fit(panel, pilot) -> None:
    bounds = panel.content_region
    for button in panel.query(Button):
        if not button.display:
            continue
        button.focus()
        await pilot.pause()
        assert button.has_focus
        label = str(button.label)
        assert button.render().plain == label
        assert cell_len(label) <= button.content_region.width
        assert button.region.x >= bounds.x
        assert button.region.right <= bounds.right
        assert button.region.y >= bounds.y
        assert button.region.bottom <= bounds.bottom
        assert not button.styles.outline
        assert button.styles.background == Color.parse("#51677e")
        assert button.styles.text_style.bold
        assert button.styles.text_style.underline


@pytest.mark.asyncio
async def test_panel_renders_repository_scope_and_complete_file_state() -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(_row("unstaged", stage_action="stage")))
        await pilot.pause()

        assert (
            _text(panel.query_one("#file-notes-git-title", Static))
            == "Prepare session for commit"
        )
        assert "/canonical/repository" in _text(
            panel.query_one("#file-notes-git-repository", Static)
        )
        assert "feature/session-git" in _text(
            panel.query_one("#file-notes-git-repository", Static)
        )
        assert (
            _text(panel.query_one("#file-notes-git-scope", Static))
            == "Session paths only · stages complete file state"
        )
        assert _text(panel.query_one("#file-notes-git-guide", Static)) == (
            "Up/Down Select | Tab Actions | Enter Run | Esc Back"
        )
        assert _text(panel.query_one("#file-notes-git-status", Static)).startswith(
            "Status: CURRENT · READY"
        )


@pytest.mark.parametrize(
    ("latest_action", "verb"),
    [
        ("created", "CREATED"),
        ("modified", "EDITED"),
        ("moved", "MOVED"),
        ("deleted", "DELETED"),
        ("restored", "RESTORED"),
    ],
)
@pytest.mark.asyncio
async def test_rows_project_note_intent_on_a_separate_primary_line(
    latest_action: SessionChangeAction,
    verb: str,
) -> None:
    row = _row(
        "unstaged",
        latest_action=latest_action,
        source_path="folder/before.md",
        destination_path=("folder/after.md" if latest_action == "moved" else None),
        stage_action="stage",
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(row))
        await pilot.pause()

        primary = _text(panel.query_one(".file-notes-git-row-primary", Static))
        semantic = _text(panel.query_one(".file-notes-git-row-secondary", Static))
        assert primary.startswith(verb)
        assert "folder/before.md" in primary
        if latest_action == "moved":
            assert "-> folder/after.md" in primary
        assert semantic == "READY TO STAGE · Git: unstaged"


@pytest.mark.parametrize(
    ("changes", "state", "expected_primary", "expected_secondary"),
    [
        (
            (
                SessionChange("created", "draft.md"),
                SessionChange("deleted", "draft.md"),
            ),
            "unstaged",
            "DELETED   draft.md",
            "READY TO STAGE · Git: unstaged",
        ),
        (
            (
                SessionChange("deleted", "restored.md"),
                SessionChange("restored", "restored.md"),
            ),
            "unstaged",
            "RESTORED  restored.md",
            "READY TO STAGE · Git: unstaged",
        ),
        (
            (SessionChange("modified", "unchanged.md"),),
            "clean",
            "EDITED    unchanged.md",
            "NO ACTION · matches HEAD",
        ),
        (
            (
                SessionChange("moved", "original.md", "middle.md"),
                SessionChange("moved", "middle.md", "final.md"),
            ),
            "unstaged",
            "MOVED     original.md -> final.md",
            "READY TO STAGE · Git: unstaged",
        ),
    ],
)
@pytest.mark.asyncio
async def test_coalesced_histories_project_latest_note_intent(
    changes: tuple[SessionChange, ...],
    state: SessionGitRowState,
    expected_primary: str,
    expected_secondary: str,
) -> None:
    (group,) = coalesce_session_changes(
        tuple(
            SequencedSessionChange(sequence, change)
            for sequence, change in enumerate(changes, 1)
        )
    )
    row = SessionGitRow(
        group,
        state,
        stage_action="stage" if state == "unstaged" else None,
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(row))
        await pilot.pause()

        assert (
            _text(panel.query_one(".file-notes-git-row-primary", Static))
            == expected_primary
        )
        assert (
            _text(panel.query_one(".file-notes-git-row-secondary", Static))
            == expected_secondary
        )


@pytest.mark.asyncio
async def test_repository_path_controls_and_rich_markup_are_display_only() -> None:
    malicious_path = "/repo[bold red]OWNED[/bold red]\nFAKE TRUST\t\x1b"
    identity = FileSystemIdentity(1, 2)
    repository = RepositoryIdentity(
        worktree_root=malicious_path,
        git_dir=f"{malicious_path}/.git",
        git_common_dir=f"{malicious_path}/.git",
        worktree_identity=identity,
        git_dir_identity=identity,
        git_common_dir_identity=identity,
    )
    status = SessionGitStatus(
        binding_generation=1,
        status_generation=1,
        state="ready",
        rows=(_row("unstaged", stage_action="stage"),),
        repository=repository,
        head=HeadIdentity.attached("main", "a" * 40),
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(status)
        await pilot.pause()

        rendered = _text(panel.query_one("#file-notes-git-repository", Static))
        assert "repo[bold red]OWNED[/bold red]\\nFAKE TRUST\\t\\x1b" in rendered
        assert "\x1b" not in rendered
        assert status.repository is not None
        assert status.repository.worktree_root == malicious_path


@pytest.mark.asyncio
async def test_trust_dialog_escapes_repository_path_controls_and_markup() -> None:
    malicious_path = "/repo[bold red]OWNED[/bold red]\nFAKE TRUST\t\x1b"
    app = _DialogHarness(SessionGitTrustDialog(malicious_path))
    async with app.run_test() as pilot:
        await pilot.pause()

        rendered = _text(app.dialog.query_one(".dialog-message", Label))
        assert (
            r"repo\[bold red]OWNED\[/bold red]\nFAKE TRUST\t\x1b"
            in rendered
        )
        assert "\x1b" not in rendered


@pytest.mark.parametrize(
    "size",
    [(150, 42), (70, 28), (70, 24), (40, 20)],
)
@pytest.mark.asyncio
async def test_focused_controls_keep_complete_labels_and_fit(
    size: tuple[int, int],
) -> None:
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=size) as pilot:
        panel.render_untrusted("/repo")
        await pilot.pause()
        await _assert_visible_panel_buttons_fit(panel, pilot)

        panel.render_status(
            _status(
                _row(
                    "owned_newer_edits",
                    stage_action="stage_update",
                    unstage_eligible=True,
                )
            )
        )
        await pilot.pause()
        await _assert_visible_panel_buttons_fit(panel, pilot)


@pytest.mark.asyncio
async def test_action_controls_fit_from_visible_label_cells_and_recompute() -> None:
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=(70, 28)) as pilot:
        panel.render_untrusted("/repo")
        await pilot.pause()
        assert not panel.has_class("-stack-actions")

        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert panel.has_class("-stack-actions")
        await _assert_visible_panel_buttons_fit(panel, pilot)

        panel.render_status(
            _status(
                _row(
                    "owned_newer_edits",
                    stage_action="stage_update",
                    unstage_eligible=True,
                )
            )
        )
        await pilot.pause()
        assert not panel.has_class("-stack-actions")
        await _assert_visible_panel_buttons_fit(panel, pilot)


def test_middle_elide_cells_preserves_graphemes_width_and_path_ends() -> None:
    text = "notes/资料/emoji-👩‍💻-very-long-folder/final.md"

    result = _middle_elide_cells(text, 24)

    assert cell_len(result) <= 24
    assert result.startswith("notes/")
    assert result.endswith("final.md")
    assert "..." in result
    assert "👩" not in result or "👩‍💻" in result
    assert "💻" not in result or "👩‍💻" in result
    assert "\u200d" not in result or "👩‍💻" in result


@pytest.mark.asyncio
async def test_middle_elide_recomputes_mounted_path_labels_after_resize() -> None:
    source = "source/" + "before-folder/" * 2 + "before-note.md"
    destination = "destination/" + "after-folder/" * 2 + "final-note.md"
    row = _row(
        "unstaged",
        stage_action="stage",
        latest_action="moved",
        source_path=source,
        destination_path=destination,
    )
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=(150, 42)) as pilot:
        panel.render_status(_status(row))
        await pilot.pause()
        primary = panel.query_one(".file-notes-git-row-primary", Static)
        selected = panel.query_one("#file-notes-git-selected-note", Static)
        assert _text(primary) == f"MOVED     {source} -> {destination}"
        assert _text(selected) == f"Selected note: {source} -> {destination}"

        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert "..." in _text(primary)
        assert _text(primary).startswith("MOVED")
        assert _text(primary).endswith("final-note.md")
        assert "..." in _text(selected)
        assert _text(selected).startswith("Selected note: ")
        assert _text(selected).endswith("final-note.md")

        await pilot.resize_terminal(150, 42)
        await pilot.pause()
        assert _text(primary) == f"MOVED     {source} -> {destination}"
        assert _text(selected) == f"Selected note: {source} -> {destination}"


@pytest.mark.asyncio
async def test_prepare_session_fixed_regions_remain_visible_at_40_by_20() -> None:
    long_source = "source-leading/" + "deep-session-folder/" * 8 + "before-note.md"
    long_destination = (
        "destination-leading/" + "another-deep-folder/" * 8 + "final-note.md"
    )
    long_reason = "index diagnostic " * 18
    rows = (
        _row(
            "owned_newer_edits",
            group_id=1,
            stage_action="stage_update",
            unstage_eligible=True,
            latest_action="moved",
            source_path=long_source,
            destination_path=long_destination,
        ),
        _row(
            "error",
            group_id=2,
            disabled_reason=long_reason,
        ),
        _row(
            "conflict",
            group_id=3,
            disabled_reason=long_reason,
        ),
        _row(
            "unavailable",
            group_id=4,
            disabled_reason=long_reason,
        ),
        *tuple(
            _row(
                "unstaged",
                group_id=group_id,
                stage_action="stage",
            )
            for group_id in range(5, 12)
        ),
    )
    repository = _repository(
        "/repository/" + "very-long-authority-path/" * 10 + "notes"
    )
    status = SessionGitStatus(
        binding_generation=1,
        status_generation=1,
        state="ready",
        rows=rows,
        repository=repository,
        head=HeadIdentity.attached(
            "feature/a-very-long-prepare-session-branch",
            "a" * 40,
        ),
    )
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=(40, 20)) as pilot:
        panel.render_status(status)
        panel.set_current_status(
            "Status: STALE · ERROR — " + long_reason + "Retry Refresh."
        )
        panel.set_last_action(
            "Last action: FAILED — " + long_reason + "retry the action, then Refresh"
        )
        await pilot.pause()

        bounds = panel.content_region
        fixed_selectors = (
            "#file-notes-git-repository",
            "#file-notes-git-status",
            "#file-notes-git-action-status",
            "#file-notes-git-selected-note",
        )
        for selector in fixed_selectors:
            widget = panel.query_one(selector, Static)
            assert widget.display
            assert 0 < widget.region.height <= 2
            assert widget.region.y >= bounds.y
            assert widget.region.bottom <= bounds.bottom
            assert cell_len(_text(widget)) <= (
                widget.content_region.width * widget.region.height
            )

        list_view = panel.query_one("#file-notes-git-rows", ListView)
        assert list_view.region.height >= 1
        assert (
            list_view.region.bottom
            <= panel.query_one(
                "#file-notes-git-status",
                Static,
            ).region.y
        )
        for item in panel.query(".file-notes-git-row"):
            assert item.region.height == 2

        primary = _text(panel.query_one(".file-notes-git-row-primary", Static))
        assert primary.startswith("MOVED")
        assert "source-" in primary
        assert primary.endswith("final-note.md")
        assert "..." in primary
        primary_widget = panel.query_one(
            ".file-notes-git-row-primary",
            Static,
        )
        assert cell_len(primary) <= primary_widget.content_region.width

        selected = panel.query_one("#file-notes-git-selected-note", Static)
        selected_text = _text(selected)
        assert selected_text.startswith("Selected note: ")
        assert selected_text.endswith("final-note.md")
        assert "..." in selected_text

        failure = next(
            widget
            for widget in panel.query(".file-notes-git-row-secondary")
            if _text(widget).startswith("FAILED")
        )
        failure_text = _text(failure)
        assert failure_text.endswith("; retry, Refresh")
        assert cell_len(failure_text) <= failure.content_region.width

        conflict, unavailable = tuple(
            widget
            for widget in panel.query(".file-notes-git-row-secondary")
            if _text(widget).startswith("BLOCKED")
        )
        conflict_text = _text(conflict)
        assert conflict_text == "BLOCKED · Conflict: use Git; Refresh"
        assert cell_len(conflict_text) <= conflict.content_region.width

        unavailable_text = _text(unavailable)
        assert unavailable_text == "BLOCKED · Restore Git first; Refresh"
        assert cell_len(unavailable_text) <= unavailable.content_region.width

        assert _text(panel.query_one("#file-notes-git-status", Static)).endswith(
            "Retry Refresh."
        )
        assert _text(panel.query_one("#file-notes-git-action-status", Static)).endswith(
            "then Refresh"
        )
        await _assert_visible_panel_buttons_fit(panel, pilot)


@pytest.mark.asyncio
async def test_ready_status_without_rows_renders_explicit_empty_state() -> None:
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status())
        await pilot.pause()
        empty = panel.query_one("#file-notes-git-empty", Static)
        assert empty.display
        assert _text(empty) == "No current-session Git changes."


_ROW_REASONS: dict[SessionGitRowState, str] = {
    "conflict": "conflict",
    "unsupported": "skip-worktree",
    "nested_repository": "nested repository",
    "unsafe_closure": "outside session lineage",
    "ambiguous_lineage": "ambiguous lineage",
    "unavailable": "Git unavailable",
    "error": "status failed",
}
_ROW_SEMANTICS: dict[SessionGitRowState, tuple[str, ...]] = {
    "unstaged": ("READY TO STAGE", "Git: unstaged"),
    "owned": ("STAGED", "by Chatbook"),
    "owned_newer_edits": ("UPDATE AVAILABLE", "newer note edits are not staged"),
    "owned_topology_changed": (
        "UPDATE REQUIRED",
        "stage the moved note before unstaging",
    ),
    "external_staged": ("BLOCKED", "already staged outside Chatbook", "Refresh"),
    "external_partial": ("BLOCKED", "already staged outside Chatbook", "Refresh"),
    "clean": ("NO ACTION", "matches HEAD"),
    "ignored": ("BLOCKED", "ignored by Git", "ignore rule", "Refresh"),
    "conflict": ("BLOCKED", "conflict", "outside Chatbook", "Refresh"),
    "unsupported": ("BLOCKED", "skip-worktree", "outside Chatbook", "Refresh"),
    "nested_repository": (
        "BLOCKED",
        "nested repository",
        "outside Chatbook",
        "Refresh",
    ),
    "unsafe_closure": (
        "BLOCKED",
        "outside session lineage",
        "outside Chatbook",
        "Refresh",
    ),
    "ambiguous_lineage": (
        "BLOCKED",
        "ambiguous lineage",
        "outside Chatbook",
        "Refresh",
    ),
    "unavailable": ("BLOCKED", "Git unavailable", "restore Git", "Refresh"),
    "error": ("FAILED", "status failed", "retry", "Refresh"),
}


@pytest.mark.parametrize(
    ("state", "stage_action", "unstage_eligible"),
    [
        ("unstaged", "stage", False),
        ("owned", None, True),
        ("owned_newer_edits", "stage_update", True),
        ("owned_topology_changed", "stage_update", False),
        ("external_staged", None, False),
        ("external_partial", None, False),
        ("clean", None, False),
        ("ignored", None, False),
        ("conflict", None, False),
        ("unsupported", None, False),
        ("nested_repository", None, False),
        ("unsafe_closure", None, False),
        ("ambiguous_lineage", None, False),
        ("unavailable", None, False),
        ("error", None, False),
    ],
)
@pytest.mark.asyncio
async def test_row_action_table_is_driven_by_row_policy(
    state: SessionGitRowState,
    stage_action: SessionGitStageAction | None,
    unstage_eligible: bool,
) -> None:
    row = _row(
        state,
        stage_action=stage_action,
        unstage_eligible=unstage_eligible,
        disabled_reason=_ROW_REASONS.get(state),
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test(size=(150, 42)) as pilot:
        panel.render_status(_status(row))
        await pilot.pause()

        stage = panel.query_one("#file-notes-git-stage-selected", Button)
        unstage = panel.query_one("#file-notes-git-unstage-selected", Button)
        assert stage.display is (stage_action is not None)
        if stage_action is not None:
            expected = "Stage update" if stage_action == "stage_update" else "Stage"
            assert str(stage.label) == expected
            assert not stage.disabled
        assert unstage.display is unstage_eligible
        assert panel.query_one("#file-notes-git-stage-all", Button).disabled is (
            stage_action is None
        )
        assert panel.query_one("#file-notes-git-unstage-all", Button).disabled is (
            not unstage_eligible
        )
        row_text = _text(panel.query_one(".file-notes-git-row-secondary", Static))
        assert all(fragment in row_text for fragment in _ROW_SEMANTICS[state])


@pytest.mark.asyncio
async def test_selection_uses_stable_group_id_across_refresh() -> None:
    first = _row("unstaged", group_id=11, stage_action="stage")
    selected = _row("unstaged", group_id=22, stage_action="stage")
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(first, selected))
        await pilot.pause()
        rows = panel.query_one("#file-notes-git-rows", ListView)
        rows.index = 1
        await pilot.pause()
        assert panel.selected_group_id == 22

        expanded = SessionGitRow(
            SessionChangeGroup(
                group_id=22,
                endpoints=("before.md", "middle.md", "after.md"),
                source_path="before.md",
                destination_path="after.md",
                current_path="after.md",
                latest_action="moved",
                latest_sequence=30,
                move_edges=(
                    ("before.md", "middle.md"),
                    ("middle.md", "after.md"),
                ),
            ),
            "owned_topology_changed",
            stage_action="stage_update",
            disabled_reason="Unstage requires Stage update",
        )
        panel.render_status(_status(expanded, first))
        await pilot.pause()

        assert panel.selected_group_id == 22
        assert rows.index == 0


@pytest.mark.asyncio
async def test_selected_and_bulk_labels_report_selection_and_independent_counts() -> None:
    first = _row(
        "unstaged",
        group_id=11,
        stage_action="stage",
        source_path="folder/first.md",
    )
    selected = _row(
        "owned_newer_edits",
        group_id=22,
        stage_action="stage_update",
        unstage_eligible=True,
        source_path="folder/second.md",
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(first, selected))
        await pilot.pause()

        selected_note = panel.query_one("#file-notes-git-selected-note", Static)
        stage_selected = panel.query_one(
            "#file-notes-git-stage-selected",
            Button,
        )
        unstage_selected = panel.query_one(
            "#file-notes-git-unstage-selected",
            Button,
        )
        assert _text(selected_note).startswith("Selected note: ")
        assert "folder/first.md" in _text(selected_note)
        assert str(stage_selected.label) == "Stage"
        assert (
            str(panel.query_one("#file-notes-git-stage-all", Button).label)
            == "Stage all (2)"
        )
        assert (
            str(panel.query_one("#file-notes-git-unstage-all", Button).label)
            == "Unstage all (1)"
        )

        rows = panel.query_one("#file-notes-git-rows", ListView)
        rows.index = 1
        await pilot.pause()
        assert "folder/second.md" in _text(selected_note)
        assert str(stage_selected.label) == "Stage update"
        assert str(unstage_selected.label) == "Unstage"


@pytest.mark.parametrize("state", ["stale", "error"])
@pytest.mark.asyncio
async def test_stale_and_error_retain_rows_but_only_refresh_is_available(
    state: str,
) -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.set_last_action("Last action: STAGED — 1 session note staged.")
        panel.render_status(
            _status(
                _row("unstaged", stage_action="stage"),
                state=state,
                message="status failed; retry",
            ),
            retain_rows=True,
        )
        await pilot.pause()

        assert len(panel.query(".file-notes-git-row")) == 1
        assert not panel.query_one("#file-notes-git-refresh", Button).disabled
        assert panel.query_one("#file-notes-git-stage-selected", Button).disabled
        assert panel.query_one("#file-notes-git-stage-all", Button).disabled
        expected = "STALE" if state == "stale" else "STALE · ERROR"
        assert expected in _text(panel.query_one("#file-notes-git-status", Static))
        assert "status failed; retry" in _text(
            panel.query_one("#file-notes-git-status", Static)
        )
        assert (
            _text(panel.query_one("#file-notes-git-action-status", Static))
            == "Last action: STAGED — 1 session note staged."
        )


@pytest.mark.parametrize(
    "authority_loss",
    [
        lambda panel: panel.render_untrusted("/different/repository"),
        lambda panel: panel.render_unavailable(
            "This notes folder is not in a Git worktree."
        ),
        lambda panel: panel.render_status(
            _status(
                state="unavailable",
                message="Git is unavailable; restore Git, then Refresh.",
            )
        ),
    ],
    ids=("untrusted", "discovery-unavailable", "status-unavailable"),
)
@pytest.mark.asyncio
async def test_authority_loss_clears_rows_selection_and_mutation_actions(
    authority_loss: Callable[[LibraryFileNotesGitPanel], None],
) -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(
            _status(
                _row("unstaged", group_id=11, stage_action="stage"),
                _row(
                    "owned",
                    group_id=22,
                    unstage_eligible=True,
                ),
            )
        )
        await pilot.pause()
        rows = panel.query_one("#file-notes-git-rows", ListView)
        rows.index = 1
        await pilot.pause()
        assert panel.selected_group_id == 22

        authority_loss(panel)
        await _wait_until(
            pilot,
            lambda: len(panel.query(".file-notes-git-row")) == 0,
            "authority loss did not clear rendered rows",
        )

        assert panel.rows == ()
        assert panel.selected_group_id is None
        for selector in (
            "#file-notes-git-stage-selected",
            "#file-notes-git-unstage-selected",
            "#file-notes-git-stage-all",
            "#file-notes-git-unstage-all",
        ):
            button = panel.query_one(selector, Button)
            assert not button.display


@pytest.mark.asyncio
async def test_untrusted_shows_only_trust_action_and_checking_keeps_back_enabled() -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_untrusted("/canonical/repository")
        await pilot.pause()
        visible_actions = {
            button.id
            for button in panel.query(Button)
            if button.display
        }
        assert visible_actions == {
            "file-notes-git-back",
            "file-notes-git-trust",
        }

        panel.render_checking("/canonical/repository")
        await pilot.pause()
        assert not panel.query_one("#file-notes-git-back", Button).disabled
        assert panel.query_one("#file-notes-git-stage-all", Button).disabled
        assert "Checking" in _text(
            panel.query_one("#file-notes-git-status", Static)
        )


@pytest.mark.asyncio
async def test_buttons_emit_typed_messages_with_selected_and_bulk_group_ids() -> None:
    panel = LibraryFileNotesGitPanel()
    app = _PanelHarness(panel)
    async with app.run_test() as pilot:
        panel.render_status(
            _status(
                _row("unstaged", group_id=4, stage_action="stage"),
                _row("owned", group_id=8, unstage_eligible=True),
            )
        )
        await pilot.pause()

        for selector in (
            "#file-notes-git-stage-selected",
            "#file-notes-git-stage-all",
            "#file-notes-git-unstage-all",
            "#file-notes-git-refresh",
            "#file-notes-git-back",
        ):
            expected = len(app.messages) + 1
            panel.query_one(selector, Button).press()
            await _wait_until(
                pilot,
                lambda: len(app.messages) == expected,
                f"{selector} message not posted",
            )

        assert isinstance(app.messages[0], LibraryFileNotesGitPanel.StageRequested)
        assert app.messages[0].group_ids == (4,)
        assert not app.messages[0].bulk
        assert isinstance(app.messages[1], LibraryFileNotesGitPanel.StageRequested)
        assert app.messages[1].group_ids == (4,)
        assert app.messages[1].bulk
        assert isinstance(app.messages[2], LibraryFileNotesGitPanel.UnstageRequested)
        assert app.messages[2].group_ids == (8,)
        assert app.messages[2].bulk
        assert isinstance(app.messages[3], LibraryFileNotesGitPanel.RefreshRequested)
        assert isinstance(app.messages[4], LibraryFileNotesGitPanel.BackRequested)


@pytest.mark.asyncio
async def test_keyboard_moves_rows_and_focus_without_implicit_enter_action() -> None:
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _PanelHarness(panel)
    async with app.run_test() as pilot:
        panel.render_status(
            _status(
                _row("unstaged", group_id=4, stage_action="stage"),
                _row("unstaged", group_id=8, stage_action="stage"),
            )
        )
        await pilot.pause()
        rows = panel.query_one("#file-notes-git-rows", ListView)
        rows.focus()
        await pilot.press("down")
        assert panel.selected_group_id == 8
        await pilot.press("enter")
        await pilot.pause()
        assert app.messages == []

        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is not rows
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.focused is rows

        panel.query_one("#file-notes-git-stage-all", Button).focus()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(app.messages) == 1, "Enter did not act")
        assert isinstance(app.messages[0], LibraryFileNotesGitPanel.StageRequested)
        assert app.messages[0].group_ids == (4, 8)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: len(app.messages) == 2, "Escape did not Back")
        assert isinstance(app.messages[1], LibraryFileNotesGitPanel.BackRequested)


@pytest.mark.asyncio
async def test_trust_dialog_is_explicit_and_cancel_focused() -> None:
    app = _DialogHarness(SessionGitTrustDialog("/canonical/repository"))
    async with app.run_test() as pilot:
        await pilot.pause()
        dialog = app.dialog
        assert dialog.query_one("#cancel-button", Button).has_focus
        message = _text(dialog.query_one(".dialog-message", Label))
        assert "/canonical/repository" in message
        assert "application process" in message
        assert "configured Git filters" in message
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is False


@pytest.mark.asyncio
async def test_workspace_retains_files_search_and_git_modes_with_back_focus(
    tmp_path: Path,
) -> None:
    root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        files_tree = workspace.query_one("#file-notes-tree", Tree)
        search_tree = workspace.query_one("#file-notes-search-results", Tree)
        search = workspace.query_one("#file-notes-search", Input)
        panel = workspace.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        entry = workspace.query_one("#file-notes-session-changes", Button)
        assert str(entry.label) == "Session Git (2)"
        assert entry.can_focus

        search.value = "needle"
        await _wait_until(pilot, lambda: search_tree.display, "search mode not shown")
        entry.press()
        await _wait_until(
            pilot,
            lambda: panel.display and len(git_service.status_calls) == 1,
            "Session Git mode did not open and refresh",
        )
        assert workspace.query_one("#file-notes-tree", Tree) is files_tree
        assert workspace.query_one("#file-notes-search-results", Tree) is search_tree
        assert not files_tree.display
        assert not search_tree.display
        assert not search.display

        panel.query_one("#file-notes-git-back", Button).press()
        await _wait_until(
            pilot,
            lambda: search_tree.display and entry.has_focus,
            "Back did not restore search mode and Session Git focus",
        )
        assert search.value == "needle"
        assert workspace.query_one("#file-notes-git-panel") is panel
        assert not panel.display
    await workspace.shutdown()
    owner.shutdown()
    replica.close()
    assert root.exists()


@pytest.mark.asyncio
async def test_thousand_unrelated_notes_send_only_three_session_groups_and_restore_files_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    unrelated_root = root / "archive"
    unrelated_root.mkdir(parents=True)
    (root / ".git").mkdir()
    for index in range(1_005):
        (unrelated_root / f"unrelated-{index:04d}.md").write_text(
            f"scale marker {index:04d}\n",
            encoding="utf-8",
        )
    session_paths = ("note-1.md", "note-2.md", "note-3.md")
    for path in session_paths:
        (root / path).write_text(f"{path}\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    for path in session_paths:
        assert owner.record_change(binding, SessionChange("modified", path))
    runner = _PathspecRecordingRunner(root)
    git_service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    owner.attach_git_service(git_service)
    discovery = await git_service.discover(binding)
    assert discovery.repository is not None
    assert owner.publish_trust(binding, discovery.repository)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
        autosave_delay=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        files_tree = workspace.query_one("#file-notes-tree", Tree)
        search_tree = workspace.query_one("#file-notes-search-results", Tree)
        archive_node = next(
            node
            for node in files_tree.root.children
            if node.data == ("folder", "archive")
        )
        search = workspace.query_one("#file-notes-search", Input)
        search.value = "scale marker 1004"
        await _wait_until(pilot, lambda: search_tree.display, "search mode not shown")
        files_tree.select_node(archive_node)
        await pilot.pause()
        tree_children_before = tuple(
            node.data for node in files_tree.root.children
        )
        entry = workspace.query_one("#file-notes-session-changes", Button)
        assert str(entry.label) == "Session Git (3)"

        entry.press()
        await _wait_until(
            pilot,
            lambda: len(workspace._git_panel_widget.rows) == 3,
            "Session Git did not request the three session groups",
        )
        status_calls = tuple(
            call
            for call in runner.calls
            if "status" in tuple(os.fsdecode(argument) for argument in call)
        )
        assert len(status_calls) == 1
        status_argv = status_calls[0]
        boundary = status_argv.index("--")
        assert status_argv[boundary + 1 :] == tuple(
            os.fsencode(path) for path in session_paths
        )
        assert len(workspace._git_panel_widget.rows) == 3
        assert {
            row.group.current_path for row in workspace._git_panel_widget.rows
        } == set(session_paths)
        assert all(
            b"unrelated-" not in os.fsencode(argument)
            for argument in status_argv
        )

        workspace.query_one("#file-notes-git-back", Button).press()
        await _wait_until(
            pilot,
            lambda: search_tree.display and entry.has_focus,
            "Back did not restore the prior search view",
        )
        assert workspace.query_one("#file-notes-tree", Tree) is files_tree
        assert workspace.query_one("#file-notes-search-results", Tree) is search_tree
        assert search.value == "scale marker 1004"
        assert files_tree.cursor_node is archive_node
        assert tuple(node.data for node in files_tree.root.children) == (
            tree_children_before
        )

    await workspace.shutdown()
    await owner.shutdown_async()
    replica.close()


@pytest.mark.asyncio
async def test_opening_session_git_moves_focus_to_a_visible_ready_control(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.status_release = asyncio.Event()
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        entry = workspace.query_one("#file-notes-session-changes", Button)
        entry.focus()
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: workspace.query_one("#file-notes-git-back", Button).has_focus,
            "first open did not focus the visible Back control",
        )
        assert not entry.display

        git_service.status_release.set()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1
            and len(workspace._git_panel_widget.rows) == 2,
            "initial status did not finish",
        )
        workspace.query_one("#file-notes-git-back", Button).press()
        await pilot.pause()
        entry.focus()
        entry.press()
        rows = workspace.query_one("#file-notes-git-rows", ListView)
        await _wait_until(
            pilot,
            lambda: rows.has_focus,
            "reopen with retained rows did not focus the row list",
        )
        await pilot.press("down")
        assert workspace._git_panel_widget.selected_group_id == 2
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_reopening_cached_status_keeps_mutation_controls_disabled(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.action_release = asyncio.Event()
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        entry = workspace.query_one("#file-notes-session-changes", Button)
        entry.press()
        await _wait_until(
            pilot,
            lambda: len(workspace._git_panel_widget.rows) == 2,
            "initial status did not finish",
        )
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: owner.mutation_active(binding),
            "Stage did not retain the mutation gate",
        )

        try:
            workspace.query_one("#file-notes-git-back", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._navigator_mode != "git",
                "Back did not hide Session Git",
            )
            entry.press()
            await _wait_until(
                pilot,
                lambda: len(git_service.discovery_calls) >= 2,
                "Session Git did not rediscover on reopen",
            )
            await pilot.pause()

            assert workspace.query_one(
                "#file-notes-git-stage-selected",
                Button,
            ).disabled
            assert workspace.query_one(
                "#file-notes-git-stage-all",
                Button,
            ).disabled
            assert git_service.stage_calls == [(1,)]
        finally:
            git_service.action_release.set()
            await _wait_until(
                pilot,
                lambda: not owner.mutation_active(binding),
                "Stage did not settle during cleanup",
            )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_reopening_cached_status_keeps_controls_disabled_during_refresh(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        entry = workspace.query_one("#file-notes-session-changes", Button)
        entry.press()
        await _wait_until(
            pilot,
            lambda: len(workspace._git_panel_widget.rows) == 2,
            "initial status did not finish",
        )
        git_service.status_release = asyncio.Event()
        workspace.query_one("#file-notes-git-refresh", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 2,
            "retained refresh did not start",
        )

        try:
            workspace.query_one("#file-notes-git-back", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._navigator_mode != "git",
                "Back did not hide Session Git",
            )
            entry.press()
            await _wait_until(
                pilot,
                lambda: len(git_service.discovery_calls) >= 2,
                "Session Git did not rediscover on reopen",
            )
            await pilot.pause()

            assert workspace.query_one(
                "#file-notes-git-stage-selected",
                Button,
            ).disabled
            assert workspace.query_one(
                "#file-notes-git-stage-all",
                Button,
            ).disabled
            assert len(git_service.status_calls) == 2
        finally:
            git_service.status_release.set()
            await _wait_until(
                pilot,
                lambda: owner.snapshot(binding).git_status is not None
                and owner.snapshot(binding).git_status.status_generation >= 2,
                "retained refresh did not settle during cleanup",
            )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_fresh_workspace_attaches_retained_status_before_cached_ready_state(
    tmp_path: Path,
) -> None:
    root, owner, binding, replica, git_service, _first_workspace = (
        _workspace_fixture(tmp_path)
    )
    changes = owner.snapshot(binding).changes
    await git_service.start_status(binding, changes)
    git_service.status_release = asyncio.Event()
    retained_task = git_service.start_status(binding, changes)
    assert git_service.retained_status(binding) is retained_task
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
        autosave_delay=10,
    )
    try:
        async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "fresh workspace scan did not finish",
            )
            workspace.query_one("#file-notes-session-changes", Button).press()
            await _wait_until(
                pilot,
                lambda: "Checking Session Git status"
                in _text(
                    workspace.query_one(
                        "#file-notes-git-action-status",
                        Static,
                    )
                ),
                "fresh workspace rendered cached status during retained work",
            )

            _assert_git_mutations_disabled(workspace)
            assert len(git_service.status_calls) == 2

            assert git_service.status_release is not None
            git_service.status_release.set()
            await _wait_until(
                pilot,
                lambda: owner.snapshot(binding).git_status is not None
                and owner.snapshot(binding).git_status.status_generation >= 2
                and len(workspace._git_panel_widget.rows) == 2,
                "fresh workspace did not render the retained status result",
            )
            assert len(git_service.status_calls) == 2
    finally:
        assert git_service.status_release is not None
        git_service.status_release.set()
        await asyncio.shield(retained_task)
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_repository_retrust_consumes_rejected_status_before_fresh_refresh(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    original_repository = git_service.repository
    original_release = asyncio.Event()
    git_service.status_release = original_release
    replacement_repository = _repository(
        "/canonical/replacement",
        identity=FileSystemIdentity(3, 4),
    )
    replacement_release = asyncio.Event()

    try:
        async with _WorkspaceHarness(workspace).run_test(
            size=(120, 40)
        ) as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "scan did not finish",
            )
            entry = workspace.query_one("#file-notes-session-changes", Button)
            entry.press()
            await _wait_until(
                pilot,
                lambda: len(git_service.status_calls) == 1,
                "original status did not start",
            )
            rejected_task = git_service._status_task
            assert rejected_task is not None

            assert owner.clear_trust_if_matches(binding, original_repository)
            git_service.repository = replacement_repository
            original_release.set()
            await asyncio.shield(rejected_task)
            await pilot.pause()
            assert owner.snapshot(binding).git_status is None

            workspace.query_one("#file-notes-git-back", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._navigator_mode != "git",
                "Back did not hide Session Git",
            )
            git_service.status_release = replacement_release
            entry.press()
            await _wait_until(
                pilot,
                lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
                "replacement repository trust prompt did not open",
            )
            workspace.app.screen.query_one("#confirm-button", Button).press()
            await _wait_until(
                pilot,
                lambda: len(git_service.status_calls) == 2,
                "replacement trust did not start a fresh status",
            )

            assert (
                owner.snapshot(binding).trusted_repository
                == replacement_repository
            )
            _assert_git_mutations_disabled(workspace)

            replacement_release.set()
            await _wait_until(
                pilot,
                lambda: owner.snapshot(binding).git_status is not None
                and owner.snapshot(binding).git_status.repository
                == replacement_repository
                and "Status ready"
                in _text(
                    workspace.query_one(
                        "#file-notes-git-action-status",
                        Static,
                    )
                ),
                "replacement repository status did not render",
            )
            assert len(git_service.status_calls) == 2
    finally:
        original_release.set()
        replacement_release.set()
        retained_task = git_service._status_task
        if retained_task is not None and not retained_task.done():
            await asyncio.shield(retained_task)
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_hidden_action_summary_is_presented_after_reopen(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.action_release = asyncio.Event()
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        entry = workspace.query_one("#file-notes-session-changes", Button)
        entry.press()
        await _wait_until(
            pilot,
            lambda: len(workspace._git_panel_widget.rows) == 2,
            "initial status did not finish",
        )
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: owner.mutation_active(binding),
            "Stage did not retain the mutation gate",
        )
        workspace.query_one("#file-notes-git-back", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace._navigator_mode != "git",
            "Back did not hide Session Git",
        )
        git_service.action_release.set()
        await _wait_until(
            pilot,
            lambda: not owner.mutation_active(binding),
            "hidden Stage did not settle",
        )
        assert len(git_service.status_calls) == 1

        entry.press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 2,
            "reopen did not refresh the hidden action result",
        )
        await _wait_until(
            pilot,
            lambda: "Staged 1 · clean 0 · blocked 0"
            in _text(
                workspace.query_one("#file-notes-git-action-status", Static)
            ),
            "reopen did not present the retained action summary",
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_remount_rehydrates_status_that_finished_while_unmounted(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.status_release = asyncio.Event()
    app = _RemountWorkspaceHarness(workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "retained status did not start",
        )
        assert "Checking Session Git status" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )

        host = app.query_one("#remount-workspace-host", Vertical)
        await host.remove_children()
        await _wait_until(
            pilot,
            lambda: not workspace._active and not tuple(host.children),
            "workspace did not unmount",
        )
        git_service.status_release.set()
        await _wait_until(
            pilot,
            lambda: owner.snapshot(binding).git_status is not None,
            "service did not publish status while unmounted",
        )
        await host.mount(workspace)
        await _wait_until(
            pilot,
            lambda: workspace.is_mounted and workspace._active,
            "workspace did not remount",
        )
        await pilot.pause()

        assert len(workspace._git_panel_widget.rows) == 2
        assert "Status ready" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )
        assert len(git_service.status_calls) == 1
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_workspace_trust_decline_runs_no_status_and_retry_revalidates(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path,
        trusted=False,
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
            "initial trust dialog did not open",
        )
        await pilot.press("escape")
        await pilot.pause()
        assert git_service.status_calls == []
        panel = workspace.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        assert panel.display
        assert panel.query_one("#file-notes-git-trust", Button).display
        assert owner.snapshot(binding).trusted_repository is None

        panel.query_one("#file-notes-git-trust", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
            "cancel retry did not reopen the prompt",
        )
        workspace.app.screen.query_one("#cancel-button", Button).press()
        await pilot.pause()
        assert git_service.status_calls == []

        panel.query_one("#file-notes-git-trust", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
            "close retry did not reopen the prompt",
        )
        workspace.app.pop_screen()
        await pilot.pause()
        assert git_service.status_calls == []

        panel.query_one("#file-notes-git-trust", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
            "trust retry did not reopen the prompt",
        )
        workspace.app.screen.query_one("#confirm-button", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "accepted trust did not start status",
        )
        assert git_service.revalidate_calls == [(binding, git_service.repository)]
        assert owner.snapshot(binding).trusted_repository == git_service.repository
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_hidden_session_mutation_marks_stale_without_status_until_reopen(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        workspace.query_one("#file-notes-git-back", Button).press()
        await pilot.pause()
        assert owner.record_change(binding, SessionChange("modified", "late.md"))
        workspace._refresh_session_changes()
        await pilot.pause(0.1)
        assert len(git_service.status_calls) == 1
        assert owner.snapshot(binding).git_status is None

        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 2,
            "reopening stale Session Git did not refresh",
        )
        assert git_service.status_calls[-1][-1].change.relative_path == "late.md"
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_stage_flushes_then_gate_keeps_editor_back_and_one_latest_refresh(
    tmp_path: Path,
) -> None:
    root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("folder/one.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        editor.select_all()
        editor.replace("saved before stage", editor.selection.start, editor.selection.end)
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        git_service.action_release = asyncio.Event()
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: git_service.stage_calls == [(1,)]
            and owner.mutation_active(binding),
            "Stage did not flush and acquire the mutation gate",
        )
        assert (root / "folder" / "one.md").read_text(encoding="utf-8") == (
            "saved before stage"
        )
        assert not editor.read_only
        editor.focus()
        await pilot.press("x")
        assert not workspace.query_one("#file-notes-git-back", Button).disabled
        assert workspace.query_one("#file-notes-new", Button).disabled
        assert workspace.query_one("#file-notes-move", Button).disabled
        assert not workspace.query_one("#file-notes-protect", Button).disabled
        assert not await workspace.flush_pending_work()

        assert owner.record_change(binding, SessionChange("modified", "latest.md"))
        workspace._refresh_session_changes()
        workspace.query_one("#file-notes-git-refresh", Button).press()
        workspace._refresh_session_changes()
        await pilot.pause(0.1)
        assert len(git_service.status_calls) == 1

        git_service.action_release.set()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 2
            and not owner.mutation_active(binding),
            "postflight did not perform exactly one latest visible refresh",
        )
        await pilot.pause(0.1)
        assert len(git_service.status_calls) == 2
        assert git_service.status_calls[-1][-1].change.relative_path == "latest.md"
        assert "Staged 1 · clean 0 · blocked 0" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_stage_all_summary_counts_the_complete_displayed_snapshot(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.rows = (
        _row("unstaged", group_id=1, stage_action="stage"),
        _row("owned", group_id=2, unstage_eligible=True),
        _row("clean", group_id=3),
        _row("conflict", group_id=4, disabled_reason="conflict"),
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )

        workspace.query_one("#file-notes-git-stage-all", Button).press()
        await _wait_until(
            pilot,
            lambda: git_service.stage_calls == [(1,)]
            and len(git_service.status_calls) == 2,
            "Stage All did not settle and refresh",
        )

        assert (
            _text(workspace.query_one("#file-notes-git-action-status", Static))
            == "Staged 1 · already staged 1 · clean 1 · blocked 1"
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_unstage_all_summary_counts_the_complete_displayed_snapshot(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.rows = (
        _row("owned", group_id=1, unstage_eligible=True),
        _row("unstaged", group_id=2, stage_action="stage"),
        _row("clean", group_id=3),
        _row("conflict", group_id=4, disabled_reason="conflict"),
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )

        workspace.query_one("#file-notes-git-unstage-all", Button).press()
        await _wait_until(
            pilot,
            lambda: git_service.unstage_calls == [(1,)]
            and len(git_service.status_calls) == 2,
            "Unstage All did not settle and refresh",
        )

        assert (
            _text(workspace.query_one("#file-notes-git-action-status", Static))
            == "Unstaged 1 · skipped 1 · clean 1 · blocked 1"
        )
        selected_context = workspace._git_action_summary_context(
            "stage",
            (1,),
            bulk=False,
        )
        assert (
            workspace._git_action_summary(
                GitActionResult(
                    "stage",
                    "uncertain",
                    (1,),
                    blocked_group_ids=(1,),
                    message="Git Stage outcome is uncertain",
                ),
                selected_context,
            )
            == "Git Stage outcome is uncertain"
        )
        uncertain = workspace._git_action_summary(
            GitActionResult(
                "stage",
                "uncertain",
                (1,),
                blocked_group_ids=(1,),
            ),
            selected_context,
        )
        assert uncertain == "Stage uncertain · clean 0 · blocked 1"
        assert "Staged" not in uncertain
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_unavailable_discovery_exposes_no_trust_or_mutation_action(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path,
        trusted=False,
    )
    git_service.discovery_result = DiscoveryResult(
        "unavailable",
        message="Git is not installed",
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        panel = workspace.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        await _wait_until(
            pilot,
            lambda: "Git is not installed"
            in _text(panel.query_one("#file-notes-git-action-status", Static)),
            "Git discovery failure was not rendered",
        )
        visible_actions = {
            button.id
            for button in panel.query(Button)
            if button.display
        }
        assert visible_actions == {"file-notes-git-back"}
        assert git_service.status_calls == []
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_hidden_postflight_starts_no_status_and_transition_wins_admission(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        transition = owner.try_acquire_transition(binding, "path")
        assert transition is not None
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await pilot.pause(0.1)
        assert git_service.stage_calls == []
        transition.release()

        git_service.action_release = asyncio.Event()
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: owner.mutation_active(binding),
            "Stage mutation did not start",
        )
        workspace.query_one("#file-notes-git-back", Button).press()
        await pilot.pause()
        git_service.action_release.set()
        await _wait_until(
            pilot,
            lambda: not owner.mutation_active(binding),
            "hidden mutation did not settle",
        )
        await pilot.pause(0.1)
        assert len(git_service.status_calls) == 1
        assert owner.snapshot(binding).git_status is None
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_identity_change_after_trust_runs_no_status(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path,
        trusted=False,
    )
    git_service.revalidate_result = False
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(workspace.app.screen, SessionGitTrustDialog),
            "trust dialog did not open",
        )
        workspace.app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause(0.1)
        assert git_service.status_calls == []
        assert owner.snapshot(binding).trusted_repository is None
        assert "identity changed" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_unstage_selected_reports_counts_and_refreshes_once(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.rows = (_row("owned", group_id=1, unstage_eligible=True),)
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        workspace.query_one("#file-notes-git-unstage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: git_service.unstage_calls == [(1,)]
            and len(git_service.status_calls) == 2,
            "Unstage did not settle and refresh once",
        )
        assert "Unstaged 1 · clean 0 · blocked 0" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_back_during_status_and_file_poll_start_no_extra_git_work(
    tmp_path: Path,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    git_service.status_release = asyncio.Event()
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "delayed status did not start",
        )
        workspace.query_one("#file-notes-git-back", Button).press()
        await pilot.pause()
        assert workspace.query_one("#file-notes-session-changes", Button).has_focus
        assert owner.snapshot(binding).git_status is None
        git_service.status_release.set()
        await _wait_until(
            pilot,
            lambda: owner.snapshot(binding).git_status is not None,
            "retained hidden status did not finish into the owner",
        )
        workspace._start_poll()
        await _wait_until(
            pilot,
            lambda: workspace._poll_worker is not None
            and workspace._poll_worker.is_finished,
            "File Notes poll did not settle",
        )
        assert len(git_service.status_calls) == 1
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_mutation_blocks_root_and_path_while_path_lease_blocks_mutation(
    tmp_path: Path,
) -> None:
    root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("two.md")
        delete_button = workspace.query_one("#file-notes-delete", Button)
        delete_button.press()
        await _wait_until(
            pilot,
            lambda: workspace._delete_confirmation_path == "two.md",
            "delete did not enter its confirmed state",
        )
        with workspace._hold_path_transition() as transition:
            assert transition is not None
            assert owner.try_acquire_mutation(binding) is None

        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        git_service.action_release = asyncio.Event()
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await _wait_until(
            pilot,
            lambda: owner.mutation_active(binding),
            "mutation did not start",
        )
        assert not await workspace.set_root(replacement, persist=False)
        assert workspace.root == root.resolve()
        assert (
            _text(workspace.query_one("#file-notes-action-status", Static))
            == "Session Git mutation in progress; structural actions are busy."
        )

        workspace._set_action_status("")
        await workspace._delete_file(Button.Pressed(delete_button))
        assert (root / "two.md").exists()
        assert workspace._opened is not None
        assert workspace._opened.relative_path == "two.md"
        assert (
            _text(workspace.query_one("#file-notes-action-status", Static))
            == "Session Git mutation in progress; structural actions are busy."
        )

        workspace._set_action_status("")
        assert not await workspace.open_path("two.md")
        assert (
            _text(workspace.query_one("#file-notes-action-status", Static))
            == "Session Git mutation in progress; structural actions are busy."
        )
        assert workspace.query_one("#file-notes-choose-root", Button).disabled
        assert workspace.query_one("#file-notes-save-copy", Button).disabled
        git_service.action_release.set()
        await _wait_until(
            pilot,
            lambda: not owner.mutation_active(binding),
            "mutation did not settle",
        )
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_stage_rechecks_transition_admission_after_flush_await(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, owner, binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "initial status did not finish",
        )
        lease = None

        async def flush_then_transition() -> bool:
            nonlocal lease
            await asyncio.sleep(0)
            lease = owner.try_acquire_transition(binding, "path")
            assert lease is not None
            return True

        monkeypatch.setattr(workspace, "flush_pending_work", flush_then_transition)
        workspace.query_one("#file-notes-git-stage-selected", Button).press()
        await pilot.pause(0.1)
        assert git_service.stage_calls == []
        assert "blocked" in _text(
            workspace.query_one("#file-notes-git-action-status", Static)
        )
        assert lease is not None
        lease.release()
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_narrow_git_navigation_retains_editor_search_tree_and_row_selection(
    tmp_path: Path,
) -> None:
    _root, owner, _binding, replica, git_service, workspace = _workspace_fixture(
        tmp_path
    )
    async with _WorkspaceHarness(workspace).run_test(size=(70, 35)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("folder/one.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        editor.cursor_location = (0, 3)
        editor.selection = editor.selection.__class__((0, 1), (0, 5))
        body = editor.text
        cursor = editor.cursor_location
        selection = editor.selection
        workspace.query_one("#file-notes-back", Button).press()
        await pilot.pause()
        search = workspace.query_one("#file-notes-search", Input)
        search.value = "needle"
        await pilot.pause()
        tree = workspace.query_one("#file-notes-tree", Tree)
        folder = next(node for node in tree.root.children if node.data == ("folder", "folder"))
        folder.expand()
        workspace.query_one("#file-notes-session-changes", Button).press()
        await _wait_until(
            pilot,
            lambda: len(git_service.status_calls) == 1,
            "Session Git did not refresh",
        )
        panel = workspace.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        panel.query_one("#file-notes-git-rows", ListView).index = 1
        await pilot.pause()
        assert panel.selected_group_id == 2
        panel.query_one("#file-notes-git-back", Button).press()
        await pilot.pause()

        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        assert editor.text == body
        assert editor.cursor_location == cursor
        assert editor.selection == selection
        assert search.value == "needle"
        assert folder.is_expanded
        assert panel.selected_group_id == 2
    await workspace.shutdown()
    owner.shutdown()
    replica.close()
