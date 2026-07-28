"""Mounted behavior tests for the File Notes Session Git navigator."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label, ListView, Static, TextArea, Tree

sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from tldw_chatbook.Notes.file_notes_git_service import (  # noqa: E402
    DiscoveryResult,
    GitActionResult,
    GitMutationAdmissionError,
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
    SessionChangeGroup,
    SessionGitRow,
    SessionGitRowState,
    SessionGitStageAction,
    SessionGitStatus,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (  # noqa: E402
    LibraryFileNotesGitPanel,
    SessionGitTrustDialog,
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

    def start_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
    ) -> asyncio.Task[SessionGitStatus]:
        self.status_calls.append(tuple(changes))

        async def finish() -> SessionGitStatus:
            if self.status_release is not None:
                await self.status_release.wait()
            generation = self.owner.next_status_generation(binding)
            assert generation is not None
            status = SessionGitStatus(
                binding_generation=binding.generation,
                status_generation=generation,
                state="ready",
                rows=self.rows,
                repository=self.repository,
                head=self.head,
            )
            assert self.owner.publish_status(binding, status)
            return status

        return asyncio.create_task(finish())

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


def _repository() -> RepositoryIdentity:
    identity = FileSystemIdentity(1, 2)
    return RepositoryIdentity(
        worktree_root="/canonical/repository",
        git_dir="/canonical/repository/.git",
        git_common_dir="/canonical/repository/.git",
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
) -> SessionGitRow:
    return SessionGitRow(
        SessionChangeGroup(
            group_id=group_id,
            endpoints=(f"note-{group_id}.md",),
            source_path=f"note-{group_id}.md",
            destination_path=None,
            current_path=f"note-{group_id}.md",
            latest_action="modified",
            latest_sequence=group_id,
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


@pytest.mark.asyncio
async def test_panel_renders_repository_scope_and_complete_file_state() -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(_row("unstaged", stage_action="stage")))
        await pilot.pause()

        assert "/canonical/repository" in _text(
            panel.query_one("#file-notes-git-repository", Static)
        )
        assert "feature/session-git" in _text(
            panel.query_one("#file-notes-git-repository", Static)
        )
        assert (
            _text(panel.query_one("#file-notes-git-scope", Static))
            == "Session paths only"
        )
        assert "content, deletion, and mode" in _text(
            panel.query_one("#file-notes-git-complete-state", Static)
        )


@pytest.mark.parametrize(
    (
        "row",
        "stage_label",
        "unstage_enabled",
        "stage_all",
        "unstage_all",
        "reason",
    ),
    [
        (_row("unstaged", stage_action="stage"), "Stage selected", False, True, False, None),
        (_row("owned", unstage_eligible=True), None, True, False, True, None),
        (
            _row(
                "owned_newer_edits",
                stage_action="stage_update",
                unstage_eligible=True,
            ),
            "Stage update",
            True,
            True,
            True,
            None,
        ),
        (
            _row(
                "owned_topology_changed",
                stage_action="stage_update",
                disabled_reason="Unstage requires Stage update",
            ),
            "Stage update",
            False,
            True,
            False,
            "Unstage requires Stage update",
        ),
        (_row("external_staged", disabled_reason="external index state"), None, False, False, False, "external index state"),
        (_row("external_partial", disabled_reason="external index state"), None, False, False, False, "external index state"),
        (_row("clean"), None, False, False, False, None),
        (_row("ignored", disabled_reason="ignored"), None, False, False, False, "ignored"),
        (_row("conflict", disabled_reason="conflict"), None, False, False, False, "conflict"),
        (_row("unsupported", disabled_reason="skip-worktree"), None, False, False, False, "skip-worktree"),
        (_row("nested_repository", disabled_reason="nested repository"), None, False, False, False, "nested repository"),
        (_row("unsafe_closure", disabled_reason="outside session lineage"), None, False, False, False, "outside session lineage"),
        (_row("ambiguous_lineage", disabled_reason="ambiguous lineage"), None, False, False, False, "ambiguous lineage"),
        (_row("unavailable", disabled_reason="Git unavailable"), None, False, False, False, "Git unavailable"),
        (_row("error", disabled_reason="status failed"), None, False, False, False, "status failed"),
    ],
)
@pytest.mark.asyncio
async def test_row_action_table_is_driven_by_row_policy(
    row: SessionGitRow,
    stage_label: str | None,
    unstage_enabled: bool,
    stage_all: bool,
    unstage_all: bool,
    reason: str | None,
) -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(row))
        await pilot.pause()

        stage = panel.query_one("#file-notes-git-stage-selected", Button)
        unstage = panel.query_one("#file-notes-git-unstage-selected", Button)
        assert stage.display is (stage_label is not None)
        if stage_label is not None:
            assert str(stage.label) == stage_label
            assert not stage.disabled
        assert unstage.display is unstage_enabled
        assert panel.query_one("#file-notes-git-stage-all", Button).disabled is (
            not stage_all
        )
        assert panel.query_one("#file-notes-git-unstage-all", Button).disabled is (
            not unstage_all
        )
        row_text = _text(panel.query_one(".file-notes-git-row-copy", Static))
        if reason is not None:
            assert reason in row_text


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


@pytest.mark.parametrize("state", ["stale", "error"])
@pytest.mark.asyncio
async def test_stale_and_error_retain_rows_but_only_refresh_is_available(
    state: str,
) -> None:
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(
            _status(
                _row("unstaged", stage_action="stage"),
                state=state,
                message="status failed; retry",
            )
        )
        await pilot.pause()

        assert len(panel.query(".file-notes-git-row")) == 1
        assert not panel.query_one("#file-notes-git-refresh", Button).disabled
        assert panel.query_one("#file-notes-git-stage-selected", Button).disabled
        assert panel.query_one("#file-notes-git-stage-all", Button).disabled
        assert "status failed; retry" in _text(
            panel.query_one("#file-notes-git-action-status", Static)
        )


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
            panel.query_one("#file-notes-git-action-status", Static)
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
        assert isinstance(app.messages[1], LibraryFileNotesGitPanel.StageRequested)
        assert app.messages[1].group_ids == (4,)
        assert isinstance(app.messages[2], LibraryFileNotesGitPanel.UnstageRequested)
        assert app.messages[2].group_ids == (8,)
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
        assert not await workspace.open_path("two.md")
        assert "mutation in progress" in _text(
            workspace.query_one("#file-notes-action-status", Static)
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
