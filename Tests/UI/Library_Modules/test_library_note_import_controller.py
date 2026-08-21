from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_chatbook.Notes.note_import_execution_models import (
    ImportExecutionReceipt,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
    ProposedFolderMembership,
)
from tldw_chatbook.UI.Library_Modules.library_note_import_controller import (
    LibraryNoteImportController,
)


BOUNDS = ImportBounds(
    max_files=20,
    max_file_bytes=1024 * 1024,
    max_total_bytes=4 * 1024 * 1024,
    max_depth=8,
)


def _plan(path: Path) -> NoteImportPlan:
    item = ImportPreviewItem(
        item_id="item-000001",
        source=ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path=path.name,
            source_path=path,
        ),
        payloads=(ParsedNotePayload(title="One", content="Body"),),
        memberships=(
            ProposedFolderMembership(payload_index=0, folder_segments=("Inbox",)),
        ),
        classification=ImportClassification.NEW,
        reason="New source",
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
        match=None,
        replace_content=False,
        add_membership=True,
    )
    return NoteImportPlan(
        bounds=BOUNDS,
        items=(item,),
        proposed_folder_paths=(("Inbox",),),
    )


class _FolderRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def list_children(self, *, parent_id: None, limit: int, offset: int):
        assert parent_id is None
        self.calls.append((limit, offset))
        folders = (
            (SimpleNamespace(name="Inbox"), SimpleNamespace(name="Reference"))
            if offset == 0
            else (SimpleNamespace(name="Later"),)
        )
        return SimpleNamespace(
            folders=folders,
            next_folder_offset=2 if offset == 0 else None,
        )


class _Executor:
    def __init__(
        self,
        calls: list[object],
        receipt: ImportExecutionReceipt,
        error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.receipt = receipt
        self.error = error

    async def execute_async(self, approved, *, cancel_event, progress_callback):
        self.calls.append(("execute", approved, cancel_event))
        if self.error is not None:
            raise self.error
        return replace(self.receipt, approval_id=approved.approval_id)

    def retry_failed(self, approved, *, cancel_event, progress_callback):
        self.calls.append(("retry", approved, cancel_event))
        return replace(self.receipt, approval_id=approved.approval_id)


def _receipt(*, retryable: int = 0) -> ImportExecutionReceipt:
    failed = retryable
    state = (
        ImportSessionState.NEEDS_ATTENTION if failed else ImportSessionState.COMPLETED
    )
    return ImportExecutionReceipt(
        approval_id=str(uuid4()),
        state=state,
        total=1,
        completed=1,
        imported=0 if failed else 1,
        updated=0,
        skipped=0,
        failed=failed,
        retryable=retryable,
    )


def _controller(
    *,
    plan: NoteImportPlan,
    calls: list[object],
    repository: _FolderRepository,
    receipt: ImportExecutionReceipt | None = None,
    executor_error: Exception | None = None,
    planning_error: Exception | None = None,
) -> LibraryNoteImportController:
    published: list[object] = []
    executor = _Executor(calls, receipt or _receipt(), executor_error)

    def discover(paths, bounds):
        calls.append(("discover", tuple(paths), bounds))
        if planning_error is not None:
            raise planning_error
        return "discovery"

    def parse(discovery, bounds, *, destination_folder_segments=None):
        calls.append(
            ("parse", discovery, bounds, tuple(destination_folder_segments or ()))
        )
        return "batch"

    def classify(batch, bounds, *, prior_observations=()):
        calls.append(("classify", batch, bounds, tuple(prior_observations)))
        return plan

    class Receipts:
        def prior_observations_for_plan_read_only(self, preview):
            calls.append(("observe-read-only", preview))
            return ("prior",)

    def analyze(preview, names):
        calls.append(("analyze", preview, tuple(names)))
        return preview

    def approve(preview):
        approved = approve_note_import_plan(preview)
        calls.append(("approve", preview, approved))
        return approved

    controller = LibraryNoteImportController(
        bounds=BOUNDS,
        database=lambda: "database",
        folder_repository=lambda: repository,
        receipt_repository=lambda: Receipts(),
        discover_import_sources=discover,
        parse_import_sources=parse,
        classify_import_batch=classify,
        analyze_root_collision=analyze,
        resolve_root_collision=lambda *args, **kwargs: args[0],
        confirm_uncertain_match=lambda preview, item_id: preview,
        apply_item_override=lambda preview, item_id, action, **kwargs: preview,
        approve_note_import_plan=approve,
        executor_factory=lambda db, folders, receipts: executor,
        publish_snapshot=published.append,
        refresh_after_settlement=lambda: calls.append("refresh"),
    )
    controller._published_for_test = published
    return controller


@pytest.mark.asyncio
async def test_check_runs_exact_read_only_planning_sequence_and_pages_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    calls: list[object] = []
    repository = _FolderRepository()
    controller = _controller(plan=_plan(source), calls=calls, repository=repository)

    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()

    assert [entry[0] for entry in calls if isinstance(entry, tuple)] == [
        "discover",
        "parse",
        "classify",
        "observe-read-only",
        "classify",
        "analyze",
    ]
    assert repository.calls == [(500, 0), (500, 2)]
    assert controller.snapshot.phase.value == "review"


@pytest.mark.asyncio
async def test_execution_crosses_approval_only_on_import_and_uses_exact_object(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    calls: list[object] = []
    controller = _controller(
        plan=_plan(source), calls=calls, repository=_FolderRepository()
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()
    assert not any(call[0] == "approve" for call in calls if isinstance(call, tuple))

    await controller.approve_and_execute()

    approved = next(
        call[2] for call in calls if isinstance(call, tuple) and call[0] == "approve"
    )
    executed = next(
        call[1] for call in calls if isinstance(call, tuple) and call[0] == "execute"
    )
    assert executed is approved
    assert controller.snapshot.phase.value == "receipt"
    assert calls[-1] == "refresh"


def test_file_accumulation_and_folder_selection_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    controller = _controller(
        plan=_plan(tmp_path / "one.md"),
        calls=calls,
        repository=_FolderRepository(),
    )
    controller.begin_selection()
    controller.accept_selected_path(tmp_path / "one.md", is_folder=False)
    controller.accept_selected_path(tmp_path / "two.md", is_folder=False)
    assert controller.snapshot.selected_count == 2
    assert controller.snapshot.requires_destination is True

    controller.begin_selection()
    controller.accept_selected_path(tmp_path / "folder", is_folder=True)
    assert controller.snapshot.selected_count == 1
    assert controller.snapshot.selection_is_folder is True
    with pytest.raises(ValueError, match="folder"):
        controller.accept_selected_path(tmp_path / "three.md", is_folder=False)


@pytest.mark.asyncio
async def test_retry_uses_retained_approved_plan_and_refreshes_again(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    calls: list[object] = []
    controller = _controller(
        plan=_plan(source),
        calls=calls,
        repository=_FolderRepository(),
        receipt=_receipt(retryable=1),
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()
    await controller.approve_and_execute()
    executed = next(
        call[1] for call in calls if isinstance(call, tuple) and call[0] == "execute"
    )

    await controller.retry_failed()

    retried = next(
        call[1] for call in calls if isinstance(call, tuple) and call[0] == "retry"
    )
    assert retried is executed
    assert calls.count("refresh") == 2


@pytest.mark.asyncio
async def test_executor_failure_restores_review_with_bounded_recovery_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    calls: list[object] = []
    controller = _controller(
        plan=_plan(source),
        calls=calls,
        repository=_FolderRepository(),
        executor_error=RuntimeError("private path must not surface"),
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()

    with pytest.raises(RuntimeError):
        await controller.approve_and_execute()

    assert controller.snapshot.phase.value == "review"
    assert controller.presentation_snapshot.status_line == (
        "Import could not start or finish safely. Review the plan and try again."
    )
    assert "private path" not in repr(controller.presentation_snapshot)


@pytest.mark.asyncio
async def test_planning_failure_returns_to_destination_with_actionable_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    controller = _controller(
        plan=_plan(source),
        calls=[],
        repository=_FolderRepository(),
        planning_error=OSError("secret filename"),
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")

    with pytest.raises(OSError):
        await controller.check()

    assert controller.snapshot.phase.value == "destination"
    assert controller.presentation_snapshot.status_line == (
        "Could not check these sources. Review the selection and try again."
    )
    assert "secret filename" not in repr(controller.presentation_snapshot)


@pytest.mark.asyncio
async def test_review_mutation_preserves_the_current_bounded_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    base = _plan(source)
    items = tuple(
        replace(
            base.items[0],
            item_id=f"item-{number:06d}",
            source=replace(
                base.items[0].source,
                display_path=f"note-{number:02d}.md",
            ),
        )
        for number in range(1, 27)
    )
    plan = replace(base, items=items)
    controller = _controller(plan=plan, calls=[], repository=_FolderRepository())
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()
    controller.set_page(2)

    controller.set_item_action("item-000026", "skip")

    assert controller.snapshot.page.page_number == 2


@pytest.mark.asyncio
async def test_executor_construction_failure_restores_review_and_clears_cancel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    controller = _controller(
        plan=_plan(source), calls=[], repository=_FolderRepository()
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()

    def unavailable(*args):
        raise RuntimeError("private authority detail")

    controller._executor_factory = unavailable
    with pytest.raises(RuntimeError):
        await controller.approve_and_execute()

    assert controller.snapshot.phase.value == "review"
    assert controller._cancel_event is None
    assert "private authority" not in repr(controller.presentation_snapshot)


@pytest.mark.asyncio
async def test_cancel_sets_the_executor_event_and_waits_for_partial_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one.md"
    source.write_text("# One\nBody", encoding="utf-8")
    controller = _controller(
        plan=_plan(source), calls=[], repository=_FolderRepository()
    )
    controller.begin_selection()
    controller.accept_selected_path(source, is_folder=False)
    controller.set_destination("Inbox")
    await controller.check()
    started = asyncio.Event()
    release = asyncio.Event()
    passed_cancel_event = None

    class GatedExecutor:
        async def execute_async(self, approved, *, cancel_event, progress_callback):
            nonlocal passed_cancel_event
            passed_cancel_event = cancel_event
            started.set()
            await release.wait()
            return replace(_receipt(), approval_id=approved.approval_id)

    controller._executor_factory = lambda *args: GatedExecutor()
    execution = asyncio.create_task(controller.approve_and_execute())
    await started.wait()

    controller.cancel()

    assert passed_cancel_event is not None and passed_cancel_event.is_set()
    assert controller.snapshot.phase.value == "importing"
    assert controller.snapshot.cancel_requested is True
    assert controller.presentation_snapshot.can_cancel is False
    assert controller.presentation_snapshot.status_line == (
        "Stopping after the current item…"
    )

    release.set()
    await execution
    assert controller.snapshot.phase.value == "receipt"
