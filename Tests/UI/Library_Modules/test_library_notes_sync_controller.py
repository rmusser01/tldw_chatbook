"""Controller tests against a recording lasting-sync runtime port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import signature

import pytest

from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictApplyResult,
    ConflictComparison,
    ConflictSelection,
    NotesSyncConflictChoice,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutionResult
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncAction,
    NotesSyncActionKind,
    NotesSyncOperationState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_reconciler import (
    ReconciliationAttention,
    ReconciliationAttentionKind,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncControlResult,
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
    RuntimeConflictHistoryRow,
    RuntimeConflictReceipt,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)

pytestmark = pytest.mark.asyncio
TOKEN = "b" * 64
TOKEN_2 = "c" * 64


async def test_runtime_snapshot_is_the_only_availability_source() -> None:
    assert "lasting_available" not in signature(LibraryNotesSyncController).parameters


@dataclass
class _ImportController:
    calls: int = 0

    def begin_selection(self) -> None:
        self.calls += 1


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.stale = False
        self.receipts: tuple[RuntimeConflictReceipt, ...] = ()
        self.history: dict[int, tuple[RuntimeConflictHistoryRow, ...]] = {}
        self.comparison_gate: asyncio.Event | None = None
        self.check_plan: ReconciliationPlan | None = None
        self.apply_result: ConflictApplyResult | None = None
        self.comparison_started = asyncio.Event()

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot(
            "active",
            "sync_now",
            (NotesSyncRootRuntimeSnapshot("root-1", "up_to_date", "sync_now"),),
        )

    async def check_root(self, root_id: str) -> ReconciliationPlan:
        self.calls.append(("check_root", root_id))
        return self.check_plan or ReconciliationPlan(
            root_id=root_id,
            observation_token=TOKEN,
            safe_actions=(
                NotesSyncAction("act-1", NotesSyncActionKind.UPDATE_NOTE, "bind-1"),
            ),
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
        )

    async def abandon_setup(self, root_id: str) -> None:
        self.calls.append(("abandon_setup", root_id))

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan:
        self.calls.append(("request_sync_now", root_id))
        return await self.check_root(root_id)

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> object:
        self.calls.append(("resolve_cleanup", root_id, operation_id))
        return object()

    async def apply_reviewed(
        self,
        root_id: str,
        token: str,
        action_ids: tuple[str, ...],
        selections: tuple[ConflictSelection, ...] = (),
    ) -> ConflictApplyResult:
        self.calls.append(("apply_reviewed", root_id, token, action_ids, selections))
        if self.stale:
            raise ValueError("stale_review")
        if self.apply_result is not None:
            return self.apply_result
        result = NotesSyncExecutionResult(
            "operation-1",
            NotesSyncOperationState.COMPLETED,
            False,
        )
        fresh = ReconciliationPlan(
            root_id=root_id,
            observation_token=TOKEN_2,
            safe_actions=(),
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
        )
        return ConflictApplyResult((result,), 1, 0, 0, False, False, False, fresh)

    async def compare_conflict(
        self, root_id: str, token: str, binding_id: str
    ) -> ConflictComparison:
        self.calls.append(("compare_conflict", root_id, token, binding_id))
        self.comparison_started.set()
        if self.comparison_gate is not None:
            await self.comparison_gate.wait()
        return ConflictComparison(
            binding_id=binding_id,
            note_title="Note",
            relative_path="note.md",
            note_version=1,
            note_updated_at=None,
            file_modified_ns=2,
            note_character_count=4,
            note_line_count=1,
            file_character_count=4,
            file_line_count=1,
            diff="",
            input_elided=False,
            output_elided=False,
        )

    async def active_conflict_receipts(
        self, root_id: str
    ) -> tuple[RuntimeConflictReceipt, ...]:
        self.calls.append(("active_conflict_receipts", root_id))
        return self.receipts

    def dismiss_conflict_receipt(self, root_id: str, operation_id: str) -> None:
        self.calls.append(("dismiss_conflict_receipt", root_id, operation_id))
        self.receipts = tuple(
            receipt for receipt in self.receipts if receipt.operation_id != operation_id
        )

    async def undo_resolution(
        self, root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        self.calls.append(("undo_resolution", root_id, operation_id))
        self.receipts = tuple(
            receipt for receipt in self.receipts if receipt.operation_id != operation_id
        )
        return NotesSyncExecutionResult(
            "undo-operation-1",
            NotesSyncOperationState.COMPLETED,
            False,
        )

    async def resolution_history(
        self,
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        self.calls.append(("resolution_history", root_id, limit, offset))
        return self.history.get(offset, ())

    async def activate_root(
        self, root_id: str, authorization: object
    ) -> NotesSyncControlResult:
        self.calls.append(("activate_root", root_id, authorization))
        return NotesSyncControlResult(False, "needs_attention", "review_settings")

    async def pause_root(self, root_id: str) -> NotesSyncControlResult:
        self.calls.append(("pause_root", root_id))
        return NotesSyncControlResult(True, "paused", "resume_sync")

    async def resume_root(self, root_id: str) -> NotesSyncControlResult:
        self.calls.append(("resume_root", root_id))
        return NotesSyncControlResult(False, "needs_attention", "review_settings")

    async def retarget_root(self, root_id: str, target: str) -> NotesSyncControlResult:
        self.calls.append(("retarget_root", root_id, target))
        return NotesSyncControlResult(False, "needs_attention", "review_settings")

    async def disconnect_root(self, root_id: str, keep: bool) -> NotesSyncControlResult:
        self.calls.append(("disconnect_root", root_id, keep))
        return NotesSyncControlResult(False, "needs_attention", "review_settings")


def _conflict_plan(
    *,
    root_id: str = "root-1",
    token: str = TOKEN,
    bindings: tuple[str, ...] = ("bind-1",),
    safe: bool = False,
    page_size: int = 100,
) -> ReconciliationPlan:
    return ReconciliationPlan(
        root_id=root_id,
        observation_token=token,
        safe_actions=(
            (NotesSyncAction("act-1", NotesSyncActionKind.UPDATE_NOTE, "bind-safe"),)
            if safe
            else ()
        ),
        attention=tuple(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                binding_id,
            )
            for binding_id in bindings
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
        page_size=page_size,
    )


async def test_import_once_routes_exactly_once_to_existing_import_controller() -> None:
    runtime = _Runtime()
    importer = _ImportController()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=importer,
    )

    assert controller.choose_relationship("import_once") == "import"
    assert importer.calls == 1
    assert runtime.calls == []


async def test_refresh_after_runtime_start_updates_availability_and_candidates() -> (
    None
):
    runtime = _Runtime()
    current = NotesSyncRuntimeSnapshot("starting", "wait")
    runtime.snapshot = lambda: current
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    assert controller.snapshot.lasting_available is False
    assert controller.snapshot.roots == ()

    current = NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        (
            NotesSyncRootRuntimeSnapshot(
                "legacy-root-" + "a" * 40,
                "paused",
                "review_migration",
            ),
        ),
    )
    controller.refresh_roots()

    assert controller.snapshot.lasting_available is True
    assert controller.snapshot.roots[0].next_action == "review_migration"


async def test_check_and_apply_reviewed_use_observation_token_and_selected_actions() -> (
    None
):
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.check_root("root-1")
    await controller.apply_reviewed()

    assert runtime.calls == [
        ("check_root", "root-1"),
        ("apply_reviewed", "root-1", TOKEN, ("act-1",), ()),
        ("active_conflict_receipts", "root-1"),
    ]
    assert controller.snapshot.phase == "receipt"
    assert "1 applied" in controller.snapshot.receipt_line


async def test_migration_review_is_activation_typed_and_uses_activate_path() -> None:
    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        (
            NotesSyncRootRuntimeSnapshot(
                "legacy-root-" + "a" * 40,
                "paused",
                "review_migration",
            ),
        ),
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    root_id = "legacy-root-" + "a" * 40

    await controller.check_migration(root_id)
    await controller.activate_root(root_id)

    assert controller.snapshot.review.activation is True
    assert runtime.calls == [
        ("check_root", root_id),
        ("activate_root", root_id, TOKEN),
    ]


async def test_stale_review_returns_to_review_with_check_again_and_does_not_retry() -> (
    None
):
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    runtime.stale = True

    await controller.apply_reviewed()

    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.next_action == "Check again"
    assert [call[0] for call in runtime.calls].count("apply_reviewed") == 1


async def test_stale_review_paging_cannot_restore_apply_eligibility() -> None:
    runtime = _Runtime()

    async def paged(root_id: str) -> ReconciliationPlan:
        return ReconciliationPlan(
            root_id=root_id,
            observation_token=TOKEN,
            safe_actions=tuple(
                NotesSyncAction(
                    f"act-{index}", NotesSyncActionKind.UPDATE_NOTE, f"bind-{index}"
                )
                for index in range(3)
            ),
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
            page_size=1,
        )

    runtime.check_root = paged
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    runtime.stale = True
    await controller.apply_reviewed()

    controller.set_review_page(2)

    assert controller.snapshot.review.page == 2
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.next_action == "Check again"


async def test_root_controls_are_explicit_and_disconnect_copy_promises_no_deletion() -> (
    None
):
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.pause_root("root-1")
    await controller.retarget_root("root-1", "/tmp/new")
    await controller.disconnect_root("root-1", keep_folder_organization=True)

    assert runtime.calls[-3:] == [
        ("pause_root", "root-1"),
        ("retarget_root", "root-1", "/tmp/new"),
        ("disconnect_root", "root-1", True),
    ]
    assert "never deletes files or notes" in controller.snapshot.status_line


async def test_inert_controller_never_calls_activation() -> None:
    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "awaiting_cutover", "finish_upgrade"
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    result = await controller.activate_root("root-1")

    assert result is False
    assert runtime.calls == []
    assert "unavailable" in controller.snapshot.status_line.casefold()


async def test_active_fake_publishes_activating_before_truthful_review_required_result() -> (
    None
):
    runtime = _Runtime()
    phases: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: phases.append(snapshot.phase),
    )

    accepted = await controller.activate_root("root-1")

    assert accepted is False
    assert "activating" in phases
    assert controller.snapshot.phase == "roots"
    assert "needs attention" in controller.snapshot.status_line.casefold()


async def test_activation_failure_is_bounded_redacted_and_leaves_activating_phase() -> (
    None
):
    runtime = _Runtime()

    async def fail(_root_id: str, _authorization: object) -> NotesSyncControlResult:
        raise RuntimeError("/Users/private/notes.md secret body")

    runtime.activate_root = fail
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    accepted = await controller.activate_root("root-1")

    assert accepted is False
    assert controller.snapshot.phase == "review"
    assert "failed" in controller.snapshot.status_line.casefold()
    assert "/Users" not in repr(controller.snapshot)
    assert "secret body" not in repr(controller.snapshot)


@pytest.mark.parametrize(
    ("status", "next_action", "label", "action_label"),
    (
        ("active", "sync_now", "Active", "Check changes"),
        ("awaiting_cutover", "finish_upgrade", "Awaiting Cutover", "Finish upgrade"),
        ("up_to_date", "sync_now", "Up to date", "Check changes"),
        ("paused", "resume_sync", "Paused", "Resume"),
        ("offline", "reconnect_folder", "Offline", "Reconnect folder"),
        ("passive", "open_active_process", "another process", "Open active process"),
        ("failed", "review_changes", "Failed", "Review changes"),
        ("partial", "review_changes", "Partial", "Review changes"),
        ("stopped", "review_settings", "Stopped", "Review settings"),
        ("stopping", "wait", "Stopping", "Wait"),
    ),
)
async def test_root_status_projection_names_bounded_next_action(
    status: str, next_action: str, label: str, action_label: str
) -> None:
    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        (NotesSyncRootRuntimeSnapshot("root:opaque.v1", status, next_action),),
    )

    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    root = controller.snapshot.roots[0]
    assert label in root.status_label
    assert root.next_action_label == action_label
    assert "/" not in repr(root)


async def test_runtime_failure_is_bounded_redacted_and_leaves_checking_phase() -> None:
    runtime = _Runtime()

    async def fail(_root_id: str):
        raise RuntimeError("/Users/private/notes.md secret body")

    runtime.check_root = fail
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.check_root("root-1")

    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.next_action == "Check again"
    assert "failed" in controller.snapshot.status_line.casefold()
    assert "/Users" not in repr(controller.snapshot)
    assert "secret body" not in repr(controller.snapshot)


async def test_control_failure_publishes_explicit_bounded_recovery_action() -> None:
    runtime = _Runtime()

    async def fail(_root_id: str):
        raise RuntimeError("/private/root leaked")

    runtime.pause_root = fail
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.pause_root("root-1")

    assert controller.snapshot.phase == "roots"
    assert "Review root status" in controller.snapshot.status_line
    assert "/private/root" not in repr(controller.snapshot)


async def test_root_list_is_bounded_and_pageable_without_paths() -> None:
    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        tuple(
            NotesSyncRootRuntimeSnapshot(f"root-{index}", "up_to_date", "sync_now")
            for index in range(25)
        ),
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    assert len(controller.snapshot.roots) == 20
    assert controller.snapshot.root_page_count == 2
    controller.set_root_page(2)
    assert len(controller.snapshot.roots) == 5
    assert controller.snapshot.root_page == 2


async def test_rejected_resume_is_not_reported_as_success() -> None:
    controller = LibraryNotesSyncController(
        runtime=_Runtime(),
        import_controller=_ImportController(),
    )

    await controller.resume_root("root-1")

    assert controller.snapshot.phase == "roots"
    assert "needs attention" in controller.snapshot.status_line.casefold()


async def test_controller_privately_applies_all_safe_actions_across_review_pages() -> (
    None
):
    runtime = _Runtime()

    async def many(root_id: str) -> ReconciliationPlan:
        return ReconciliationPlan(
            root_id=root_id,
            observation_token=TOKEN,
            safe_actions=tuple(
                NotesSyncAction(
                    f"act-{index}", NotesSyncActionKind.UPDATE_NOTE, f"bind-{index}"
                )
                for index in range(101)
            ),
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
            page_size=2,
        )

    runtime.check_root = many
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")

    await controller.apply_reviewed()

    apply_call = next(call for call in runtime.calls if call[0] == "apply_reviewed")
    assert len(apply_call[3]) == 101
    assert len(repr(controller.snapshot)) < 1_000


async def test_manual_sync_now_and_operation_recovery_use_existing_runtime_methods() -> (
    None
):
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.sync_now("root-1")
    await controller.resolve_cleanup("root-1", "operation-1")

    assert ("request_sync_now", "root-1") in runtime.calls
    assert ("resolve_cleanup", "root-1", "operation-1") in runtime.calls
    assert "Recovery reviewed" in controller.snapshot.status_line


async def test_not_configured_runtime_offers_lasting_setup() -> None:
    """TASK-21112: a boot-deferred runtime must still offer first-time setup."""

    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot("not_configured", "none")
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    assert controller.snapshot.lasting_available is True
    assert controller.snapshot.roots == ()
    assert controller.choose_relationship("keep_synced") == "configure"


async def test_not_configured_refresh_keeps_setup_available() -> None:
    runtime = _Runtime()
    current = NotesSyncRuntimeSnapshot("starting", "wait")
    runtime.snapshot = lambda: current
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    assert controller.snapshot.lasting_available is False

    current = NotesSyncRuntimeSnapshot("not_configured", "none")
    controller.refresh_roots()

    assert controller.snapshot.lasting_available is True


async def test_conflict_selection_stages_without_runtime_and_survives_paging() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan(bindings=("bind-1", "bind-2"), page_size=1)
    publications: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: publications.append(snapshot.status_line),
    )
    await controller.check_root("root-1")
    calls_before = tuple(runtime.calls)
    publications.clear()

    controller.stage_attention_choice("bind-1", "Keep both")

    assert tuple(runtime.calls) == calls_before
    assert publications == ["Choice staged. No changes yet."]
    assert controller.snapshot.review.rows[0].selected_choice is (
        NotesSyncConflictChoice.KEEP_BOTH
    )
    controller.set_review_page(2)
    assert controller.snapshot.review.rows[0].item_id == "bind-2"
    controller.set_review_page(1)
    assert controller.snapshot.review.rows[0].selected_label == "Selected: Keep both"


async def test_selection_is_keyed_by_observation_token_and_clears_on_stale_back_root_and_remount() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("bind-1", "Keep file")
    assert controller._selections == {  # noqa: SLF001 - token key is the contract
        (TOKEN, "bind-1"): NotesSyncConflictChoice.KEEP_FILE
    }

    runtime.check_plan = _conflict_plan(token=TOKEN_2)
    await controller.check_root("root-1")
    assert controller.snapshot.review.rows[0].selected_choice is None

    controller.stage_attention_choice("bind-1", "Keep note")
    runtime.stale = True
    await controller.apply_reviewed()
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.rows[0].selected_choice is None

    runtime.stale = False
    runtime.check_plan = _conflict_plan(root_id="root-2", token=TOKEN)
    await controller.check_root("root-2")
    controller.stage_attention_choice("bind-1", "Keep both")
    await controller.abandon_setup()
    assert controller.snapshot.review.rows[0].selected_choice is None

    remounted = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    assert remounted.snapshot.review.rows == ()


async def test_comparison_return_and_page_change_release_only_comparison() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan(bindings=("bind-1", "bind-2"), page_size=1)
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("bind-1", "Keep file")

    await controller.show_conflict_comparison("bind-1")
    assert controller.snapshot.comparison is not None
    controller.return_to_conflict_choices()
    assert controller.snapshot.comparison is None
    assert controller.snapshot.review.rows[0].selected_choice is (
        NotesSyncConflictChoice.KEEP_FILE
    )

    await controller.show_conflict_comparison("bind-1")
    controller.set_review_page(2)
    assert controller.snapshot.comparison is None
    controller.set_review_page(1)
    assert controller.snapshot.review.rows[0].selected_choice is (
        NotesSyncConflictChoice.KEEP_FILE
    )


async def test_delayed_comparison_cannot_publish_after_controller_remount_invalidation() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    comparison_calls = 0

    async def delayed(root_id: str, token: str, binding_id: str) -> ConflictComparison:
        nonlocal comparison_calls
        comparison_calls += 1
        call = comparison_calls
        (first_started if call == 1 else second_started).set()
        await (release_first if call == 1 else release_second).wait()
        value = await _Runtime().compare_conflict(root_id, token, binding_id)
        return ConflictComparison(
            binding_id=value.binding_id,
            note_title="Old" if call == 1 else "Fresh",
            relative_path=value.relative_path,
            note_version=value.note_version,
            note_updated_at=value.note_updated_at,
            file_modified_ns=value.file_modified_ns,
            note_character_count=value.note_character_count,
            note_line_count=value.note_line_count,
            file_character_count=value.file_character_count,
            file_line_count=value.file_line_count,
            diff=value.diff,
            input_elided=value.input_elided,
            output_elided=value.output_elided,
        )

    runtime.compare_conflict = delayed
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")

    pending = asyncio.create_task(controller.show_conflict_comparison("bind-1"))
    await first_started.wait()
    controller.invalidate_for_remount()
    current = asyncio.create_task(controller.show_conflict_comparison("bind-1"))
    await second_started.wait()
    release_second.set()
    await current
    assert controller.snapshot.comparison is not None
    assert controller.snapshot.comparison.note_title == "Fresh"
    release_first.set()
    await pending

    assert controller.snapshot.comparison is not None
    assert controller.snapshot.comparison.note_title == "Fresh"


async def test_apply_sends_safe_ids_and_token_typed_conflict_choices() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan(safe=True)
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("bind-1", "Keep note")

    await controller.apply_reviewed()

    apply_call = next(call for call in runtime.calls if call[0] == "apply_reviewed")
    assert apply_call == (
        "apply_reviewed",
        "root-1",
        TOKEN,
        ("act-1",),
        (ConflictSelection("bind-1", NotesSyncConflictChoice.KEEP_NOTE),),
    )


async def test_terminal_subset_projects_receipts_fresh_review_and_one_status_update() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan(safe=True)
    result = NotesSyncExecutionResult(
        "operation-1", NotesSyncOperationState.COMPLETED, False
    )
    runtime.apply_result = ConflictApplyResult(
        (result, result),
        1,
        1,
        1,
        True,
        False,
        False,
        _conflict_plan(token=TOKEN_2),
    )
    runtime.receipts = (
        RuntimeConflictReceipt(
            "operation-1",
            "Note · note.md",
            NotesSyncConflictChoice.KEEP_FILE,
            "completed",
            True,
        ),
    )
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("bind-1", "Keep file")
    statuses.clear()

    await controller.apply_reviewed()

    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.observation_token == TOKEN_2
    assert len(controller.snapshot.receipts) == 1
    assert statuses == ["2 applied · 1 conflict remains."]


async def test_nonterminal_apply_routes_to_recovery_without_fresh_review() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    # Rebuild through the executor's canonical choices rather than duplicating them.
    from tldw_chatbook.Notes.notes_sync_executor import NotesSyncRecoveryChoice

    attention = NotesSyncExecutionResult(
        "operation-1",
        NotesSyncOperationState.NEEDS_ATTENTION,
        True,
        "operation_failed",
        tuple(NotesSyncRecoveryChoice),
    )
    runtime.apply_result = ConflictApplyResult(
        (attention,), 0, 0, 1, True, False, True, None
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("bind-1", "Keep file")

    await controller.apply_reviewed()

    assert controller.snapshot.phase == "roots"
    assert "recovery needs attention" in controller.snapshot.status_line


async def test_receipt_undo_dismiss_and_history_paging_use_fresh_runtime_projections() -> (
    None
):
    runtime = _Runtime()
    receipt = RuntimeConflictReceipt(
        "operation-1",
        "Note · note.md",
        NotesSyncConflictChoice.KEEP_BOTH,
        "completed",
        True,
    )
    runtime.receipts = (receipt,)
    runtime.history[0] = (
        RuntimeConflictHistoryRow(
            "operation-1",
            "Note · note.md",
            NotesSyncConflictChoice.KEEP_BOTH,
            "completed",
            "2026-08-22T12:00:00+00:00",
            "2026-08-22T12:00:00+00:00",
            True,
        ),
    )
    runtime.history[100] = (
        RuntimeConflictHistoryRow(
            "operation-2",
            "Other · other.md",
            NotesSyncConflictChoice.KEEP_NOTE,
            "completed",
            "2026-08-21T12:00:00+00:00",
            "2026-08-21T12:00:00+00:00",
            False,
            "Undo expired",
        ),
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.refresh_conflict_receipts("root-1")
    assert controller.snapshot.receipts[0].operation_id == "operation-1"
    await controller.dismiss_conflict_receipt("root-1", "operation-1")
    assert controller.snapshot.receipts == ()

    runtime.receipts = (receipt,)
    await controller.undo_conflict_resolution("root-1", "operation-1")
    assert controller.snapshot.receipts == ()
    assert ("undo_resolution", "root-1", "operation-1") in runtime.calls

    await controller.show_resolution_history("root-1", page=1)
    assert controller.snapshot.phase == "history"
    assert controller.snapshot.history.rows[0].operation_id == "operation-1"
    await controller.show_resolution_history("root-1", page=2)
    assert controller.snapshot.history.page == 2
    assert controller.snapshot.history.rows[0].operation_id == "operation-2"
    assert ("resolution_history", "root-1", 100, 100) in runtime.calls
